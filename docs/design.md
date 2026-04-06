# hipflex Design

A cdylib loaded via `LD_PRELOAD` that intercepts HIP GPU memory allocation APIs, enforces per-process VRAM limits, and tracks usage through shared memory.

## How It Works

```
Application (hipMalloc, hipFree, …)
        │
        ▼
LD_PRELOAD → libhipflex.so (Frida GUM inline hooks)
        │
        ├─ Reserve-then-allocate (atomic fetch_add on SHM)
        ├─ Call real HIP API in libamdhip64.so
        ├─ Track allocation in DashMap
        └─ On free: remove from DashMap, saturating_fetch_sub on SHM
```

The limiter is transparent to applications. Frameworks see the configured VRAM limit as the total GPU memory (via spoofed `hipMemGetInfo`/`hipDeviceTotalMem`/SMI queries), and allocations that exceed the limit return `hipErrorOutOfMemory`.

## Operating Modes

| Priority | Mode | Trigger | SHM Source | Config Source |
|----------|------|---------|------------|---------------|
| 1 | **Standalone** | `FH_MEMORY_LIMIT` set | Self-created | GPU auto-discovery via KFD sysfs |
| 2 | **Mock/test** | `FH_SHM_FILE` + `FH_VISIBLE_DEVICES` set | Local file | `FH_VISIBLE_DEVICES` env var |

If both `FH_MEMORY_LIMIT` and `FH_SHM_FILE` are set, standalone wins with a warning.

### Standalone Mode

Enables VRAM enforcement with just `LD_PRELOAD` + `FH_MEMORY_LIMIT` — no daemon, no config files, no kernel modules.

Init flow:
1. Parse `FH_MEMORY_LIMIT` via `size_parser::parse_memory_limit()` → bytes
2. Enumerate all visible GPUs via KFD sysfs (`/sys/class/kfd/kfd/topology/nodes/`), falling back to HIP runtime if sysfs fails
3. Build `DeviceConfig` per GPU with `mem_limit` from env var, `up_limit: 100`
4. Create SHM via `SharedMemoryHandle::create(shm_path, &configs)` at `{SHM_PATH}/shm` (default `/dev/shm/hipflex/shm`). If SHM already exists, joins it without reinitializing — preserving runtime state from concurrent processes
5. Claim a proc slot for per-process overhead tracking
6. Install Frida GUM hooks on `libamdhip64.so`

**SHM multi-process safety:** `create()` handles the race where the segment already exists by opening it instead of failing. Only the first creator writes initial state; subsequent joiners preserve existing runtime counters so concurrent processes don't stomp each other's `pod_memory_used`.

### Mock/Test Mode

Reads `FH_SHM_FILE` and `FH_VISIBLE_DEVICES` env vars. SHM is lazily opened on first hook invocation via `OnceCell`. Used by gpu-tests for test harness setup.

## Init Flow

1. **Library load** — `#[ctor] entry_point()` runs when `LD_PRELOAD` loads the cdylib. If `FH_ENABLE_HOOKS=false`, returns immediately (full passthrough). Otherwise, installs the `dlsym` hook and returns — no logging, no limiter init, no HIP hook installation. This is critical: logging setup during `.init_array` corrupts HIP/ROCr internal state, breaking rocFFT's JIT kernel compilation.
2. **Deferred init** — on the first `dlsym` call for a HIP or SMI symbol (after `.init_array` completes), the `dlsym` detour triggers `init_hooks()` which runs the full init sequence.
3. **Logging** — `logging::init()` sets up the tracing subscriber. Must run after `.init_array` completes.
4. **Config resolution** — selects operating mode per the priority table above.
5. **Device mapping** — enumerates GPUs, matches against config UUIDs by PCI BDF normalization.
6. **SHM attach** — created and injected eagerly in standalone mode, lazily opened in mock mode.
7. **Hook installation** — creates a Frida GUM `HookManager`, replaces 24 symbols in `libamdhip64.so` via inline hooks (15 alloc + 7 free + 2 info spoofing). The remaining 4 hooks (3 SMI spoofing + 1 `dlsym`) operate at the `dlsym`-interception level. Guarded by `catch_unwind` to prevent panics from crashing the host application. If `libamdhip64.so` is not yet loaded, inline hook installation is skipped; subsequent `dlsym` calls retry until the library appears.

## Reserve-Then-Allocate

Eliminates the TOCTOU race in check-then-allocate:

```
1. fetch_add(size) on pod_memory_used     → atomically reserve space
2. if new_total > limit → fetch_sub(size) → roll back, return OOM
3. call real hipMalloc (native allocator)
4. if native fails → fetch_sub(size)      → roll back reservation
5. on success → insert (pointer, size) into DashMap
```

The under-utilization window (between reserve and native call) is bounded by one allocation's duration. This is the accepted tradeoff for eliminating overcommit.

**Pitched allocations** (hipMallocPitch, hipMemAllocPitch, hipMalloc3D) use a two-phase variant: reserve an estimate (width × height), call native to learn actual pitch, then reserve the extra difference (pitch − width) × height. If the extra pushes over the limit, the initial reservation is rolled back and the native allocation is freed.

**Free path** uses conservative ordering: call native free first, then decrement SHM via `saturating_fetch_sub`. A crash between free and decrement causes over-reporting (safe direction — prevents overcommit, never allows silent over-allocation).

## KFD Sysfs Overhead Tracking

Not all GPU VRAM usage goes through HIP allocation APIs. The ROCm runtime stack and kernel driver allocate memory for code objects, scratch buffers, page tables, and HSA state that is invisible to the limiter's hooks. On MI325X with PyTorch workloads, this overhead ranges from 3.5–9 GiB per process.

**How it works:**

1. **KFD sysfs measurement** — `read_kfd_vram_for_pid()` reads `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>` to get per-process per-GPU physical VRAM from the kernel.
2. **Overhead calculation** — For each mapped device: `non_hip_bytes = vram_resident - tracked_hipMalloc`.
3. **Per-process proc_slots** — Each process writes its per-device overhead into a slot in the proc_slots SHM segment (128 slots, each with PID + `[AtomicU64; MAX_DEVICES]`).
4. **Effective limit** — `effective_mem_limit = mem_limit - total_overhead` (summed across all live processes). The `try_reserve` fast path uses `effective_mem_limit` when non-zero, falling back to `mem_limit` before the first reconciliation.

**All-devices reconciliation:** A single KFD sysfs read and DashMap pass updates overhead and effective limits for ALL mapped devices — not just the device being allocated on. This prevents stale effective limits on multi-GPU setups.

## Process Lifecycle

**Clean exit:** An `atexit` handler (`drain_allocations`) iterates the process-local `allocation_tracker`, aggregates per-device totals, and does one bulk `saturating_fetch_sub` per device. This handles frameworks like PyTorch whose caching allocator may not free all GPU memory before process exit.

**Abnormal exit (SIGKILL, OOM-kill):** The `atexit` handler never runs, leaving `pod_memory_used` inflated. The proc_slots mechanism handles this — on each new process init (and on OOM in `try_reserve`), `reap_dead_pids()`:
1. Scans all claimed slots for PIDs that no longer exist (`kill(pid, 0)` liveness check)
2. Atomically claims dead slots via CAS (prevents double-reap by concurrent processes)
3. Subtracts the dead process's tracked per-device usage from `pod_memory_used`
4. Recalculates effective limits for all mapped devices
5. Zeros the slot for reuse

## Atomics

**Reservation (alloc path):**
```rust
let previous = pod_memory_used.fetch_add(size, Ordering::AcqRel);
if previous.saturating_add(size) > limit {
    pod_memory_used.fetch_sub(size, Ordering::AcqRel);
    return Err(OverLimit);
}
```

`MAX_ALLOC_SIZE = u64::MAX / 2` guard rejects pathological sizes before `fetch_add` to prevent transient wrapping.

**Free path (saturating CAS loop):**
```rust
loop {
    let current = pod_memory_used.load(Ordering::Acquire);
    let new_value = current.saturating_sub(size);
    match pod_memory_used.compare_exchange_weak(
        current, new_value, Ordering::AcqRel, Ordering::Acquire
    ) {
        Ok(_) => break,
        Err(_) => continue,
    }
}
```

Saturating subtraction prevents underflow wrapping. Logs a warning if underflow would have occurred.

## Shared Memory Layout

```
SharedDeviceStateV2 (35632 bytes total):
  devices: [DeviceEntryV2; 16]       — per-device state (16 × 144 = 2304 bytes)
    uuid: [u8; 64]                   — GPU UUID string
    device_info: SharedDeviceInfoV2 (72 bytes)
      up_limit: AtomicU32           — utilization percentage
      mem_limit: AtomicU64          — VRAM limit in bytes
      total_cuda_cores: AtomicU32   — compute cores
      pod_memory_used: AtomicU64    — current usage (read/write)
      erl_*: 4 × AtomicU64         — reserved for future compute enforcement
      effective_mem_limit: AtomicU64 — mem_limit minus non-HIP overhead (0 = not yet computed)
    is_active: AtomicU32             — device active flag
  device_count: AtomicU32
  last_heartbeat: AtomicU64
  padding: [u8; 512]
```

SHM path: `{SHM_PATH}/shm` (default `/dev/shm/hipflex/shm`). On tmpfs, cleaned up on reboot. In containers, cleaned up on container termination.

## Allocation Tracker

`DashMap<usize, (usize, u64)>` — maps pointer address → (device_idx, allocation_size).

- Process-local: pointer addresses are virtual, only meaningful within one process
- The SHM `pod_memory_used` counter is the cross-process source of truth
- All pointer types share the same keyspace (device VA, host pointers, `hipArray_t`, `hipMipmappedArray_t`, `hipMemGenericAllocationHandle_t`) — their address spaces don't collide
- Zero-size allocations succeed without tracking (matches native HIP behavior)
- Free of unknown pointer is a no-op (handles zero-size ptrs and double-free gracefully)

## Device Mapping

GPUs are discovered via KFD sysfs (`/sys/class/kfd/kfd/topology/nodes/`) — fork-safe, no HIP runtime init required, and respects `/dev/dri/renderD*` restrictions.

AMD GPU UUIDs are PCI BDF-based. `normalize_uuid_to_bdf()` unifies naming conventions by lowercasing and stripping the `amd-gpu-` prefix. Device resolution at runtime is cached in a `DashMap`.

## Failure Modes

| Scenario | Behavior |
|----------|----------|
| **SHM unavailable at runtime** | Hook falls through to the native call (passthrough). Subsequent calls retry. |
| **SHM unavailable during free** | Pointer removed from tracker but `pod_memory_used` never decremented — permanent accounting leak for that allocation. |
| **`libamdhip64.so` not loaded** | Hooks deferred until `dlsym` resolves a HIP symbol. If never loaded, the limiter is a silent no-op. |
| **Hook installation panic** | Caught by `catch_unwind`. Application continues without enforcement. |
| **Crash between free and decrement** | Over-reports memory (safe direction). Requires restart to reset. |
| **Invalid `FH_MEMORY_LIMIT`** | Limiter not initialized, all hooks passthrough. |
| **`FH_MEMORY_LIMIT` with no visible GPUs** | Limiter not initialized. Logged as error. |
| **SHM directory not writable** | Limiter not initialized, passthrough. |

## Components

| File | Purpose |
|------|---------|
| `hipflex.rs` | Entry point, `#[ctor]`, dlsym detour, init orchestration |
| `limiter.rs` | Limiter struct, SHM accounting, device mapping, KFD sysfs reconciliation |
| `config.rs` | PodConfig struct |
| `detour/mem.rs` | 22 Frida inline hooks on `libamdhip64.so` (15 alloc + 7 free), size computation helpers |
| `detour/smi.rs` | SMI spoofing via dlsym-level interception (rocm-smi, amd-smi) |
| `hiplib.rs` | Thin `libloading` wrapper around `libamdhip64.so` for non-hooked HIP queries |
| `kfd.rs` | KFD sysfs GPU discovery, post-init BDF verification |
| `size_parser.rs` | Human-readable memory limit parser (bytes, SI, binary, fractional) |

See [hook-coverage.md](hook-coverage.md) for the full hook inventory and known gaps.
