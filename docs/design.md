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
2. **Deferred init** — triggered by `ensure_init()`, which fires on every LD_PRELOAD export call and every `dlsym` override. Uses atomic guards (`CTOR_COMPLETE`, `INIT_HOOKS_ATTEMPTED`, `HOOKS_INITIALIZED`) with `compare_exchange` for at-most-once semantics. Safe to call from multiple threads and from both entry paths concurrently.
3. **Logging** — `logging::init()` sets up the tracing subscriber. Must run after `.init_array` completes.
4. **Config resolution** — selects operating mode per the priority table above.
5. **Device mapping** — enumerates GPUs, matches against config UUIDs by PCI BDF normalization.
6. **SHM attach** — created and injected eagerly in standalone mode, lazily opened in mock mode.
7. **Hook installation** — creates a Frida GUM `HookManager`, replaces 27 symbols in `libamdhip64.so` via inline hooks (15 alloc + 7 free + 5 info spoofing). The remaining 4 hooks (3 SMI spoofing + 1 `dlsym`) operate at the `dlsym`-interception level. Guarded by `catch_unwind` to prevent panics from crashing the host application. If `libamdhip64.so` is not yet loaded, inline hook installation is skipped; subsequent `dlsym` calls retry until the library appears.

## Interception Paths

There are three independent interception mechanisms, ensuring coverage regardless of how the application resolves HIP symbols:

```
Application
  │
  ├─ PLT (direct link against libamdhip64.so)
  │    → LD_PRELOAD export (27 #[no_mangle] symbols)
  │    → ensure_init() → detour fn → Frida trampoline (real fn)
  │
  ├─ dlsym(handle, "hipMalloc")
  │    → dlsym override
  │    → ensure_init() → returns detour fn pointer
  │
  └─ Internal libamdhip64 calls (e.g. hipMallocPitch → hipMalloc)
       → Frida GUM inline hook (patches function prologue)
       → detour fn → Frida trampoline (real fn)
```

**LD_PRELOAD exports** — 27 `#[no_mangle] pub unsafe extern "C"` functions that intercept PLT-resolved calls. This is the primary path for frameworks like PyTorch that link directly against `libamdhip64.so` and never call `dlsym`. Each export calls `ensure_init()`, then forwards to the corresponding detour function. If hooks aren't initialized yet (e.g., called before `.init_array` completes), the export falls through to the real function via `real_dlsym(RTLD_NEXT, ...)`.

**dlsym override** — intercepts `dlsym` calls and returns detour function pointers for HIP/SMI symbols. This is the path for applications that `dlopen`/`dlsym` the HIP library at runtime (e.g., SMI tools). Also triggers `ensure_init()`.

**Frida GUM inline hooks** — patches the prologue of functions inside `libamdhip64.so` to redirect to detour functions. This catches internal call chains within the HIP runtime (e.g., `hipMallocPitch` calling `hipMalloc` internally). The Frida trampoline preserves the original function entry point for calling the real implementation.

All three paths converge on the same detour functions — accounting and enforcement logic is written once. The detour calls the original function through the Frida trampoline (original function entry point), not back through the export symbol, preventing infinite recursion.

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

Not all GPU VRAM usage goes through HIP allocation APIs. The ROCm runtime stack and kernel driver allocate memory for code objects, scratch buffers, page tables, and HSA state that is invisible to the limiter's hooks. On MI325X with PyTorch workloads, this overhead is typically 1–3 GiB per process, reaching up to ~10 GiB for test suites that compile hundreds of unique GPU kernels (e.g., inductor, dynamo).

### Why This VRAM Is Invisible to HIP Hooks

Compiled GPU kernels (code objects) are loaded into VRAM through an entirely separate path that bypasses all public HIP memory APIs. The full call chain from ROCm source (`refs/rocm-systems/`):

```
hipModuleLoadData()                                    [clr/hipamd/src/hip_module.cpp]
  → PlatformState::loadModule()                        [clr/hipamd/src/hip_platform.cpp]
    → DynCO::loadCodeObject()                          [clr/hipamd/src/hip_code_object.cpp]
      → Program::setKernels()                          [clr/rocclr/device/rocm/rocprogram.cpp]
        → hsa_executable_load_agent_code_object()      [rocr-runtime/.../hsa.cpp]
          → ExecutableImpl::LoadSegmentV1()             [rocr-runtime/.../executable.cpp]
            → LoaderContext::SegmentAlloc()             [rocr-runtime/.../amd_loader_context.cpp]
              → Runtime::AllocateMemory()              [HSA internal, NOT hsa_amd_memory_pool_allocate]
                → KfdDriver::AllocateKfdMemory()       [rocr-runtime/.../amd_kfd_driver.cpp]
                  → hsaKmtAllocMemory()                [KFD ioctl: AMDKFD_IOC_ALLOC_MEMORY_OF_GPU]
```

There is no `hipMalloc`, `hipMemCreate`, or any public HIP allocation API anywhere in this chain. The HSA runtime's loader uses its internal `LoaderContext::SegmentAlloc()` which calls the KFD kernel driver directly via `hsaKmtAllocMemory()`. This is architecturally separate from user-facing allocations — even perfect HIP API hooking cannot observe these allocations.

The same KFD ioctl path is used for scratch buffers (wavefront spill memory), page tables, and HSA queue state. All are allocated by the runtime/driver stack, not by application code.

**Measured impact** (MI325X, compiled with `torch.compile`):

| Workload | Unique GPU kernels | non-hipMalloc overhead |
|----------|-------------------|----------------------|
| Single model inference (Qwen 7B) | ~20–30 | 1.2–1.3 GiB |
| PyTorch CI test suite (inductor, dynamo, nn) | Hundreds–thousands | 5–10 GiB |

Each compiled kernel contributes ~1–50 MiB as a code object in VRAM. The overhead scales with kernel diversity, not model size.

**Why hooking `hipModuleLoadData` is not feasible:** The function receives the binary image (ELF bytes), not the VRAM footprint. The actual VRAM consumed depends on segment layout, alignment, and page table overhead determined by the HSA loader. Denying the call would prevent the workload from running entirely. The reconciliation approach — measuring actual VRAM via KFD sysfs and tightening `effective_mem_limit` — handles all overhead sources uniformly without needing to intercept internal loading paths.

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
| `detour/mem.rs` | 27 Frida inline hooks (15 alloc + 7 free + 5 info spoofing), 27 LD_PRELOAD exports, size computation helpers |
| `detour/smi.rs` | SMI spoofing via dlsym-level interception (rocm-smi, amd-smi) |
| `hiplib.rs` | Thin `libloading` wrapper around `libamdhip64.so` for non-hooked HIP queries |
| `kfd.rs` | KFD sysfs GPU discovery, post-init BDF verification |
| `size_parser.rs` | Human-readable memory limit parser (bytes, SI, binary, fractional) |

See [hook-coverage.md](hook-coverage.md) for the full hook inventory and known gaps.
