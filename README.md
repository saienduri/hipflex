# hipflex

Flexible GPU fractionalization — run more workloads per GPU. Transparently intercepts HIP memory allocation APIs and enforces per-process VRAM limits using shared memory counters — no kernel driver modifications, no daemon, no config files.

## Quick start

### Build

```bash
cargo build --release -p hipflex
# Output: target/release/libhipflex.so
```

### Run

```bash
# Limit each GPU to 4 GiB
FH_MEMORY_LIMIT=4GiB LD_PRELOAD=./target/release/libhipflex.so python3 train.py

# Limit to 75% of a 64 GiB GPU
FH_MEMORY_LIMIT=48GiB LD_PRELOAD=./target/release/libhipflex.so python3 train.py
```

Accepts bytes (`137438953472`), SI (`128G`, `512MB`), binary (`128GiB`, `512MiB`), or fractional (`1.5G`).

## Architecture

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ Application (PyTorch, JAX, etc.)                                │
 │   hipMalloc(&ptr, size)    hipMemGetInfo()    rocm-smi query    │
 └──────────┬──────────────────────┬──────────────────┬────────────┘
            │                      │                  │
            │ LD_PRELOAD           │ LD_PRELOAD       │ dlsym override
            ▼                      ▼                  ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ libhipflex.so                                                   │
 │                                                                 │
 │  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐ │
 │  │ Alloc/Free Hooks │  │ Info Spoofing │  │ SMI Spoofing      │ │
 │  │ 15 alloc + 7 free│  │ hipMemGetInfo │  │ rocm-smi, amd-smi │ │
 │  │ (Frida GUM)      │  │ hipDevTotal   │  │ (dlsym override)  │ │
 │  └────────┬─────────┘  │ hipGetDevProp │  └────────────────────┘ │
 │           │             │ (Frida GUM)  │                        │
 │           ▼             └──────────────┘                         │
 │  ┌────────────────────────────────────────────┐                 │
 │  │ Limiter (reserve-then-allocate)            │                 │
 │  │                                            │                 │
 │  │  1. fetch_add(size) on SHM counter         │                 │
 │  │  2. over limit? → fetch_sub, deny          │                 │
 │  │  3. call real HIP API                      │                 │
 │  │  4. native fail? → fetch_sub, rollback     │                 │
 │  │  5. track (pointer → device, size)         │                 │
 │  └────────┬───────────────────────────────────┘                 │
 └──────────┬──────────────────────────────────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Shared Memory (/dev/shm/hipflex)                                │
 │                                                                 │
 │  per-device: pod_memory_used (atomic u64), mem_limit, UUID      │
 │  per-process: PID slot table with non-HIP overhead per device   │
 └──────────────────────────────────────────────────────────────────┘
```

Non-HIP memory (code objects, page tables, scratch buffers) is tracked via KFD sysfs and reconciled into the effective limit. Crashed or killed processes are automatically reaped so stale overhead doesn't reduce capacity. Each GPU gets independent accounting — `FH_MEMORY_LIMIT` applies per device. Structured logging via `tracing` covers VRAM usage, overhead, and slot state with configurable levels and file rotation.

## Configuration

All configuration is via environment variables.

### Core

| Variable | Description |
|----------|-------------|
| `FH_MEMORY_LIMIT` | Per-GPU memory limit. Accepts bytes, SI, binary, or fractional sizes. |
| `FH_ENABLE_HOOKS` | Set to `false` to disable all hooking (library becomes a no-op). Default: `true`. |
| `FH_HIP_LIB_PATH` | Override path to `libamdhip64.so`. |
| `FH_SHM_PATH` | Override shared memory directory. Default: `/dev/shm/hipflex`. |

### Logging

| Variable | Description |
|----------|-------------|
| `FH_ENABLE_LOG` | Set to `off`, `0`, or `false` to disable logging. |
| `FH_LOG_PATH` | Log destination. `stderr` for stderr, or a file/directory path. Default: `/tmp/hipflex/hipflex.log.*` (daily rotation). |
| `FH_LOG_LEVEL` | Tracing filter directive (e.g. `debug`, `hipflex=trace`). Default: `info`. |

## Project structure

```
crates/
  hipflex/           # Main cdylib — hooks, limiter, reconciliation, reaping, device mapping
  hipflex-internal/  # Shared internals — SHM types, proc slots, logging, hook manager
  hipflex-macro/     # Proc macro for hook function generation
  hipflex-fuzz/      # Property-based fuzzer (proptest) + simulated limiter
tests/
  gpu-tests/         # Python GPU conformance tests (requires AMD GPU + Docker)
scripts/
  run-gpu-tests.sh   # Test runner (fuzzer + GPU tests)
```

## Testing

### Unit + fuzzer tests (no GPU required)

```bash
cargo test -p hipflex -p hipflex-fuzz -p hipflex-internal
```

Runs unit tests, property-based fuzz tests (accounting invariants, concurrent stress, edge cases, multi-process lifecycle), and SHM compatibility checks.

### GPU conformance tests (requires AMD GPU)

```bash
bash scripts/run-gpu-tests.sh
```

See [`scripts/run-gpu-tests.sh`](scripts/run-gpu-tests.sh) for setup instructions and options (`--fuzz-only`, `--gpu-only`, `--full`).

## Documentation

- [Design](docs/design.md) — architecture, operating modes, atomics, SHM layout, failure modes
- [Hook coverage](docs/hook-coverage.md) — full inventory of hooked HIP/SMI APIs, exclusions, and known gaps

## License

[Apache-2.0](LICENSE)
