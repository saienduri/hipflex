# hipflex Agent Guidelines

## What This Is

A cdylib (`libhipflex.so`) loaded via `LD_PRELOAD` that hooks HIP memory APIs with Frida GUM to enforce per-process VRAM limits. Four crates:

- **hipflex** — hooks, limiter, device mapping, standalone init (`src/hipflex.rs` is the entry point)
- **hipflex-internal** — SHM types, proc slot table, logging, hook manager
- **hipflex-macro** — `#[hook_fn]` proc macro generating hook boilerplate
- **hipflex-fuzz** — simulated limiter + proptest fuzzer for offline correctness testing

## Key Constraints

- The library **must not** initialize logging or the HIP runtime during `.init_array` — this corrupts ROCr state and breaks rocFFT JIT. All init is deferred to first `dlsym` call.
- Hook code that may panic must be wrapped in `catch_unwind` — crashing the host process is never acceptable.
- Atomic counters use saturating arithmetic (CAS loop for subtract, `MAX_ALLOC_SIZE` guard for add) to prevent wrapping.
- Environment variables use the `FH_` prefix (`FH_MEMORY_LIMIT`, `FH_LOG_PATH`, `FH_HIP_LIB_PATH`).
- GPU UUIDs are PCI BDF-based (`0000:03:00.0`), normalized by stripping `amd-gpu-` prefix and lowercasing.
- Device discovery uses KFD sysfs (`/sys/class/kfd/`) — not the HIP runtime — for fork safety.

## Adding a Hook

1. Look up the HIP API signature in `refs/rocm-systems/projects/hip/include/hip/hip_runtime_api.h`
2. Add the hook in `crates/hipflex/src/detour/mem.rs` using the `check_and_alloc!` or `check_and_free!` macro
3. Add the function name to the Frida attach list in `enable_hooks()`
4. Add a GPU conformance test in `tests/gpu-tests/`
5. Add a fuzzer test case if it has interesting edge cases (pitched, array, etc.)

## Code Style

- Correctness and clarity first
- No `unwrap()` / `expect()` — use `?` and proper error types
- No silent error discarding (`let _ =` on fallible ops) — propagate, log, or handle explicitly
- Guard against panicking indexing — prefer `.get()` or iterators
- Prefer borrowing (`&T`, `&str`) — avoid unnecessary `clone`, `Arc`, `String`
- Prefer iterators and combinators over manual loops
- Use `Result` / `Option` to express correct semantics
- Full descriptive names (`device_index` not `idx`) — consistent across the entire project
- Comments explain *why*, never *what* — no organizational headers or summaries
- Prefer adding to existing files over creating new ones
- No `mod.rs` — use `src/module_name.rs`
- New crates set `[lib] path = "src/crate_name.rs"` in Cargo.toml
- No speculative abstractions — add traits/generics only when there's a concrete second use case
- Don't repeat logic — extract common behavior into helpers
- Prefer existing crates (`anyhow`, `thiserror`, `dashmap`, `tracing`, `libc`, etc.) over hand-rolled utilities

## Tests

- Test critical paths, boundaries, and error behavior
- Don't test trivial getters or obvious logic
- Don't couple tests to internal implementation details

## Verification

```bash
# Must all pass before any change is considered complete
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test -p hipflex -p hipflex-fuzz -p hipflex-internal

# GPU conformance (requires AMD GPU + Docker)
bash scripts/run-gpu-tests.sh
```
