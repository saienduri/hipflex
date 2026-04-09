# HIP Memory Hook Coverage

All HIP memory allocation APIs that consume physical VRAM are hooked. This document tracks what is hooked, what was evaluated and excluded, and known gaps.

## Interception Layers

Each hooked HIP function is intercepted at up to three levels:

1. **LD_PRELOAD exports** (27 `#[no_mangle]` symbols) — intercepts PLT-resolved calls from applications that link directly against `libamdhip64.so` (e.g., PyTorch). Primary path for most frameworks.
2. **dlsym override** — intercepts `dlsym` lookups and returns detour function pointers. Covers applications that `dlopen`/`dlsym` the HIP library at runtime (e.g., SMI tools).
3. **Frida GUM inline hooks** — patches function prologues inside `libamdhip64.so`. Catches internal call chains within the HIP runtime.

All three paths converge on the same detour functions — the accounting and enforcement logic is written once. LD_PRELOAD exports and dlsym overrides call the detour directly; Frida inline hooks redirect to the detour via prologue patching. The detour calls the original function through the Frida trampoline, not back through the export symbol, preventing infinite recursion.

A thread-local reentrancy guard prevents double-counting when hooked HIP functions internally call other hooked functions (e.g., `hipMemAllocPitch` → `hipMallocPitch` in CLR). The inner call falls through to native without accounting.

All 27 HIP hooks are declared in a single `hip_hooks!` macro table (`detour/mem.rs`) that generates all three interception mechanisms from one source of truth. Adding or removing a hook is a one-line change.

## Hook Coverage (31 hooks, 27 LD_PRELOAD exports)

**Alloc (15):** hipMalloc, hipExtMallocWithFlags, hipMallocManaged, hipMallocAsync, hipMallocFromPoolAsync, hipMallocPitch, hipMemAllocPitch, hipMalloc3D, hipMemCreate, hipMallocArray, hipMalloc3DArray, hipArrayCreate, hipArray3DCreate, hipMallocMipmappedArray, hipMipmappedArrayCreate

**Free (7):** hipFree, hipFreeAsync, hipMemRelease, hipFreeArray, hipArrayDestroy, hipFreeMipmappedArray, hipMipmappedArrayDestroy

**Spoofing — inline (5):** hipMemGetInfo, hipDeviceTotalMem, hipGetDeviceProperties, hipGetDevicePropertiesR0600, hipGetDevicePropertiesR0000 (Frida GUM + LD_PRELOAD export). The `hipGetDeviceProperties` variants spoof both `totalGlobalMem` (memory limit) and `multiProcessorCount` (CU range, when `FH_CU_RANGE` is set)

**Spoofing — dlsym-level (3):** rsmi_dev_memory_total_get, amdsmi_get_gpu_memory_total, amdsmi_get_gpu_vram_info

**System (1):** dlsym (catches late-loaded libraries)

## Conservative hooks

Two hooked APIs don't deterministically consume VRAM at allocation time but are hooked to prevent silent limit bypass:

| API | Why hooked despite ambiguity |
|-----|------------------------------|
| `hipMallocManaged` | Uses HMM (Heterogeneous Memory Management). Pages start in system RAM and migrate to VRAM on GPU access via page faults. No VRAM consumed at `hipMallocManaged` time, but the full allocation can end up resident on the GPU. We account for the full requested size at allocation time because: (1) there is no hook point for page migration — it happens in the kernel driver, (2) the worst case is the full allocation landing on VRAM, and (3) under-accounting would allow silent overcommit. |
| `hipExtMallocWithFlags` | Most flags allocate VRAM (e.g., `hipDeviceMallocDefault`, `hipDeviceMallocFinegrained`). The exception is `hipMallocSignalMemory` (flag `0x2`), which allocates 8 bytes of host-side doorbell memory for IPC signaling — no VRAM. We hook unconditionally because: (1) the common-case flags do allocate VRAM, (2) the 8-byte signal memory over-count is negligible, and (3) flag-conditional bypass would add complexity for no practical benefit. |

## Evaluated and correctly NOT hooked

| API | Reason |
|-----|--------|
| `hipHostMalloc` / `hipHostAlloc` / `hipMallocHost` / `hipMemAllocHost` | Host pinned memory — allocates from system RAM, not VRAM. All four route through `ihipMalloc` with `CL_MEM_SVM_FINE_GRAIN_BUFFER` on the host context. The GPU can DMA to/from host pinned memory but it does not consume any VRAM capacity. |
| `hipHostFree` / `hipFreeHost` | Corresponding free APIs for host pinned memory. Not hooked because the alloc side is not hooked. |
| `hipMemAddressFree` / `hipMemAddressReserve` / `hipMemMap` / `hipMemUnmap` | VMM address management only — no physical VRAM. Physical backing comes via `hipMemCreate`/`hipMemRelease` which ARE hooked. |
| `hipExternalMemoryGetMappedBuffer` | Maps memory allocated by another API (Vulkan, OpenGL, DMA-buf). No new physical allocation — the foreign API owns the VRAM. |
| `hipMemPoolImportPointer` | Imports a pointer from another process's pool. No new allocation — memory was already allocated and accounted for in the exporting process. |

## Known gap: graph memory nodes

`hipGraphAddMemAllocNode` and `hipGraphAddMemFreeNode` are NOT hooked. These APIs use an internal CLR pool allocator (`MemoryPool::AllocateMemory` → `amd::SvmBuffer::malloc()`) that bypasses all public HIP APIs — neither `hipMalloc` nor `hipFree` is called during graph execution (`hipGraphLaunch`). Physical VRAM is allocated at launch time, retained in a per-device pool across graph lifetimes, and only released to the OS via `hipDeviceGraphMemTrim()`.

**Why this is accepted:**
1. Major ML frameworks (PyTorch, JAX, TensorFlow, ONNX Runtime) currently pre-allocate memory via private pools or arena allocators before graph capture, then capture only kernel launches. The one edge case is cuBLAS/hipBLAS 12+, which can unintentionally produce alloc nodes via internal `cudaMallocAsync` workspace calls during stream capture — frameworks work around this by pre-setting workspace via `cublasSetWorkspace()`.
2. No production GPU limiter (HAMi-core, tkestack/vcuda, NVIDIA MPS) hooks graph memory.
3. Synchronous enforcement is impossible — the internal allocator has no public interception point, and before/after queries on `hipDeviceGetGraphMemAttribute` are racy under concurrent graph launches.

**Monitoring path if this becomes relevant:**
- `hipDeviceGetGraphMemAttribute(hipGraphMemAttrReservedMemCurrent)` queries actual graph pool VRAM usage
- `hipDeviceGraphMemTrim(device)` reclaims inactive pool memory back to OS
- These could be integrated as a periodic reconciliation mechanism without per-node hooking
