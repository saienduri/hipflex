use std::cell::Cell;
use std::env;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::sync::OnceLock;

use ctor::ctor;
use hipflex_internal::hooks::HookManager;
use hipflex_internal::logging;
use hipflex_internal::shared_memory::handle::SharedMemoryHandle;
use hipflex_internal::shared_memory::proc_slots::ProcSlotHandle;
use limiter::Limiter;

mod config;
pub(crate) mod detour;
mod hiplib;
mod kfd;
mod limiter;
mod size_parser;

/// The real `dlsym` function pointer, resolved once via `dlvsym(RTLD_NEXT)`.
///
/// Our `#[no_mangle] dlsym` export overrides the PLT entry, so calling
/// `libc::dlsym` directly would recurse into our override. We break the
/// cycle by resolving the real dlsym through `dlvsym` with an explicit
/// GLIBC version, which is not overridden.
static REAL_DLSYM: OnceLock<unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void> =
    OnceLock::new();

extern "C" {
    fn dlvsym(handle: *mut c_void, symbol: *const c_char, version: *const c_char) -> *mut c_void;
}

/// Resolve the real `dlsym` function pointer. Called once, result is cached.
/// Uses `dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5")` which is not subject to
/// our LD_PRELOAD override.
pub(crate) unsafe fn real_dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void {
    let real_fn = REAL_DLSYM.get_or_init(|| {
        let ptr = dlvsym(libc::RTLD_NEXT, c"dlsym".as_ptr(), c"GLIBC_2.2.5".as_ptr());
        assert!(!ptr.is_null(), "dlvsym failed to resolve real dlsym");
        std::mem::transmute(ptr)
    });
    real_fn(handle, symbol)
}

static GLOBAL_LIMITER: OnceLock<Limiter> = OnceLock::new();
static GLOBAL_LIMITER_ERROR: OnceLock<String> = OnceLock::new();
static HOOKS_INITIALIZED: AtomicBool = AtomicBool::new(false);
static LIMITER_ERROR_REPORTED: AtomicBool = AtomicBool::new(false);
/// Set at the end of the ctor to signal that .init_array processing is complete.
/// The dlsym detour skips full init_hooks() until this is true, because
/// logging::init() and other heavy initialization during .init_array corrupts
/// HIP/ROCr internal state (breaks rocFFT's JIT → HIPFFT_PARSE_ERROR).
static CTOR_COMPLETE: AtomicBool = AtomicBool::new(false);
/// Tracks whether init_hooks() has been attempted (success or failure).
/// Prevents repeated calls from the dlsym detour when the limiter fails to init.
static INIT_HOOKS_ATTEMPTED: AtomicBool = AtomicBool::new(false);
/// BDFs from sysfs enumeration, stored for post-init verification.
static SYSFS_BDFS: OnceLock<Vec<String>> = OnceLock::new();
/// Guards one-time post-init verification of sysfs vs HIP device order.
static SYSFS_VERIFICATION_DONE: AtomicBool = AtomicBool::new(false);

#[ctor]
unsafe fn entry_point() {
    // Do NOT call logging::init() or init_hooks() here. Setting up the tracing
    // subscriber during .init_array corrupts HIP/ROCr internal state, causing
    // rocFFT's JIT kernel compilation to fail with HIPFFT_PARSE_ERROR.
    //
    // Full initialization (logging + limiter + HIP hooks) is deferred to the first
    // HIP/SMI symbol lookup via the #[no_mangle] dlsym override, AFTER .init_array
    // completes. The dlsym override is active automatically via LD_PRELOAD — no
    // explicit installation needed.
    let enable_hip_hooks = env::var("FH_ENABLE_HOOKS")
        .map(|value| value != "false")
        .unwrap_or(true);

    if !enable_hip_hooks {
        INIT_HOOKS_ATTEMPTED.store(true, Ordering::Release);
        HOOKS_INITIALIZED.store(true, Ordering::Release);
        CTOR_COMPLETE.store(true, Ordering::Release);
        return;
    }

    CTOR_COMPLETE.store(true, Ordering::Release);
}

fn should_skip_hooks_on_no_limit() -> bool {
    static SKIP_HOOKS_ON_NO_LIMIT: OnceLock<bool> = OnceLock::new();
    *SKIP_HOOKS_ON_NO_LIMIT.get_or_init(|| {
        env::var("FH_SKIP_HOOKS_IF_NO_LIMIT")
            .map(|value| value == "true" || value == "1")
            .unwrap_or(false)
    })
}

fn record_limiter_error(message: impl Into<String>) {
    let message = message.into();
    tracing::warn!("{message}");
    if GLOBAL_LIMITER_ERROR.set(message).is_err() {
        tracing::debug!("Limiter error already recorded");
    }
}

pub(crate) fn report_limiter_not_initialized() {
    if LIMITER_ERROR_REPORTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        if let Some(reason) = GLOBAL_LIMITER_ERROR.get() {
            tracing::warn!("Limiter not initialized: {reason}");
        } else {
            tracing::warn!("Limiter not initialized; init has not run");
        }
    }
}

/// Run sysfs-vs-HIP verification once, on the first hooked HIP call.
/// This is deferred from init because calling hipGetDeviceCount during init
/// would initialize the HIP runtime, breaking fork safety.
pub(crate) fn maybe_run_sysfs_verification() {
    if SYSFS_VERIFICATION_DONE.swap(true, Ordering::AcqRel) {
        return; // Already done
    }
    if let Some(sysfs_bdfs) = SYSFS_BDFS.get() {
        kfd::verify_against_hip(sysfs_bdfs);
    }
}

pub(crate) fn mock_shm_path() -> Option<PathBuf> {
    env::var("FH_SHM_FILE")
        .map(PathBuf::from)
        .map(|mut path| {
            path.pop();
            path
        })
        .ok()
}

/// SHM directory default for standalone mode.
const STANDALONE_SHM_DIR: &str = "/dev/shm/hipflex";

pub(crate) fn resolve_shm_path() -> String {
    env::var("FH_SHM_PATH").unwrap_or_else(|_| STANDALONE_SHM_DIR.to_string())
}

/// Result of standalone mode initialization.
struct StandaloneConfig {
    pod_config: config::PodConfig,
    shm_handle: SharedMemoryHandle,
    proc_slots: ProcSlotHandle,
    /// Devices from sysfs enumeration; `None` when HIP runtime was used as fallback.
    pre_enumerated: Option<Vec<limiter::EnumeratedDevice>>,
    /// KFD GPU devices for per-process VRAM reads in reconciliation.
    kfd_devices: Vec<kfd::GpuDevice>,
}

/// Build a BDF+DeviceConfig pair from a device index and PCI bus ID.
fn build_standalone_device(
    index: u32,
    pci_bdf: String,
    mem_limit: u64,
) -> (String, hipflex_internal::shared_memory::DeviceConfig) {
    let config = hipflex_internal::shared_memory::DeviceConfig::memory_only(
        index,
        pci_bdf.clone(),
        mem_limit,
    );
    (pci_bdf, config)
}

/// Standalone mode: parse FH_MEMORY_LIMIT, enumerate GPUs, create SHM.
fn init_standalone_config(mem_limit_str: &str) -> Result<StandaloneConfig, String> {
    if mock_shm_path().is_some() {
        tracing::warn!("FH_MEMORY_LIMIT is set, ignoring FH_SHM_FILE");
    }

    let mem_limit = size_parser::parse_memory_limit(mem_limit_str)
        .ok_or_else(|| format!("invalid FH_MEMORY_LIMIT value: '{mem_limit_str}'"))?;

    // Try KFD sysfs first (fork-safe), fall back to HIP runtime.
    // NOTE: Sysfs does not respect HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES.
    // In K8s, GPU restriction is via /dev/dri/renderD* mounting (handled by sysfs
    // render device check). The post-init verification will log an error if the
    // sysfs device list diverges from HIP's view.
    let (gpu_uuids, configs, pre_enumerated, kfd_devices) = match kfd::enumerate_gpu_devices() {
        Ok(devices) => {
            tracing::info!(
                device_count = devices.len(),
                "Standalone mode: enumerated GPUs via KFD sysfs (fork-safe)"
            );
            let mut uuids = Vec::with_capacity(devices.len());
            let mut cfgs = Vec::with_capacity(devices.len());
            let mut enumerated = Vec::with_capacity(devices.len());
            for (idx, device) in devices.iter().enumerate() {
                let (uuid, config) =
                    build_standalone_device(idx as u32, device.pci_bdf.clone(), mem_limit);
                enumerated.push((idx as i32, uuid.clone()));
                uuids.push(uuid);
                cfgs.push(config);
            }
            let kfd_devs = devices;
            (uuids, cfgs, Some(enumerated), kfd_devs)
        }
        Err(e) => {
            tracing::warn!("KFD sysfs enumeration failed ({e}), falling back to HIP runtime");
            let hip = hiplib::hiplib();
            let device_count = hip
                .get_device_count()
                .map_err(|e| format!("failed to enumerate GPUs: {e}"))?;

            if device_count == 0 {
                return Err("FH_MEMORY_LIMIT set but no GPUs visible".to_string());
            }

            let mut uuids = Vec::with_capacity(device_count as usize);
            let mut cfgs = Vec::with_capacity(device_count as usize);
            for device_index in 0..device_count {
                let pci_bus_id = hip.get_pci_bus_id(device_index).map_err(|e| {
                    format!("failed to get PCI bus ID for device {device_index}: {e}")
                })?;
                let (uuid, config) =
                    build_standalone_device(device_index as u32, pci_bus_id, mem_limit);
                uuids.push(uuid);
                cfgs.push(config);
            }
            (uuids, cfgs, None, Vec::new())
        }
    };

    debug_assert!(
        !gpu_uuids.is_empty(),
        "both sysfs and HIP paths guarantee at least one device"
    );

    let shm_path = resolve_shm_path();
    let shm_handle = SharedMemoryHandle::create(&shm_path, &configs)
        .map_err(|e| format!("failed to create SHM: {e}"))?;

    let proc_slots = ProcSlotHandle::create_and_claim(&shm_path)
        .map_err(|e| format!("failed to create proc slots: {e}"))?;

    tracing::info!(
        mem_limit_bytes = mem_limit,
        mem_limit_str = %mem_limit_str,
        device_count = gpu_uuids.len(),
        "Standalone mode: created SHM with per-GPU limit"
    );

    Ok(StandaloneConfig {
        pod_config: config::PodConfig {
            gpu_uuids,
            isolation: Some(limiter::ISOLATION_SOFT.to_string()),
        },
        shm_handle,
        proc_slots,
        pre_enumerated,
        kfd_devices,
    })
}

/// Mock/test mode: read GPU UUIDs from env.
fn init_mock_config() -> Result<config::PodConfig, String> {
    let uuids = env::var("FH_VISIBLE_DEVICES")
        .map_err(|_| {
            "FH_VISIBLE_DEVICES not set in mock/test mode; skipping limiter init".to_string()
        })?
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();

    Ok(config::PodConfig {
        gpu_uuids: uuids,
        isolation: None,
    })
}

fn init_limiter() {
    static LIMITER_INITIALIZED: Once = Once::new();
    LIMITER_INITIALIZED.call_once(|| {
        if let Err(error) = hiplib::init_hiplib() {
            record_limiter_error(format!("failed to initialize HIP library: {error}"));
            return;
        }

        // Config priority: FH_MEMORY_LIMIT (standalone) > FH_SHM_FILE (mock).
        let (config, standalone_shm, proc_slots, pre_enumerated, kfd_devices) =
            if let Ok(mem_limit_str) = env::var("FH_MEMORY_LIMIT") {
                match init_standalone_config(&mem_limit_str) {
                    Ok(sc) => (
                        sc.pod_config,
                        Some(sc.shm_handle),
                        Some(sc.proc_slots),
                        sc.pre_enumerated,
                        sc.kfd_devices,
                    ),
                    Err(e) => {
                        record_limiter_error(e);
                        return;
                    }
                }
            } else {
                match init_mock_config() {
                    Ok(config) => (config, None, None, None, Vec::new()),
                    Err(e) => {
                        record_limiter_error(e);
                        return;
                    }
                }
            };

        // NOTE: Device visibility is the platform's responsibility (K8s device plugin),
        // not the limiter's. The limiter enforces memory limits via SHM hooks on whichever
        // GPUs are visible. We do not set HIP_VISIBLE_DEVICES here because HIP only reads
        // it at first initialization, and in the HIP fallback path hipGetDeviceCount is
        // called before we could set it. The sysfs path doesn't respect HIP_VISIBLE_DEVICES
        // at all (see init_standalone_config comment).

        let is_standalone = standalone_shm.is_some();

        let sysfs_bdfs: Option<Vec<String>> = pre_enumerated
            .as_ref()
            .map(|devices| devices.iter().map(|(_, bdf)| bdf.clone()).collect());

        let limiter = match Limiter::new(
            config.gpu_uuids,
            config.isolation,
            is_standalone,
            proc_slots,
            pre_enumerated,
            kfd_devices,
        ) {
            Ok(limiter) => limiter,
            Err(error) => {
                record_limiter_error(format!("failed to initialize limiter: {error}"));
                return;
            }
        };

        // In standalone mode, eagerly inject the SHM handle we just created
        // (in other modes, SHM is lazily opened on first hook invocation)
        if let Some(shm_handle) = standalone_shm {
            if let Err(e) = limiter.set_shared_memory_handle(shm_handle) {
                record_limiter_error(format!("failed to set SHM handle: {e}"));
                return;
            }
        }

        if GLOBAL_LIMITER.set(limiter).is_err() {
            record_limiter_error("GLOBAL_LIMITER already initialized");
            return;
        }

        if let Some(bdfs) = sysfs_bdfs {
            if SYSFS_BDFS.set(bdfs).is_err() {
                tracing::debug!("SYSFS_BDFS already set");
            }
        }

        // Register atexit handler to drain tracked allocations and decrement
        // SHM counters. Without this, processes that exit without calling
        // hipFree (e.g., PyTorch's caching allocator) leave stale
        // pod_memory_used in SHM, causing OOM for subsequent processes.
        //
        // catch_unwind: by the time atexit runs, Rust's TLS destructors may
        // have already fired, causing panics in tracing or DashMap. The SHM
        // atomic ops themselves are safe (no TLS), so we use eprintln for
        // logging in drain_allocations and catch any residual panics here.
        extern "C" fn on_exit() {
            let _ = std::panic::catch_unwind(|| {
                if let Some(limiter) = GLOBAL_LIMITER.get() {
                    limiter.drain_allocations();
                }
            });
        }
        unsafe {
            libc::atexit(on_exit);
        }

        // Reap dead process slots: subtract their stale usage from pod_memory_used.
        // Must run after SHM handle is set and atexit is registered.
        if let Some(limiter) = GLOBAL_LIMITER.get() {
            limiter.reap_dead_pids();
        }
    });
}

fn try_install_hip_hooks() {
    if HOOKS_INITIALIZED.load(Ordering::Acquire) {
        return;
    }

    if !hipflex_internal::hooks::is_module_loaded("libamdhip64.") {
        return;
    }

    // Use Once to ensure only one thread attempts hook installation, even if
    // multiple threads race past the HOOKS_INITIALIZED fast-path check above.
    static INSTALL_ONCE: Once = Once::new();
    INSTALL_ONCE.call_once(|| {
        tracing::debug!("Installing HIP hooks...");

        let install_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            let mut hook_manager = HookManager::default();
            detour::mem::enable_hooks(&mut hook_manager)?;
            Ok::<(), hipflex_internal::HookError>(())
        }));

        match install_result {
            Ok(Ok(())) => {
                HOOKS_INITIALIZED.store(true, Ordering::Release);
                tracing::debug!("HIP hooks installed successfully");
            }
            Ok(Err(error)) => {
                tracing::error!("HIP hooks installation failed: {error}");
            }
            Err(error) => {
                tracing::error!("HIP hooks installation panicked: {error:?}");
            }
        }
    });
}

fn init_hooks() {
    // Ensure init_hooks() runs at most once, even if limiter init fails.
    // Without this guard, every dlsym call for HIP symbols would retry
    // init_hooks() when GLOBAL_LIMITER is None.
    if INIT_HOOKS_ATTEMPTED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_err()
    {
        return;
    }

    if cfg!(test) {
        return;
    }

    logging::init();
    init_limiter();

    let limiter = match GLOBAL_LIMITER.get() {
        Some(limiter) => limiter,
        None => {
            // Limiter failed to initialize (e.g., missing config env vars).
            // Gracefully skip hooks — the library becomes a passthrough.
            // Set HOOKS_INITIALIZED to prevent the dlsym detour from
            // repeatedly calling try_install_hip_hooks() on every lookup.
            HOOKS_INITIALIZED.store(true, Ordering::Release);
            report_limiter_not_initialized();
            return;
        }
    };

    let isolation = limiter.isolation();
    let should_skip_isolation = isolation.is_some_and(|iso| iso != limiter::ISOLATION_SOFT);

    if should_skip_isolation {
        tracing::info!(
            "Isolation level '{}' detected (non-soft), skipping hook initialization",
            isolation.expect("isolation checked above")
        );
        HOOKS_INITIALIZED.store(true, Ordering::Release);
        return;
    }

    if should_skip_hooks_on_no_limit() && limiter.all_devices_unlimited() {
        tracing::info!("All devices unlimited, skipping hooks installation");
        HOOKS_INITIALIZED.store(true, Ordering::Release);
        return;
    }

    // Try to install hooks immediately if libraries are already loaded
    if hipflex_internal::hooks::is_module_loaded("libamdhip64.") {
        try_install_hip_hooks();
    }

    tracing::debug!("Hook initialization completed");
}

/// Ensure hipflex is fully initialized (logging, limiter, Frida hooks).
///
/// Called from LD_PRELOAD symbol exports to trigger lazy init when applications
/// link directly against libamdhip64.so (PLT resolution bypasses our `dlsym`
/// override). The `CTOR_COMPLETE` guard prevents running during `.init_array`,
/// which would corrupt ROCr state.
pub(crate) fn ensure_init() {
    if CTOR_COMPLETE.load(Ordering::Acquire) && !INIT_HOOKS_ATTEMPTED.load(Ordering::Acquire) {
        init_hooks();
    }
    // Sequential (not else-if): init_hooks() sets INIT_HOOKS_ATTEMPTED, so the second
    // branch can fire in the same call — completing full init in one pass rather than
    // requiring two separate dlsym calls like the dlsym override path.
    if INIT_HOOKS_ATTEMPTED.load(Ordering::Acquire) && !HOOKS_INITIALIZED.load(Ordering::Acquire) {
        try_install_hip_hooks();
    }
}

thread_local! {
    static IN_DLSYM_DETOUR: Cell<bool> = const { Cell::new(false) };
}

fn call_original_dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void {
    unsafe { real_dlsym(handle, symbol) }
}

/// Check if a NUL-terminated C string starts with the given prefix by reading
/// raw bytes. Does NOT call `strlen` or any PLT-resolved function.
///
/// # Safety
///
/// `ptr` must point to a valid NUL-terminated C string. The NUL terminator
/// guarantees short-circuit safety: for a string shorter than `prefix`, the
/// loop hits `\0 != expected_byte` and returns `false` before reading past
/// the string's allocation.
#[inline(always)]
unsafe fn cstr_starts_with(ptr: *const c_char, prefix: &[u8]) -> bool {
    debug_assert!(!prefix.is_empty());
    let p = ptr as *const u8;
    for (i, &expected) in prefix.iter().enumerate() {
        if *p.add(i) != expected {
            return false;
        }
    }
    true
}

/// LD_PRELOAD symbol override for `dlsym`.
///
/// The dynamic linker resolves all `dlsym` calls to this function automatically
/// because the limiter .so is loaded via LD_PRELOAD. No Frida code patching needed.
///
/// This serves two purposes:
/// 1. Lazy init trigger — first HIP/SMI symbol lookup after ctor triggers `init_hooks()`
/// 2. SMI interception — returns our spoofed function pointers for rocm-smi/amd-smi symbols
///
/// IMPORTANT: This function must NOT call `strlen`, `CStr::from_ptr`, or any libc
/// function resolved via PLT on every invocation. Doing so triggers dynamic linker
/// re-entrancy during early process startup, corrupting HSA vmem state and causing
/// hipMallocManaged/hipMallocAsync/hipMemCreate to fail with OOM. Use
/// [`cstr_starts_with`] for prefix matching instead.
/// # Safety
///
/// `symbol` must be a valid null-terminated C string or null.
#[no_mangle]
pub unsafe extern "C" fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void {
    if symbol.is_null() {
        return call_original_dlsym(handle, symbol);
    }

    let is_hip_symbol = cstr_starts_with(symbol, b"hip");
    let is_smi_symbol = cstr_starts_with(symbol, b"rsmi_") || cstr_starts_with(symbol, b"amdsmi_");

    if !is_hip_symbol && !is_smi_symbol {
        return call_original_dlsym(handle, symbol);
    }

    // Prevent recursion
    if IN_DLSYM_DETOUR.with(|flag| flag.get()) {
        return call_original_dlsym(handle, symbol);
    }

    IN_DLSYM_DETOUR.with(|flag| flag.set(true));

    struct ResetGuard;
    impl Drop for ResetGuard {
        fn drop(&mut self) {
            IN_DLSYM_DETOUR.with(|flag| flag.set(false));
        }
    }
    let _guard = ResetGuard;

    // On first HIP/SMI symbol lookup after .init_array completes, run full
    // initialization (logging + limiter + hooks). Deferred from the ctor because
    // logging::init() during .init_array corrupts HIP/ROCr state, breaking
    // rocFFT's JIT compilation. During .init_array (CTOR_COMPLETE=false), dlsym
    // calls pass through without initialization; the first post-.init_array
    // lookup triggers init_hooks().
    if CTOR_COMPLETE.load(Ordering::Acquire) {
        if !INIT_HOOKS_ATTEMPTED.load(Ordering::Acquire) {
            init_hooks();
        } else if is_hip_symbol && !HOOKS_INITIALIZED.load(Ordering::Acquire) {
            try_install_hip_hooks();
        }
    }

    // For SMI symbols, intercept at the dlsym level: resolve the original
    // address and return our detour's address so callers invoke our hook directly.
    // CStr::from_ptr is safe here — SMI lookups happen after initialization, not
    // during early startup when dynamic linker re-entrancy is a concern.
    let original = call_original_dlsym(handle, symbol);
    if is_smi_symbol && !original.is_null() {
        let symbol_str = CStr::from_ptr(symbol).to_str().unwrap_or("");
        if let Some(detour) =
            detour::smi::try_intercept_smi_symbol(symbol_str, original as *const c_void)
        {
            return detour as *mut c_void;
        }
    }

    original
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Helper: run cstr_starts_with on a Rust string.
    fn starts_with(s: &str, prefix: &[u8]) -> bool {
        let cs = CString::new(s).unwrap();
        unsafe { cstr_starts_with(cs.as_ptr(), prefix) }
    }

    // --- cstr_starts_with correctness ---

    #[test]
    fn test_cstr_starts_with_exact_match() {
        assert!(starts_with("hip", b"hip"));
        assert!(starts_with("rsmi_", b"rsmi_"));
        assert!(starts_with("amdsmi_", b"amdsmi_"));
    }

    #[test]
    fn test_cstr_starts_with_longer_string() {
        assert!(starts_with("hipMalloc", b"hip"));
        assert!(starts_with("rsmi_dev_memory_total_get", b"rsmi_"));
        assert!(starts_with("amdsmi_get_gpu_memory_total", b"amdsmi_"));
    }

    #[test]
    fn test_cstr_starts_with_no_match() {
        assert!(!starts_with("pthread_create", b"hip"));
        assert!(!starts_with("dlopen", b"rsmi_"));
        assert!(!starts_with("cudaMalloc", b"amdsmi_"));
    }

    #[test]
    fn test_cstr_starts_with_short_strings() {
        // Strings shorter than prefix — NUL terminator causes mismatch.
        assert!(!starts_with("h", b"hip"));
        assert!(!starts_with("hi", b"hip"));
        assert!(!starts_with("rs", b"rsmi_"));
        assert!(!starts_with("a", b"amdsmi_"));
        assert!(!starts_with("am", b"amdsmi_"));
        assert!(!starts_with("amdsmi", b"amdsmi_"));
    }

    #[test]
    fn test_cstr_starts_with_empty_string() {
        assert!(!starts_with("", b"hip"));
        assert!(!starts_with("", b"rsmi_"));
    }

    #[test]
    fn test_cstr_starts_with_single_byte_prefix() {
        assert!(starts_with("hipMalloc", b"h"));
        assert!(!starts_with("malloc", b"h"));
    }

    // --- dlsym symbol classification (integration of cstr_starts_with) ---

    /// Classify a symbol the same way the dlsym override does.
    fn classify(s: &str) -> (bool, bool) {
        let cs = CString::new(s).unwrap();
        let ptr = cs.as_ptr();
        unsafe {
            let is_hip = cstr_starts_with(ptr, b"hip");
            let is_smi = cstr_starts_with(ptr, b"rsmi_") || cstr_starts_with(ptr, b"amdsmi_");
            (is_hip, is_smi)
        }
    }

    #[test]
    fn test_classify_hip_symbols() {
        assert_eq!(classify("hipMalloc"), (true, false));
        assert_eq!(classify("hipMallocManaged"), (true, false));
        assert_eq!(classify("hipFree"), (true, false));
        assert_eq!(classify("hipMemGetInfo"), (true, false));
    }

    #[test]
    fn test_classify_smi_symbols() {
        assert_eq!(classify("rsmi_dev_memory_total_get"), (false, true));
        assert_eq!(classify("amdsmi_get_gpu_memory_total"), (false, true));
        assert_eq!(classify("amdsmi_get_gpu_vram_info"), (false, true));
    }

    #[test]
    fn test_classify_unrelated_symbols() {
        assert_eq!(classify("pthread_create"), (false, false));
        assert_eq!(classify("malloc"), (false, false));
        assert_eq!(classify("dlopen"), (false, false));
        assert_eq!(classify(""), (false, false));
        assert_eq!(classify("h"), (false, false));
    }

    // --- regression guard: prevent reintroduction of CStr/strlen in dlsym fast path ---

    #[test]
    fn test_no_cstr_in_dlsym_fast_path() {
        // CStr::from_ptr calls strlen, which triggers dynamic linker re-entrancy
        // during early startup, corrupting HSA vmem and causing hipMallocManaged/
        // hipMallocAsync/hipMemCreate to fail with OOM. CStr::from_ptr is only
        // safe AFTER the `is_smi_symbol` check (post-initialization lookups).
        // This test ensures no one adds it back in the fast path.
        let source = include_str!("hipflex.rs");
        let dlsym_start = source
            .find("pub unsafe extern \"C\" fn dlsym(")
            .expect("dlsym function not found");
        let smi_branch_offset = source[dlsym_start..]
            .find("if is_smi_symbol")
            .expect("is_smi_symbol branch not found");
        let fast_path = &source[dlsym_start..dlsym_start + smi_branch_offset];
        // Check non-comment lines only — doc comments legitimately mention CStr::from_ptr.
        let has_cstr_code = fast_path.lines().any(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("//") && trimmed.contains("CStr::from_ptr")
        });
        assert!(
            !has_cstr_code,
            "CStr::from_ptr must not appear in the dlsym fast path (before is_smi_symbol check). \
             It calls strlen via PLT, causing dynamic linker re-entrancy. \
             Use cstr_starts_with instead."
        );
    }
}
