use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub mod handle;
pub mod proc_slots;

/// Maximum GPU devices tracked in a single SHM segment.
pub const MAX_DEVICES: usize = 16;

/// Fixed byte width of the UUID field in each [`DeviceEntry`].
const UUID_BYTES: usize = 64;

/// Tail padding in [`SharedDeviceState`] to allow future field additions
/// without changing the mapped size.
const STATE_PADDING: usize = 512;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_else(|err| {
            tracing::warn!(%err, "system clock before UNIX epoch, using 0");
            0
        })
}

/// Walk up from `path` removing empty directories until `stop` (exclusive) or a
/// non-empty directory is reached.
pub fn cleanup_empty_parent_directories(path: &Path, stop: Option<&Path>) -> std::io::Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if stop.is_some_and(|s| s == parent) {
        return Ok(());
    }
    let is_empty = std::fs::read_dir(parent).map(|mut it| it.next().is_none());
    if matches!(is_empty, Ok(true)) {
        tracing::info!("removing empty directory: {}", parent.display());
        std::fs::remove_dir(parent).map_err(|err| {
            tracing::debug!("could not remove {}: {err}", parent.display());
            err
        })?;
        cleanup_empty_parent_directories(parent, stop)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PodIdentifier
// ---------------------------------------------------------------------------

/// Namespace + name pair that identifies a pod within the cluster.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PodIdentifier {
    pub namespace: String,
    pub name: String,
}

impl PodIdentifier {
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
        }
    }

    /// Build the SHM directory path: `<base>/<namespace>/<name>`.
    pub fn to_path(&self, base: impl AsRef<Path>) -> PathBuf {
        base.as_ref().join(&self.namespace).join(&self.name)
    }

    /// Extract namespace/name from a path ending in `…/<ns>/<name>/shm`.
    pub fn from_shm_file_path(path: &str) -> Option<Self> {
        let components: Vec<&str> = Path::new(path)
            .components()
            .filter_map(|c| c.as_os_str().to_str())
            .collect();
        if components.len() < 3 {
            return None;
        }
        let n = components.len();
        Some(Self::new(components[n - 3], components[n - 2]))
    }
}

impl std::fmt::Display for PodIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.namespace, self.name)
    }
}

// ---------------------------------------------------------------------------
// DeviceConfig (construction-time input)
// ---------------------------------------------------------------------------

/// Parameters needed to initialise one device slot in shared memory.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceConfig {
    pub device_idx: u32,
    pub device_uuid: String,
    pub mem_limit: u64,
}

impl DeviceConfig {
    pub fn memory_only(device_idx: u32, device_uuid: String, mem_limit: u64) -> Self {
        Self {
            device_idx,
            device_uuid,
            mem_limit,
        }
    }
}

// ---------------------------------------------------------------------------
// SharedDeviceInfo (per-device counters, lives inside SHM)
// ---------------------------------------------------------------------------

/// Atomic counters for a single device: limit, current usage, and effective limit.
///
/// All fields are updated with `Acquire`/`Release` ordering so that a store on one
/// process is visible to a load on another.
#[repr(C)]
#[derive(Debug)]
pub struct SharedDeviceInfo {
    pub mem_limit: AtomicU64,
    pub pod_memory_used: AtomicU64,
    /// `mem_limit` minus total non-HIP overhead across all processes.
    /// Zero means "not yet computed" — callers fall back to `mem_limit`.
    pub effective_mem_limit: AtomicU64,
}

impl SharedDeviceInfo {
    pub fn new(limit: u64) -> Self {
        Self {
            mem_limit: AtomicU64::new(limit),
            pod_memory_used: AtomicU64::new(0),
            effective_mem_limit: AtomicU64::new(0),
        }
    }

    pub fn get_mem_limit(&self) -> u64 {
        self.mem_limit.load(Ordering::Acquire)
    }

    pub fn set_mem_limit(&self, v: u64) {
        self.mem_limit.store(v, Ordering::Release);
    }

    pub fn get_pod_memory_used(&self) -> u64 {
        self.pod_memory_used.load(Ordering::Acquire)
    }

    pub fn set_pod_memory_used(&self, v: u64) {
        self.pod_memory_used.store(v, Ordering::Release);
    }

    pub fn get_effective_mem_limit(&self) -> u64 {
        self.effective_mem_limit.load(Ordering::Acquire)
    }

    pub fn set_effective_mem_limit(&self, v: u64) {
        self.effective_mem_limit.store(v, Ordering::Release);
    }

    /// CAS loop that subtracts `amount` from `pod_memory_used`, clamping at zero.
    ///
    /// Returns the value **before** subtraction. Logs a warning if the subtraction
    /// would have wrapped.
    pub fn saturating_fetch_sub_pod_memory_used(&self, amount: u64) -> u64 {
        loop {
            let prev = self.pod_memory_used.load(Ordering::Acquire);
            let next = prev.saturating_sub(amount);
            if self
                .pod_memory_used
                .compare_exchange_weak(prev, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                if amount > prev {
                    tracing::warn!(
                        prev,
                        amount,
                        "pod_memory_used underflow prevented: \
                         tried to subtract {amount} from {prev}, clamped to 0"
                    );
                }
                return prev;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DeviceEntry (one slot in the fixed array)
// ---------------------------------------------------------------------------

/// One device slot in shared memory: UUID bytes, counters, and an active flag.
#[repr(C)]
#[derive(Debug)]
pub struct DeviceEntry {
    uuid_buf: [u8; UUID_BYTES],
    pub device_info: SharedDeviceInfo,
    active: AtomicU32,
}

impl DeviceEntry {
    pub fn new() -> Self {
        Self {
            uuid_buf: [0; UUID_BYTES],
            device_info: SharedDeviceInfo::new(0),
            active: AtomicU32::new(0),
        }
    }

    /// Write a UUID string into the fixed buffer. Only safe during single-threaded init.
    pub fn set_uuid(&mut self, uuid: &str) {
        self.uuid_buf = [0; UUID_BYTES];
        let bytes = uuid.as_bytes();
        let n = bytes.len().min(UUID_BYTES - 1);
        self.uuid_buf[..n].copy_from_slice(&bytes[..n]);
    }

    /// Read the UUID as a `&str`.
    pub fn get_uuid(&self) -> &str {
        let end = self
            .uuid_buf
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(UUID_BYTES - 1);
        // Safety: we only ever write valid UTF-8 via `set_uuid`.
        unsafe { std::str::from_utf8_unchecked(&self.uuid_buf[..end]) }
    }

    pub fn get_uuid_owned(&self) -> String {
        self.get_uuid().to_owned()
    }

    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire) != 0
    }

    pub fn set_active(&self, yes: bool) {
        self.active.store(u32::from(yes), Ordering::Release);
    }
}

impl Default for DeviceEntry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SharedDeviceState (top-level SHM layout)
// ---------------------------------------------------------------------------

/// Root structure mapped into shared memory. Fixed-size so that multiple
/// processes can mmap the same region.
#[repr(C)]
pub struct SharedDeviceState {
    pub devices: [DeviceEntry; MAX_DEVICES],
    pub device_count: AtomicU32,
    pub last_heartbeat: AtomicU64,
    pub _padding: [u8; STATE_PADDING],
}

impl SharedDeviceState {
    /// Initialise from a list of device configs. Only the specified slots are
    /// activated; the rest remain zeroed.
    pub fn new(configs: &[DeviceConfig]) -> Self {
        let mut state = Self {
            devices: std::array::from_fn(|_| DeviceEntry::new()),
            device_count: AtomicU32::new(0),
            last_heartbeat: AtomicU64::new(unix_now()),
            _padding: [0; STATE_PADDING],
        };

        let mut count = 0u32;
        for cfg in configs {
            let idx = cfg.device_idx as usize;
            if idx >= MAX_DEVICES {
                tracing::warn!(
                    device_idx = idx,
                    "device index exceeds MAX_DEVICES ({MAX_DEVICES}), skipping"
                );
                continue;
            }
            let slot = &mut state.devices[idx];
            if !slot.is_active() {
                count = count.saturating_add(1);
            }
            slot.set_uuid(&cfg.device_uuid);
            slot.device_info
                .mem_limit
                .store(cfg.mem_limit, Ordering::Relaxed);
            slot.set_active(true);
        }
        state.device_count.store(count, Ordering::Release);
        state
    }

    pub fn has_device(&self, index: usize) -> bool {
        self.devices.get(index).is_some_and(DeviceEntry::is_active)
    }

    pub fn device_count(&self) -> usize {
        self.device_count.load(Ordering::Acquire) as usize
    }

    pub fn update_heartbeat(&self, ts: u64) {
        self.last_heartbeat.store(ts, Ordering::Release);
    }

    pub fn get_last_heartbeat(&self) -> u64 {
        self.last_heartbeat.load(Ordering::Acquire)
    }

    /// Returns `true` if the last heartbeat is within `timeout` of now.
    pub fn is_healthy(&self, timeout: Duration) -> bool {
        let now = unix_now();
        if now == 0 {
            return false;
        }
        let hb = self.get_last_heartbeat();
        if hb == 0 || hb > now {
            return false;
        }
        tracing::debug!(
            last_heartbeat = hb,
            now,
            timeout_secs = timeout.as_secs(),
            "health check"
        );
        now.saturating_sub(hb) <= timeout.as_secs()
    }

    /// Run `f` on the device at `index` if it is active.
    pub fn with_device<T>(&self, index: usize, f: impl FnOnce(&DeviceEntry) -> T) -> Option<T> {
        self.devices.get(index).filter(|d| d.is_active()).map(f)
    }

    /// Convenience: set `pod_memory_used` on device `index`. Returns `false` if inactive.
    pub fn set_pod_memory_used(&self, index: usize, bytes: u64) -> bool {
        self.with_device(index, |d| d.device_info.set_pod_memory_used(bytes))
            .is_some()
    }

    /// Iterate over `(index, entry)` for active devices only.
    pub fn iter_active_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntry)> {
        self.devices
            .iter()
            .enumerate()
            .filter(|(_, d)| d.is_active())
    }

    /// Iterate over all device slots (active or not).
    pub fn iter_all_devices(&self) -> impl Iterator<Item = (usize, &DeviceEntry)> {
        self.devices.iter().enumerate()
    }

    /// Run `f` for each active device.
    pub fn for_each_active_device(&self, mut f: impl FnMut(usize, &DeviceEntry)) {
        for (i, d) in self.iter_active_devices() {
            f(i, d);
        }
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefCountError {
    Underflow,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::shared_memory::handle::{SharedMemoryHandle, SHM_PATH_SUFFIX};

    const SHM_BASE: &str = "/tmp/shm";
    const ONE_GIB: u64 = 1024 * 1024 * 1024;

    fn single_gpu_config() -> Vec<DeviceConfig> {
        vec![DeviceConfig {
            device_idx: 0,
            device_uuid: "test-device-uuid".into(),
            mem_limit: ONE_GIB,
        }]
    }

    // -- DeviceEntry ---------------------------------------------------------

    #[test]
    fn device_entry_basic_operations() {
        let mut entry = DeviceEntry::new();

        entry.set_uuid("test-uuid-123");
        assert_eq!(entry.get_uuid(), "test-uuid-123");

        assert!(!entry.is_active());
        entry.set_active(true);
        assert!(entry.is_active());
        entry.set_active(false);
        assert!(!entry.is_active());

        let overflow = "x".repeat(UUID_BYTES + 10);
        entry.set_uuid(&overflow);
        assert!(entry.get_uuid().len() < UUID_BYTES);
        assert!(entry.get_uuid().starts_with('x'));
    }

    // -- SharedDeviceState ---------------------------------------------------

    #[test]
    fn shared_device_state_creation_and_basic_ops() {
        let cfgs = single_gpu_config();
        let state = SharedDeviceState::new(&cfgs);

        assert_eq!(state.device_count(), 1);

        let hb = state.get_last_heartbeat();
        assert!(hb > 0);
        let now = unix_now();
        assert!(now.saturating_sub(hb) < 2);
        assert!(state.is_healthy(Duration::from_secs(30)));
        assert!(state.has_device(0));
    }

    #[test]
    fn shared_device_state_heartbeat_functionality() {
        let state = SharedDeviceState::new(&[]);
        assert!(state.is_healthy(Duration::from_secs(30)));

        let now = unix_now();
        state.update_heartbeat(now);
        assert_eq!(state.get_last_heartbeat(), now);
        assert!(state.is_healthy(Duration::from_secs(30)));

        state.update_heartbeat(now - 60);
        assert!(!state.is_healthy(Duration::from_secs(30)));
    }

    #[test]
    fn shared_device_info_atomic_operations() {
        let info = SharedDeviceInfo::new(ONE_GIB);
        info.set_pod_memory_used(512 * 1024 * 1024);
        assert_eq!(info.get_pod_memory_used(), 512 * 1024 * 1024);

        info.set_mem_limit(2 * ONE_GIB);
        assert_eq!(info.get_mem_limit(), 2 * ONE_GIB);
    }

    // -- SharedMemoryHandle --------------------------------------------------

    #[test]
    fn shared_memory_handle_create_and_open() {
        let id = PodIdentifier::new("handle_create_open", "test");
        let dir = id.to_path(SHM_BASE);
        let _ = std::fs::remove_dir_all(&dir);

        let cfgs = single_gpu_config();
        let h1 = SharedMemoryHandle::create(&dir, &cfgs).unwrap();
        assert_eq!(h1.get_state().device_count(), 1);

        assert!(dir.exists());

        let h2 = SharedMemoryHandle::open(&dir).unwrap();
        assert_eq!(h2.get_state().device_count(), 1);

        h1.get_state()
            .with_device(0, |d| d.device_info.set_mem_limit(42));

        let limit = h2
            .get_state()
            .with_device(0, |d| d.device_info.get_mem_limit())
            .unwrap();
        assert_eq!(limit, 42);
    }

    #[test]
    fn shared_memory_handle_error_handling() {
        assert!(SharedMemoryHandle::open("non_existent_memory").is_err());
    }

    // -- Concurrency ---------------------------------------------------------

    #[test]
    fn concurrent_device_access() {
        let id = PodIdentifier::new("concurrent_access", "test");
        let dir = id.to_path(SHM_BASE);
        let _ = std::fs::remove_dir_all(&dir);

        let cfgs = single_gpu_config();
        let handle = Arc::new(SharedMemoryHandle::create(&dir, &cfgs).unwrap());

        let threads: Vec<_> = (0..5)
            .map(|tid| {
                let h = Arc::clone(&handle);
                thread::spawn(move || {
                    let st = h.get_state();
                    for iter in 0..20 {
                        let val = tid * 20 + iter;
                        st.with_device(0, |d| d.device_info.set_mem_limit(val));
                        thread::sleep(Duration::from_millis(1));
                        let read = st
                            .with_device(0, |d| d.device_info.get_mem_limit())
                            .unwrap();
                        assert!((0..100).contains(&read));
                    }
                })
            })
            .collect();

        for t in threads {
            t.join().unwrap();
        }
    }

    // -- Iteration -----------------------------------------------------------

    #[test]
    fn device_iteration_methods() {
        let cfgs = vec![
            DeviceConfig {
                device_idx: 0,
                device_uuid: "device-0".into(),
                mem_limit: ONE_GIB,
            },
            DeviceConfig {
                device_idx: 2,
                device_uuid: "device-2".into(),
                mem_limit: 2 * ONE_GIB,
            },
        ];
        let state = SharedDeviceState::new(&cfgs);

        let active: Vec<_> = state.iter_active_devices().collect();
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].0, 0);
        assert_eq!(active[0].1.get_uuid(), "device-0");
        assert_eq!(active[1].0, 2);
        assert_eq!(active[1].1.get_uuid(), "device-2");

        assert_eq!(state.iter_all_devices().count(), MAX_DEVICES);
        assert_eq!(
            state
                .iter_all_devices()
                .filter(|(_, d)| d.is_active())
                .count(),
            2
        );

        let mut found = Vec::new();
        state.for_each_active_device(|i, d| found.push((i, d.get_uuid_owned())));
        assert_eq!(found, vec![(0, "device-0".into()), (2, "device-2".into())]);

        state.devices[2].set_active(false);
        let after: Vec<_> = state.iter_active_devices().collect();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].0, 0);
    }

    // -- Directory cleanup ---------------------------------------------------

    #[test]
    fn test_cleanup_empty_parent_directories() {
        let tmp = tempfile::TempDir::new().unwrap();
        let ns = tmp.path().join("ns");
        let pod = ns.join("pod");
        std::fs::create_dir_all(&pod).unwrap();

        let file = pod.join(SHM_PATH_SUFFIX);
        std::fs::write(&file, "data").unwrap();
        std::fs::remove_file(&file).unwrap();

        cleanup_empty_parent_directories(&file, None).unwrap();
        assert!(!pod.exists());
        assert!(!ns.exists());
    }

    #[test]
    fn test_cleanup_empty_parent_directories_with_stop_at_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let base = tmp.path();
        let ns = base.join("ns");
        let pod = ns.join("pod");
        std::fs::create_dir_all(&pod).unwrap();

        let file = pod.join(SHM_PATH_SUFFIX);
        std::fs::write(&file, "data").unwrap();
        std::fs::remove_file(&file).unwrap();

        cleanup_empty_parent_directories(&file, Some(base)).unwrap();
        assert!(!pod.exists());
        assert!(!ns.exists());
        assert!(base.exists());
    }

    #[test]
    fn test_cleanup_empty_parent_directories_stops_at_non_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let base = tmp.path();
        let ns = base.join("ns");
        let pod = ns.join("pod");
        std::fs::create_dir_all(&pod).unwrap();

        let f1 = pod.join(SHM_PATH_SUFFIX);
        let f2 = pod.join("other");
        std::fs::write(&f1, "a").unwrap();
        std::fs::write(&f2, "b").unwrap();
        std::fs::remove_file(&f1).unwrap();

        cleanup_empty_parent_directories(&f1, Some(base)).unwrap();
        assert!(pod.exists());
        assert!(f2.exists());
    }

    #[test]
    fn test_cleanup_empty_parent_directories_with_nested_stop_path() {
        let tmp = tempfile::TempDir::new().unwrap();
        let base = tmp.path();
        let ns = base.join("ns");
        let pod = ns.join("pod");
        std::fs::create_dir_all(&pod).unwrap();

        let file = pod.join(SHM_PATH_SUFFIX);
        std::fs::write(&file, "data").unwrap();
        std::fs::remove_file(&file).unwrap();

        cleanup_empty_parent_directories(&file, Some(&ns)).unwrap();
        assert!(!pod.exists());
        assert!(ns.exists());
        assert!(base.exists());
    }
}
