use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result};
use shared_memory::{Mode, Shmem, ShmemConf, ShmemError};
use tracing::info;

use super::{DeviceConfig, SharedDeviceState};

/// Filename used for the SHM segment within its directory.
pub const SHM_PATH_SUFFIX: &str = "shm";

/// RAII wrapper around a POSIX shared memory segment backed by tmpfs.
///
/// The underlying memory is mapped as a [`SharedDeviceState`] and accessed via
/// raw pointer. Atomic fields in `SharedDeviceState` guarantee safe concurrent
/// access across processes; this struct itself is `Send + Sync`.
pub struct SharedMemoryHandle {
    shmem: RefCell<Shmem>,
    state: *mut SharedDeviceState,
}

unsafe impl Send for SharedMemoryHandle {}
unsafe impl Sync for SharedMemoryHandle {}

impl SharedMemoryHandle {
    /// Open an existing segment at `dir_path/shm`. Fails if it doesn't exist.
    pub fn open(dir_path: impl AsRef<Path>) -> Result<Self> {
        let dir = dir_path.as_ref();
        let mut segment = ShmemConf::new()
            .size(std::mem::size_of::<SharedDeviceState>())
            .use_tmpfs_with_dir(dir)
            .os_id(SHM_PATH_SUFFIX)
            .open()
            .with_context(|| format!("failed to open SHM at {}", dir.display()))?;

        segment.set_owner(false);
        let state = segment.as_ptr() as *mut SharedDeviceState;

        Ok(Self {
            shmem: RefCell::new(segment),
            state,
        })
    }

    /// Create a new segment, or join one that already exists.
    ///
    /// When joining an existing segment (created by another process), the contents
    /// are **not** reinitialized — runtime state like `pod_memory_used` is preserved.
    /// Each process is responsible for draining its own allocations at exit.
    pub fn create(dir_path: impl AsRef<Path>, configs: &[DeviceConfig]) -> Result<Self> {
        let dir = dir_path.as_ref();
        std::fs::create_dir_all(dir)?;

        let prev_umask = unsafe { libc::umask(0) };
        let result = Self::create_inner(dir, configs);
        unsafe { libc::umask(prev_umask) };

        result
    }

    fn create_inner(dir: &Path, configs: &[DeviceConfig]) -> Result<Self> {
        let size = std::mem::size_of::<SharedDeviceState>();
        let permissions = Mode::S_IRUSR
            | Mode::S_IWUSR
            | Mode::S_IRGRP
            | Mode::S_IWGRP
            | Mode::S_IROTH
            | Mode::S_IWOTH;

        let (mut segment, fresh) = match ShmemConf::new()
            .size(size)
            .use_tmpfs_with_dir(dir)
            .os_id(SHM_PATH_SUFFIX)
            .mode(permissions)
            .create()
        {
            Ok(seg) => (seg, true),
            Err(ShmemError::LinkExists | ShmemError::MappingIdExists) => {
                let seg = ShmemConf::new()
                    .size(size)
                    .use_tmpfs_with_dir(dir)
                    .os_id(SHM_PATH_SUFFIX)
                    .open()
                    .context("failed to open existing SHM after create race")?;
                (seg, false)
            }
            Err(err) => return Err(anyhow::anyhow!("failed to create SHM: {err}")),
        };

        segment.set_owner(false);
        let state = segment.as_ptr() as *mut SharedDeviceState;

        if fresh {
            unsafe { state.write(SharedDeviceState::new(configs)) };
            info!(path = ?dir, "created shared memory segment");
        } else {
            info!(path = ?dir, "joined existing shared memory segment");
        }

        Ok(Self {
            shmem: RefCell::new(segment),
            state,
        })
    }

    /// Create a handle backed by real SHM for use in tests.
    pub fn mock(shm_dir: impl AsRef<Path>, gpu_map: Vec<(usize, String)>) -> Self {
        let dir = shm_dir.as_ref();
        let configs: Vec<DeviceConfig> = gpu_map
            .into_iter()
            .map(|(idx, uuid)| DeviceConfig {
                device_idx: idx as u32,
                device_uuid: uuid,
                mem_limit: 8 * 1024 * 1024 * 1024,
            })
            .collect();

        let size = std::mem::size_of::<SharedDeviceState>();

        let segment = ShmemConf::new()
            .size(size)
            .use_tmpfs_with_dir(dir)
            .os_id(SHM_PATH_SUFFIX)
            .open()
            .unwrap_or_else(|_| {
                std::fs::create_dir_all(dir).expect("create mock SHM dir");
                let seg = ShmemConf::new()
                    .size(size)
                    .use_tmpfs_with_dir(dir)
                    .os_id(SHM_PATH_SUFFIX)
                    .create()
                    .expect("create mock SHM");
                let ptr = seg.as_ptr() as *mut SharedDeviceState;
                unsafe { ptr.write(SharedDeviceState::new(&configs)) };
                seg
            });

        let state = segment.as_ptr() as *mut SharedDeviceState;
        Self {
            shmem: RefCell::new(segment),
            state,
        }
    }

    /// Raw pointer to the mapped state. Prefer [`get_state`](Self::get_state).
    pub fn get_ptr(&self) -> *mut SharedDeviceState {
        self.state
    }

    /// Toggle whether this handle owns (and will unlink) the SHM file on drop.
    pub fn set_owner(&self, owns: bool) {
        self.shmem.borrow_mut().set_owner(owns);
    }

    /// Borrow the shared device state. Valid for the lifetime of this handle.
    pub fn get_state(&self) -> &SharedDeviceState {
        unsafe { &*self.state }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn one_gpu() -> Vec<DeviceConfig> {
        vec![DeviceConfig {
            device_idx: 0,
            device_uuid: "GPU-test-uuid".into(),
            mem_limit: 4 * 1024 * 1024 * 1024,
        }]
    }

    #[test]
    fn test_open_fails_when_not_exists() {
        let tmp = TempDir::new().unwrap();
        assert!(SharedMemoryHandle::open(tmp.path().join("nope")).is_err());
    }

    #[test]
    fn test_open_existing_shared_memory() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("open_existing");
        let cfgs = one_gpu();

        let h1 = SharedMemoryHandle::create(&dir, &cfgs).unwrap();
        assert_eq!(h1.get_state().device_count(), 1);

        let h2 = SharedMemoryHandle::open(&dir).unwrap();
        assert_eq!(h2.get_state().device_count(), 1);

        let uuid = h2
            .get_state()
            .with_device(0, |d| d.get_uuid_owned())
            .unwrap();
        assert_eq!(uuid, "GPU-test-uuid");
    }

    #[test]
    fn test_open_multiple_times() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("open_multi");

        SharedMemoryHandle::create(&dir, &[]).unwrap();

        let h1 = SharedMemoryHandle::open(&dir).unwrap();
        assert_eq!(h1.get_state().device_count(), 0);

        let h2 = SharedMemoryHandle::open(&dir).unwrap();
        assert_eq!(h2.get_state().device_count(), 0);

        drop(h1);

        let h3 = SharedMemoryHandle::open(&dir).unwrap();
        assert_eq!(h3.get_state().device_count(), 0);
    }

    #[test]
    fn test_create_twice_joins_existing_and_preserves_state() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("create_twice");
        let cfgs = one_gpu();

        let h1 = SharedMemoryHandle::create(&dir, &cfgs).unwrap();
        assert_eq!(h1.get_state().device_count(), 1);

        let usage: u64 = 500 * 1024 * 1024;
        assert!(h1.get_state().set_pod_memory_used(0, usage));

        let h2 = SharedMemoryHandle::create(&dir, &cfgs).unwrap();
        assert_eq!(h2.get_state().device_count(), 1);

        let observed = h2
            .get_state()
            .with_device(0, |d| d.device_info.get_pod_memory_used())
            .unwrap();
        assert_eq!(observed, usage, "second create must not reinitialize SHM");
    }

    #[test]
    fn test_open_with_nested_path() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("a").join("b").join("c");

        SharedMemoryHandle::create(&dir, &[]).unwrap();
        let h = SharedMemoryHandle::open(&dir).unwrap();
        assert_eq!(h.get_state().device_count(), 0);
        assert!(dir.exists());
    }
}
