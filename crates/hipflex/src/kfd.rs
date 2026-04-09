//! KFD sysfs enumeration — discover GPU devices without touching the HIP runtime.
//!
//! This avoids initializing GPU context during standalone mode init, which would
//! break fork safety (child inherits stale GPU state).

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

/// A GPU device discovered via KFD sysfs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    /// PCI Bus/Device/Function address, e.g. "0000:75:00.0"
    pub pci_bdf: String,
    /// DRM render node minor number, e.g. 128 → /dev/dri/renderD128
    pub render_minor: u32,
    /// KFD gpu_id — used as suffix in `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`.
    pub gpu_id: u64,
    /// Number of Compute Units (simd_count / simd_per_cu). `None` if fields missing (CPU node or older kernel).
    pub cu_count: Option<u32>,
    /// GFX target version (e.g. 90402 = gfx942/CDNA3 MI325X). `None` if missing.
    pub gfx_target_version: Option<u32>,
}

/// Parsed fields from a KFD topology node's `properties` file.
#[derive(Debug, Clone)]
pub struct NodeProperties {
    pub domain: u32,
    pub location_id: u32,
    pub drm_render_minor: u32,
    pub simd_count: Option<u32>,
    pub simd_per_cu: Option<u32>,
    pub gfx_target_version: Option<u32>,
}

/// Errors that can occur during GPU enumeration.
#[derive(Debug)]
pub enum EnumerationError {
    /// The KFD sysfs topology path does not exist or is inaccessible.
    SysfsNotAvailable(String),
    /// A required field is missing from a node's properties file.
    MissingField(String),
    /// A field value could not be parsed as an integer.
    ParseError(String),
    /// No GPU devices were found after scanning all nodes.
    NoDevicesFound,
}

impl fmt::Display for EnumerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SysfsNotAvailable(path) => write!(f, "KFD sysfs not available: {path}"),
            Self::MissingField(field) => write!(f, "missing required field: {field}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::NoDevicesFound => write!(f, "no GPU devices found"),
        }
    }
}

impl std::error::Error for EnumerationError {}

/// Decode a PCI Bus/Device/Function string from KFD domain and location_id.
///
/// KFD encodes BDF in `location_id` as:
/// - bits [15:8] → bus
/// - bits [7:3]  → device
/// - bits [2:0]  → function
pub fn decode_pci_bdf(domain: u32, location_id: u32) -> String {
    let bus = (location_id >> 8) & 0xFF;
    let device = (location_id >> 3) & 0x1F;
    let function = location_id & 0x7;
    format!("{domain:04x}:{bus:02x}:{device:02x}.{function}")
}

/// Parse required fields from the content of a KFD node `properties` file.
///
/// The file is line-oriented: `key value\n`. Values are decimal integers.
pub fn parse_properties(content: &str) -> Result<NodeProperties, EnumerationError> {
    let mut domain: Option<u32> = None;
    let mut location_id: Option<u32> = None;
    let mut drm_render_minor: Option<u32> = None;
    let mut simd_count: Option<u32> = None;
    let mut simd_per_cu: Option<u32> = None;
    let mut gfx_target_version: Option<u32> = None;

    let parse_u32 = |key: &str, value: &str| -> Result<u32, EnumerationError> {
        value
            .parse()
            .map_err(|_| EnumerationError::ParseError(format!("{key}: {value}")))
    };

    for line in content.lines() {
        let mut parts = line.split_whitespace();
        let Some(key) = parts.next() else { continue };
        let Some(value) = parts.next() else { continue };

        match key {
            "domain" => domain = Some(parse_u32(key, value)?),
            "location_id" => location_id = Some(parse_u32(key, value)?),
            "drm_render_minor" => drm_render_minor = Some(parse_u32(key, value)?),
            "simd_count" => simd_count = Some(parse_u32(key, value)?),
            "simd_per_cu" => simd_per_cu = Some(parse_u32(key, value)?),
            "gfx_target_version" => gfx_target_version = Some(parse_u32(key, value)?),
            _ => {}
        }
    }

    Ok(NodeProperties {
        domain: domain.ok_or_else(|| EnumerationError::MissingField("domain".into()))?,
        location_id: location_id
            .ok_or_else(|| EnumerationError::MissingField("location_id".into()))?,
        drm_render_minor: drm_render_minor
            .ok_or_else(|| EnumerationError::MissingField("drm_render_minor".into()))?,
        simd_count,
        simd_per_cu,
        gfx_target_version,
    })
}

const KFD_TOPOLOGY_PATH: &str = "/sys/class/kfd/kfd/topology/nodes";
const DRI_PATH: &str = "/dev/dri";

/// Enumerate GPU devices visible via KFD sysfs.
///
/// This is the public entry point. For tests, use [`enumerate_gpu_devices_from`].
pub fn enumerate_gpu_devices() -> Result<Vec<GpuDevice>, EnumerationError> {
    enumerate_gpu_devices_from(Path::new(KFD_TOPOLOGY_PATH), Path::new(DRI_PATH))
}

/// Testable enumeration function that accepts custom sysfs and dri paths.
pub fn enumerate_gpu_devices_from(
    topology_path: &Path,
    dri_path: &Path,
) -> Result<Vec<GpuDevice>, EnumerationError> {
    let entries = fs::read_dir(topology_path)
        .map_err(|_| EnumerationError::SysfsNotAvailable(topology_path.display().to_string()))?;

    // Collect node directory names and sort numerically.
    let mut node_ids: Vec<u32> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name();
            name.to_str()?.parse::<u32>().ok()
        })
        .collect();
    node_ids.sort_unstable();

    let mut devices = Vec::new();

    for node_id in node_ids {
        let node_dir = topology_path.join(node_id.to_string());

        // Skip CPUs (gpu_id == 0) and inaccessible nodes (EPERM).
        let gpu_id_path = node_dir.join("gpu_id");
        let gpu_id_str = match fs::read_to_string(&gpu_id_path) {
            Ok(s) => s,
            Err(_) => continue, // EPERM or missing — skip
        };
        let gpu_id: u64 = match gpu_id_str.trim().parse() {
            Ok(id) => id,
            Err(_) => continue,
        };
        if gpu_id == 0 {
            continue; // CPU node
        }

        // Parse properties for this GPU node.
        let props_path = node_dir.join("properties");
        let props_content = match fs::read_to_string(&props_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let props = match parse_properties(&props_content) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(node_id, "skipping KFD node: {e}");
                continue;
            }
        };

        // Check that the render device exists.
        let render_device = dri_path.join(format!("renderD{}", props.drm_render_minor));
        if !render_device.exists() {
            continue;
        }

        let pci_bdf = decode_pci_bdf(props.domain, props.location_id);
        let cu_count = match (props.simd_count, props.simd_per_cu) {
            (Some(simds), Some(per_cu)) if per_cu > 0 => Some(simds / per_cu),
            _ => None,
        };
        devices.push(GpuDevice {
            pci_bdf,
            render_minor: props.drm_render_minor,
            gpu_id,
            cu_count,
            gfx_target_version: props.gfx_target_version,
        });
    }

    if devices.is_empty() {
        return Err(EnumerationError::NoDevicesFound);
    }

    Ok(devices)
}

const KFD_PROC_PATH: &str = "/sys/class/kfd/kfd/proc";

/// Read per-process VRAM usage from KFD sysfs, returning a map of PCI BDF → bytes.
///
/// Reads `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>` for each known GPU device.
/// The `pid` must be a **host PID** (KFD sysfs is not PID-namespace-aware).
/// Use [`resolve_host_pid`] to translate container PIDs before calling this.
pub fn read_kfd_vram_for_pid(pid: u32, gpu_devices: &[GpuDevice]) -> HashMap<String, u64> {
    read_kfd_vram_for_pid_from(Path::new(KFD_PROC_PATH), pid, gpu_devices)
}

fn read_kfd_vram_for_pid_from(
    kfd_proc_base: &Path,
    pid: u32,
    gpu_devices: &[GpuDevice],
) -> HashMap<String, u64> {
    let mut result = HashMap::new();
    let proc_dir = kfd_proc_base.join(pid.to_string());
    for device in gpu_devices {
        let vram_path = proc_dir.join(format!("vram_{}", device.gpu_id));
        let bytes = match fs::read_to_string(&vram_path) {
            Ok(content) => match content.trim().parse::<u64>() {
                Ok(v) => v,
                Err(_) => continue,
            },
            Err(_) => continue,
        };
        *result.entry(device.pci_bdf.clone()).or_insert(0) += bytes;
    }
    result
}

const DEFAULT_HOST_PROC: &str = "/host/proc";

/// Resolve the host PID for the current process.
///
/// In containers with a separate PID namespace, `getpid()` returns the
/// container-local PID which doesn't exist in KFD sysfs (which uses host PIDs).
///
/// Resolution strategy:
/// 1. Check if our PID exists in KFD sysfs → we're on bare metal, return as-is.
/// 2. Scan `host_proc_path` (typically `/host/proc`, mounted via `-v /proc:/host/proc:ro`)
///    for a process whose `NSpid` line ends with our container PID → return the host PID.
/// 3. If neither works, return `None` (overhead tracking disabled).
pub fn resolve_host_pid(gpu_devices: &[GpuDevice]) -> Option<u32> {
    resolve_host_pid_with(
        std::process::id(),
        Path::new(KFD_PROC_PATH),
        &host_proc_path(),
        gpu_devices,
    )
}

fn host_proc_path() -> std::path::PathBuf {
    std::env::var("FH_HOST_PROC_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from(DEFAULT_HOST_PROC))
}

fn resolve_host_pid_with(
    container_pid: u32,
    kfd_proc_base: &Path,
    host_proc_base: &Path,
    gpu_devices: &[GpuDevice],
) -> Option<u32> {
    // Fast path: our PID already works in KFD sysfs (bare metal or --pid=host).
    if let Some(first_device) = gpu_devices.first() {
        let probe_path = kfd_proc_base
            .join(container_pid.to_string())
            .join(format!("vram_{}", first_device.gpu_id));
        if probe_path.parent().is_some_and(|d| d.exists()) {
            return Some(container_pid);
        }
    }

    // Container path: scan host procfs for our NSpid.
    scan_host_proc_for_nspid(host_proc_base, container_pid)
}

/// Scan `/host/proc/*/status` to find the host PID whose NSpid ends with
/// our container PID.
///
/// The `NSpid` line in `/proc/<pid>/status` lists the PID in each namespace
/// from outermost to innermost. When reading from the host's procfs mount,
/// the last value is the container PID.
fn scan_host_proc_for_nspid(host_proc_base: &Path, container_pid: u32) -> Option<u32> {
    let entries = match fs::read_dir(host_proc_base) {
        Ok(e) => e,
        Err(_) => return None,
    };

    let target = container_pid.to_string();

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Only look at numeric directories (PID entries).
        if !name_str.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }

        let status_path = entry.path().join("status");
        let content = match fs::read_to_string(&status_path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("NSpid:") {
                let pids: Vec<&str> = rest.split_whitespace().collect();
                // Last value is the innermost (container) PID.
                // First value is the host PID (from the host procfs perspective).
                if pids.len() >= 2 && pids.last() == Some(&target.as_str()) {
                    if let Ok(host_pid) = pids[0].parse::<u32>() {
                        return Some(host_pid);
                    }
                }
                break;
            }
        }
    }

    None
}

/// One-time verification: compare sysfs-enumerated BDFs against HIP runtime.
///
/// Called after HIP hooks are installed (meaning HIP library is loaded).
/// Uses hiplib's raw function pointers (not hooked) to query device info.
/// Logs an error on mismatch but doesn't change behavior.
pub(crate) fn verify_against_hip(sysfs_bdfs: &[String]) {
    let hip = crate::hiplib::hiplib();

    let device_count = match hip.get_device_count() {
        Ok(count) => count,
        Err(e) => {
            tracing::warn!("Post-init verification: hipGetDeviceCount failed: {e}");
            return;
        }
    };

    let mut hip_bdfs = Vec::with_capacity(device_count as usize);
    for i in 0..device_count {
        match hip.get_pci_bus_id(i) {
            Ok(bdf) => hip_bdfs.push(bdf.to_lowercase()),
            Err(e) => {
                tracing::warn!("Post-init verification: hipDeviceGetPCIBusId({i}) failed: {e}");
                return;
            }
        }
    }

    let sysfs_lower: Vec<String> = sysfs_bdfs.iter().map(|b| b.to_lowercase()).collect();

    if sysfs_lower.len() != hip_bdfs.len() {
        tracing::error!(
            sysfs_count = sysfs_lower.len(),
            hip_count = hip_bdfs.len(),
            sysfs_bdfs = ?sysfs_lower,
            hip_bdfs = ?hip_bdfs,
            "DEVICE MISMATCH: sysfs and HIP report different device counts"
        );
        return;
    }

    if sysfs_lower != hip_bdfs {
        tracing::error!(
            sysfs_bdfs = ?sysfs_lower,
            hip_bdfs = ?hip_bdfs,
            "DEVICE MISMATCH: sysfs and HIP report different device ordering"
        );
    } else {
        tracing::debug!(
            device_count,
            "Post-init verification: sysfs matches HIP device order"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- BDF decoding tests --

    #[test]
    fn test_decode_bdf_mi325x_device0() {
        assert_eq!(decode_pci_bdf(0, 0x7500), "0000:75:00.0");
    }

    #[test]
    fn test_decode_bdf_high_bus() {
        assert_eq!(decode_pci_bdf(0, 0xf500), "0000:f5:00.0");
    }

    #[test]
    fn test_decode_bdf_with_device_and_function() {
        // location_id=0x050b → bus=0x05, device=(0x0b >> 3)=0x01, function=0x0b & 0x7=0x3
        assert_eq!(decode_pci_bdf(0, 0x050b), "0000:05:01.3");
    }

    #[test]
    fn test_decode_bdf_nonzero_domain() {
        assert_eq!(decode_pci_bdf(1, 0x7500), "0001:75:00.0");
    }

    #[test]
    fn test_decode_bdf_zero_location() {
        assert_eq!(decode_pci_bdf(0, 0), "0000:00:00.0");
    }

    // -- Properties parsing tests --

    #[test]
    fn test_parse_properties_mi325x() {
        let content = "\
cpu_cores_count 0
simd_count 304
mem_banks_count 1
caches_count 228
io_links_count 3
p2p_links_count 0
max_waves_per_simd 16
lds_size_in_kb 64
gds_size_in_kb 0
num_gws 64
wave_front_size 64
array_count 38
simd_arrays_per_engine 2
cu_per_simd_array 2
simd_per_cu 4
max_slots_scratch_cu 32
gfx_target_version 90402
vendor_id 4098
device_id 29856
location_id 29952
domain 0
drm_render_minor 128
hive_id 11638818706288400384
num_sdma_engines 2
num_sdma_xgmi_engines 6
num_sdma_queues_per_engine 8
num_cp_queues 24
max_engine_clk_fcompute 2100
local_mem_size 206142726144
fw_version 262
capability 540676
sdma_fw_version 24
max_engine_clk_ccompute 2100
num_xcc 1
debug_prop 32768
unique_id 9895604649984";

        let props = parse_properties(content).unwrap();
        assert_eq!(props.domain, 0);
        assert_eq!(props.location_id, 29952);
        assert_eq!(props.drm_render_minor, 128);
        assert_eq!(props.simd_count, Some(304));
        assert_eq!(props.simd_per_cu, Some(4));
        assert_eq!(props.gfx_target_version, Some(90402));
    }

    #[test]
    fn test_parse_properties_cu_count() {
        let content = "\
domain 0
location_id 29952
drm_render_minor 128
simd_count 304
simd_per_cu 4
gfx_target_version 90402";

        let props = parse_properties(content).unwrap();
        // cu_count is computed in enumerate, but we can verify the raw fields
        assert_eq!(props.simd_count, Some(304));
        assert_eq!(props.simd_per_cu, Some(4));
        // 304 / 4 = 76 CUs — verified by enumerate_gpu_devices_from
    }

    #[test]
    fn test_parse_properties_no_cu_fields() {
        let content = "domain 0\nlocation_id 29952\ndrm_render_minor 128\n";
        let props = parse_properties(content).unwrap();
        assert_eq!(props.simd_count, None);
        assert_eq!(props.simd_per_cu, None);
        assert_eq!(props.gfx_target_version, None);
    }

    #[test]
    fn test_parse_properties_missing_field() {
        let content = "domain 0\nlocation_id 29952\n";
        let result = parse_properties(content);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, EnumerationError::MissingField(ref f) if f == "drm_render_minor"),
            "expected MissingField(drm_render_minor), got: {err}"
        );
    }

    #[test]
    fn test_parse_properties_empty() {
        let result = parse_properties("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_properties_malformed_value() {
        let content = "domain abc\nlocation_id 29952\ndrm_render_minor 128\n";
        let result = parse_properties(content);
        assert!(matches!(result, Err(EnumerationError::ParseError(_))));
    }

    // -- Sysfs enumeration tests --

    /// Helper: create a mock KFD topology node directory.
    fn create_node(topology: &Path, node_id: u32, gpu_id: u64, properties: Option<&str>) {
        let node_dir = topology.join(node_id.to_string());
        fs::create_dir_all(&node_dir).unwrap();
        fs::write(node_dir.join("gpu_id"), gpu_id.to_string()).unwrap();
        if let Some(props) = properties {
            fs::write(node_dir.join("properties"), props).unwrap();
        }
    }

    fn gpu_properties(domain: u32, location_id: u32, render_minor: u32) -> String {
        format!("domain {domain}\nlocation_id {location_id}\ndrm_render_minor {render_minor}\n")
    }

    fn create_render_device(dri: &Path, minor: u32) {
        fs::write(dri.join(format!("renderD{minor}")), "").unwrap();
    }

    #[test]
    fn test_enumerate_single_gpu() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Node 0: CPU
        create_node(&topology, 0, 0, None);
        // Node 1: GPU
        let props = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 1, 12345, Some(&props));
        create_render_device(&dri, 128);

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
        assert_eq!(devices[0].render_minor, 128);
        assert_eq!(devices[0].gpu_id, 12345);
        assert_eq!(devices[0].cu_count, None);
        assert_eq!(devices[0].gfx_target_version, None);
    }

    #[test]
    fn test_enumerate_gpu_with_cu_count() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        create_node(&topology, 0, 0, None);
        let props = "domain 0\nlocation_id 29952\ndrm_render_minor 128\nsimd_count 304\nsimd_per_cu 4\ngfx_target_version 90402\n";
        create_node(&topology, 1, 12345, Some(props));
        create_render_device(&dri, 128);

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].cu_count, Some(76)); // 304 / 4
        assert_eq!(devices[0].gfx_target_version, Some(90402));
    }

    #[test]
    fn test_enumerate_multiple_gpus_sorted() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Node 0: CPU
        create_node(&topology, 0, 0, None);
        // Node 2: GPU (created before node 1 to test sorting)
        let props2 = gpu_properties(0, 0xf500, 129);
        create_node(&topology, 2, 67890, Some(&props2));
        create_render_device(&dri, 129);
        // Node 1: GPU
        let props1 = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 1, 12345, Some(&props1));
        create_render_device(&dri, 128);

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 2);
        // Should be sorted by node ID: node 1 first, node 2 second.
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
        assert_eq!(devices[0].render_minor, 128);
        assert_eq!(devices[0].gpu_id, 12345);
        assert_eq!(devices[1].pci_bdf, "0000:f5:00.0");
        assert_eq!(devices[1].render_minor, 129);
        assert_eq!(devices[1].gpu_id, 67890);
    }

    #[test]
    fn test_enumerate_skips_missing_render_device() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        create_node(&topology, 0, 0, None);

        // GPU 1 — has render device
        let props1 = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 1, 12345, Some(&props1));
        create_render_device(&dri, 128);

        // GPU 2 — NO render device
        let props2 = gpu_properties(0, 0xf500, 129);
        create_node(&topology, 2, 67890, Some(&props2));

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
    }

    #[test]
    fn test_enumerate_no_gpus_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Only a CPU node
        create_node(&topology, 0, 0, None);

        let result = enumerate_gpu_devices_from(&topology, &dri);
        assert!(matches!(result, Err(EnumerationError::NoDevicesFound)));
    }

    #[test]
    fn test_enumerate_skips_node_without_properties() {
        let tmp = tempfile::tempdir().unwrap();
        let topology = tmp.path().join("nodes");
        let dri = tmp.path().join("dri");
        fs::create_dir_all(&topology).unwrap();
        fs::create_dir_all(&dri).unwrap();

        // Node 0: CPU
        create_node(&topology, 0, 0, None);
        // Node 1: GPU with gpu_id but no properties file
        create_node(&topology, 1, 12345, None);
        // Node 2: Valid GPU
        let props = gpu_properties(0, 0x7500, 128);
        create_node(&topology, 2, 67890, Some(&props));
        create_render_device(&dri, 128);

        let devices = enumerate_gpu_devices_from(&topology, &dri).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].pci_bdf, "0000:75:00.0");
    }

    #[test]
    fn test_enumerate_sysfs_not_available() {
        let nonexistent = Path::new("/tmp/does-not-exist-kfd-test");
        let dri = Path::new("/tmp/does-not-exist-dri-test");

        let result = enumerate_gpu_devices_from(nonexistent, dri);
        assert!(matches!(
            result,
            Err(EnumerationError::SysfsNotAvailable(_))
        ));
    }

    // -- KFD per-process VRAM read tests --

    fn create_kfd_vram_file(kfd_proc: &Path, pid: u32, gpu_id: u64, bytes: u64) {
        let proc_dir = kfd_proc.join(pid.to_string());
        fs::create_dir_all(&proc_dir).unwrap();
        fs::write(proc_dir.join(format!("vram_{gpu_id}")), bytes.to_string()).unwrap();
    }

    #[test]
    fn test_read_kfd_vram_single_device() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("proc");
        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 51110,
            cu_count: None,
            gfx_target_version: None,
        }];
        create_kfd_vram_file(&kfd_proc, 1234, 51110, 1_073_741_824);

        let result = read_kfd_vram_for_pid_from(&kfd_proc, 1234, &devices);
        assert_eq!(result.len(), 1);
        assert_eq!(result["0000:75:00.0"], 1_073_741_824);
    }

    #[test]
    fn test_read_kfd_vram_multiple_devices() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("proc");
        let devices = vec![
            GpuDevice {
                pci_bdf: "0000:05:00.0".into(),
                render_minor: 128,
                gpu_id: 51110,
                cu_count: None,
                gfx_target_version: None,
            },
            GpuDevice {
                pci_bdf: "0000:46:00.0".into(),
                render_minor: 129,
                gpu_id: 22099,
                cu_count: None,
                gfx_target_version: None,
            },
            GpuDevice {
                pci_bdf: "0000:85:00.0".into(),
                render_minor: 130,
                gpu_id: 39621,
                cu_count: None,
                gfx_target_version: None,
            },
            GpuDevice {
                pci_bdf: "0000:c6:00.0".into(),
                render_minor: 131,
                gpu_id: 47737,
                cu_count: None,
                gfx_target_version: None,
            },
        ];
        create_kfd_vram_file(&kfd_proc, 100, 51110, 5_000_000_000);
        create_kfd_vram_file(&kfd_proc, 100, 22099, 0);
        create_kfd_vram_file(&kfd_proc, 100, 39621, 3_000_000_000);
        // gpu_id 47737 not written — simulates no VRAM used on that GPU

        let result = read_kfd_vram_for_pid_from(&kfd_proc, 100, &devices);
        assert_eq!(
            result.get("0000:05:00.0").copied().unwrap_or(0),
            5_000_000_000
        );
        assert_eq!(result.get("0000:46:00.0").copied().unwrap_or(0), 0);
        assert_eq!(
            result.get("0000:85:00.0").copied().unwrap_or(0),
            3_000_000_000
        );
        assert_eq!(result.get("0000:c6:00.0"), None); // file not present
    }

    #[test]
    fn test_read_kfd_vram_missing_proc_dir() {
        let nonexistent = Path::new("/tmp/does-not-exist-kfd-proc-test");
        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 51110,
            cu_count: None,
            gfx_target_version: None,
        }];

        let result = read_kfd_vram_for_pid_from(nonexistent, 9999, &devices);
        assert!(result.is_empty());
    }

    #[test]
    fn test_read_kfd_vram_empty_device_list() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("proc");
        let result = read_kfd_vram_for_pid_from(&kfd_proc, 1234, &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_read_kfd_vram_malformed_content() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("proc");
        let proc_dir = kfd_proc.join("1234");
        fs::create_dir_all(&proc_dir).unwrap();
        fs::write(proc_dir.join("vram_51110"), "not_a_number\n").unwrap();

        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 51110,
            cu_count: None,
            gfx_target_version: None,
        }];
        let result = read_kfd_vram_for_pid_from(&kfd_proc, 1234, &devices);
        assert!(result.is_empty());
    }

    // -- Host PID resolution tests --

    fn create_host_proc_status(host_proc: &Path, host_pid: u32, nspid_line: &str) {
        let pid_dir = host_proc.join(host_pid.to_string());
        fs::create_dir_all(&pid_dir).unwrap();
        fs::write(pid_dir.join("status"), nspid_line).unwrap();
    }

    #[test]
    fn test_resolve_host_pid_bare_metal() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("kfd_proc");
        let host_proc = tmp.path().join("host_proc");

        // Our PID exists in KFD sysfs → bare metal, return as-is.
        let our_pid = 1234u32;
        let kfd_pid_dir = kfd_proc.join(our_pid.to_string());
        fs::create_dir_all(&kfd_pid_dir).unwrap();
        fs::write(kfd_pid_dir.join("vram_9091"), "0").unwrap();

        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 9091,
            cu_count: None,
            gfx_target_version: None,
        }];

        let result = resolve_host_pid_with(our_pid, &kfd_proc, &host_proc, &devices);
        assert_eq!(result, Some(1234));
    }

    #[test]
    fn test_resolve_host_pid_container_nspid() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("kfd_proc");
        let host_proc = tmp.path().join("host_proc");
        fs::create_dir_all(&kfd_proc).unwrap();
        fs::create_dir_all(&host_proc).unwrap();

        // Container PID 42 maps to host PID 98765.
        create_host_proc_status(
            &host_proc,
            98765,
            "Name:\tpython3\nPid:\t98765\nNSpid:\t98765\t42\n",
        );
        // Another process (not us).
        create_host_proc_status(
            &host_proc,
            11111,
            "Name:\tbash\nPid:\t11111\nNSpid:\t11111\t1\n",
        );

        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 9091,
            cu_count: None,
            gfx_target_version: None,
        }];

        let result = resolve_host_pid_with(42, &kfd_proc, &host_proc, &devices);
        assert_eq!(result, Some(98765));
    }

    #[test]
    fn test_resolve_host_pid_no_host_proc() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("kfd_proc");
        let nonexistent = tmp.path().join("no_such_dir");
        fs::create_dir_all(&kfd_proc).unwrap();

        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 9091,
            cu_count: None,
            gfx_target_version: None,
        }];

        let result = resolve_host_pid_with(42, &kfd_proc, &nonexistent, &devices);
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_host_pid_no_matching_nspid() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("kfd_proc");
        let host_proc = tmp.path().join("host_proc");
        fs::create_dir_all(&kfd_proc).unwrap();
        fs::create_dir_all(&host_proc).unwrap();

        // No process has container PID 42.
        create_host_proc_status(
            &host_proc,
            11111,
            "Name:\tbash\nPid:\t11111\nNSpid:\t11111\t99\n",
        );

        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 9091,
            cu_count: None,
            gfx_target_version: None,
        }];

        let result = resolve_host_pid_with(42, &kfd_proc, &host_proc, &devices);
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_host_pid_single_nspid_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("kfd_proc");
        let host_proc = tmp.path().join("host_proc");
        fs::create_dir_all(&kfd_proc).unwrap();
        fs::create_dir_all(&host_proc).unwrap();

        // Host process (not in a PID namespace) has single NSpid value.
        create_host_proc_status(&host_proc, 42, "Name:\tpython3\nPid:\t42\nNSpid:\t42\n");

        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 9091,
            cu_count: None,
            gfx_target_version: None,
        }];

        // Single NSpid = no namespace nesting, skip (need len >= 2).
        let result = resolve_host_pid_with(42, &kfd_proc, &host_proc, &devices);
        assert_eq!(result, None);
    }

    #[test]
    fn test_resolve_host_pid_nested_namespaces() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("kfd_proc");
        let host_proc = tmp.path().join("host_proc");
        fs::create_dir_all(&host_proc).unwrap();

        // DinD scenario: host → K8s pod → Docker container (3 PID namespaces).
        // NSpid shows all three: host_pid, pod_pid, container_pid.
        create_host_proc_status(
            &host_proc,
            1287304,
            "Name:\tpython3\nPid:\t1287304\nNSpid:\t1287304\t5617\t1\n",
        );

        let devices = vec![GpuDevice {
            pci_bdf: "0000:75:00.0".into(),
            render_minor: 128,
            gpu_id: 9091,
            cu_count: None,
            gfx_target_version: None,
        }];

        // Container PID is 1 (innermost), should resolve to host PID 1287304.
        let result = resolve_host_pid_with(1, &kfd_proc, &host_proc, &devices);
        assert_eq!(result, Some(1287304));
    }

    #[test]
    fn test_resolve_host_pid_empty_devices_skips_kfd_check() {
        let tmp = tempfile::tempdir().unwrap();
        let kfd_proc = tmp.path().join("kfd_proc");
        let host_proc = tmp.path().join("host_proc");
        fs::create_dir_all(&host_proc).unwrap();

        // No KFD devices → skip KFD fast path, go straight to host proc scan.
        create_host_proc_status(
            &host_proc,
            99999,
            "Name:\tpython3\nPid:\t99999\nNSpid:\t99999\t7\n",
        );

        let result = resolve_host_pid_with(7, &kfd_proc, &host_proc, &[]);
        assert_eq!(result, Some(99999));
    }
}
