/// Pod configuration for device setup.
#[derive(Debug, Clone)]
pub struct PodConfig {
    pub gpu_uuids: Vec<String>,
    pub isolation: Option<String>,
}
