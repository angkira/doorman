//! No shipped config may select a GPU. A GPU needs an explicit owner edit:
//! `ml.device` rocm/cuda/gpu, `ml.backend = "ncnn"`, `ml.gpu_uuid` or
//! `ml.vulkan_device` (see docs/host-facts-answers.md, "Round 3").

use doorman_shared::config_check::check_str;
use doorman_shared::Config;
use std::path::PathBuf;

#[test]
fn shipped_configs_select_no_gpu() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut configs = vec![("built-in defaults".to_string(), Config::default())];
    for rel in ["doorman.toml", "packaging/doorman.toml.example"] {
        let text = std::fs::read_to_string(root.join(rel)).unwrap();
        let report = check_str(&text).unwrap_or_else(|e| panic!("{rel}: {e}"));
        configs.push((rel.to_string(), report.config));
    }
    for (name, config) in configs {
        let ml = &config.ml;
        assert_eq!(ml.device, "cpu", "{name}: ml.device");
        assert_ne!(ml.backend, "ncnn", "{name}: ml.backend");
        assert!(
            ml.gpu_uuid.as_deref().unwrap_or("").is_empty(),
            "{name}: ml.gpu_uuid"
        );
        assert!(
            ml.vulkan_device.as_deref().unwrap_or("").is_empty(),
            "{name}: ml.vulkan_device"
        );
    }
}
