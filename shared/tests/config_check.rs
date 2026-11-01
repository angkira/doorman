use doorman_shared::config::ConfigError;
use doorman_shared::config_check::{check_str, render_reference_markdown};
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/config")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

#[test]
fn old_schema_fixture_passes_with_defaults_for_new_sections() {
    let report = check_str(&fixture("old_schema.toml")).expect("old config must stay valid");
    assert!(report.unknown_keys.is_empty(), "{:?}", report.unknown_keys);
    assert!(report.deprecated_keys.is_empty());
    assert_eq!(report.config.schema_version, 1);
    assert_eq!(report.config.decision.accept_n, 3);
}

#[test]
fn unknown_keys_fixture_reports_every_unknown_path() {
    let report = check_str(&fixture("unknown_keys.toml")).expect("still valid");
    assert_eq!(
        report.unknown_keys,
        vec!["camera.widht", "mystery_top_level", "not_a_section"]
    );
}

#[test]
fn invalid_fixture_is_an_invalid_config_error() {
    let err = check_str(&fixture("invalid.toml")).unwrap_err();
    assert!(matches!(err, ConfigError::Invalid(_)), "{err}");
}

#[test]
fn shipped_example_configs_have_no_unknown_keys() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    for rel in ["doorman.toml", "packaging/doorman.toml.example"] {
        let text = std::fs::read_to_string(root.join(rel)).unwrap();
        let report = check_str(&text).unwrap_or_else(|e| panic!("{rel}: {e}"));
        assert!(
            report.unknown_keys.is_empty(),
            "{rel}: {:?}",
            report.unknown_keys
        );
    }
}

/// docs/CONFIG.md is generated: regenerate with `doorman config dump-defaults > docs/CONFIG.md`.
#[test]
fn config_reference_doc_is_up_to_date() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/CONFIG.md");
    let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
    assert_eq!(
        on_disk,
        render_reference_markdown(),
        "docs/CONFIG.md is stale; run: doorman config dump-defaults > docs/CONFIG.md"
    );
}

#[test]
fn removed_dataset_key_is_reported_as_deprecated_only() {
    let report = check_str(&fixture("deprecated_keys.toml")).expect("still valid");
    // Reported once, as deprecated; not additionally as unknown.
    assert!(report.unknown_keys.is_empty(), "{:?}", report.unknown_keys);
    assert_eq!(report.deprecated_keys.len(), 1);
    let (key, hint) = &report.deprecated_keys[0];
    assert_eq!(key, "dataset.raw_face_frame_bundle_size");
    assert!(hint.contains("raw_frame_bundle_size"), "{hint}");
    assert!(hint.contains("ignored"), "{hint}");
    // The key it points to stays valid.
    assert_eq!(report.config.dataset.raw_frame_bundle_size, 5);
}
