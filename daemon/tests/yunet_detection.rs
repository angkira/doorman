//! End-to-end YuNet detection test through the real ORT backend.
//!
//! Loads a real face JPEG and the YuNet ONNX model from the runtime models
//! directory (`~/.local/share/doorman/models`) and asserts that:
//!   * a face is detected with a plausible, in-frame, normalized bbox,
//!   * the TV-test-pattern frame yields NO detection.
//!
//! The reference (Python onnxruntime) decode for `face_lena.jpg` (512x512)
//! produces normalized bbox ~ (0.409, 0.355, 0.277, 0.402), score ~0.91.
//! The test asserts the Rust path lands close to those values.

use doorman_shared::Config;
use doormand::ml::MLPipeline;
use std::env;
use std::path::Path;

fn models_present(config: &Config) -> bool {
    Path::new(&config.ml.models_dir)
        .join("face_detection_yunet_2023mar.onnx")
        .exists()
}

fn test_config() -> Config {
    let home = env::var("HOME").unwrap();
    let mut config = Config::default();
    config.ml.models_dir = format!("{}/.local/share/doorman/models", home);
    config
}

#[tokio::test]
async fn yunet_detects_real_face() {
    let config = test_config();
    if !models_present(&config) {
        eprintln!("SKIP: YuNet model not present in {}", config.ml.models_dir);
        return;
    }

    let pipeline = MLPipeline::new(&config)
        .await
        .expect("Failed to create ML pipeline");

    let img = image::open("tests/fixtures/face_lena.jpg").expect("load face fixture");

    let face = pipeline
        .detect_face(&img)
        .await
        .expect("detect_face errored")
        .expect("expected a face on the lena fixture, got None");

    let (x, y, w, h) = face.bbox;
    println!(
        "lena detection: bbox_norm=({:.3},{:.3},{:.3},{:.3}) conf={:.3} landmarks={:?}",
        x, y, w, h, face.confidence, face.landmarks
    );

    // In-frame, sane size.
    assert!((0.0..=1.0).contains(&x), "x out of range: {}", x);
    assert!((0.0..=1.0).contains(&y), "y out of range: {}", y);
    assert!(w > 0.05 && w < 0.95, "width implausible: {}", w);
    assert!(h > 0.05 && h < 0.95, "height implausible: {}", h);
    assert!(x + w <= 1.001 && y + h <= 1.001, "bbox extends past frame");
    assert!(face.confidence > 0.6, "confidence below threshold: {}", face.confidence);

    // Match the Python reference decode (tolerance for resize-filter diffs).
    assert!((x - 0.409).abs() < 0.06, "x far from reference: {}", x);
    assert!((y - 0.355).abs() < 0.06, "y far from reference: {}", y);
    assert!((w - 0.277).abs() < 0.06, "w far from reference: {}", w);
    assert!((h - 0.402).abs() < 0.06, "h far from reference: {}", h);

    // 5 landmarks present and inside the frame.
    let lms = face.landmarks.expect("YuNet must supply landmarks");
    for (lx, ly) in lms.iter() {
        assert!((0.0..=1.0).contains(lx) && (0.0..=1.0).contains(ly), "landmark out of frame: ({lx},{ly})");
    }
}

#[tokio::test]
async fn yunet_no_face_on_test_pattern() {
    let config = test_config();
    if !models_present(&config) {
        eprintln!("SKIP: YuNet model not present in {}", config.ml.models_dir);
        return;
    }

    let pipeline = MLPipeline::new(&config)
        .await
        .expect("Failed to create ML pipeline");

    let img = image::open("tests/fixtures/no_face_pattern.jpg").expect("load no-face fixture");
    let result = pipeline.detect_face(&img).await.expect("detect_face errored");
    println!("test-pattern detection: {:?}", result.as_ref().map(|f| f.bbox));
    assert!(result.is_none(), "expected no face on TV test pattern");
}
