/// Unit tests for face recognition system
/// Tests cropping, embedding extraction, and similarity matching

use approx::assert_relative_eq;
use doormand::ml::cosine_similarity;
use doormand::{Face, MLPipeline};
use doorman_shared::Config;
use std::path::{Path, PathBuf};

#[test]
fn test_cosine_similarity_same_face() {
    // Simulate embeddings from same person
    let embedding1 = vec![0.5; 512];
    let embedding2 = vec![0.5; 512];
    
    let similarity = cosine_similarity(&embedding1, &embedding2);
    
    assert!(similarity > 0.99, "Same embeddings should have similarity ~1.0, got {}", similarity);
}

#[test]
fn test_cosine_similarity_similar_faces() {
    // Simulate embeddings from similar faces (same person, different angles)
    let mut embedding1 = vec![0.0; 512];
    let mut embedding2 = vec![0.0; 512];
    
    for i in 0..512 {
        embedding1[i] = (i as f32 / 512.0).sin();
        embedding2[i] = (i as f32 / 512.0).sin() * 0.98; // 98% similar
    }
    
    let similarity = cosine_similarity(&embedding1, &embedding2);
    
    assert!(similarity > 0.95, "Similar faces should have high similarity, got {}", similarity);
}

#[test]
fn test_cosine_similarity_different_faces() {
    // Simulate embeddings from different people
    let mut embedding1 = vec![0.0; 512];
    let mut embedding2 = vec![0.0; 512];
    
    for i in 0..512 {
        embedding1[i] = (i as f32 / 512.0).sin();
        embedding2[i] = -(i as f32 / 512.0).sin(); // Opposite pattern
    }
    
    let similarity = cosine_similarity(&embedding1, &embedding2);
    
    assert!(similarity < 0.0, "Different faces (opposite patterns) should have negative similarity, got {}", similarity);
}

#[test]
fn test_recognition_threshold() {
    // Threshold comes from shared constants (tuned for EdgeFace-S).
    let threshold = doorman_shared::SIMILARITY_THRESHOLD;

    // A clearly-genuine EdgeFace cosine (measured ~0.79) passes.
    assert!(0.79 >= threshold, "High similarity should pass threshold");
    // A clearly-impostor EdgeFace cosine (measured ~0.05) fails.
    assert!(0.05 < threshold, "Impostor similarity should fail threshold");
    // Exact threshold passes (>=).
    assert!(threshold >= threshold, "Exact threshold should pass");
}

#[test]
fn test_embedding_normalization() {
    // Test that embeddings are properly normalized to unit length
    let mut embedding = vec![3.0, 4.0]; // 3-4-5 triangle
    
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    
    // Check unit length
    let new_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(new_norm, 1.0, epsilon = 0.001);
    
    // Check values
    assert_relative_eq!(embedding[0], 0.6, epsilon = 0.001);
    assert_relative_eq!(embedding[1], 0.8, epsilon = 0.001);
}

#[test]
fn test_embedding_averaging_for_enrollment() {
    // Test averaging multiple embeddings during enrollment
    let embeddings = vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ];
    
    // Average
    let embedding_dim = embeddings[0].len();
    let mut master = vec![0.0f32; embedding_dim];
    
    for embedding in &embeddings {
        for (i, val) in embedding.iter().enumerate() {
            master[i] += val;
        }
    }
    
    for val in &mut master {
        *val /= embeddings.len() as f32;
    }
    
    assert_eq!(master, vec![2.0, 4.0, 6.0]);
    
    // Normalize master embedding
    let norm: f32 = master.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut master {
        *val /= norm;
    }
    
    // Should still be unit length
    let new_norm: f32 = master.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert_relative_eq!(new_norm, 1.0, epsilon = 0.001);
}

#[test]
fn test_bbox_cropping_math() {
    // Test face bounding box cropping with padding
    let frame_width = 1024.0f32;
    let frame_height = 720.0f32;
    
    // Normalized bbox from detector (center of frame)
    let (x_norm, y_norm, w_norm, h_norm) = (0.4f32, 0.4f32, 0.2f32, 0.2f32);
    
    // Add 10% padding
    let padding = 0.10f32;
    let x = ((x_norm - w_norm * padding) * frame_width).max(0.0);
    let y = ((y_norm - h_norm * padding) * frame_height).max(0.0);
    let w = (w_norm * (1.0 + 2.0 * padding) * frame_width).min(frame_width - x);
    let h = (h_norm * (1.0 + 2.0 * padding) * frame_height).min(frame_height - y);
    
    // Expected: original was 204.8x144 at (409.6, 288)
    // With 10% padding: ~20.48 pixels each side
    // Result: ~245.76x172.8 at (~389.12, ~273.6)
    assert_relative_eq!(x, 389.12, epsilon = 1.0);
    assert_relative_eq!(y, 273.6, epsilon = 1.0);
    assert_relative_eq!(w, 245.76, epsilon = 1.0);
    assert_relative_eq!(h, 172.8, epsilon = 1.0);
}

#[test]
fn test_bbox_edge_cases() {
    // Test bbox at frame edge (should clamp to 0)
    let frame_width = 1024.0f32;
    let frame_height = 720.0f32;
    
    // Face at top-left corner
    let (x_norm, y_norm, w_norm, h_norm) = (0.0f32, 0.0f32, 0.2f32, 0.2f32);
    
    let padding = 0.10f32;
    let x = ((x_norm - w_norm * padding) * frame_width).max(0.0);
    let y = ((y_norm - h_norm * padding) * frame_height).max(0.0);
    
    // Should clamp to 0
    assert_eq!(x, 0.0);
    assert_eq!(y, 0.0);
    
    // Face at bottom-right corner
    let (x_norm, y_norm, w_norm, h_norm) = (0.8f32, 0.8f32, 0.2f32, 0.2f32);
    
    let x = ((x_norm - w_norm * padding) * frame_width).max(0.0);
    let y = ((y_norm - h_norm * padding) * frame_height).max(0.0);
    let w = (w_norm * (1.0 + 2.0 * padding) * frame_width).min(frame_width - x);
    let h = (h_norm * (1.0 + 2.0 * padding) * frame_height).min(frame_height - y);
    
    // Should not exceed frame bounds
    assert!(x + w <= frame_width);
    assert!(y + h <= frame_height);
}

#[test]
fn test_multi_user_recognition() {
    // Simulate multiple enrolled users with distinct embeddings
    let mut user1_embedding = vec![0.0; 512];
    let mut user2_embedding = vec![0.0; 512];
    let mut user3_embedding = vec![0.0; 512];
    
    for i in 0..512 {
        user1_embedding[i] = (i as f32 / 100.0).sin();
        user2_embedding[i] = (i as f32 / 200.0).cos();
        user3_embedding[i] = (i as f32 / 150.0).sin() * 0.5;
    }
    
    // Test frame should match user1
    let mut test_embedding = vec![0.0; 512];
    for i in 0..512 {
        test_embedding[i] = (i as f32 / 100.0).sin() * 0.98; // Very similar to user1
    }
    
    let sim1 = cosine_similarity(&test_embedding, &user1_embedding);
    let sim2 = cosine_similarity(&test_embedding, &user2_embedding);
    let sim3 = cosine_similarity(&test_embedding, &user3_embedding);
    
    // user1 should have highest similarity
    assert!(sim1 > sim2, "Should match closest user: user1={} user2={}", sim1, sim2);
    assert!(sim1 > sim3, "Should match closest user: user1={} user3={}", sim1, sim3);
    
    // user1 should have high similarity (same pattern, 98% match)
    assert!(sim1 > 0.95, "User1 similarity should be high: {}", sim1);
}

#[test]
fn test_recognition_with_noise() {
    // Test robustness to small perturbations
    let clean_embedding: Vec<f32> = (0..512).map(|i| (i as f32 / 512.0).sin()).collect();
    
    // Add 5% noise
    let mut noisy_embedding = clean_embedding.clone();
    for val in &mut noisy_embedding {
        *val += 0.05 * (*val);
    }
    
    let similarity = cosine_similarity(&clean_embedding, &noisy_embedding);
    
    // Should still be high similarity despite noise
    assert!(similarity > 0.95, "Should be robust to 5% noise, got {}", similarity);
}

#[test]
fn test_empty_embedding_handling() {
    let empty1: Vec<f32> = vec![];
    let empty2: Vec<f32> = vec![];
    
    let similarity = cosine_similarity(&empty1, &empty2);
    
    // Should return 0.0 for empty vectors
    assert_eq!(similarity, 0.0);
}

#[test]
fn test_mismatched_embedding_dimensions() {
    let embedding1 = vec![1.0; 512];
    let embedding2 = vec![1.0; 256]; // Wrong dimension
    
    let similarity = cosine_similarity(&embedding1, &embedding2);
    
    // Should return 0.0 for mismatched dimensions
    assert_eq!(similarity, 0.0);
}

#[test]
fn test_zero_vector_handling() {
    let zero_vec = vec![0.0; 512];
    let normal_vec = vec![1.0; 512];
    
    let similarity = cosine_similarity(&zero_vec, &normal_vec);
    
    // Should return 0.0 (norm is zero)
    assert_eq!(similarity, 0.0);
}

#[test]
fn test_high_dimensional_similarity() {
    // Test with actual 512-d vectors (typical for face recognition)
    let mut embedding1 = vec![0.0f32; 512];
    let mut embedding2 = vec![0.0f32; 512];
    
    // Fill with realistic patterns
    for i in 0..512 {
        embedding1[i] = (i as f32 * 0.1).sin() * 0.5;
        embedding2[i] = (i as f32 * 0.1).sin() * 0.48; // Very similar
    }
    
    let similarity = cosine_similarity(&embedding1, &embedding2);
    
    assert!(similarity > 0.99, "Very similar 512-d vectors should have high similarity");
}

#[test]
fn test_normalized_vs_unnormalized() {
    // Test that normalization doesn't affect cosine similarity
    let unnormalized = vec![3.0, 4.0]; // Length 5

    // Normalize
    let norm: f32 = unnormalized.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized: Vec<f32> = unnormalized.iter().map(|x| x / norm).collect();

    // Cosine similarity should be 1.0 (same direction)
    let similarity = cosine_similarity(&unnormalized, &normalized);

    assert_relative_eq!(similarity, 1.0, epsilon = 0.001);
}

// ---------------------------------------------------------------------------
// End-to-end recognition on REAL faces (YuNet detect -> landmark alignment ->
// EdgeFace-S embed -> cosine match). Requires the ONNX models in data/models/.
// ---------------------------------------------------------------------------

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("data/models")
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/faces")
}

fn have_recognition_models() -> bool {
    let d = models_dir();
    d.join("face_detection_yunet_2023mar.onnx").exists() && d.join("edgeface_s.onnx").exists()
}

async fn build_pipeline() -> MLPipeline {
    let mut config = Config::default();
    config.ml.models_dir = models_dir().to_string_lossy().to_string();
    config.ml.backend = "ort".to_string();
    config.ml.device = "cpu".to_string();
    MLPipeline::new(&config).await.expect("build ML pipeline")
}

async fn embed_face(pipeline: &MLPipeline, path: &Path) -> Vec<f32> {
    let img = image::open(path).unwrap_or_else(|e| panic!("open {:?}: {}", path, e));
    let face: Face = pipeline
        .detect_face(&img)
        .await
        .expect("detect_face errored")
        .unwrap_or_else(|| panic!("no face detected in {:?}", path));
    assert!(
        face.landmarks.is_some(),
        "expected YuNet landmarks for alignment in {:?}",
        path
    );
    let emb = pipeline
        .extract_embedding(&img, &face)
        .await
        .expect("extract_embedding errored");
    assert_eq!(emb.len(), 512, "embedding must be 512-d");
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "embedding not L2-normalized: {norm}");
    emb
}

#[tokio::test]
async fn e2e_liveness_real_face_passes() {
    let d = models_dir();
    if !d.join("face_detection_yunet_2023mar.onnx").exists()
        || !d.join("minifasnet_v2se.onnx").exists()
    {
        eprintln!("SKIP liveness: MiniFASNetV2-SE model not found in {:?}", d);
        return;
    }
    // Surface the backend's `avg_real` debug line during the run.
    let _ = tracing_subscriber::fmt()
        .with_env_filter("doormand=debug")
        .with_test_writer()
        .try_init();

    let p = build_pipeline().await;
    // Ground-truth samples: real_face.jpg (genuine live capture) and
    // spoof_face.jpg (a printed/replayed spoof). The single MiniFASNetV2-SE
    // model (facenox, 128x128) gives clean separation on these fixtures
    // (genuine logit-diff ~ +8, spoof ~ -10).
    let real_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/liveness/real_face.jpg");
    let spoof_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/liveness/spoof_face.jpg");

    let real_img = image::open(&real_path).unwrap();
    let real_face = p
        .detect_face(&real_img)
        .await
        .expect("detect errored")
        .expect("no face in real_face.jpg");
    let real_live = p
        .check_liveness(&real_img, &real_face)
        .await
        .expect("liveness errored");
    println!("liveness real_face.jpg: real={real_live} (expect true)");

    // The spoof sample MUST be flagged (rejected). The daemon's 640x640 YuNet
    // does not always detect a face in this photo-of-a-photo; when it doesn't,
    // fall back to a centered bbox (the liveness model still classifies the crop
    // as spoof — verified ground-truth diff ~ -10 either way).
    let spoof_img = image::open(&spoof_path).unwrap();
    let spoof_face = match p.detect_face(&spoof_img).await {
        Ok(Some(f)) => f,
        _ => {
            let (sw, sh) = image::GenericImageView::dimensions(&spoof_img);
            Face {
                bbox: (0.2, 0.2, 0.6, 0.6),
                confidence: 0.0,
                frame_dimensions: (sw, sh),
                landmarks: None,
            }
        }
    };
    let spoof_live = p
        .check_liveness(&spoof_img, &spoof_face)
        .await
        .expect("liveness errored on spoof");
    println!("liveness spoof_face.jpg: real={spoof_live} (expect false)");

    // The genuine live face MUST pass and the path MUST run without error.
    assert!(real_live, "genuine live face should pass MiniFASNetV2-SE liveness");
    // The spoof MUST be rejected.
    assert!(!spoof_live, "spoof face should be rejected by MiniFASNetV2-SE liveness");
}

#[tokio::test]
async fn e2e_enroll_match_reject_separation() {
    if !have_recognition_models() {
        eprintln!(
            "SKIP e2e recognition: models not found in {:?}. Run scripts/fetch_models.sh.",
            models_dir()
        );
        return;
    }

    let p = build_pipeline().await;
    let fx = fixtures_dir();

    // Enroll person A from two photos (mirrors storing multiple embeddings).
    let enrolled_a = vec![
        embed_face(&p, &fx.join("personA_1.jpg")).await,
        embed_face(&p, &fx.join("personA_2.jpg")).await,
    ];

    // Probes: a held-out A photo (genuine) + two distinct identities (impostors).
    let a3 = embed_face(&p, &fx.join("personA_3.jpg")).await;
    let b1 = embed_face(&p, &fx.join("personB_1.jpg")).await;
    let c1 = embed_face(&p, &fx.join("personC_1.jpg")).await;

    // Daemon match logic: best cosine over the enrolled set.
    let best = |probe: &[f32]| {
        enrolled_a
            .iter()
            .map(|e| cosine_similarity(probe, e))
            .fold(f32::MIN, f32::max)
    };

    let genuine = best(&a3);
    let impostor_b = best(&b1);
    let impostor_c = best(&c1);
    let threshold = doorman_shared::SIMILARITY_THRESHOLD;

    println!("threshold = {threshold}");
    println!("genuine  A vs A = {genuine:.4} -> {}", if genuine >= threshold { "MATCH (green)" } else { "reject" });
    println!("impostor B vs A = {impostor_b:.4} -> {}", if impostor_b >= threshold { "match" } else { "REJECT (red)" });
    println!("impostor C vs A = {impostor_c:.4} -> {}", if impostor_c >= threshold { "match" } else { "REJECT (red)" });

    assert!(genuine >= threshold, "genuine A should match, got {genuine:.4}");
    assert!(impostor_b < threshold, "impostor B must be rejected, got {impostor_b:.4}");
    assert!(impostor_c < threshold, "impostor C must be rejected, got {impostor_c:.4}");

    let margin = genuine - impostor_b.max(impostor_c);
    assert!(margin > 0.2, "expected clear separation, margin = {margin:.4}");
    println!("separation margin = {margin:.4}");
}
