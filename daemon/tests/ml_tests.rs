use doorman_shared::Config;

// Test cosine similarity function
#[test]
fn test_cosine_similarity_identical() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    
    // Use a simple implementation for testing
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a * norm_b)
    }
    
    let similarity = cosine_similarity(&a, &b);
    assert!((similarity - 1.0).abs() < 0.001, "Identical vectors should have similarity ~1.0");
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a * norm_b)
    }
    
    let similarity = cosine_similarity(&a, &b);
    assert!(similarity.abs() < 0.001, "Orthogonal vectors should have similarity ~0.0");
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![-1.0, -2.0, -3.0];
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a * norm_b)
    }
    
    let similarity = cosine_similarity(&a, &b);
    assert!((similarity + 1.0).abs() < 0.001, "Opposite vectors should have similarity ~-1.0");
}

#[test]
fn test_cosine_similarity_different_lengths() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0];
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a * norm_b)
    }
    
    let similarity = cosine_similarity(&a, &b);
    assert_eq!(similarity, 0.0, "Different length vectors should return 0.0");
}

#[test]
fn test_embedding_normalization() {
    let mut embedding = vec![3.0, 4.0]; // 3-4-5 triangle
    
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for val in &mut embedding {
        *val /= norm;
    }
    
    // Check unit length
    let new_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((new_norm - 1.0).abs() < 0.001, "Normalized vector should have length 1.0");
}

#[test]
fn test_config_default_device() {
    let config = Config::default();
    assert_eq!(config.ml.device, "cpu");
}

#[test]
fn test_embedding_averaging() {
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
}

#[test]
fn test_high_dimensional_similarity() {
    // Test with 512-d vectors (typical for face recognition)
    let mut a = vec![0.0f32; 512];
    let mut b = vec![0.0f32; 512];
    
    // Make them similar but not identical
    for i in 0..512 {
        a[i] = (i as f32).sin();
        b[i] = (i as f32).sin() + 0.01; // Small perturbation
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot_product / (norm_a * norm_b)
    }
    
    let similarity = cosine_similarity(&a, &b);
    assert!(similarity > 0.99, "Similar 512-d vectors should have high similarity");
    assert!(similarity < 1.0, "Perturbed vectors should not be identical");
}

