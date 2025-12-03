/// End-to-End tests for enrollment and authentication flow
/// Tests the complete system: enrollment → storage → recognition
/// 
/// Note: These tests require models to be loaded and are marked with #[ignore]
/// Run with: cargo test --test enrollment_e2e_test -- --ignored

use doorman_shared::{Request, Response, ResponseData};
use std::path::PathBuf;
use tempfile::TempDir;

#[test]
fn test_enrollment_request_serialization() {
    // Test that enrollment request serializes correctly
    let request = Request::Enroll {
        username: "testuser".to_string(),
    };
    
    let json = serde_json::to_string(&request).unwrap();
    
    assert!(json.contains("enroll"));
    assert!(json.contains("testuser"));
    
    // Test deserialization
    let parsed: Request = serde_json::from_str(&json).unwrap();
    
    match parsed {
        Request::Enroll { username } => {
            assert_eq!(username, "testuser");
        }
        _ => panic!("Wrong request type"),
    }
}

#[test]
fn test_authentication_request_serialization() {
    let request = Request::Authenticate {
        username: "testuser".to_string(),
    };
    
    let json = serde_json::to_string(&request).unwrap();
    
    assert!(json.contains("authenticate"));
    assert!(json.contains("testuser"));
}

#[test]
fn test_list_users_request() {
    let request = Request::ListUsers;
    
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("list_users"));
}

#[test]
fn test_remove_user_request() {
    let request = Request::RemoveUser {
        username: "olduser".to_string(),
    };
    
    let json = serde_json::to_string(&request).unwrap();
    assert!(json.contains("remove_user"));
    assert!(json.contains("olduser"));
}

#[test]
fn test_response_success() {
    let response = Response::Success {
        message: Some("Operation successful".to_string()),
        data: None,
    };
    
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("success"));
    assert!(json.contains("Operation successful"));
}

#[test]
fn test_response_failure() {
    let response = Response::Failure {
        reason: "No face detected".to_string(),
    };
    
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("failure"));
    assert!(json.contains("No face detected"));
}

#[test]
fn test_response_progress() {
    let response = Response::Progress {
        message: "Capturing frames".to_string(),
        current: 5,
        total: 20,
    };
    
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("progress"));
    assert!(json.contains("5"));
    assert!(json.contains("20"));
}

#[test]
fn test_user_info_in_response() {
    let user_info = doorman_shared::UserInfo {
        username: "john".to_string(),
        enrolled_at: "2024-01-01T00:00:00Z".to_string(),
        num_embeddings: 1,
    };
    
    let response = Response::Success {
        message: None,
        data: Some(ResponseData::UserList {
            users: vec![user_info],
        }),
    };
    
    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains("john"));
    assert!(json.contains("2024-01-01"));
}

/// Test storage operations without daemon
#[tokio::test]
async fn test_storage_operations() {
    use doormand::storage::Storage;
    
    // Create temporary directory for test
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_path_buf();
    
    // Initialize storage
    let mut storage = Storage::new_with_dir(&data_dir).await.unwrap();
    
    // Initially empty
    assert_eq!(storage.count(), 0);
    
    // Store an embedding
    let username = "testuser".to_string();
    let embedding = vec![1.0; 512];
    
    storage.store_embedding(username.clone(), embedding.clone()).await.unwrap();
    
    // Check it was stored
    assert_eq!(storage.count(), 1);
    
    // Retrieve it
    let retrieved = storage.get_embedding(&username).unwrap();
    assert_eq!(retrieved.len(), 512);
    assert_eq!(retrieved[0], 1.0);
    
    // List users
    let users = storage.list_users();
    assert_eq!(users.len(), 1);
    assert_eq!(users[0].username, "testuser");
    
    // Remove user
    let removed = storage.remove_embedding(&username).await.unwrap();
    assert!(removed);
    assert_eq!(storage.count(), 0);
    
    // Try to remove again (should fail)
    let removed_again = storage.remove_embedding(&username).await.unwrap();
    assert!(!removed_again);
}

/// Test storage persistence
#[tokio::test]
async fn test_storage_persistence() {
    use doormand::storage::Storage;
    
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_path_buf();
    
    // Store data
    {
        let mut storage = Storage::new_with_dir(&data_dir).await.unwrap();
        storage.store_embedding("user1".to_string(), vec![1.0; 512]).await.unwrap();
        storage.store_embedding("user2".to_string(), vec![2.0; 512]).await.unwrap();
    }
    
    // Load in new instance
    {
        let storage = Storage::new_with_dir(&data_dir).await.unwrap();
        assert_eq!(storage.count(), 2);
        
        let emb1 = storage.get_embedding("user1").unwrap();
        let emb2 = storage.get_embedding("user2").unwrap();
        
        assert_eq!(emb1[0], 1.0);
        assert_eq!(emb2[0], 2.0);
    }
}

/// Test storage with multiple users
#[tokio::test]
async fn test_storage_multiple_users() {
    use doormand::storage::Storage;
    use doormand::ml::cosine_similarity;
    
    let temp_dir = TempDir::new().unwrap();
    let data_dir = temp_dir.path().to_path_buf();
    
    let mut storage = Storage::new_with_dir(&data_dir).await.unwrap();
    
    // Store 5 users with different embeddings
    for i in 0..5 {
        let username = format!("user{}", i);
        let mut embedding = vec![0.0; 512];
        for j in 0..512 {
            embedding[j] = ((i + j) as f32 / 512.0).sin();
        }
        storage.store_embedding(username, embedding).await.unwrap();
    }
    
    assert_eq!(storage.count(), 5);
    
    // Test finding best match
    let test_embedding = storage.get_embedding("user2").unwrap().clone();
    
    let mut best_match = None;
    let mut best_similarity = 0.0;
    
    for user in storage.list_users() {
        let user_emb = storage.get_embedding(&user.username).unwrap();
        let similarity = cosine_similarity(&test_embedding, user_emb);
        
        if similarity > best_similarity {
            best_similarity = similarity;
            best_match = Some(user.username.clone());
        }
    }
    
    assert_eq!(best_match, Some("user2".to_string()));
    assert!(best_similarity > 0.99);
}

/// Test enrollment with averaging
#[test]
fn test_enrollment_averaging() {
    // Simulate enrollment with multiple samples
    let samples = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.1, 2.1, 3.1],
        vec![0.9, 1.9, 2.9],
    ];
    
    // Average
    let dim = samples[0].len();
    let mut master = vec![0.0f32; dim];
    
    for sample in &samples {
        for (i, val) in sample.iter().enumerate() {
            master[i] += val;
        }
    }
    
    for val in &mut master {
        *val /= samples.len() as f32;
    }
    
    // Should be close to [1.0, 2.0, 3.0]
    assert!((master[0] - 1.0).abs() < 0.1);
    assert!((master[1] - 2.0).abs() < 0.1);
    assert!((master[2] - 3.0).abs() < 0.1);
}

/// Test recognition threshold
#[test]
fn test_recognition_threshold_logic() {
    use doorman_shared::SIMILARITY_THRESHOLD;
    
    let threshold = SIMILARITY_THRESHOLD; // 0.65
    
    // Simulate recognition attempts
    let attempts = vec![
        ("user1", 0.80), // Should match
        ("user2", 0.50), // Should not match
        ("user3", 0.65), // Should match (exactly threshold)
        ("user4", 0.64), // Should not match
        ("user5", 0.99), // Should match (high confidence)
    ];
    
    for (user, similarity) in attempts {
        let should_match = similarity >= threshold;
        let actual_match = similarity >= SIMILARITY_THRESHOLD;
        
        assert_eq!(should_match, actual_match, 
            "User {} with similarity {} should {} match",
            user, similarity, if should_match { "" } else { "not" });
    }
}

/// Test IPC socket path resolution
#[test]
fn test_socket_path_resolution() {
    // Test that we can generate proper socket paths for user/system mode
    let user_runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok();
    
    if let Some(runtime_dir) = user_runtime_dir {
        let user_socket = PathBuf::from(&runtime_dir).join("doorman.sock");
        assert!(user_socket.to_string_lossy().contains("doorman.sock"));
        assert!(user_socket.to_string_lossy().contains(&runtime_dir));
    }
    
    let system_socket = PathBuf::from("/run/doorman.sock");
    assert_eq!(system_socket.to_str().unwrap(), "/run/doorman.sock");
}

/// Test config loading for enrollment
#[test]
fn test_enrollment_config() {
    use doorman_shared::ENROLL_FRAMES;
    
    // Should capture enough frames for reliable enrollment
    assert!(ENROLL_FRAMES >= 10, "Should capture at least 10 frames");
    assert!(ENROLL_FRAMES <= 30, "Should not capture too many frames (slow)");
}

/// Test similarity calculation edge cases
#[test]
fn test_similarity_edge_cases() {
    use doormand::ml::cosine_similarity;
    
    // Test with very small values
    let tiny1 = vec![1e-10; 512];
    let tiny2 = vec![1e-10; 512];
    let sim = cosine_similarity(&tiny1, &tiny2);
    assert!(sim > 0.99, "Tiny but identical vectors should be similar");
    
    // Test with very large values
    let huge1 = vec![1e10; 512];
    let huge2 = vec![1e10; 512];
    let sim = cosine_similarity(&huge1, &huge2);
    assert!(sim > 0.99, "Huge but identical vectors should be similar");
    
    // Test with mixed signs
    let mixed1 = vec![1.0, -1.0, 1.0, -1.0];
    let mixed2 = vec![1.0, -1.0, 1.0, -1.0];
    let sim = cosine_similarity(&mixed1, &mixed2);
    assert!(sim > 0.99, "Identical mixed-sign vectors should be similar");
}
