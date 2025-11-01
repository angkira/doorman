use tempfile::TempDir;
use std::path::PathBuf;

// Mock storage for testing
mod storage_mock {
    use super::*;
    use std::collections::HashMap;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct UserEmbedding {
        pub username: String,
        pub embedding: Vec<f32>,
        pub enrolled_at: String,
    }

    pub struct Storage {
        embeddings: HashMap<String, UserEmbedding>,
        file_path: PathBuf,
    }

    impl Storage {
        pub async fn new_with_path(path: PathBuf) -> anyhow::Result<Self> {
            let embeddings = if path.exists() {
                Self::load_embeddings(&path)?
            } else {
                HashMap::new()
            };

            Ok(Self {
                embeddings,
                file_path: path,
            })
        }

        fn load_embeddings(path: &PathBuf) -> anyhow::Result<HashMap<String, UserEmbedding>> {
            let data = std::fs::read(path)?;
            let embeddings: Vec<UserEmbedding> = bincode::deserialize(&data)?;
            let map = embeddings
                .into_iter()
                .map(|e| (e.username.clone(), e))
                .collect();
            Ok(map)
        }

        fn save_embeddings(&self) -> anyhow::Result<()> {
            let embeddings_vec: Vec<UserEmbedding> = self.embeddings.values().cloned().collect();
            let data = bincode::serialize(&embeddings_vec)?;
            std::fs::write(&self.file_path, data)?;
            Ok(())
        }

        pub async fn store_embedding(&mut self, username: String, embedding: Vec<f32>) -> anyhow::Result<()> {
            let enrolled_at = chrono::Local::now().to_rfc3339();
            let user_embedding = UserEmbedding {
                username: username.clone(),
                embedding,
                enrolled_at,
            };
            self.embeddings.insert(username, user_embedding);
            self.save_embeddings()?;
            Ok(())
        }

        pub fn get_embedding(&self, username: &str) -> Option<&Vec<f32>> {
            self.embeddings.get(username).map(|e| &e.embedding)
        }

        pub async fn remove_embedding(&mut self, username: &str) -> anyhow::Result<bool> {
            if self.embeddings.remove(username).is_some() {
                self.save_embeddings()?;
                Ok(true)
            } else {
                Ok(false)
            }
        }

        pub fn count(&self) -> usize {
            self.embeddings.len()
        }
    }
}

#[tokio::test]
async fn test_storage_create_and_store() {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().join("test_embeddings.bin");

    let mut storage = storage_mock::Storage::new_with_path(storage_path.clone())
        .await
        .unwrap();

    // Create a test embedding
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    storage
        .store_embedding("testuser".to_string(), embedding.clone())
        .await
        .unwrap();

    // Verify it was stored
    let retrieved = storage.get_embedding("testuser").unwrap();
    assert_eq!(retrieved, &embedding);
    assert_eq!(storage.count(), 1);
}

#[tokio::test]
async fn test_storage_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().join("test_embeddings.bin");

    // Store in first instance
    {
        let mut storage = storage_mock::Storage::new_with_path(storage_path.clone())
            .await
            .unwrap();
        let embedding = vec![1.0, 2.0, 3.0];
        storage
            .store_embedding("alice".to_string(), embedding)
            .await
            .unwrap();
    }

    // Load in second instance
    {
        let storage = storage_mock::Storage::new_with_path(storage_path.clone())
            .await
            .unwrap();
        let retrieved = storage.get_embedding("alice").unwrap();
        assert_eq!(retrieved, &vec![1.0, 2.0, 3.0]);
    }
}

#[tokio::test]
async fn test_storage_remove() {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().join("test_embeddings.bin");

    let mut storage = storage_mock::Storage::new_with_path(storage_path)
        .await
        .unwrap();

    // Store two users
    storage
        .store_embedding("user1".to_string(), vec![1.0])
        .await
        .unwrap();
    storage
        .store_embedding("user2".to_string(), vec![2.0])
        .await
        .unwrap();

    assert_eq!(storage.count(), 2);

    // Remove one
    let removed = storage.remove_embedding("user1").await.unwrap();
    assert!(removed);
    assert_eq!(storage.count(), 1);
    assert!(storage.get_embedding("user1").is_none());
    assert!(storage.get_embedding("user2").is_some());

    // Try to remove non-existent
    let removed = storage.remove_embedding("user3").await.unwrap();
    assert!(!removed);
}

#[tokio::test]
async fn test_storage_multiple_users() {
    let temp_dir = TempDir::new().unwrap();
    let storage_path = temp_dir.path().join("test_embeddings.bin");

    let mut storage = storage_mock::Storage::new_with_path(storage_path)
        .await
        .unwrap();

    // Store multiple users
    for i in 0..10 {
        let username = format!("user{}", i);
        let embedding = vec![i as f32; 512];
        storage
            .store_embedding(username, embedding)
            .await
            .unwrap();
    }

    assert_eq!(storage.count(), 10);

    // Verify all can be retrieved
    for i in 0..10 {
        let username = format!("user{}", i);
        let embedding = storage.get_embedding(&username).unwrap();
        assert_eq!(embedding.len(), 512);
        assert_eq!(embedding[0], i as f32);
    }
}

