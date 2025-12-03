use std::time::Instant;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use image::DynamicImage;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use async_trait::async_trait;
use crate::ml::backend::{MLBackend, Face};

#[derive(Serialize)]
struct InferenceRequest {
    image: String,  // Base64 encoded
}

#[derive(Deserialize)]
struct DetectionResponse {
    boxes: Vec<Vec<f32>>,
    scores: Vec<f32>,
}

#[derive(Deserialize)]
struct LivenessResponse {
    score: f32,
    is_live: bool,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f32>,
}

pub struct DockerBackend {
    client: reqwest::blocking::Client,
    endpoint: String,
}

impl DockerBackend {
    pub fn new(endpoint: &str) -> Result<Self> {
        tracing::info!("Initializing Docker ML backend...");
        tracing::info!("Endpoint: {}", endpoint);
        
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .context("Failed to create HTTP client")?;
        
        // Wait for service to be ready
        let health_url = format!("{}/health", endpoint);
        let start = Instant::now();
        let timeout = std::time::Duration::from_secs(30);
        
        tracing::info!("Waiting for Docker inference service to be ready...");
        loop {
            match client.get(&health_url).send() {
                Ok(resp) if resp.status().is_success() => {
                    tracing::info!("✓ Docker inference service is ready!");
                    break;
                }
                Ok(resp) => {
                    tracing::warn!("Service responded with status: {}", resp.status());
                }
                Err(e) => {
                    tracing::debug!("Waiting for service: {}", e);
                }
            }
            
            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for Docker inference service. Is container running?");
            }
            
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
        
        Ok(Self {
            client,
            endpoint: endpoint.to_string(),
        })
    }
    
    fn encode_image(&self, image: &DynamicImage) -> Result<String> {
        let mut buffer = Vec::new();
        image.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
            .context("Failed to encode image to PNG")?;
        Ok(BASE64.encode(&buffer))
    }
}

#[async_trait]
impl MLBackend for DockerBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let image_b64 = self.encode_image(image)?;
        let request = InferenceRequest { image: image_b64 };
        
        let url = format!("{}/detect", self.endpoint);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send detection request")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Detection request failed: {}", response.status());
        }
        
        let detection: DetectionResponse = response.json()
            .context("Failed to parse detection response")?;
        
        // Return first face above threshold
        for (bbox, &score) in detection.boxes.iter().zip(detection.scores.iter()) {
            if score > 0.5 {
                let (width, height) = image.dimensions();
                return Ok(Some(Face {
                    bbox: (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                    confidence: score,
                    frame_dimensions: (width, height),
                }));
            }
        }
        
        Ok(None)
    }
    
    async fn check_liveness(&self, image: &DynamicImage, _face: &Face) -> Result<bool> {
        let image_b64 = self.encode_image(face_image)?;
        let request = InferenceRequest { image: image_b64 };
        
        let url = format!("{}/liveness", self.endpoint);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send liveness request")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Liveness request failed: {}", response.status());
        }
        
        let liveness: LivenessResponse = response.json()
            .context("Failed to parse liveness response")?;
        
        Ok(liveness.is_live)
    }
    
    async fn extract_embedding(&self, image: &DynamicImage, _face: &Face) -> Result<Vec<f32>> {
        let image_b64 = self.encode_image(face_image)?;
        let request = InferenceRequest { image: image_b64 };
        
        let url = format!("{}/embed", self.endpoint);
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .context("Failed to send embedding request")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Embedding request failed: {}", response.status());
        }
        
        let embedding: EmbeddingResponse = response.json()
            .context("Failed to parse embedding response")?;
        
        Ok(embedding.embedding)
    }
    
    fn is_ready(&self) -> bool {
        true
    }
    
    fn name(&self) -> &'static str {
        "Docker (ONNX Runtime + ROCm)"
    }
}
