use anyhow::{Context, Result};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView};
use tracing::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::os::unix::net::UnixStream;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Mutex;

use super::{Face, MLBackend};

/// Unix Domain Socket backend for ML inference
/// Communicates with Python ONNX Runtime server via binary protocol
pub struct SocketBackend {
    socket_path: String,
    stream: Mutex<Option<UnixStream>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonResponse {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    detections: Option<Vec<DetectionData>>,
    #[serde(default)]
    is_live: Option<bool>,
    #[serde(default)]
    score: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DetectionData {
    bbox: [f32; 4],
    confidence: f32,
}

impl SocketBackend {
    pub fn new(socket_path: impl AsRef<Path>) -> Result<Self> {
        let socket_path = socket_path.as_ref().to_string_lossy().into_owned();
        
        info!("Initializing Socket ML backend...");
        info!("Socket path: {}", socket_path);
        
        let backend = Self {
            socket_path: socket_path.clone(),
            stream: Mutex::new(None),
        };
        
        // Connect
        backend.connect()?;
        
        // Health check
        backend.ping()?;
        
        info!("Socket backend initialized successfully!");
        
        Ok(backend)
    }
    
    fn connect(&self) -> Result<()> {
        debug!("Connecting to socket: {}", self.socket_path);
        
        let stream = UnixStream::connect(&self.socket_path)
            .context("Failed to connect to ML inference socket")?;
        
        // Set timeouts
        use std::time::Duration;
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        
        *self.stream.lock().unwrap() = Some(stream);
        
        debug!("Connected successfully");
        Ok(())
    }
    
    fn ensure_connected(&self) -> Result<()> {
        let mut guard = self.stream.lock().unwrap();
        if guard.is_none() {
            drop(guard);
            self.connect()?;
        }
        Ok(())
    }
    
    fn ping(&self) -> Result<()> {
        self.ensure_connected()?;
        
        let mut guard = self.stream.lock().unwrap();
        let stream = guard.as_mut().unwrap();
        
        // Send ping request (type=0)
        stream.write_all(&[0u8])?;
        
        // Receive response
        let response = Self::recv_json_response_from(stream)?;
        
        if response.status.as_deref() != Some("ok") {
            anyhow::bail!("Ping failed: unexpected response");
        }
        
        debug!("Ping successful");
        Ok(())
    }
    
    fn send_frame_to(stream: &mut UnixStream, frame: &DynamicImage) -> Result<()> {
        // Convert to RGB8
        let rgb = frame.to_rgb8();
        let width = rgb.width();
        let height = rgb.height();
        let channels = 3u32;
        
        // Send header: [width:u32][height:u32][channels:u32]
        let header = [
            width.to_le_bytes(),
            height.to_le_bytes(),
            channels.to_le_bytes(),
        ].concat();
        
        stream.write_all(&header)?;
        
        // Send frame data
        stream.write_all(rgb.as_raw())?;
        
        Ok(())
    }
    
    fn recv_json_response_from(stream: &mut UnixStream) -> Result<JsonResponse> {
        // Read response type and length: [type:u8][len:u32]
        let mut header = [0u8; 5];
        stream.read_exact(&mut header)?;
        
        let response_type = header[0];
        let data_len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
        
        if response_type != 1 {
            anyhow::bail!("Expected JSON response (type=1), got type={}", response_type);
        }
        
        // Read JSON data
        let mut data = vec![0u8; data_len];
        stream.read_exact(&mut data)?;
        
        // Parse JSON
        let response: JsonResponse = serde_json::from_slice(&data)
            .context("Failed to parse JSON response")?;
        
        Ok(response)
    }
    
    fn recv_binary_response_from(stream: &mut UnixStream) -> Result<Vec<f32>> {
        // Read response type and length: [type:u8][len:u32]
        let mut header = [0u8; 5];
        stream.read_exact(&mut header)?;
        
        let response_type = header[0];
        let data_len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
        
        if response_type != 2 {
            anyhow::bail!("Expected binary response (type=2), got type={}", response_type);
        }
        
        // Read binary data
        let mut data = vec![0u8; data_len];
        stream.read_exact(&mut data)?;
        
        // Convert to f32 array
        let floats = data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Ok(floats)
    }
}

#[async_trait]
impl MLBackend for SocketBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        // TODO: Implement properly - for now return None
        // Need to make self mutable or use interior mutability (Mutex)
        warn!("Socket backend detect_face not yet implemented");
        Ok(None)
    }
    
    async fn check_liveness(&self, _image: &DynamicImage, _face: &Face) -> Result<bool> {
        // TODO: Implement properly
        warn!("Socket backend check_liveness not yet implemented");
        Ok(true)
    }
    
    async fn extract_embedding(&self, _image: &DynamicImage, _face: &Face) -> Result<Vec<f32>> {
        // TODO: Implement properly
        warn!("Socket backend extract_embedding not yet implemented");
        Ok(vec![0.0; 128])
    }
    
    fn is_ready(&self) -> bool {
        self.stream.lock().unwrap().is_some()
    }
    
    fn name(&self) -> &'static str {
        "Socket (Unix Domain Socket)"
    }
}

impl Drop for SocketBackend {
    fn drop(&mut self) {
        if let Some(stream) = self.stream.lock().unwrap().take() {
            drop(stream);
            debug!("Socket backend connection closed");
        }
    }
}
