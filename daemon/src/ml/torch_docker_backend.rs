//! PyTorch backend using Docker container with CUDA support
//! 
//! Uses shared memory for zero-copy image transfer and Unix socket for control

use super::backend::{Face, MLBackend};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use shared_memory::{Shmem, ShmemConf};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Child, Command};
use std::time::Duration;
use tracing::{info, warn};

const SOCKET_PATH: &str = "/tmp/doorman-inference.sock";
const SHM_SIZE: usize = 1280 * 720 * 3 + 8; // width * height * RGB + metadata

#[derive(Debug, Serialize, Deserialize)]
struct InferenceRequest {
    #[serde(rename = "type")]
    req_type: String,
    buffer: u8, // 0 or 1
}

#[derive(Debug, Deserialize)]
struct InferenceResponse {
    result: serde_json::Value,
}

pub struct TorchDockerBackend {
    container_id: String,
    shm_buffers: [Shmem; 2],
    socket: Option<UnixStream>,
    current_buffer: u8,
}

impl TorchDockerBackend {
    pub fn new(models_dir: &str, device: &str) -> Result<Self> {
        info!("Initializing PyTorch Docker backend with CUDA...");
        info!("Models directory: {}", models_dir);
        info!("Device: {}", device);

        let pid = std::process::id();
        let shm_name_0 = format!("doorman_shm_{}_0", pid);
        let shm_name_1 = format!("doorman_shm_{}_1", pid);

        // Create shared memory buffers
        let shm_0 = ShmemConf::new()
            .size(SHM_SIZE)
            .os_id(&shm_name_0)
            .create()
            .context("Failed to create shared memory buffer 0")?;
        
        let shm_1 = ShmemConf::new()
            .size(SHM_SIZE)
            .os_id(&shm_name_1)
            .create()
            .context("Failed to create shared memory buffer 1")?;

        info!("Created shared memory buffers: {} and {}", shm_name_0, shm_name_1);

        // Start Docker container
        info!("Starting CUDA Docker container...");
        let output = Command::new("docker")
            .args(&[
                "run",
                "-d",                    // Detached
                "--rm",                  // Auto-remove
                "--gpus", "all",         // GPU support
                "--ipc=host",            // Shared memory
                "-v", "/tmp:/tmp",       // Socket
                "-v", &format!("{}:/app/models:ro", models_dir),  // Models (read-only)
                "-e", &format!("MODELS_DIR=/app/models"),
                "-e", &format!("DEVICE={}", device),
                "-e", &format!("SHM_NAME_0={}", shm_name_0),
                "-e", &format!("SHM_NAME_1={}", shm_name_1),
                "-e", &format!("SOCKET_PATH={}", SOCKET_PATH),
                "doorman-cuda:latest",
                "python3", "torch_inference_shm.py"
            ])
            .output()
            .context("Failed to start Docker container")?;

        if !output.status.success() {
            return Err(anyhow!(
                "Docker failed to start: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let container_id = String::from_utf8(output.stdout)
            .context("Invalid container ID")?
            .trim()
            .to_string();

        info!("Started Docker container: {}", container_id);

        // Wait for inference server to be ready
        info!("Waiting for inference server...");
        let mut retries = 0;
        let socket = loop {
            if retries > 50 {
                return Err(anyhow!("Inference server failed to start after 5 seconds"));
            }
            
            match UnixStream::connect(SOCKET_PATH) {
                Ok(stream) => {
                    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                    break stream;
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(100));
                    retries += 1;
                }
            }
        };

        info!("Connected to inference server via Unix socket");

        Ok(Self {
            container_id,
            shm_buffers: [shm_0, shm_1],
            socket: Some(socket),
            current_buffer: 0,
        })
    }

    fn write_image_to_shm(&mut self, image: &DynamicImage) -> Result<()> {
        let rgb = image.to_rgb8();
        let (width, height) = (rgb.width(), rgb.height());
        
        let buffer_idx = self.current_buffer as usize;
        let shm = &self.shm_buffers[buffer_idx];
        
        unsafe {
            let ptr = shm.as_ptr() as *mut u8;
            // Write dimensions
            std::ptr::write(ptr as *mut u32, width);
            std::ptr::write(ptr.add(4) as *mut u32, height);
            // Write image data
            std::ptr::copy_nonoverlapping(
                rgb.as_raw().as_ptr(),
                ptr.add(8),
                rgb.as_raw().len()
            );
        }
        
        // Swap buffers for next call
        self.current_buffer = 1 - self.current_buffer;
        
        Ok(())
    }

    fn send_request(&mut self, req_type: &str) -> Result<serde_json::Value> {
        let socket = self.socket.as_mut()
            .ok_or_else(|| anyhow!("Socket not connected"))?;

        let buffer = 1 - self.current_buffer; // Use the buffer we just wrote to
        let request = InferenceRequest {
            req_type: req_type.to_string(),
            buffer,
        };

        let request_json = serde_json::to_string(&request)?;
        socket.write_all(request_json.as_bytes())?;
        socket.write_all(b"\n")?;
        socket.flush()?;

        let mut response_buf = String::new();
        let mut reader = std::io::BufReader::new(socket);
        use std::io::BufRead;
        reader.read_line(&mut response_buf)?;

        let response: InferenceResponse = serde_json::from_str(&response_buf)?;
        Ok(response.result)
    }
}

#[async_trait]
impl MLBackend for TorchDockerBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        // TODO: Implement detection
        Ok(None)
    }

    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        // TODO: Implement liveness
        Ok(true)
    }

    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        // TODO: Implement embedding
        Ok(vec![0.0; 512])
    }

    fn is_ready(&self) -> bool {
        self.socket.is_some()
    }

    fn name(&self) -> &'static str {
        "PyTorch Docker CUDA"
    }
}

impl Drop for TorchDockerBackend {
    fn drop(&mut self) {
        info!("Stopping Docker container: {}", self.container_id);
        let _ = Command::new("docker")
            .args(&["stop", &self.container_id])
            .output();
        info!("PyTorch Docker backend shut down");
    }
}
