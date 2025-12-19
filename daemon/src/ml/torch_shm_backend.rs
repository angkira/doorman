use crate::ml::backend::{Face, MLBackend};
use anyhow::{Context, Result};
use async_trait::async_trait;
use image::DynamicImage;
use shared_memory::{Shmem, ShmemConf};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use tracing::{debug, info};

const SHM_SIZE: usize = 1920 * 1080 * 3; // Max frame size (RGB)
const SOCKET_PATH: &str = "/tmp/doorman-inference.sock";

/// Shared memory segment for zero-copy frame transfer
/// Wrapped in unsafe impl Send/Sync since we control access via Mutex
struct ShmSegment {
    shmem: Shmem,
    name: String,
}

unsafe impl Send for ShmSegment {}
unsafe impl Sync for ShmSegment {}

impl ShmSegment {
    fn new(name: &str) -> Result<Self> {
        let shmem = ShmemConf::new()
            .size(SHM_SIZE)
            .os_id(name)
            .create()
            .context("Failed to create shared memory")?;
        
        Ok(Self {
            shmem,
            name: name.to_string(),
        })
    }

    fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        let ptr = self.shmem.as_ptr() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }
}

/// PyTorch backend with shared memory optimization
pub struct TorchShmBackend {
    process: Mutex<Option<Child>>,
    socket: Mutex<UnixStream>,
    shm_buffers: [ShmSegment; 2],
    current_buffer: AtomicUsize,
    device: String,
}

impl TorchShmBackend {
    pub fn new(models_dir: &str, device: &str) -> Result<Self> {
        info!("Initializing PyTorch Shared Memory backend...");
        info!("Models directory: {:?}", models_dir);
        info!("Device: {}", device);

        // Create two shared memory segments for double buffering
        let shm_name_0 = format!("doorman_shm_{}_0", std::process::id());
        let shm_name_1 = format!("doorman_shm_{}_1", std::process::id());
        let shm0 = ShmSegment::new(&shm_name_0)?;
        let shm1 = ShmSegment::new(&shm_name_1)?;
        info!("Created shared memory buffers: {} and {}", shm_name_0, shm_name_1);

        // Remove old socket if exists
        let _ = std::fs::remove_file(SOCKET_PATH);

        // Start inference subprocess with venv activated
        let project_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let venv_path = project_root.join(".venv");
        let python_bin = venv_path.join("bin/python3");
        let script_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/ml/torch_inference_shm.py");
        
        if !python_bin.exists() {
            anyhow::bail!("Python venv not found at {}. Run: uv sync", python_bin.display());
        }
        
        debug!("Using Python: {}", python_bin.display());
        debug!("Script: {}", script_path.display());
        
        // Set VIRTUAL_ENV to activate venv properly
        let mut cmd = Command::new(&python_bin);
        cmd.arg("-u")
            .arg(script_path)
            .env("VIRTUAL_ENV", &venv_path)
            .env("PATH", format!("{}:{}", venv_path.join("bin").display(), std::env::var("PATH").unwrap_or_default()))
            .env("DOORMAN_MODELS_DIR", models_dir)
            .env("DOORMAN_DEVICE", device)
            .env("DOORMAN_SHM_NAME_0", &shm_name_0)
            .env("DOORMAN_SHM_NAME_1", &shm_name_1)
            .env("DOORMAN_SOCKET_PATH", SOCKET_PATH);
        
        // Activate venv if using it (DO NOT set PYTHONPATH - venv handles it!)
        if venv_path.exists() {
            cmd.env("VIRTUAL_ENV", venv_path.display().to_string());
            let venv_bin = venv_path.join("bin");
            if let Ok(path) = std::env::var("PATH") {
                cmd.env("PATH", format!("{}:{}", venv_bin.display(), path));
            } else {
                cmd.env("PATH", venv_bin.display().to_string());
            }
            // Clear PYTHONPATH to let venv work properly
            cmd.env_remove("PYTHONPATH");
        }
        
        let process = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to start inference subprocess")?;

        info!("Started inference subprocess (PID: {})", process.id());

        // Wait for socket to be ready
        let mut attempts = 0;
        let socket = loop {
            if attempts >= 50 {
                return Err(anyhow::anyhow!("Inference server failed to start after 5 seconds"));
            }
            
            match UnixStream::connect(SOCKET_PATH) {
                Ok(s) => break s,
                Err(_) => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    attempts += 1;
                }
            }
        };

        info!("Connected to inference server via Unix socket");

        Ok(Self {
            process: Mutex::new(Some(process)),
            socket: Mutex::new(socket),
            shm_buffers: [shm0, shm1],
            current_buffer: AtomicUsize::new(0),
            device: device.to_string(),
        })
    }

    fn send_command(&self, cmd: &str, width: u32, height: u32, buffer_index: usize) -> Result<String> {
        let mut socket = self.socket.lock().unwrap();
        
        // Send command: "command width height buffer_index\n"
        let msg = format!("{} {} {} {}\n", cmd, width, height, buffer_index);
        socket.write_all(msg.as_bytes())?;
        socket.flush()?;
        
        // Read response (newline-terminated JSON)
        let mut response = String::new();
        let mut buf = [0u8; 1];
        loop {
            socket.read_exact(&mut buf)?;
            if buf[0] == b'\n' {
                break;
            }
            response.push(buf[0] as char);
        }
        
        Ok(response)
    }
}

#[async_trait]
impl MLBackend for TorchShmBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let faces = self.detect_faces_sync(image)?;
        Ok(faces.into_iter().next())
    }
    
    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        let (is_live, _score) = self.check_liveness_sync(image, face)?;
        Ok(is_live)
    }
    
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        self.extract_embedding_sync(image, face)
    }
    
    fn is_ready(&self) -> bool {
        self.socket.lock().is_ok()
    }
    
    fn name(&self) -> &'static str {
        "TorchShm"
    }
}

impl TorchShmBackend {
    fn detect_faces_sync(&self, image: &DynamicImage) -> Result<Vec<Face>> {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();
        let data = rgb.as_raw();

        // Write frame to current shared memory buffer
        let buffer_index = self.current_buffer.load(Ordering::Relaxed);
        {
            let shm = self.shm_buffers[buffer_index].shmem.as_ptr() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), shm, data.len());
            }
        }

        // Send detect command with buffer index
        let response = self.send_command("detect", width, height, buffer_index)?;
        
        // Switch to other buffer for next operation
        self.current_buffer.store(1 - buffer_index, Ordering::Relaxed);
        
        // Parse JSON response
        let result: serde_json::Value = serde_json::from_str(&response)
            .context("Failed to parse detection response")?;
        
        if let Some(error) = result.get("error") {
            return Err(anyhow::anyhow!("Detection failed: {}", error));
        }
        
        let detections = result["detections"].as_array()
            .context("Invalid detection response format")?;
        
        let mut faces = Vec::new();
        for det in detections {
            let bbox = det["bbox"].as_array().unwrap();
            let x = bbox[0].as_f64().unwrap() as f32;
            let y = bbox[1].as_f64().unwrap() as f32;
            let w = bbox[2].as_f64().unwrap() as f32;
            let h = bbox[3].as_f64().unwrap() as f32;
            
            faces.push(Face {
                bbox: (x, y, w, h),
                confidence: det["confidence"].as_f64().unwrap() as f32,
                frame_dimensions: (width, height),
            });
        }
        
        debug!("Detected {} faces", faces.len());
        Ok(faces)
    }

    fn check_liveness_sync(&self, image: &DynamicImage, face: &Face) -> Result<(bool, f32)> {
        let (x, y, w, h) = face.bbox;
        let face_img = image.crop_imm(x as u32, y as u32, w as u32, h as u32);
        let rgb = face_img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let data = rgb.as_raw();

        // Write face to current shared memory buffer
        let buffer_index = self.current_buffer.load(Ordering::Relaxed);
        {
            let shm = self.shm_buffers[buffer_index].shmem.as_ptr() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), shm, data.len());
            }
        }

        // Send liveness command with buffer index
        let response = self.send_command("liveness", width, height, buffer_index)?;
        
        // Switch to other buffer for next operation
        self.current_buffer.store(1 - buffer_index, Ordering::Relaxed);
        
        let result: serde_json::Value = serde_json::from_str(&response)?;
        
        if let Some(error) = result.get("error") {
            return Err(anyhow::anyhow!("Liveness check failed: {}", error));
        }
        
        let score = result["score"].as_f64().unwrap() as f32;
        let is_live = score > 0.5;
        
        Ok((is_live, score))
    }

    fn extract_embedding_sync(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        let (x, y, w, h) = face.bbox;
        let face_img = image.crop_imm(x as u32, y as u32, w as u32, h as u32);
        let rgb = face_img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let data = rgb.as_raw();

        // Write face to current shared memory buffer
        let buffer_index = self.current_buffer.load(Ordering::Relaxed);
        {
            let shm = self.shm_buffers[buffer_index].shmem.as_ptr() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), shm, data.len());
            }
        }

        // Send embedding command with buffer index
        let response = self.send_command("embedding", width, height, buffer_index)?;
        
        // Switch to other buffer for next operation
        self.current_buffer.store(1 - buffer_index, Ordering::Relaxed);
        
        let result: serde_json::Value = serde_json::from_str(&response)?;
        
        if let Some(error) = result.get("error") {
            return Err(anyhow::anyhow!("Embedding extraction failed: {}", error));
        }
        
        let embedding = result["embedding"].as_array()
            .context("Invalid embedding response format")?
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        
        Ok(embedding)
    }

    fn warmup(&self) -> Result<()> {
        info!("Warming up PyTorch Shared Memory backend...");
        
        // Send warmup command
        let response = self.send_command("warmup", 0, 0, 0)?;
        
        let result: serde_json::Value = serde_json::from_str(&response)?;
        
        if let Some(error) = result.get("error") {
            return Err(anyhow::anyhow!("Warmup failed: {}", error));
        }
        
        info!("✓ PyTorch backend warmed up successfully");
        Ok(())
    }
}

impl Drop for TorchShmBackend {
    fn drop(&mut self) {
        // Send shutdown command
        let _ = self.send_command("shutdown", 0, 0, 0);
        
        // Kill subprocess
        if let Ok(mut process_opt) = self.process.lock() {
            if let Some(mut process) = process_opt.take() {
                let _ = process.kill();
                let _ = process.wait();
            }
        }
        
        // Cleanup socket
        let _ = std::fs::remove_file(SOCKET_PATH);
        
        info!("PyTorch Shared Memory backend shut down");
    }
}
