use anyhow::{Context, Result, anyhow};
use image::DynamicImage;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use base64::{Engine as _, engine::general_purpose};
use async_trait::async_trait;

use super::backend::{Face, MLBackend};

pub struct TorchBackend {
    process: Mutex<PythonProcess>,
}

struct PythonProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    request_id: u64,
}

#[derive(Serialize)]
struct JsonRpcRequest {
    id: u64,
    method: String,
    params: serde_json::Value,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    id: Option<u64>,
    result: Option<serde_json::Value>,
    error: Option<String>,
}

impl TorchBackend {
    pub fn new(models_dir: &Path) -> Result<Self> {
        tracing::info!("Initializing PyTorch backend with ROCm...");
        
        // Find Python script
        let script_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/ml/torch_inference.py");
        
        if !script_path.exists() {
            return Err(anyhow!("PyTorch inference script not found: {:?}", script_path));
        }
        
        // Find Python from venv or use system python
        let python_cmd = std::env::var("VIRTUAL_ENV")
            .map(|venv| format!("{}/bin/python3", venv))
            .unwrap_or_else(|_| "python3".to_string());
        
        // Start Python process
        tracing::info!("Starting Python inference process with: {}", python_cmd);
        let mut child = Command::new(&python_cmd)
            .arg(&script_path)
            .arg(models_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn Python inference process")?;
        
        let stdin = child.stdin.take().context("Failed to get stdin")?;
        let stdout = child.stdout.take().context("Failed to get stdout")?;
        let stdout = BufReader::new(stdout);
        
        let process = PythonProcess {
            child,
            stdin,
            stdout,
            request_id: 0,
        };
        
        let backend = Self {
            process: Mutex::new(process),
        };
        
        // Warm up models
        tracing::info!("Warming up PyTorch models...");
        backend.warmup_models()?;
        tracing::info!("✓ PyTorch models warmed up and ready");
        
        Ok(backend)
    }
    
    fn warmup_models(&self) -> Result<()> {
        // Create dummy image for warmup
        let dummy_img = DynamicImage::new_rgb8(640, 480);
        
        // Warmup: detect_face
        match self.call_method("detect_face", serde_json::json!({
            "image_base64": Self::image_to_base64(&dummy_img)?
        })) {
            Ok(_) => tracing::info!("  ✓ Face detector warmed up"),
            Err(e) => tracing::warn!("  ✗ Failed to warm up face detector: {}", e),
        }
        
        // Warmup: check_liveness (with dummy face)
        match self.call_method("check_liveness", serde_json::json!({
            "image_base64": Self::image_to_base64(&dummy_img)?,
            "bbox": [100, 100, 200, 200]
        })) {
            Ok(_) => tracing::info!("  ✓ Liveness detector warmed up"),
            Err(e) => tracing::warn!("  ✗ Failed to warm up liveness detector: {}", e),
        }
        
        // Warmup: extract_embedding (with dummy face)
        match self.call_method("extract_embedding", serde_json::json!({
            "image_base64": Self::image_to_base64(&dummy_img)?,
            "bbox": [100, 100, 200, 200]
        })) {
            Ok(_) => tracing::info!("  ✓ Face recognizer warmed up"),
            Err(e) => tracing::warn!("  ✗ Failed to warm up face recognizer: {}", e),
        }
        
        Ok(())
    }
    
    fn image_to_base64(image: &DynamicImage) -> Result<String> {
        let mut buffer = Vec::new();
        image.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)?;
        Ok(general_purpose::STANDARD.encode(&buffer))
    }
    
    fn call_method(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let mut process = self.process.lock().unwrap();
        process.request_id += 1;
        let id = process.request_id;
        
        let request = JsonRpcRequest {
            id,
            method: method.to_string(),
            params,
        };
        
        // Send request
        let request_json = serde_json::to_string(&request)?;
        writeln!(process.stdin, "{}", request_json)?;
        process.stdin.flush()?;
        
        // Read response
        let mut response_line = String::new();
        process.stdout.read_line(&mut response_line)?;
        
        let response: JsonRpcResponse = serde_json::from_str(&response_line)?;
        
        if let Some(error) = response.error {
            return Err(anyhow!("Python error: {}", error));
        }
        
        response.result.ok_or_else(|| anyhow!("No result in response"))
    }
}

#[async_trait]
impl MLBackend for TorchBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        // Encode image as JPEG
        let mut buffer = Vec::new();
        image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Jpeg
        )?;
        let image_base64 = general_purpose::STANDARD.encode(&buffer);
        
        let params = serde_json::json!({
            "image_data": image_base64,
            "width": image.width(),
            "height": image.height(),
        });
        
        let result = self.call_method("detect_faces", params)?;
        let detections = result["detections"].as_array()
            .ok_or_else(|| anyhow!("Invalid detections format"))?;
        
        if detections.is_empty() {
            return Ok(None);
        }
        
        // Return first detection
        let det = &detections[0];
        let x = det["x"].as_f64().unwrap() as f32;
        let y = det["y"].as_f64().unwrap() as f32;
        let w = det["width"].as_f64().unwrap() as f32;
        let h = det["height"].as_f64().unwrap() as f32;
        let confidence = det["confidence"].as_f64().unwrap() as f32;
        
        // Normalize to [0, 1]
        let img_w = image.width() as f32;
        let img_h = image.height() as f32;
        
        Ok(Some(Face {
            bbox: (x / img_w, y / img_h, w / img_w, h / img_h),
            confidence,
            frame_dimensions: (image.width(), image.height()),
        }))
    }

    async fn check_liveness(&self, image: &DynamicImage, _face: &Face) -> Result<bool> {
        // Encode image
        let mut buffer = Vec::new();
        image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Jpeg
        )?;
        let image_base64 = general_purpose::STANDARD.encode(&buffer);
        
        let params = serde_json::json!({
            "face_crop": image_base64,
        });
        
        let result = self.call_method("check_liveness", params)?;
        Ok(result["is_live"].as_bool().unwrap_or(false))
    }

    async fn extract_embedding(&self, image: &DynamicImage, _face: &Face) -> Result<Vec<f32>> {
        // Encode image
        let mut buffer = Vec::new();
        image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Jpeg
        )?;
        let image_base64 = general_purpose::STANDARD.encode(&buffer);
        
        let params = serde_json::json!({
            "face_crop": image_base64,
        });
        
        let result = self.call_method("extract_embedding", params)?;
        let embedding_vec = result["embedding"].as_array()
            .ok_or_else(|| anyhow!("Invalid embedding format"))?;
        
        let embedding: Vec<f32> = embedding_vec.iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        
        Ok(embedding)
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "PyTorch + ROCm"
    }
}

impl Drop for TorchBackend {
    fn drop(&mut self) {
        if let Ok(mut process) = self.process.lock() {
            let _ = process.child.kill();
        }
    }
}
