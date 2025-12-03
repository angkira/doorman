use anyhow::{Context, Result};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView};
use serde_json::json;
use base64::Engine;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use tracing::info;

use super::{Face, MLBackend};

pub struct MIGraphXBackend {
    process: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<ChildStdin>>,
    stdout: Arc<Mutex<BufReader<ChildStdout>>>,
}

impl MIGraphXBackend {
    pub fn new(models_dir: &str) -> Result<Self> {
        info!("Initializing MIGraphX backend...");
        info!("Models directory: {:?}", models_dir);

        // Start Python inference server with uv run
        let script_path = "tools/torch_rocm_inference.py";
        let mut child = Command::new("uv")
            .arg("run")
            .arg("python")
            .arg(script_path)
            .arg(models_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())  // Ignore warnings
            .spawn()
            .context("Failed to start MIGraphX/ROCm inference server")?;

        let stdin = child.stdin.take().context("Failed to get stdin")?;
        let stdout = child.stdout.take().context("Failed to get stdout")?;
        let mut reader = BufReader::new(stdout);

        // Wait for ready signal (may take a few seconds due to model loading)
        let mut line = String::new();
        let max_attempts = 50;  // 50 attempts * 100ms = 5 seconds
        let mut attempts = 0;
        
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    anyhow::bail!("Inference server closed unexpectedly");
                }
                Ok(_) => {
                    // Try to parse as JSON
                    if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                        if msg["status"] == "ready" {
                            info!("✓ MIGraphX inference server started");
                            break;
                        }
                        if let Some(error) = msg.get("error") {
                            anyhow::bail!("MIGraphX server error: {}", error);
                        }
                    }
                    // Ignore non-JSON lines (warnings, etc.)
                }
                Err(e) => {
                    attempts += 1;
                    if attempts >= max_attempts {
                        anyhow::bail!("Timeout waiting for ready signal: {}", e);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }

        let backend = Self {
            process: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            stdout: Arc::new(Mutex::new(reader)),
        };

        info!("✓ MIGraphX backend initialized (models loaded in Python)");
        Ok(backend)
    }

    fn load_model(&self, name: &str, path: &str) -> Result<()> {
        let req = json!({
            "cmd": "load",
            "name": name,
            "path": path
        });

        let resp = self.send_request(&req)?;
        if resp.get("error").is_some() {
            anyhow::bail!("Failed to load model {}: {:?}", name, resp["error"]);
        }

        info!("✓ Loaded model: {}", name);
        Ok(())
    }

    fn send_request(&self, req: &serde_json::Value) -> Result<serde_json::Value> {
        let mut stdin = self.stdin.lock().unwrap();
        let mut stdout = self.stdout.lock().unwrap();

        // Send request
        let req_str = serde_json::to_string(req)?;
        writeln!(stdin, "{}", req_str)?;
        stdin.flush()?;

        // Read response
        let mut line = String::new();
        stdout.read_line(&mut line)?;
        let resp: serde_json::Value = serde_json::from_str(&line)?;

        Ok(resp)
    }

    fn preprocess_image(&self, image: &DynamicImage, target_size: (u32, u32)) -> Vec<f32> {
        let resized = image.resize_exact(
            target_size.0,
            target_size.1,
            image::imageops::FilterType::Triangle,
        );
        let rgb = resized.to_rgb8();

        let mut data = Vec::with_capacity((target_size.0 * target_size.1 * 3) as usize);
        for pixel in rgb.pixels() {
            data.push(pixel[0] as f32 / 255.0);
            data.push(pixel[1] as f32 / 255.0);
            data.push(pixel[2] as f32 / 255.0);
        }

        data
    }
}

#[async_trait]
impl MLBackend for MIGraphXBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let (width, height) = image.dimensions();
        
        // Encode image as JPEG base64
        let mut buf = Vec::new();
        image.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)?;
        let image_b64 = base64::engine::general_purpose::STANDARD.encode(&buf);

        let req = json!({
            "type": "detect",
            "image": image_b64
        });

        let resp = self.send_request(&req)?;
        if let Some(error) = resp.get("error") {
            anyhow::bail!("Detection failed: {:?}", error);
        }

        // Parse result
        if let Some(result) = resp.get("result") {
            let bbox = result["bbox"].as_array().context("Invalid bbox")?;
            let confidence = result["confidence"].as_f64().unwrap_or(0.0) as f32;
            
            Ok(Some(Face {
                bbox: (
                    bbox[0].as_f64().unwrap() as f32,
                    bbox[1].as_f64().unwrap() as f32,
                    bbox[2].as_f64().unwrap() as f32,
                    bbox[3].as_f64().unwrap() as f32,
                ),
                confidence,
                frame_dimensions: (width, height),
            }))
        } else {
            Ok(None)
        }
    }

    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        // Crop face from image
        let (x, y, w, h) = face.bbox;
        let (img_w, img_h) = image.dimensions();
        let crop_x = (x * img_w as f32) as u32;
        let crop_y = (y * img_h as f32) as u32;
        let crop_w = (w * img_w as f32) as u32;
        let crop_h = (h * img_h as f32) as u32;

        let face_crop = image.crop_imm(crop_x, crop_y, crop_w, crop_h);
        
        // Encode crop as JPEG base64
        let mut buf = Vec::new();
        face_crop.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)?;
        let crop_b64 = base64::engine::general_purpose::STANDARD.encode(&buf);

        let req = json!({
            "type": "liveness",
            "crop": crop_b64
        });

        let resp = self.send_request(&req)?;
        if let Some(error) = resp.get("error") {
            anyhow::bail!("Liveness check failed: {:?}", error);
        }

        // Parse result
        Ok(resp["result"]["is_live"].as_bool().unwrap_or(false))
    }

    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        // Crop face from image
        let (x, y, w, h) = face.bbox;
        let (img_w, img_h) = image.dimensions();
        let crop_x = (x * img_w as f32) as u32;
        let crop_y = (y * img_h as f32) as u32;
        let crop_w = (w * img_w as f32) as u32;
        let crop_h = (h * img_h as f32) as u32;

        let face_crop = image.crop_imm(crop_x, crop_y, crop_w, crop_h);
        
        // Encode crop as JPEG base64
        let mut buf = Vec::new();
        face_crop.write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Jpeg)?;
        let crop_b64 = base64::engine::general_purpose::STANDARD.encode(&buf);

        let req = json!({
            "type": "embed",
            "crop": crop_b64
        });

        let resp = self.send_request(&req)?;
        if let Some(error) = resp.get("error") {
            anyhow::bail!("Embedding extraction failed: {:?}", error);
        }

        let embedding = resp["embedding"]
            .as_array()
            .context("Invalid embedding output")?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        Ok(embedding)
    }

    fn is_ready(&self) -> bool {
        // Check if process is still alive
        if let Ok(process) = self.process.lock() {
            process.id() > 0
        } else {
            false
        }
    }

    fn name(&self) -> &'static str {
        "MIGraphX (AMD ROCm iGPU)"
    }
}

impl Drop for MIGraphXBackend {
    fn drop(&mut self) {
        // Send exit command
        let req = json!({"cmd": "exit"});
        let _ = self.send_request(&req);

        // Kill process
        if let Ok(mut process) = self.process.lock() {
            let _ = process.kill();
        }
    }
}
