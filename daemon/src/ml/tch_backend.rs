//! PyTorch backend using tch-rs (libtorch bindings)
//! Provides CUDA-accelerated inference with automatic GPU detection

use super::backend::{Face, MLBackend};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use image::DynamicImage;
use std::path::Path;
use tch::{nn, Device, Kind, Tensor};
use tracing::{info, warn};

pub struct TchBackend {
    detector: Option<nn::CModule>,
    liveness: Option<nn::CModule>,
    recognizer: Option<nn::CModule>,
    device: Device,
}

impl TchBackend {
    pub fn new(models_dir: &Path, device_str: &str) -> Result<Self> {
        info!("Initializing tch-rs (PyTorch) backend...");
        
        // Determine device
        let device = if device_str == "cuda" {
            if tch::Cuda::is_available() {
                info!("✓ CUDA available, device count: {}", tch::Cuda::device_count());
                Device::Cuda(0)
            } else {
                warn!("CUDA requested but not available, falling back to CPU");
                Device::Cpu
            }
        } else {
            Device::Cpu
        };
        info!("Using device: {:?}", device);
        
        // Load models (expecting .pt files)
        let detector_path = models_dir.join("blazeface.pt");
        let detector = if detector_path.exists() {
            match nn::CModule::load(&detector_path) {
                Ok(mut model) => {
                    model.set_eval();
                    info!("✓ Loaded face detector: {:?}", detector_path);
                    Some(model)
                }
                Err(e) => {
                    warn!("✗ Failed to load detector: {}", e);
                    None
                }
            }
        } else {
            warn!("✗ Detector model not found: {:?}", detector_path);
            None
        };
        
        let liveness_path = models_dir.join("liveness.pt");
        let liveness = if liveness_path.exists() {
            match nn::CModule::load(&liveness_path) {
                Ok(mut model) => {
                    model.set_eval();
                    info!("✓ Loaded liveness detector: {:?}", liveness_path);
                    Some(model)
                }
                Err(e) => {
                    warn!("✗ Failed to load liveness: {}", e);
                    None
                }
            }
        } else {
            warn!("✗ Liveness model not found: {:?}", liveness_path);
            None
        };
        
        let recognizer_path = models_dir.join("mobilefacenet.pt");
        let recognizer = if recognizer_path.exists() {
            match nn::CModule::load(&recognizer_path) {
                Ok(mut model) => {
                    model.set_eval();
                    info!("✓ Loaded face recognizer: {:?}", recognizer_path);
                    Some(model)
                }
                Err(e) => {
                    warn!("✗ Failed to load recognizer: {}", e);
                    None
                }
            }
        } else {
            warn!("✗ Recognizer model not found: {:?}", recognizer_path);
            None
        };
        
        let loaded = [&detector, &liveness, &recognizer]
            .iter()
            .filter(|m| m.is_some())
            .count();
        info!("tch-rs backend: loaded {}/3 models on {:?}", loaded, device);
        
        Ok(Self {
            detector,
            liveness,
            recognizer,
            device,
        })
    }
    
    fn image_to_tensor(&self, image: &DynamicImage, width: u32, height: u32) -> Result<Tensor> {
        let resized = image.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        
        // Convert to CHW format and normalize to [0, 1]
        let mut data = Vec::with_capacity((3 * width * height) as usize);
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb.get_pixel(x, y);
                    data.push(pixel[c] as f32 / 255.0);
                }
            }
        }
        
        let tensor = Tensor::of_slice(&data)
            .view([1, 3, height as i64, width as i64])
            .to(self.device);
        
        Ok(tensor)
    }
}

#[async_trait]
impl MLBackend for TchBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let detector = self.detector.as_ref()
            .ok_or_else(|| anyhow!("Detector not loaded"))?;
        
        // Prepare input (320x240 for BlazeFace)
        let input = self.image_to_tensor(image, 320, 240)?;
        
        // Run inference
        let output = tch::no_grad(|| detector.forward_ts(&[input]))?;
        
        // Parse BlazeFace output: [scores, boxes]
        // For now, return dummy face (need to implement NMS decoding)
        // TODO: Implement proper BlazeFace decoder
        warn!("BlazeFace decoder not yet implemented, returning None");
        Ok(None)
    }
    
    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        let liveness = self.liveness.as_ref()
            .ok_or_else(|| anyhow!("Liveness detector not loaded"))?;
        
        // Crop face region
        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(
            x.max(0.0) as u32,
            y.max(0.0) as u32,
            w as u32,
            h as u32
        );
        
        // Prepare input (224x224 for liveness)
        let input = self.image_to_tensor(&face_crop, 224, 224)?;
        
        // Run inference
        let output = tch::no_grad(|| liveness.forward_ts(&[input]))?;
        
        // Get prediction (sigmoid output > 0.5 means real)
        let score = f32::try_from(output.sigmoid())?;
        Ok(score > 0.5)
    }
    
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        let recognizer = self.recognizer.as_ref()
            .ok_or_else(|| anyhow!("Recognizer not loaded"))?;
        
        // Crop face region
        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(
            x.max(0.0) as u32,
            y.max(0.0) as u32,
            w as u32,
            h as u32
        );
        
        // Prepare input (112x112 for MobileFaceNet)
        let input = self.image_to_tensor(&face_crop, 112, 112)?;
        
        // Run inference
        let output = tch::no_grad(|| recognizer.forward_ts(&[input]))?;
        
        // Extract embedding vector
        let embedding: Vec<f32> = output.try_into()?;
        Ok(embedding)
    }
    
    fn is_ready(&self) -> bool {
        self.detector.is_some() && self.liveness.is_some() && self.recognizer.is_some()
    }
    
    fn name(&self) -> &'static str {
        "PyTorch (tch-rs)"
    }
}
