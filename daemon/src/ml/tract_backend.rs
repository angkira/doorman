use super::backend::{Face, MLBackend};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView};
use std::path::Path;
use tracing::{debug, info, warn};

/// Tract-based ML backend (pure Rust, no external deps)
pub struct TractBackend {
    detector: Option<tract_onnx::prelude::TypedModel>,
    liveness: Option<tract_onnx::prelude::TypedModel>,
    recognizer: Option<tract_onnx::prelude::TypedModel>,
}

impl TractBackend {
    pub fn new(models_dir: &Path) -> Result<Self> {
        info!("Initializing Tract backend...");
        
        let detector = Self::load_model(&models_dir.join("blazeface.onnx")).ok();
        let liveness = Self::load_model(&models_dir.join("liveness.onnx")).ok();
        let recognizer = Self::load_model(&models_dir.join("mobilefacenet.onnx")).ok();
        
        let loaded = [&detector, &liveness, &recognizer]
            .iter()
            .filter(|m| m.is_some())
            .count();
        
        info!("Tract backend: loaded {}/3 models", loaded);
        
        Ok(Self {
            detector,
            liveness,
            recognizer,
        })
    }
    
    fn load_model(path: &Path) -> Result<tract_onnx::prelude::TypedModel> {
        use tract_onnx::prelude::*;
        
        debug!("Loading model with Tract: {:?}", path);
        
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;
        
        Ok(model)
    }
    
    fn image_to_tensor(&self, image: &DynamicImage, width: u32, height: u32, normalize: bool) -> tract_ndarray::Array4<f32> {
        use tract_ndarray::Array4;
        
        let resized = image.resize_exact(width, height, image::imageops::FilterType::Lanczos3);
        let rgb = resized.to_rgb8();
        
        let mut tensor = Array4::zeros((1, 3, height as usize, width as usize));
        
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                for c in 0..3 {
                    let val = pixel[c] as f32;
                    tensor[[0, c, y as usize, x as usize]] = if normalize {
                        val / 255.0
                    } else {
                        (val / 127.5) - 1.0 // [-1, 1] normalization
                    };
                }
            }
        }
        
        tensor
    }
}

#[async_trait]
impl MLBackend for TractBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let detector = match &self.detector {
            Some(d) => d,
            None => {
                warn!("No detector model loaded");
                return Ok(None);
            }
        };
        
        let (width, height) = image.dimensions();
        let tensor = self.image_to_tensor(image, width, height, true);
        
        let result = detector.run(tvec![tensor.into()])?;
        
        // Parse outputs (boxes, scores)
        let boxes = result[0]
            .to_array_view::<f32>()?
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get boxes"))?;
        let scores = result[1]
            .to_array_view::<f32>()?
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get scores"))?;
        
        // Find best detection
        let mut best_idx = 0;
        let mut best_score = 0.0f32;
        
        for (i, &score) in scores.iter().enumerate() {
            if score > best_score && score > 0.5 {
                best_score = score;
                best_idx = i;
            }
        }
        
        if best_score > 0.5 {
            let box_offset = best_idx * 4;
            Ok(Some(Face {
                bbox: (
                    boxes[box_offset] * width as f32,
                    boxes[box_offset + 1] * height as f32,
                    boxes[box_offset + 2] * width as f32,
                    boxes[box_offset + 3] * height as f32,
                ),
                confidence: best_score,
            }))
        } else {
            Ok(None)
        }
    }
    
    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        let liveness = match &self.liveness {
            Some(l) => l,
            None => {
                warn!("No liveness model loaded, assuming live");
                return Ok(true);
            }
        };
        
        // Crop face
        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(
            x.max(0.0) as u32,
            y.max(0.0) as u32,
            w as u32,
            h as u32,
        );
        
        let tensor = self.image_to_tensor(&face_crop, 224, 224, true);
        let result = liveness.run(tvec![tensor.into()])?;
        
        let scores = result[0]
            .to_array_view::<f32>()?
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get liveness scores"))?;
        
        let real_score = scores[1];
        Ok(real_score > 0.5)
    }
    
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        let recognizer = match &self.recognizer {
            Some(r) => r,
            None => return Err(anyhow!("No recognizer model loaded")),
        };
        
        // Crop face
        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(
            x.max(0.0) as u32,
            y.max(0.0) as u32,
            w as u32,
            h as u32,
        );
        
        let tensor = self.image_to_tensor(&face_crop, 112, 112, false);
        let result = recognizer.run(tvec![tensor.into()])?;
        
        let embedding = result[0]
            .to_array_view::<f32>()?
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get embedding"))?
            .to_vec();
        
        // Normalize to unit length
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };
        
        Ok(normalized)
    }
    
    fn is_ready(&self) -> bool {
        self.detector.is_some() && self.liveness.is_some() && self.recognizer.is_some()
    }
    
    fn name(&self) -> &'static str {
        "Tract (Pure Rust)"
    }
}

