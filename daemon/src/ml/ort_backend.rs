use super::backend::{Face, MLBackend};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use doorman_shared::Config;
use image::{DynamicImage, GenericImageView};
use ort::{GraphOptimizationLevel, Session};
use std::path::Path;
use tracing::{debug, info, warn};

/// ONNX Runtime backend (supports GPU via ROCm/CUDA)
pub struct OrtBackend {
    detector: Option<Session>,
    liveness: Option<Session>,
    recognizer: Option<Session>,
}

impl OrtBackend {
    pub fn new(models_dir: &Path, config: &Config) -> Result<Self> {
        info!("Initializing ONNX Runtime backend...");
        
        // Initialize with device selection
        let init = ort::init().with_name("doorman");
        
        let init = match config.ml.device.as_str() {
            "cuda" => {
                info!("Using CUDA execution provider");
                init.with_execution_providers([
                    ort::CUDAExecutionProvider::default()
                        .with_device_id(config.ml.gpu_device_id)
                        .build(),
                    ort::CPUExecutionProvider::default().build(),
                ])
            }
            "rocm" => {
                info!("Using ROCm execution provider");
                init.with_execution_providers([
                    ort::ROCmExecutionProvider::default()
                        .with_device_id(config.ml.gpu_device_id)
                        .build(),
                    ort::CPUExecutionProvider::default().build(),
                ])
            }
            _ => {
                info!("Using CPU execution provider");
                init.with_execution_providers([
                    ort::CPUExecutionProvider::default().build(),
                ])
            }
        };
        
        init.commit()?;
        
        let detector = Self::load_model(&models_dir.join("blazeface.onnx"), config).ok();
        let liveness = Self::load_model(&models_dir.join("liveness.onnx"), config).ok();
        let recognizer = Self::load_model(&models_dir.join("mobilefacenet.onnx"), config).ok();
        
        let loaded = [&detector, &liveness, &recognizer]
            .iter()
            .filter(|m| m.is_some())
            .count();
        
        info!("ORT backend: loaded {}/3 models", loaded);
        
        Ok(Self {
            detector,
            liveness,
            recognizer,
        })
    }
    
    fn load_model(path: &Path, config: &Config) -> Result<Session> {
        let threads = if config.ml.cpu_threads > 0 {
            config.ml.cpu_threads
        } else {
            2
        };
        
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load model: {:?}", path))
    }
}

#[async_trait]
impl MLBackend for OrtBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let detector = match &self.detector {
            Some(d) => d,
            None => return Ok(None),
        };
        
        let img = image.to_rgb8();
        let (width, height) = img.dimensions();
        
        let mut input = Vec::with_capacity((3 * width * height) as usize);
        for pixel in img.pixels() {
            input.push(pixel[0] as f32 / 255.0);
            input.push(pixel[1] as f32 / 255.0);
            input.push(pixel[2] as f32 / 255.0);
        }
        
        let input_tensor = ort::Value::from_array(([1, 3, height as usize, width as usize], input.as_slice()))?;
        let outputs = detector.run(ort::inputs![input_tensor]?)?;
        
        let boxes = outputs[0].try_extract_tensor::<f32>()?;
        let scores = outputs[1].try_extract_tensor::<f32>()?;
        
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
            None => return Ok(true),
        };
        
        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(
            x.max(0.0) as u32,
            y.max(0.0) as u32,
            w as u32,
            h as u32,
        );
        
        let face_resized = face_crop.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
        let img = face_resized.to_rgb8();
        
        let mut input = Vec::with_capacity(3 * 224 * 224);
        for pixel in img.pixels() {
            input.push(pixel[0] as f32 / 255.0);
            input.push(pixel[1] as f32 / 255.0);
            input.push(pixel[2] as f32 / 255.0);
        }
        
        let input_tensor = ort::Value::from_array(([1, 3, 224, 224], input.as_slice()))?;
        let outputs = liveness.run(ort::inputs![input_tensor]?)?;
        
        let scores = outputs[0].try_extract_tensor::<f32>()?;
        Ok(scores[1] > 0.5)
    }
    
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        let recognizer = match &self.recognizer {
            Some(r) => r,
            None => return Err(anyhow!("No recognizer model")),
        };
        
        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(
            x.max(0.0) as u32,
            y.max(0.0) as u32,
            w as u32,
            h as u32,
        );
        
        let face_resized = face_crop.resize_exact(112, 112, image::imageops::FilterType::Lanczos3);
        let img = face_resized.to_rgb8();
        
        let mut input = Vec::with_capacity(3 * 112 * 112);
        for pixel in img.pixels() {
            input.push((pixel[0] as f32 / 127.5) - 1.0);
            input.push((pixel[1] as f32 / 127.5) - 1.0);
            input.push((pixel[2] as f32 / 127.5) - 1.0);
        }
        
        let input_tensor = ort::Value::from_array(([1, 3, 112, 112], input.as_slice()))?;
        let outputs = recognizer.run(ort::inputs![input_tensor]?)?;
        
        let embedding_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let embedding: Vec<f32> = embedding_tensor.iter().copied().collect();
        
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
        "ONNX Runtime"
    }
}

