#[cfg(feature = "backend-tract")]
use super::backend::{Face, MLBackend};
#[cfg(feature = "backend-tract")]
use anyhow::{anyhow, Result};
#[cfg(feature = "backend-tract")]
use async_trait::async_trait;
#[cfg(feature = "backend-tract")]
use image::{DynamicImage, GenericImageView};
#[cfg(feature = "backend-tract")]
use std::path::Path;
#[cfg(feature = "backend-tract")]
use tracing::{debug, info, warn};
#[cfg(feature = "backend-tract")]
use tract_onnx::prelude::*;

#[cfg(feature = "backend-tract")]
/// Tract-based ML backend (pure Rust, no external deps)
pub struct TractBackend {
    detector: Option<tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>>,
    liveness: Option<tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>>,
    recognizer: Option<tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>>,
    model_config: super::model_config::ModelSet,
    decoder: Option<super::blazeface_decoder::BlazeFaceDecoder>,
}

#[cfg(feature = "backend-tract")]
impl TractBackend {
    pub fn new(models_dir: &Path) -> Result<Self> {
        info!("Initializing Tract backend...");
        info!("Models directory: {:?}", models_dir);

        // Load detector
        let detector_path = models_dir.join("blazeface.onnx");
        let detector = match Self::load_model(&detector_path) {
            Ok(model) => {
                info!("✓ Loaded face detector: {:?}", detector_path);
                Some(model)
            }
            Err(e) => {
                warn!(
                    "✗ Failed to load face detector from {:?}: {}",
                    detector_path, e
                );
                None
            }
        };

        // Load liveness detector
        let liveness_path = models_dir.join("liveness.onnx");
        let liveness = match Self::load_model(&liveness_path) {
            Ok(model) => {
                info!("✓ Loaded liveness detector: {:?}", liveness_path);
                Some(model)
            }
            Err(e) => {
                warn!(
                    "✗ Failed to load liveness detector from {:?}: {}",
                    liveness_path, e
                );
                None
            }
        };

        // Load face recognizer
        let recognizer_path = models_dir.join("mobilefacenet.onnx");
        let recognizer = match Self::load_model(&recognizer_path) {
            Ok(model) => {
                info!("✓ Loaded face recognizer: {:?}", recognizer_path);
                Some(model)
            }
            Err(e) => {
                warn!(
                    "✗ Failed to load face recognizer from {:?}: {}",
                    recognizer_path, e
                );
                None
            }
        };

        let loaded = [&detector, &liveness, &recognizer]
            .iter()
            .filter(|m| m.is_some())
            .count();

        info!("Tract backend: loaded {}/3 models", loaded);

        if loaded == 0 {
            warn!("No models loaded! Face authentication will not work.");
            warn!("Please ensure models are present in: {:?}", models_dir);
        }

        // Initialize decoder if detector is loaded
        let decoder = if detector.is_some() {
            match super::blazeface_decoder::BlazeFaceDecoder::new_default() {
                Ok(dec) => {
                    info!("✓ Initialized BlazeFace decoder with {} anchors", 896);
                    Some(dec)
                }
                Err(e) => {
                    warn!("✗ Failed to initialize BlazeFace decoder: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            detector,
            liveness,
            recognizer,
            model_config: super::model_config::ModelSet::DEFAULT,
            decoder,
        })
    }

    fn load_model(
        path: &Path,
    ) -> Result<tract_onnx::prelude::TypedRunnableModel<tract_onnx::prelude::TypedModel>> {
        debug!("Loading model with Tract: {:?}", path);

        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .into_optimized()?
            .into_runnable()?;

        debug!("Model loaded and optimized for CPU execution");
        Ok(model)
    }

    fn image_to_tensor(
        &self,
        image: &DynamicImage,
        width: u32,
        height: u32,
        normalize: bool,
    ) -> tract_onnx::prelude::Tensor {
        use tract_onnx::prelude::*;
        use image::{Rgb, RgbImage};

        // Resize with faster filter for CPU performance
        let resized = image.resize(width, height, image::imageops::FilterType::Triangle);
        let resized_rgb = resized.to_rgb8();

        // Create a black canvas of target size
        let mut canvas = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));

        // Center the resized image on the canvas
        let (resized_w, resized_h) = resized_rgb.dimensions();
        let offset_x = (width - resized_w) / 2;
        let offset_y = (height - resized_h) / 2;

        image::imageops::overlay(&mut canvas, &resized_rgb, offset_x as i64, offset_y as i64);

        let rgb = canvas;

        // Pre-allocate and use parallel processing for tensor conversion
        let total_pixels = (3 * width * height) as usize;
        let mut data = Vec::with_capacity(total_pixels);
        
        // CPU optimization: Convert HWC to CHW format with better cache locality
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb.get_pixel(x, y);
                    let val = pixel[c] as f32;
                    data.push(if normalize {
                        val / 255.0
                    } else {
                        (val / 127.5) - 1.0
                    });
                }
            }
        }

        tract_ndarray::Array4::from_shape_vec((1, 3, height as usize, width as usize), data)
            .unwrap()
            .into()
    }
}

#[cfg(feature = "backend-tract")]
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

        // Use configured detector input size
        let (orig_width, orig_height) = image.dimensions();
        let width = self.model_config.detector.input_width;
        let height = self.model_config.detector.input_height;

        // Calculate letterbox parameters (for bbox coordinate conversion later)
        let scale = (width as f32 / orig_width as f32).min(height as f32 / orig_height as f32);
        let resized_w = (orig_width as f32 * scale) as u32;
        let resized_h = (orig_height as f32 * scale) as u32;
        let offset_x = (width - resized_w) as f32 / 2.0;
        let offset_y = (height - resized_h) as f32 / 2.0;

        // Normalize to [0, 1] range (original working preprocessing)
        let image_tensor = self.image_to_tensor(image, width, height, true);

        // BlazeFace model takes single input: RGB image tensor
        use tract_onnx::prelude::*;

        debug!("Input to model:");
        debug!("  Original image: {}x{}", orig_width, orig_height);
        debug!("  Resized to: {}x{}", width, height);
        debug!("  Tensor shape: {:?}", image_tensor.shape());

        let result = detector.run(tvec![image_tensor.into()])?;

        // BlazeFace model outputs:
        // Output 0: scores [1, N, 2] - class probabilities (background, face)
        // Output 1: boxes [1, N, 4] - bounding boxes (x, y, w, h) in normalized coordinates
        if result.len() != 2 {
            return Err(anyhow!("Expected 2 outputs from model, got {}", result.len()));
        }

        let scores_view = result[0].to_array_view::<f32>()?;
        let boxes_view = result[1].to_array_view::<f32>()?;

        let scores_shape = scores_view.shape();
        let boxes_shape = boxes_view.shape();

        debug!("  Scores shape: {:?}", scores_shape);
        debug!("  Boxes shape: {:?}", boxes_shape);

        // Reshape to [N, 2] and [N, 4]
        let num_detections = scores_shape[1];
        let scores = scores_view.as_slice().ok_or_else(|| anyhow!("Failed to get scores"))?;
        let boxes = boxes_view.as_slice().ok_or_else(|| anyhow!("Failed to get boxes"))?;

        // Find best face detection
        let mut best_idx = None;
        let mut best_score = self.model_config.detector.confidence_threshold;
        let mut top_scores = Vec::new();

        for i in 0..num_detections {
            let face_score = scores[i * 2 + 1]; // Index 1 is face class
            if face_score > best_score {
                best_score = face_score;
                best_idx = Some(i);
            }
            // Track top 5 scores for debugging
            if top_scores.len() < 5 || face_score > top_scores[4] {
                top_scores.push(face_score);
                top_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
                if top_scores.len() > 5 {
                    top_scores.truncate(5);
                }
            }
        }

        let idx = match best_idx {
            Some(i) => i,
            None => {
                info!("No faces detected above threshold {}. Top 5 scores: {:?}",
                    self.model_config.detector.confidence_threshold, top_scores);
                return Ok(None);
            }
        };

        // BlazeFace outputs from PINTO Model Zoo are in format: [top_y, top_x, bot_y, bot_x]
        // All values are normalized [0, 1]
        let top_y = boxes[idx * 4];
        let top_x = boxes[idx * 4 + 1];
        let bot_y = boxes[idx * 4 + 2];
        let bot_x = boxes[idx * 4 + 3];

        // Convert to (x, y, width, height) format
        let x = top_x.clamp(0.0, 1.0);
        let y = top_y.clamp(0.0, 1.0);
        let x2 = bot_x.clamp(0.0, 1.0);
        let y2 = bot_y.clamp(0.0, 1.0);
        let w = (x2 - x).abs().clamp(0.01, 1.0);
        let h = (y2 - y).abs().clamp(0.01, 1.0);

        info!("BlazeFace raw: top_y={:.3}, top_x={:.3}, bot_y={:.3}, bot_x={:.3}, conf={:.3}",
            top_y, top_x, bot_y, bot_x, best_score);
        info!("  Converted: x={:.3}, y={:.3}, w={:.3}, h={:.3}", x, y, w, h);

        // Convert from normalized [0,1] coordinates in letterboxed image to original image coordinates
        // BlazeFace coordinates are normalized relative to the letterboxed input (e.g., 128x128)
        // We need to:
        // 1. Scale to letterbox pixel coordinates
        // 2. Remove letterbox offsets to get coordinates in the resized (but not letterboxed) image
        // 3. Scale back to original image dimensions
        
        // Step 1: Convert normalized coords to letterboxed image pixel coords
        let x_letterbox = x * width as f32;
        let y_letterbox = y * height as f32;
        let x2_letterbox = x2 * width as f32;
        let y2_letterbox = y2 * height as f32;

        // Step 2: Remove letterbox offsets to get coordinates in resized image
        let x_resized = x_letterbox - offset_x;
        let y_resized = y_letterbox - offset_y;
        let x2_resized = x2_letterbox - offset_x;
        let y2_resized = y2_letterbox - offset_y;
        
        let w_resized = x2_resized - x_resized;
        let h_resized = y2_resized - y_resized;

        // Step 3: Scale back to original image dimensions
        let x_orig = (x_resized / resized_w as f32) * orig_width as f32;
        let y_orig = (y_resized / resized_h as f32) * orig_height as f32;
        let w_orig = (w_resized / resized_w as f32) * orig_width as f32;
        let h_orig = (h_resized / resized_h as f32) * orig_height as f32;

        info!("  Letterbox params: offset_x={:.1}, offset_y={:.1}, resized={}x{}, input={}x{}",
            offset_x, offset_y, resized_w, resized_h, width, height);
        info!("  Letterbox coords: x={:.1}, y={:.1}, x2={:.1}, y2={:.1}",
            x_letterbox, y_letterbox, x2_letterbox, y2_letterbox);
        info!("  Resized coords: x={:.1}, y={:.1}, w={:.1}, h={:.1}",
            x_resized, y_resized, w_resized, h_resized);
        info!("  Final bbox in original image: x={:.1}, y={:.1}, w={:.1}, h={:.1}",
            x_orig, y_orig, w_orig, h_orig);

        // Normalize to [0, 1] range for consistent handling
        let x_norm = x_orig / orig_width as f32;
        let y_norm = y_orig / orig_height as f32;
        let w_norm = w_orig / orig_width as f32;
        let h_norm = h_orig / orig_height as f32;

        debug!("  Normalized bbox: x={:.3}, y={:.3}, w={:.3}, h={:.3}",
            x_norm, y_norm, w_norm, h_norm);

        Ok(Some(Face {
            bbox: (
                x_norm,
                y_norm,
                w_norm,
                h_norm,
            ),
            confidence: best_score,
            frame_dimensions: (orig_width, orig_height),
        }))
    }

    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        let liveness = match &self.liveness {
            Some(l) => l,
            None => {
                warn!("No liveness model loaded - cannot verify face authenticity");
                return Err(anyhow!(
                    "Liveness check unavailable: model not loaded. This is a security requirement."
                ));
            }
        };

        // Crop face with padding (10% on each side)
        let (x_norm, y_norm, w_norm, h_norm) = face.bbox;
        let img_width = image.width() as f32;
        let img_height = image.height() as f32;
        
        let padding = 0.10; // 10% padding
        let x = ((x_norm - w_norm * padding) * img_width).max(0.0);
        let y = ((y_norm - h_norm * padding) * img_height).max(0.0);
        let w = (w_norm * (1.0 + 2.0 * padding) * img_width).min(img_width - x);
        let h = (h_norm * (1.0 + 2.0 * padding) * img_height).min(img_height - y);
        
        let face_crop = image.crop_imm(x as u32, y as u32, w as u32, h as u32);

        let size = self.model_config.liveness.input_size;
        let tensor = self.image_to_tensor(&face_crop, size, size, true);
        let result = liveness.run(tvec![tensor.into()])?;

        let scores_view = result[0].to_array_view::<f32>()?;
        let scores = scores_view
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get liveness scores"))?;

        info!("Liveness output: {} scores: {:?}", scores.len(), scores);

        let live_index = self.model_config.liveness.live_class_index;
        if live_index >= scores.len() {
            return Err(anyhow!("Live class index {} out of bounds (len={})", live_index, scores.len()));
        }

        let real_score = scores[live_index];
        let threshold = self.model_config.liveness.confidence_threshold;

        let passed = real_score > threshold;
        
        if passed {
            info!("✓ Liveness check passed: index={}, score={:.3}", live_index, real_score);
            Ok(true)
        } else {
            warn!("✗ Liveness check failed: index={}, score={:.3} (threshold={})",
                live_index, real_score, threshold);
            Ok(false)
        }
    }

    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        let recognizer = match &self.recognizer {
            Some(r) => r,
            None => return Err(anyhow!("No recognizer model loaded")),
        };

        // Crop face with padding (10% on each side)
        let (x_norm, y_norm, w_norm, h_norm) = face.bbox;
        let img_width = image.width() as f32;
        let img_height = image.height() as f32;
        
        let padding = 0.10; // 10% padding
        let x = ((x_norm - w_norm * padding) * img_width).max(0.0);
        let y = ((y_norm - h_norm * padding) * img_height).max(0.0);
        let w = (w_norm * (1.0 + 2.0 * padding) * img_width).min(img_width - x);
        let h = (h_norm * (1.0 + 2.0 * padding) * img_height).min(img_height - y);
        
        let face_crop = image.crop_imm(x as u32, y as u32, w as u32, h as u32);

        let size = self.model_config.recognizer.input_size;
        let tensor = self.image_to_tensor(&face_crop, size, size, false);
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
