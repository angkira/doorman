#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use super::backend::{Face, MLBackend};
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use anyhow::{anyhow, Context, Result};
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use async_trait::async_trait;
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use doorman_shared::Config;
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use image::DynamicImage;
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use ort::value::Value;
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use std::path::Path;
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use std::sync::Mutex;
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use tracing::{info, warn};
#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use image::GenericImageView;

#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
macro_rules! ort_try {
    ($expr:expr) => {
        $expr.map_err(|e| anyhow!("ORT error: {}", e))?
    };
}

#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
/// ONNX Runtime backend (supports GPU via ROCm/CUDA)
/// Uses session pooling for concurrent requests
pub struct OrtBackend {
    detector_pool: Vec<Mutex<Session>>,
    liveness_pool: Vec<Mutex<Session>>,
    recognizer_pool: Vec<Mutex<Session>>,
    pool_index: AtomicUsize,
}

#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
impl OrtBackend {
        pub fn new(models_dir: &Path, config: &Config) -> Result<Self> {
        info!("Initializing ONNX Runtime backend with session pooling...");
        info!("Device: {}", config.ml.device);
        
        const POOL_SIZE: usize = 4; // 4 sessions per model for concurrency

        // Set environment variable for gfx1103 (Radeon 780M) if needed
        if config.ml.device == "rocm" {
            std::env::set_var("HSA_OVERRIDE_GFX_VERSION", "11.0.0");
            info!("Set HSA_OVERRIDE_GFX_VERSION=11.0.0 for gfx1103 support");
        }

        // Load detector pool
        let detector_path = models_dir.join("blazeface.onnx");
        let mut detector_pool = Vec::new();
        for i in 0..POOL_SIZE {
            match Self::load_model(&detector_path, config) {
                Ok(model) => {
                    detector_pool.push(Mutex::new(model));
                }
                Err(e) => {
                    warn!("✗ Failed to load detector session {}: {}", i, e);
                }
            }
        }
        if !detector_pool.is_empty() {
            info!("✓ Loaded {} face detector sessions", detector_pool.len());
        }

        // Load liveness pool
        let liveness_path = models_dir.join("liveness.onnx");
        let mut liveness_pool = Vec::new();
        for i in 0..POOL_SIZE {
            match Self::load_model(&liveness_path, config) {
                Ok(model) => {
                    liveness_pool.push(Mutex::new(model));
                }
                Err(e) => {
                    warn!("✗ Failed to load liveness session {}: {}", i, e);
                }
            }
        }
        if !liveness_pool.is_empty() {
            info!("✓ Loaded {} liveness detector sessions", liveness_pool.len());
        }

        // Load recognizer pool
        let recognizer_path = models_dir.join("mobilefacenet.onnx");
        let mut recognizer_pool = Vec::new();
        for i in 0..POOL_SIZE {
            match Self::load_model(&recognizer_path, config) {
                Ok(model) => {
                    recognizer_pool.push(Mutex::new(model));
                }
                Err(e) => {
                    warn!("✗ Failed to load recognizer session {}: {}", i, e);
                }
            }
        }
        if !recognizer_pool.is_empty() {
            info!("✓ Loaded {} face recognizer sessions", recognizer_pool.len());
        }

        info!(
            "ORT backend: loaded {}/{} detector, {}/{} liveness, {}/{} recognizer sessions",
            detector_pool.len(), POOL_SIZE,
            liveness_pool.len(), POOL_SIZE,
            recognizer_pool.len(), POOL_SIZE
        );

        Ok(Self {
            detector_pool,
            liveness_pool,
            recognizer_pool,
            pool_index: AtomicUsize::new(0),
        })
    }
    
    fn get_next_session<'a>(&'a self, pool: &'a [Mutex<Session>]) -> Option<&'a Mutex<Session>> {
        if pool.is_empty() {
            return None;
        }
        let idx = self.pool_index.fetch_add(1, Ordering::Relaxed) % pool.len();
        Some(&pool[idx])
    }


    fn load_model(path: &Path, config: &Config) -> Result<Session> {
        let threads = if config.ml.cpu_threads > 0 {
            config.ml.cpu_threads as usize
        } else {
            4
        };

        let builder = Session::builder()
            .map_err(|e| anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("Failed to set optimization level: {}", e))?
            .with_intra_threads(threads)
            .map_err(|e| anyhow!("Failed to set threads: {}", e))?;

        // Configure execution provider based on device
        #[cfg(feature = "backend-ort-cuda")]
        let builder = if config.ml.device == "cuda" || config.ml.device == "gpu" {
            info!("Configuring CUDA execution provider for {:?}", path);
            builder.with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
            ])
            .map_err(|e| anyhow!("Failed to set CUDA EP: {}", e))?
        } else {
            builder
        };
        
        #[cfg(feature = "backend-ort-rocm")]
        let builder = if config.ml.device == "rocm" || config.ml.device == "gpu" {
            info!("Configuring ROCm execution provider for {:?}", path);
            builder.with_execution_providers([
                ort::execution_providers::ROCmExecutionProvider::default()
                    .with_device_id(0)
                    .build(),
            ])
            .map_err(|e| anyhow!("Failed to set ROCm EP: {}", e))?
        } else {
            builder
        };

        #[cfg(not(any(feature = "backend-ort-rocm", feature = "backend-ort-cuda")))]
        let builder = builder;

        // Load model file
        let model_bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read model file: {:?}", path))?;
        
        let session = builder
            .commit_from_memory(&model_bytes)
            .map_err(|e| anyhow!("Failed to create session from model: {}", e))?;

        Ok(session)
    }
}

#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
impl OrtBackend {
    /// Preprocess image with letterboxing (preserve aspect ratio)
    fn image_to_tensor_letterbox(&self, image: &DynamicImage, target_w: u32, target_h: u32) -> (Vec<f32>, u32, u32, f32, f32) {
        use image::{Rgb, RgbImage};

        // Calculate scale to fit image in target size while preserving aspect ratio
        let (orig_w, orig_h) = image.dimensions();
        let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
        let resized_w = (orig_w as f32 * scale) as u32;
        let resized_h = (orig_h as f32 * scale) as u32;

        // Resize image WITH aspect ratio preserved
        let resized = image.resize(resized_w, resized_h, image::imageops::FilterType::Lanczos3);
        let resized_rgb = resized.to_rgb8();

        // Create black canvas
        let mut canvas = RgbImage::from_pixel(target_w, target_h, Rgb([0, 0, 0]));

        // Center resized image on canvas
        let offset_x = (target_w - resized_w) / 2;
        let offset_y = (target_h - resized_h) / 2;

        image::imageops::overlay(&mut canvas, &resized_rgb, offset_x as i64, offset_y as i64);

        // Convert to tensor (CHW format, normalized)
        let mut data = Vec::with_capacity((3 * target_w * target_h) as usize);
        for c in 0..3 {
            for y in 0..target_h {
                for x in 0..target_w {
                    let pixel = canvas.get_pixel(x, y);
                    data.push(pixel[c] as f32 / 255.0);
                }
            }
        }

        (data, resized_w, resized_h, offset_x as f32, offset_y as f32)
    }
}

#[cfg(any(feature = "backend-ort-cpu", feature = "backend-ort-cuda", feature = "backend-ort-rocm"))]
#[async_trait]
impl MLBackend for OrtBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let detector = match &self.detector {
            Some(d) => d,
            None => return Ok(None),
        };

        let (orig_width, orig_height) = image.dimensions();

        // Use letterboxing for 320x240 model input
        let (input_data, resized_w, resized_h, offset_x, offset_y) =
            self.image_to_tensor_letterbox(image, 320, 240);

        let input_tensor =
            ort_try!(Value::from_array(([1, 3, 240, 320], input_data)));
        let mut detector_lock = detector.lock().unwrap();
        let outputs = ort_try!(detector_lock.run(ort::inputs![input_tensor]));

        // BlazeFace model outputs: [scores, boxes]
        let (_, scores) = ort_try!(outputs[0].try_extract_tensor::<f32>());
        let (_, boxes) = ort_try!(outputs[1].try_extract_tensor::<f32>());

        // BlazeFace format: boxes may be fewer than scores
        // Only iterate through boxes that actually exist
        let mut best_idx = 0;
        let mut best_score = 0.0f32;

        let num_classes = 2;
        let num_boxes = boxes.len() / 4;
        let num_score_anchors = scores.len() / num_classes;

        let max_check = num_boxes.min(num_score_anchors);

        // Debug: check first few scores
        tracing::debug!("Score array size: {}, first 10 values: {:?}", 
            scores.len(), &scores[..scores.len().min(10)]);

        for box_idx in 0..max_check {
            let score_idx = box_idx * num_classes + 1; // face class
            let face_score = scores[score_idx];

            if face_score > best_score && face_score > 0.5 {
                best_score = face_score;
                best_idx = box_idx;
            }
        }

        tracing::debug!("BlazeFace detection: checked {} boxes, best_score={:.3}", max_check, best_score);

        if best_score > 0.5 {
            let box_offset = best_idx * 4;

            // BlazeFace outputs [center_x, center_y, w, h] in normalized [0,1] coordinates
            let center_x = boxes[box_offset];
            let center_y = boxes[box_offset + 1];
            let w = boxes[box_offset + 2];
            let h = boxes[box_offset + 3];

            // Convert from center coords to top-left corner coords
            let x = center_x - (w / 2.0);
            let y = center_y - (h / 2.0);

            // Convert from normalized [0,1] coordinates in letterboxed image to original image coordinates
            // Step 1: Convert normalized coords to letterboxed image pixel coords (320x240)
            let x_letterbox = x * 320.0;
            let y_letterbox = y * 240.0;
            let w_letterbox = w * 320.0;
            let h_letterbox = h * 240.0;

            // Step 2: Remove letterbox offsets
            let x_resized = x_letterbox - offset_x;
            let y_resized = y_letterbox - offset_y;

            // Step 3: Scale back to original image dimensions
            let scale = (320.0 / orig_width as f32).min(240.0 / orig_height as f32);
            let x_orig = x_resized / scale;
            let y_orig = y_resized / scale;
            let w_orig = w_letterbox / scale;
            let h_orig = h_letterbox / scale;

            Ok(Some(Face {
                bbox: (x_orig, y_orig, w_orig, h_orig),
                confidence: best_score,
                frame_dimensions: (image.width(), image.height()),
            }))
        } else {
            tracing::debug!("No face detected (best_score={:.3} < 0.5)", best_score);
            Ok(None)
        }
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

        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(x.max(0.0) as u32, y.max(0.0) as u32, w as u32, h as u32);

        let face_resized = face_crop.resize_exact(96, 96, image::imageops::FilterType::Lanczos3);
        let img = face_resized.to_rgb8();

        let mut input = Vec::with_capacity(3 * 96 * 96);
        for pixel in img.pixels() {
            input.push(pixel[0] as f32 / 255.0);
            input.push(pixel[1] as f32 / 255.0);
            input.push(pixel[2] as f32 / 255.0);
        }

        let input_tensor = ort_try!(Value::from_array(([1, 3, 96, 96], input)));
        let mut liveness_lock = liveness.lock().unwrap();
        let outputs = ort_try!(liveness_lock.run(ort::inputs![input_tensor]));

        let (_, scores) = ort_try!(outputs[0].try_extract_tensor::<f32>());
        Ok(scores[1] > 0.5)
    }

    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        let recognizer = match &self.recognizer {
            Some(r) => r,
            None => return Err(anyhow!("No recognizer model")),
        };

        let (x, y, w, h) = face.bbox;
        let face_crop = image.crop_imm(x.max(0.0) as u32, y.max(0.0) as u32, w as u32, h as u32);

        let face_resized = face_crop.resize_exact(112, 112, image::imageops::FilterType::Lanczos3);
        let img = face_resized.to_rgb8();

        let mut input = Vec::with_capacity(3 * 112 * 112);
        for pixel in img.pixels() {
            input.push((pixel[0] as f32 / 127.5) - 1.0);
            input.push((pixel[1] as f32 / 127.5) - 1.0);
            input.push((pixel[2] as f32 / 127.5) - 1.0);
        }

        let input_tensor = ort_try!(Value::from_array(([1, 3, 112, 112], input)));
        let mut recognizer_lock = recognizer.lock().unwrap();
        let outputs = ort_try!(recognizer_lock.run(ort::inputs![input_tensor]));

        let (_, embedding_data) = ort_try!(outputs[0].try_extract_tensor::<f32>());
        let embedding: Vec<f32> = embedding_data.iter().copied().collect();

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
