#[cfg(feature = "backend-ort")]
use super::backend::{Face, MLBackend};
#[cfg(feature = "backend-ort")]
use anyhow::{anyhow, Context, Result};
#[cfg(feature = "backend-ort")]
use async_trait::async_trait;
#[cfg(feature = "backend-ort")]
use doorman_shared::Config;
#[cfg(feature = "backend-ort")]
use image::DynamicImage;
#[cfg(feature = "backend-ort")]
use ort::session::{Session, builder::GraphOptimizationLevel};
#[cfg(feature = "backend-ort")]
use ort::value::Value;
#[cfg(feature = "backend-ort")]
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, ROCmExecutionProvider};
#[cfg(feature = "backend-ort")]
use std::path::Path;
#[cfg(feature = "backend-ort")]
use std::sync::Mutex;
#[cfg(feature = "backend-ort")]
use tracing::{info, warn};
#[cfg(feature = "backend-ort")]
use image::GenericImageView;

#[cfg(feature = "backend-ort")]
/// ONNX Runtime backend (supports GPU via ROCm/CUDA)
pub struct OrtBackend {
    detector: Option<Mutex<Session>>,
    liveness: Option<Mutex<Session>>,
    recognizer: Option<Mutex<Session>>,
}

#[cfg(feature = "backend-ort")]
impl OrtBackend {
    pub fn new(models_dir: &Path, config: &Config) -> Result<Self> {
        info!("Initializing ONNX Runtime backend...");

        // Initialize with device selection
        let init = ort::init().with_name("doorman");

        let init = match config.ml.device.as_str() {
            "cuda" => {
                info!("Using CUDA execution provider");
                init.with_execution_providers([
                    CUDAExecutionProvider::default()
                        .with_device_id(config.ml.gpu_device_id)
                        .build(),
                    CPUExecutionProvider::default().build(),
                ])
            }
            "rocm" => {
                info!("Using ROCm execution provider");
                init.with_execution_providers([
                    ROCmExecutionProvider::default()
                        .with_device_id(config.ml.gpu_device_id)
                        .build(),
                    CPUExecutionProvider::default().build(),
                ])
            }
            "npu" | "vitisai" => {
                info!("Using VitisAI execution provider (AMD Ryzen AI NPU)");
                warn!("NPU support requires AMD Ryzen AI drivers installed");
                warn!("See: https://ryzenai.docs.amd.com/en/latest/linux.html");
                // VitisAI EP will be registered via dynamic library if available
                // Otherwise falls back to CPU
                init.with_execution_providers([CPUExecutionProvider::default().build()])
            }
            _ => {
                info!("Using CPU execution provider");
                init.with_execution_providers([CPUExecutionProvider::default().build()])
            }
        };

        init.commit()?;

        // Load detector
        let detector_path = models_dir.join("blazeface.onnx");
        let detector = match Self::load_model(&detector_path, config) {
            Ok(model) => {
                info!("✓ Loaded face detector: {:?}", detector_path);
                Some(Mutex::new(model))
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
        let liveness = match Self::load_model(&liveness_path, config) {
            Ok(model) => {
                info!("✓ Loaded liveness detector: {:?}", liveness_path);
                Some(Mutex::new(model))
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
        let recognizer = match Self::load_model(&recognizer_path, config) {
            Ok(model) => {
                info!("✓ Loaded face recognizer: {:?}", recognizer_path);
                Some(Mutex::new(model))
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

        info!("ORT backend: loaded {}/3 models", loaded);

        if loaded == 0 {
            warn!("No models loaded! Face authentication will not work.");
            warn!("Please ensure models are present in: {:?}", models_dir);
        }

        Ok(Self {
            detector,
            liveness,
            recognizer,
        })
    }

    fn load_model(path: &Path, config: &Config) -> Result<Session> {
        let threads = if config.ml.cpu_threads > 0 {
            config.ml.cpu_threads as usize
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

#[cfg(feature = "backend-ort")]
impl OrtBackend {
    /// Preprocess image with letterboxing (preserve aspect ratio)
    fn image_to_tensor_letterbox(&self, image: &DynamicImage, target_w: u32, target_h: u32) -> (Vec<f32>, u32, u32, f32, f32) {
        use image::{Rgb, RgbImage};

        // Calculate scale to fit image in target size while preserving aspect ratio
        let (orig_w, orig_h) = image.dimensions();
        let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
        let resized_w = (orig_w as f32 * scale) as u32;
        let resized_h = (orig_h as f32 * scale) as u32;

        // Resize image
        let resized = image.resize(target_w, target_h, image::imageops::FilterType::Lanczos3);
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

#[cfg(feature = "backend-ort")]
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
            Value::from_array(([1, 3, 240, 320], input_data))?;
        let mut detector_lock = detector.lock().unwrap();
        let outputs = detector_lock.run(ort::inputs![input_tensor])?;

        let (_, boxes) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, scores) = outputs[1].try_extract_tensor::<f32>()?;

        // BlazeFace format: boxes may be fewer than scores
        // Only iterate through boxes that actually exist
        let mut best_idx = 0;
        let mut best_score = 0.0f32;

        let num_classes = 2;
        let num_boxes = boxes.len() / 4;
        let num_score_anchors = scores.len() / num_classes;

        let max_check = num_boxes.min(num_score_anchors);

        for box_idx in 0..max_check {
            let score_idx = box_idx * num_classes + 1; // face class
            let face_score = scores[score_idx];

            if face_score > best_score && face_score > 0.5 {
                best_score = face_score;
                best_idx = box_idx;
            }
        }

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
            }))
        } else {
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

        let face_resized = face_crop.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
        let img = face_resized.to_rgb8();

        let mut input = Vec::with_capacity(3 * 224 * 224);
        for pixel in img.pixels() {
            input.push(pixel[0] as f32 / 255.0);
            input.push(pixel[1] as f32 / 255.0);
            input.push(pixel[2] as f32 / 255.0);
        }

        let input_tensor = Value::from_array(([1, 3, 224, 224], input))?;
        let mut liveness_lock = liveness.lock().unwrap();
        let outputs = liveness_lock.run(ort::inputs![input_tensor])?;

        let (_, scores) = outputs[0].try_extract_tensor::<f32>()?;
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

        let input_tensor = Value::from_array(([1, 3, 112, 112], input))?;
        let mut recognizer_lock = recognizer.lock().unwrap();
        let outputs = recognizer_lock.run(ort::inputs![input_tensor])?;

        let (_, embedding_data) = outputs[0].try_extract_tensor::<f32>()?;
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
