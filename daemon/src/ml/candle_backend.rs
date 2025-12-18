use crate::ml::backend::{Face, MLBackend};
use anyhow::{Context, Result};
use async_trait::async_trait;
use image::DynamicImage;
use std::path::Path;
use tracing::{debug, info, warn};

#[cfg(feature = "backend-candle")]
use candle_core::{Device, Tensor};
#[cfg(feature = "backend-candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "backend-candle")]
use candle_onnx::onnx;

pub struct CandleBackend {
    #[cfg(feature = "backend-candle")]
    device: Device,
    #[cfg(feature = "backend-candle")]
    detector: Option<onnx::SimpleEval<'static>>,
    #[cfg(feature = "backend-candle")]
    liveness: Option<onnx::SimpleEval<'static>>,
    #[cfg(feature = "backend-candle")]
    recognizer: Option<onnx::SimpleEval<'static>>,
}

impl CandleBackend {
    pub fn new(models_dir: &Path, device_str: &str) -> Result<Self> {
        info!("Initializing Candle backend...");
        info!("Models directory: {:?}", models_dir);
        info!("Device: {}", device_str);

        #[cfg(feature = "backend-candle")]
        {
            // Initialize device (CPU or CUDA)
            let device = if device_str == "cuda" {
                #[cfg(feature = "backend-candle-cuda")]
                {
                    info!("Using CUDA device");
                    Device::new_cuda(0).context("Failed to initialize CUDA device")?
                }
                #[cfg(not(feature = "backend-candle-cuda"))]
                {
                    info!("CUDA support not enabled, falling back to CPU");
                    Device::Cpu
                }
            } else {
                info!("Using CPU device");
                Device::Cpu
            };

            // Load models
            let detector_path = models_dir.join("blazeface.onnx");
            let liveness_path = models_dir.join("liveness.onnx");
            let recognizer_path = models_dir.join("mobilefacenet.onnx");

            let detector = if detector_path.exists() {
                info!("✓ Loading face detector from {:?}", detector_path);
                match Self::load_onnx_model(&detector_path, &device) {
                    Ok(model) => {
                        info!("✓ Face detector loaded successfully");
                        Some(model)
                    }
                    Err(e) => {
                        info!("✗ Failed to load face detector: {}", e);
                        None
                    }
                }
            } else {
                info!("✗ Face detector not found at {:?}", detector_path);
                None
            };

            let liveness = if liveness_path.exists() {
                info!("✓ Loading liveness detector from {:?}", liveness_path);
                match Self::load_onnx_model(&liveness_path, &device) {
                    Ok(model) => {
                        info!("✓ Liveness detector loaded successfully");
                        Some(model)
                    }
                    Err(e) => {
                        info!("✗ Failed to load liveness detector: {}", e);
                        None
                    }
                }
            } else {
                info!("✗ Liveness detector not found at {:?}", liveness_path);
                None
            };

            let recognizer = if recognizer_path.exists() {
                info!("✓ Loading face recognizer from {:?}", recognizer_path);
                match Self::load_onnx_model(&recognizer_path, &device) {
                    Ok(model) => {
                        info!("✓ Face recognizer loaded successfully");
                        Some(model)
                    }
                    Err(e) => {
                        info!("✗ Failed to load face recognizer: {}", e);
                        None
                    }
                }
            } else {
                info!("✗ Face recognizer not found at {:?}", recognizer_path);
                None
            };

            let loaded_count = detector.is_some() as u8
                + liveness.is_some() as u8
                + recognizer.is_some() as u8;
            info!("Candle backend: loaded {}/3 models", loaded_count);

            if loaded_count == 0 {
                info!("⚠ No models loaded! Face authentication will not work.");
                info!("Please ensure models are present in: {:?}", models_dir);
            }

            Ok(Self {
                device,
                detector,
                liveness,
                recognizer,
            })
        }

        #[cfg(not(feature = "backend-candle"))]
        {
            anyhow::bail!("Candle backend not enabled. Compile with --features backend-candle");
        }
    }

    #[cfg(feature = "backend-candle")]
    fn load_onnx_model(path: &Path, device: &Device) -> Result<onnx::SimpleEval<'static>> {
        let model = onnx::read_file(path)
            .with_context(|| format!("Failed to read ONNX file: {:?}", path))?;
        
        // Create inference session
        let eval = candle_onnx::simple_eval(&model, device.clone())
            .with_context(|| format!("Failed to create inference session for {:?}", path))?;
        
        Ok(eval)
    }

    #[cfg(feature = "backend-candle")]
    fn preprocess_image_for_detection(&self, image: &DynamicImage) -> Result<Tensor> {
        // Convert to RGB
        let rgb_image = image.to_rgb8();
        let (width, height) = rgb_image.dimensions();

        // Resize to model input size (128x128 for BlazeFace)
        let resized = image::imageops::resize(
            &rgb_image,
            128,
            128,
            image::imageops::FilterType::Triangle,
        );

        // Convert to tensor [1, 3, 128, 128]
        let mut data = Vec::with_capacity(3 * 128 * 128);
        for pixel in resized.pixels() {
            data.push(pixel[0] as f32 / 255.0);
            data.push(pixel[1] as f32 / 255.0);
            data.push(pixel[2] as f32 / 255.0);
        }

        let tensor = Tensor::from_vec(data, &[1, 3, 128, 128], &self.device)
            .context("Failed to create tensor from image")?;

        Ok(tensor)
    }

    #[cfg(feature = "backend-candle")]
    fn preprocess_face_for_recognition(&self, face_image: &DynamicImage) -> Result<Tensor> {
        // Convert to RGB
        let rgb_image = face_image.to_rgb8();

        // Resize to 112x112 for MobileFaceNet
        let resized = image::imageops::resize(
            &rgb_image,
            112,
            112,
            image::imageops::FilterType::Triangle,
        );

        // Normalize: (pixel / 255.0 - 0.5) / 0.5
        let mut data = Vec::with_capacity(3 * 112 * 112);
        for pixel in resized.pixels() {
            data.push((pixel[0] as f32 / 255.0 - 0.5) / 0.5);
            data.push((pixel[1] as f32 / 255.0 - 0.5) / 0.5);
            data.push((pixel[2] as f32 / 255.0 - 0.5) / 0.5);
        }

        let tensor = Tensor::from_vec(data, &[1, 3, 112, 112], &self.device)
            .context("Failed to create tensor from face")?;

        Ok(tensor)
    }
}

#[async_trait]
impl MLBackend for CandleBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        #[cfg(feature = "backend-candle")]
        {
            if let Some(detector) = &self.detector {
                debug!("Running face detection with Candle");
                
                let input = self.preprocess_image_for_detection(image)?;
                let outputs = detector.eval(&[input])
                    .context("Failed to run detector inference")?;
                
                // Parse outputs (boxes and scores)
                // This is simplified - actual BlazeFace output parsing is more complex
                
                // TODO: Implement proper BlazeFace output parsing
                // For now, return None to avoid crashes
                debug!("Face detection completed: no faces parsed yet");
                
                Ok(None)
            } else {
                debug!("No detector model loaded");
                Ok(None)
            }
        }

        #[cfg(not(feature = "backend-candle"))]
        {
            anyhow::bail!("Candle backend not enabled");
        }
    }

    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        #[cfg(feature = "backend-candle")]
        {
            if let Some(liveness_model) = &self.liveness {
                debug!("Running liveness detection with Candle");
                
                // Extract face region
                let (x, y, w, h) = face.bbox;
                let face_img = image.crop_imm(
                    x as u32,
                    y as u32,
                    w as u32,
                    h as u32,
                );

                let input = self.preprocess_face_for_recognition(&face_img)?;
                let outputs = liveness_model.eval(&[input])
                    .context("Failed to run liveness inference")?;
                
                // Parse liveness score
                // TODO: Implement proper liveness output parsing
                let is_live = true; // Placeholder
                
                debug!("Liveness check: {}", if is_live { "LIVE" } else { "SPOOF" });
                Ok(is_live)
            } else {
                debug!("No liveness model loaded, assuming live");
                Ok(true)
            }
        }

        #[cfg(not(feature = "backend-candle"))]
        {
            anyhow::bail!("Candle backend not enabled");
        }
    }

    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        #[cfg(feature = "backend-candle")]
        {
            if let Some(recognizer) = &self.recognizer {
                debug!("Extracting face embedding with Candle");
                
                // Extract face region
                let (x, y, w, h) = face.bbox;
                let face_img = image.crop_imm(
                    x as u32,
                    y as u32,
                    w as u32,
                    h as u32,
                );

                let input = self.preprocess_face_for_recognition(&face_img)?;
                let outputs = recognizer.eval(&[input])
                    .context("Failed to run recognizer inference")?;
                
                // Extract embedding vector
                if let Some(output) = outputs.first() {
                    let embedding = output.flatten_all()?.to_vec1::<f32>()?;
                    debug!("Embedding extracted: {} dimensions", embedding.len());
                    Ok(embedding)
                } else {
                    anyhow::bail!("No output from recognizer");
                }
            } else {
                anyhow::bail!("No recognizer model loaded");
            }
        }

        #[cfg(not(feature = "backend-candle"))]
        {
            anyhow::bail!("Candle backend not enabled");
        }
    }

    fn is_ready(&self) -> bool {
        #[cfg(feature = "backend-candle")]
        {
            // Backend is ready if at least detector is loaded
            self.detector.is_some()
        }

        #[cfg(not(feature = "backend-candle"))]
        {
            false
        }
    }

    fn name(&self) -> &'static str {
        "Candle"
    }
}
