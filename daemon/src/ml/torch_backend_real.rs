use anyhow::{Context, Result};
use image::DynamicImage;
use std::path::Path;

#[cfg(feature = "backend-torch")]
use tch::{nn, Device, Kind, Tensor};

use super::{BBox, FaceEmbedding, MLBackend, ProcessedFrame};

pub struct TorchBackend {
    #[cfg(feature = "backend-torch")]
    device: Device,
    #[cfg(feature = "backend-torch")]
    face_detector: tch::CModule,
    #[cfg(feature = "backend-torch")]
    liveness_detector: tch::CModule,
    #[cfg(feature = "backend-torch")]
    face_recognizer: tch::CModule,
    anchors: Vec<(f32, f32)>,
}

impl TorchBackend {
    pub fn new(models_dir: &Path) -> Result<Self> {
        #[cfg(not(feature = "backend-torch"))]
        {
            anyhow::bail!("Torch backend not compiled. Build with --features backend-torch");
        }
        
        #[cfg(feature = "backend-torch")]
        {
            tracing::info!("Initializing PyTorch backend with ROCm...");
            
            // Check if CUDA (ROCm) is available
            let device = if tch::Cuda::is_available() {
                let device_count = tch::Cuda::device_count();
                tracing::info!("ROCm available with {} devices", device_count);
                Device::Cuda(0)
            } else {
                tracing::warn!("ROCm not available, falling back to CPU");
                Device::Cpu
            };
            
            tracing::info!("Using device: {:?}", device);
            
            // Load TorchScript models
            let detector_path = models_dir.join("blazeface.pt");
            let liveness_path = models_dir.join("liveness.pt");
            let recognizer_path = models_dir.join("mobilefacenet.pt");
            
            tracing::info!("Loading TorchScript models from {:?}", models_dir);
            
            let face_detector = tch::CModule::load(&detector_path)
                .with_context(|| format!("Failed to load face detector from {:?}", detector_path))?;
            
            let liveness_detector = tch::CModule::load(&liveness_path)
                .with_context(|| format!("Failed to load liveness detector from {:?}", liveness_path))?;
            
            let face_recognizer = tch::CModule::load(&recognizer_path)
                .with_context(|| format!("Failed to load face recognizer from {:?}", recognizer_path))?;
            
            tracing::info!("✓ Loaded all TorchScript models");
            
            let anchors = Self::generate_anchors();
            
            Ok(Self {
                device,
                face_detector,
                liveness_detector,
                face_recognizer,
                anchors,
            })
        }
    }
    
    fn generate_anchors() -> Vec<(f32, f32)> {
        // Same anchor generation as in other backends
        let mut anchors = Vec::new();
        let feature_maps = [[16, 16], [8, 8]];
        let min_sizes = [[16.0, 32.0], [64.0, 128.0]];
        let input_size = 128.0;
        
        for (fm, min_size) in feature_maps.iter().zip(min_sizes.iter()) {
            for i in 0..fm[0] {
                for j in 0..fm[1] {
                    for &min in min_size {
                        let s_kx = min / input_size;
                        let s_ky = min / input_size;
                        let cx = (j as f32 + 0.5) / fm[1] as f32;
                        let cy = (i as f32 + 0.5) / fm[0] as f32;
                        anchors.push((cx, cy));
                    }
                }
            }
        }
        
        anchors
    }
}

#[async_trait::async_trait]
impl MLBackend for TorchBackend {
    async fn process_frame(&self, frame: &DynamicImage) -> Result<ProcessedFrame> {
        #[cfg(not(feature = "backend-torch"))]
        {
            anyhow::bail!("Torch backend not compiled");
        }
        
        #[cfg(feature = "backend-torch")]
        {
            // Convert image to tensor
            let rgb_image = frame.to_rgb8();
            let (width, height) = rgb_image.dimensions();
            
            // Resize to 128x128 for detector
            let resized = image::imageops::resize(
                &rgb_image,
                128,
                128,
                image::imageops::FilterType::Nearest,
            );
            
            // Convert to tensor [1, 3, 128, 128] normalized to [-1, 1]
            let img_data: Vec<f32> = resized
                .pixels()
                .flat_map(|p| {
                    vec![
                        (p[0] as f32 / 127.5) - 1.0,
                        (p[1] as f32 / 127.5) - 1.0,
                        (p[2] as f32 / 127.5) - 1.0,
                    ]
                })
                .collect();
            
            let input = Tensor::of_slice(&img_data)
                .view([1, 3, 128, 128])
                .to_device(self.device);
            
            // Run detection
            let outputs = self.face_detector.forward_ts(&[input])?;
            
            // Parse outputs (similar to other backends)
            // TODO: Implement full detection parsing
            
            Ok(ProcessedFrame {
                detections: vec![],
                timestamp: std::time::SystemTime::now(),
            })
        }
    }
}
