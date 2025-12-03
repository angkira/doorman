use super::backend::{Face, MLBackend};
use anyhow::{Context, Result};
use async_trait::async_trait;
use image::{DynamicImage, GenericImageView};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::Path;
use std::sync::Mutex;
use tracing::info;

/// Native PyO3-based PyTorch backend
/// 
/// Uses doorman_ml_native extension for direct ONNX Runtime calls
/// without IPC overhead. Achieves ~169 FPS vs ~8 FPS with IPC.
pub struct TorchBackendNative {
    ml_instance: Mutex<Py<PyAny>>,
    ready: bool,
}

impl TorchBackendNative {
    pub fn new(models_dir: &Path, device: &str) -> Result<Self> {
        info!("Initializing Native PyTorch backend...");
        info!("Models directory: {:?}", models_dir);
        info!("Device: {}", device);

        // Set ORT_DYLIB_PATH if not set
        if std::env::var("ORT_DYLIB_PATH").is_err() {
            // Try to find libonnxruntime.so in common locations
            let possible_paths = [
                "/usr/lib/x86_64-linux-gnu/libonnxruntime.so",  // System package (preferred - no executable stack issues)
                "/usr/lib/libonnxruntime.so",
                "/usr/local/lib/libonnxruntime.so",
            ];
            
            for path in &possible_paths {
                if std::path::Path::new(path).exists() {
                    std::env::set_var("ORT_DYLIB_PATH", path);
                    info!("Set ORT_DYLIB_PATH to: {}", path);
                    break;
                }
            }
        }

        // Initialize Python with PYTHONPATH if set
        if let Ok(python_path) = std::env::var("PYTHONPATH") {
            info!("Using PYTHONPATH: {}", python_path);
        }

        let ml_instance = Python::with_gil(|py| {
            // Add PYTHONPATH to sys.path
            if let Ok(python_path) = std::env::var("PYTHONPATH") {
                let sys = py.import_bound("sys")?;
                let path = sys.getattr("path")?;
                for p in python_path.split(':') {
                    if !p.is_empty() {
                        path.call_method1("insert", (0, p))?;
                    }
                }
            }
            
            // Import native module
            let module = py.import_bound("doorman_ml_native")
                .map_err(|e| {
                    // Log the actual Python error
                    info!("Python import error: {}", e);
                    anyhow::anyhow!("Failed to import doorman_ml_native: {}. Run: cd daemon/native_ml && ./build.sh", e)
                })?;
            
            let ml_class = module.getattr("DoormanML")?;
            
            // Create instance - this loads and compiles all models
            info!("Loading ML models (this may take 3-5 minutes on first run)...");
            let ml_instance = ml_class.call1((
                models_dir.to_str().unwrap(),
                device
            ))?;

            info!("✓ Models loaded successfully");
            
            Ok::<Py<PyAny>, anyhow::Error>(ml_instance.into())
        })?;

        // Warmup: run inference on dummy image to ensure models are fully compiled
        info!("Warming up models...");
        let warmup_start = std::time::Instant::now();
        
        Python::with_gil(|py| {
            let ml = ml_instance.bind(py);
            
            // Create dummy 1024x720 image
            let dummy_data = vec![128u8; 1024 * 720 * 3];
            
            // Run detection warmup (3 iterations)
            for i in 1..=3 {
                let bytes = PyBytes::new_bound(py, &dummy_data);
                ml.call_method1("detect_faces", (bytes, 1024u32, 720u32))?;
                info!("  Warmup iteration {}/3", i);
            }
            
            // Run liveness/embedding warmup on 112x112
            let dummy_crop = vec![128u8; 112 * 112 * 3];
            
            for i in 1..=2 {
                let crop_bytes = PyBytes::new_bound(py, &dummy_crop);
                ml.call_method1("check_liveness", (crop_bytes,))?;
                
                let crop_bytes = PyBytes::new_bound(py, &dummy_crop);
                ml.call_method1("extract_embedding", (crop_bytes,))?;
                info!("  Warmup liveness/embedding {}/2", i);
            }
            
            Ok::<(), anyhow::Error>(())
        })?;

        let warmup_duration = warmup_start.elapsed();
        info!("✓ Warmup complete in {:.2}s", warmup_duration.as_secs_f32());
        info!("Native PyTorch backend ready for production use");

        Ok(Self {
            ml_instance: Mutex::new(ml_instance),
            ready: true,
        })
    }

}

#[async_trait]
impl MLBackend for TorchBackendNative {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        let (width, height) = image.dimensions();
        let rgb_data = image.to_rgb8().into_raw();
        
        // Get Py<PyAny> for thread-safe transfer
        let ml_instance = Python::with_gil(|py| {
            self.ml_instance.lock().unwrap().clone_ref(py)
        });
        
        let faces = tokio::task::spawn_blocking(move || {
            // Call detect_faces via Python GIL
            Python::with_gil(|py| {
                let ml = ml_instance.bind(py);
                let bytes = PyBytes::new_bound(py, &rgb_data);
                let result = ml.call_method1("detect_faces", (bytes, width, height))?;
                
                // Parse detections
                let detections = result.getattr("detections")?;
                let detections = detections.extract::<Vec<Py<PyAny>>>()?;
                
                let mut parsed_faces = Vec::new();
                for det in detections {
                    let det = det.bind(py);
                    let bbox = det.getattr("bbox")?.extract::<(f32, f32, f32, f32)>()?;
                    let confidence = det.getattr("confidence")?.extract::<f32>()?;
                    parsed_faces.push((bbox.0, bbox.1, bbox.2, bbox.3, confidence));
                }
                
                Ok::<Vec<(f32, f32, f32, f32, f32)>, anyhow::Error>(parsed_faces)
            })
        })
        .await??;

        if faces.is_empty() {
            return Ok(None);
        }

        // Return first face with highest confidence
        let (x1, y1, x2, y2, confidence) = faces[0];
        
        // Convert pixel coordinates to normalized [0, 1]
        let bbox = (
            x1 / width as f32,
            y1 / height as f32,
            (x2 - x1) / width as f32,
            (y2 - y1) / height as f32,
        );

        Ok(Some(Face {
            bbox,
            confidence,
            frame_dimensions: (width, height),
        }))
    }

    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        // Crop and resize face to 112x112
        let (img_w, img_h) = image.dimensions();
        let (x, y, w, h) = face.bbox;
        
        let x_px = (x * img_w as f32) as u32;
        let y_px = (y * img_h as f32) as u32;
        let w_px = (w * img_w as f32) as u32;
        let h_px = (h * img_h as f32) as u32;
        
        let face_crop = image.crop_imm(x_px, y_px, w_px, h_px);
        let face_crop = face_crop.resize_exact(112, 112, image::imageops::FilterType::Lanczos3);
        let face_data = face_crop.to_rgb8().into_raw();

        let ml_instance = Python::with_gil(|py| self.ml_instance.lock().unwrap().clone_ref(py));
        
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let ml = ml_instance.bind(py);
                let bytes = PyBytes::new_bound(py, &face_data);
                let result = ml.call_method1("check_liveness", (bytes,))?;
                let is_live = result.getattr("is_live")?.extract::<bool>()?;
                Ok::<bool, anyhow::Error>(is_live)
            })
        })
        .await?
    }

    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        // Crop and resize face to 112x112
        let (img_w, img_h) = image.dimensions();
        let (x, y, w, h) = face.bbox;
        
        let x_px = (x * img_w as f32) as u32;
        let y_px = (y * img_h as f32) as u32;
        let w_px = (w * img_w as f32) as u32;
        let h_px = (h * img_h as f32) as u32;
        
        let face_crop = image.crop_imm(x_px, y_px, w_px, h_px);
        let face_crop = face_crop.resize_exact(112, 112, image::imageops::FilterType::Lanczos3);
        let face_data = face_crop.to_rgb8().into_raw();

        let ml_instance = Python::with_gil(|py| self.ml_instance.lock().unwrap().clone_ref(py));
        
        tokio::task::spawn_blocking(move || {
            Python::with_gil(|py| {
                let ml = ml_instance.bind(py);
                let bytes = PyBytes::new_bound(py, &face_data);
                let result = ml.call_method1("extract_embedding", (bytes,))?;
                
                let embedding_bytes = result.downcast::<PyBytes>()
                    .map_err(|e| anyhow::anyhow!("Failed to downcast to PyBytes: {}", e))?;
                let embedding_slice = embedding_bytes.as_bytes();
                
                let embedding: Vec<f32> = embedding_slice
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                
                Ok::<Vec<f32>, anyhow::Error>(embedding)
            })
        })
        .await?
    }

    fn is_ready(&self) -> bool {
        self.ready
    }

    fn name(&self) -> &'static str {
        "Native PyTorch (PyO3 + ONNX Runtime)"
    }
}
