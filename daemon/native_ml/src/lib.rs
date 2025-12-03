use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::exceptions::PyRuntimeError;

mod detector;
mod liveness;
mod embedder;

use detector::FaceDetector;
use liveness::LivenessChecker;
use embedder::FaceEmbedder;

/// Face detection result
#[pyclass]
#[derive(Clone)]
pub struct DetectionResult {
    #[pyo3(get)]
    pub bbox: (f32, f32, f32, f32),  // x1, y1, x2, y2
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub landmarks: Vec<(f32, f32)>,  // 5 keypoints
}

#[pymethods]
impl DetectionResult {
    fn __repr__(&self) -> String {
        format!("DetectionResult(bbox={:?}, conf={:.3})", self.bbox, self.confidence)
    }
}

/// Liveness check result
#[pyclass]
#[derive(Clone)]
pub struct LivenessResult {
    #[pyo3(get)]
    pub is_live: bool,
    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl LivenessResult {
    fn __repr__(&self) -> String {
        format!("LivenessResult(is_live={}, conf={:.3})", self.is_live, self.confidence)
    }
}

/// Native ML backend for doorman
#[pyclass]
pub struct DoormanML {
    detector: FaceDetector,
    liveness: LivenessChecker,
    embedder: FaceEmbedder,
}

#[pymethods]
impl DoormanML {
    #[new]
    #[pyo3(signature = (models_dir, device="cuda"))]
    fn new(models_dir: &str, device: &str) -> PyResult<Self> {
        let detector = FaceDetector::new(models_dir, device)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load detector: {}", e)))?;
        
        let liveness = LivenessChecker::new(models_dir, device)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load liveness: {}", e)))?;
        
        let embedder = FaceEmbedder::new(models_dir, device)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to load embedder: {}", e)))?;

        Ok(Self {
            detector,
            liveness,
            embedder,
        })
    }

    /// Detect faces in image
    /// 
    /// Args:
    ///     image_data: RGB image bytes (height * width * 3)
    ///     width: Image width
    ///     height: Image height
    /// 
    /// Returns:
    ///     List of DetectionResult
    fn detect_faces(
        &mut self,
        image_data: &PyBytes,
        width: u32,
        height: u32,
    ) -> PyResult<Vec<DetectionResult>> {
        let data = image_data.as_bytes();
        
        self.detector
            .detect(data, width, height)
            .map_err(|e| PyRuntimeError::new_err(format!("Detection failed: {}", e)))
    }

    /// Check if face is live
    /// 
    /// Args:
    ///     face_crop: RGB face crop bytes (112 * 112 * 3)
    /// 
    /// Returns:
    ///     LivenessResult
    fn check_liveness(&mut self, face_crop: &PyBytes) -> PyResult<LivenessResult> {
        let data = face_crop.as_bytes();
        
        self.liveness
            .check(data)
            .map_err(|e| PyRuntimeError::new_err(format!("Liveness check failed: {}", e)))
    }

    /// Extract face embedding
    /// 
    /// Args:
    ///     face_crop: RGB face crop bytes (112 * 112 * 3)
    /// 
    /// Returns:
    ///     Embedding as bytes (512 floats = 2048 bytes)
    fn extract_embedding<'py>(&mut self, py: Python<'py>, face_crop: &PyBytes) -> PyResult<&'py PyBytes> {
        let data = face_crop.as_bytes();
        
        let embedding = self.embedder
            .extract(data)
            .map_err(|e| PyRuntimeError::new_err(format!("Embedding extraction failed: {}", e)))?;

        // Convert Vec<f32> to bytes
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embedding.as_ptr() as *const u8,
                embedding.len() * std::mem::size_of::<f32>()
            )
        };

        Ok(PyBytes::new(py, bytes))
    }
}

/// Native ML module for doorman
#[pymodule]
fn doorman_ml_native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DoormanML>()?;
    m.add_class::<DetectionResult>()?;
    m.add_class::<LivenessResult>()?;
    Ok(())
}
