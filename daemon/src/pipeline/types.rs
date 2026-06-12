use image::DynamicImage;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Raw camera frame with metadata
/// Image is Arc-wrapped for efficient sharing between pipeline stages
#[derive(Clone)]
pub struct RawFrame {
    pub image: Arc<DynamicImage>,
    pub timestamp: Instant,
    pub sequence: u64,
}

/// Face bounding box from detection
#[derive(Clone, Debug)]
pub struct Face {
    pub bbox: (f32, f32, f32, f32), // (x, y, width, height) normalized 0-1
    pub confidence: f32,
}

/// Detection result from ML pipeline
pub struct DetectionResult {
    pub sequence: u64,
    pub face: Option<Face>,
    pub embedding: Option<Vec<f32>>,
    pub processing_time: Duration,
    pub frame_width: u32,
    pub frame_height: u32,
    /// The source frame, carried through ONLY on quality-passed embedding frames
    /// so the recognition stage can re-run the FATAL depth-liveness gate before a
    /// proactive unlock (Stage B). `None` on detect-only / no-face frames. Kept
    /// Arc-cheap; not used by the preview overlay path.
    pub frame: Option<Arc<DynamicImage>>,
    /// The full ML-backend face (with landmarks) for the same frame, needed by
    /// `liveness_gate` / depth relief. `None` unless `frame` is `Some`.
    pub ml_face: Option<crate::ml::Face>,
}
