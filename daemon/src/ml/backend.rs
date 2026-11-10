use anyhow::Result;
use async_trait::async_trait;
use image::DynamicImage;

/// Face detection result
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Face {
    pub bbox: (f32, f32, f32, f32), // x, y, w, h in normalized [0,1] coordinates
    pub confidence: f32,
    pub frame_dimensions: (u32, u32), // (width, height) of frame these coords are for
    /// Optional 5-point facial landmarks in normalized [0,1] coordinates,
    /// ordered: right-eye, left-eye, nose, right-mouth-corner, left-mouth-corner
    /// (YuNet/OpenCV ordering). `None` when the detector does not provide them.
    pub landmarks: Option<[(f32, f32); 5]>,
    /// The other faces the detector found in the same frame (post-NMS, above
    /// the detector's own score threshold), without their own `other_faces`.
    /// Empty when the detector reports a single face or cannot tell.
    pub other_faces: Vec<Face>,
}

impl Face {
    /// Normalized box area.
    pub fn area_frac(&self) -> f32 {
        let (_, _, w, h) = self.bbox;
        (w * h).abs()
    }
}

/// Abstract ML backend trait - implements driver pattern
#[async_trait]
pub trait MLBackend: Send + Sync {
    /// Detect faces in image
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>>;

    /// Check if face is real (anti-spoofing) — legacy MiniFASNet check.
    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool>;

    /// Monocular-depth face relief score for anti-spoofing (PAD).
    ///
    /// Runs Depth-Anything-V2 over the frame, crops the depth map to the
    /// detected face bbox, and returns `std(face_depth) / (depth_range + eps)`
    /// clamped to `[0, 1]`. A real 3D face scores high; a flat screen/print
    /// replay scores near zero. Returns `Ok(None)` when the depth model is not
    /// loaded (the caller decides how to treat an unavailable gate).
    async fn depth_relief(&self, _image: &DynamicImage, _face: &Face) -> Result<Option<f32>> {
        Ok(None)
    }

    /// MiniFASNetV2-SE real-minus-spoof logit difference for the face, for
    /// SHADOW logging only (positive = judged real). Returns `Ok(None)` when the
    /// model is not loaded. Unlike [`MLBackend::check_liveness`] this never
    /// answers "pass" for a missing model, and nothing in the decision path may
    /// call it.
    async fn minifasnet_logit_diff(
        &self,
        _image: &DynamicImage,
        _face: &Face,
    ) -> Result<Option<f32>> {
        Ok(None)
    }

    /// Extract face embedding (512-d vector)
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>>;

    /// Check if backend is ready
    fn is_ready(&self) -> bool;

    /// Get backend name
    fn name(&self) -> &'static str;
}

/// Backend selection configuration.
///
/// Only the ONNX Runtime (`ort`) backend is shipped. The enum is retained so
/// the config `ml.backend` string keeps mapping to a concrete backend and so
/// additional backends can be re-introduced cleanly later.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// ONNX Runtime (ort crate)
    OnnxRuntime,
}

impl BackendType {
    // Not `FromStr`: this is infallible (every value maps to the one backend).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(_s: &str) -> Self {
        // All config values currently resolve to the single ONNX Runtime backend.
        Self::OnnxRuntime
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_from_str() {
        assert_eq!(BackendType::from_str("onnx"), BackendType::OnnxRuntime);
        assert_eq!(BackendType::from_str("ort"), BackendType::OnnxRuntime);
        assert_eq!(BackendType::from_str("ort-cpu"), BackendType::OnnxRuntime);
        // Unknown / legacy values fall back to the only available backend.
        assert_eq!(BackendType::from_str("unknown"), BackendType::OnnxRuntime);
    }
}
