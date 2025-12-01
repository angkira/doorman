/// Model-specific configuration and output parsing
///
/// Different model architectures have different input/output formats.
/// This module centralizes model-specific knowledge.

/// Face detector configuration
pub struct DetectorConfig {
    pub name: &'static str,
    pub input_width: u32,
    pub input_height: u32,
    pub num_classes: usize,
    pub confidence_threshold: f32,
    pub max_detections: i64,
    pub iou_threshold: f32,
}

/// Known face detector models
impl DetectorConfig {
    /// BlazeFace: Google's lightweight face detector
    /// - Input: 320x240 RGB (garavv/blazeface-onnx model)
    /// - Outputs: boxes [N*4], scores [N*2] where N may differ between outputs
    /// - Boxes: [x0,y0,w0,h0, ...] in normalized coordinates
    /// - Scores: [bg0,face0, bg1,face1, ...] interleaved class scores
    pub const BLAZEFACE: DetectorConfig = DetectorConfig {
        name: "BlazeFace",
        input_width: 320,
        input_height: 240,
        num_classes: 2, // background + face
        confidence_threshold: 0.1, // Lowered for testing - detecting very low confidence faces
        max_detections: 10, // Max faces to detect (authentication needs only 1)
        iou_threshold: 0.3, // IoU threshold for Non-Maximum Suppression
    };

    /// UltraFace: Ultra-lightweight face detector
    /// - Input: 320x240 RGB
    /// - Different output format than BlazeFace
    #[allow(dead_code)]
    pub const ULTRAFACE: DetectorConfig = DetectorConfig {
        name: "UltraFace",
        input_width: 320,
        input_height: 240,
        num_classes: 2,
        confidence_threshold: 0.7,
        max_detections: 10,
        iou_threshold: 0.3,
    };
}

/// Liveness detector configuration
pub struct LivenessConfig {
    pub name: &'static str,
    pub input_size: u32, // square input
    pub live_class_index: usize, // which output index is "live"
    pub confidence_threshold: f32,
}

impl LivenessConfig {
    /// Standard liveness model
    /// - Input: 96x96 RGB (actual model size)
    /// - Output: [not_live, live] probabilities
    pub const STANDARD: LivenessConfig = LivenessConfig {
        name: "Liveness",
        input_size: 96,
        live_class_index: 1, // index 1 is "live" class
        confidence_threshold: 0.5,
    };
}

/// Face recognizer configuration
pub struct RecognizerConfig {
    pub name: &'static str,
    pub input_size: u32,
    pub embedding_size: usize,
}

impl RecognizerConfig {
    /// MobileFaceNet/ArcFace: Face recognition
    /// - Input: 112x112 RGB
    /// - Output: 512-dim embedding vector (ArcFace ResNet100)
    pub const MOBILEFACENET: RecognizerConfig = RecognizerConfig {
        name: "ArcFace",
        input_size: 112,
        embedding_size: 512,
    };

    /// ArcFace: High-accuracy face recognition
    /// - Input: 112x112 RGB
    /// - Output: 512-dim embedding vector
    #[allow(dead_code)]
    pub const ARCFACE: RecognizerConfig = RecognizerConfig {
        name: "ArcFace",
        input_size: 112,
        embedding_size: 512,
    };
}

/// Complete model configuration set
pub struct ModelSet {
    pub detector: DetectorConfig,
    pub liveness: LivenessConfig,
    pub recognizer: RecognizerConfig,
}

impl ModelSet {
    /// Default model set (BlazeFace + Standard Liveness + MobileFaceNet)
    pub const DEFAULT: ModelSet = ModelSet {
        detector: DetectorConfig::BLAZEFACE,
        liveness: LivenessConfig::STANDARD,
        recognizer: RecognizerConfig::MOBILEFACENET,
    };
}
