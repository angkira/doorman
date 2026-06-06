/// Model-specific configuration and output parsing
///
/// Different model architectures have different input/output formats.
/// This module centralizes model-specific knowledge.

/// Face detector configuration
pub struct DetectorConfig {
    /// Human-readable model name.
    pub name: &'static str,
    /// Model file name expected in the models directory.
    pub model_file: &'static str,
    /// Fixed network input width (pixels). YuNet 2023mar is a FIXED-shape model.
    pub input_width: u32,
    /// Fixed network input height (pixels).
    pub input_height: u32,
    /// Minimum face score (sqrt(cls*obj)) to keep a detection.
    pub confidence_threshold: f32,
    /// IoU threshold used by Non-Maximum Suppression.
    pub iou_threshold: f32,
}

/// Known face detector models
impl DetectorConfig {
    /// YuNet (OpenCV Zoo, `face_detection_yunet_2023mar.onnx`).
    ///
    /// - License: MIT.
    /// - Input tensor `input`: FIXED `[1, 3, 640, 640]`, **BGR**, raw float
    ///   `0..255` (NO mean/std normalization), NCHW.
    /// - Outputs (12 total), one triplet+kps per stride s in {8, 16, 32} over a
    ///   `(640/s) x (640/s)` grid (row-major: idx -> row = idx / cols,
    ///   col = idx % cols):
    ///     * `cls_{s}`  `[1, N, 1]` — class score, already sigmoid-activated.
    ///     * `obj_{s}`  `[1, N, 1]` — objectness, already sigmoid-activated.
    ///     * `bbox_{s}` `[1, N, 4]` — `(dx, dy, dw, dh)` box deltas.
    ///     * `kps_{s}`  `[1, N, 10]` — 5 landmark `(dx, dy)` deltas.
    /// - Decode (per anchor): `cx = (col + dx) * s`, `cy = (row + dy) * s`,
    ///   `w = exp(dw) * s`, `h = exp(dh) * s`; top-left = `(cx - w/2, cy - h/2)`.
    ///   Landmark `k`: `lx = (col + kdx) * s`, `ly = (row + kdy) * s`.
    /// - Score: `sqrt(cls * obj)`.
    /// - Coordinates are produced in 640x640 space, then scaled back to the
    ///   original frame by `(orig_w / 640, orig_h / 640)` (plain stretch resize).
    pub const YUNET: DetectorConfig = DetectorConfig {
        name: "YuNet",
        model_file: "face_detection_yunet_2023mar.onnx",
        input_width: 640,
        input_height: 640,
        confidence_threshold: 0.6,
        iou_threshold: 0.3,
    };

    /// Detector strides used by the YuNet multi-level decode head.
    pub const YUNET_STRIDES: [u32; 3] = [8, 16, 32];
}

/// Liveness detector configuration (MiniFASNetV2-SE, facenox/face-antispoof-onnx).
///
/// - Source: <https://github.com/facenox/face-antispoof-onnx> (release v1.0.0,
///   `best_model.onnx`). License: **Apache-2.0**. ~98.2% acc / AUC 0.9984 on
///   CelebA-Spoof. Single model (NOT the older fused 2-model minivision setup).
/// - A SINGLE model fed a `128x128` **RGB** crop, normalized **`/255` -> [0,1]**
///   (the repo's `preprocess` does `img.transpose(2,0,1).astype(float32)/255.0`;
///   NO mean/std — unlike the old 80x80 minivision models which used raw 0..255),
///   NCHW float32 `[1, 3, 128, 128]`.
/// - Crop (repo `crop()`): take the face bbox, compute `side = max(w,h) * scale`
///   (`scale` = the repo's default `bbox_expansion_factor` = **1.5**), a SQUARE
///   box of that side centered on the bbox center, reflect-pad (`BORDER_REFLECT_101`)
///   any part outside the frame, then letterbox-resize to 128 (a no-op for an
///   already-square crop beyond the resize).
/// - Output `output`: `[1, 2]` raw **logits** — **index 0 = real, index 1 = spoof**.
///   The repo's decision (`process_with_logits`) is
///   `is_real = (real_logit - spoof_logit) >= logit_threshold`, where
///   `logit_threshold = ln(p/(1-p))` for a target real-probability `p`
///   (default `p = 0.5` -> threshold `0.0`, i.e. plain argmax). We store the
///   probability `p` (`real_prob_threshold`) and derive the logit threshold.
///   Empirically (this repo's fixtures): genuine live face `diff ~ +8`,
///   printed/replayed spoof `diff ~ -10` (clean separation).
pub struct LivenessConfig {
    pub name: &'static str,
    /// Square network input size (128).
    pub input_size: u32,
    /// Output index of the "real" / "live" class (0).
    pub real_class_index: usize,
    /// Output index of the "spoof" class (1).
    pub spoof_class_index: usize,
    /// Square-crop expansion factor about the bbox center (repo default 1.5).
    pub bbox_expansion_factor: f32,
    /// Target real-class probability `p` for the decision boundary; the backend
    /// accepts as real iff `softmax(real) >= p`, equivalently
    /// `(real_logit - spoof_logit) >= ln(p/(1-p))`. Default 0.5 (argmax).
    pub real_prob_threshold: f32,
    /// ONNX model file name in the models directory.
    pub model_file: &'static str,
}

impl LivenessConfig {
    /// MiniFASNetV2-SE single-model liveness (facenox, 128x128, Apache-2.0).
    pub const MINIFASNET: LivenessConfig = LivenessConfig {
        name: "MiniFASNetV2-SE (facenox, 128x128)",
        input_size: 128,
        real_class_index: 0,
        spoof_class_index: 1,
        bbox_expansion_factor: 1.5,
        // Default p=0.5 reproduces the repo's argmax decision (threshold 0.0 in
        // logit-diff space). Liveness is a non-fatal convenience deterrent.
        real_prob_threshold: 0.5,
        model_file: "minifasnet_v2se.onnx",
    };

    /// Back-compat alias.
    pub const STANDARD: LivenessConfig = Self::MINIFASNET;
}

/// Face recognizer configuration
pub struct RecognizerConfig {
    pub name: &'static str,
    /// Model file name expected in the models directory.
    pub model_file: &'static str,
    pub input_size: u32,
    pub embedding_size: usize,
}

impl RecognizerConfig {
    /// EdgeFace-S (gamma=0.5), `edgeface_s.onnx` — the DEFAULT recognizer.
    ///
    /// - Source: <https://github.com/otroshi/edgeface> (Idiap). License: the
    ///   **weights are CC-BY-NC-SA 4.0 — NON-COMMERCIAL** (the "BSD-3" sometimes
    ///   cited is the `bob` framework *code*, not these weights). For commercial
    ///   use, swap in **AuraFace-v1** (fal, native ONNX, commercial-OK).
    /// - Lightweight: ~3.65M params, ~15 MB ONNX (vs 174 MB for ArcFace).
    /// - Exported via `scripts/export_edgeface.py` (timm `edgenext_small` backbone
    ///   with low-rank linear layers, rank_ratio=0.5).
    /// - Input tensor `input`: `[1, 3, 112, 112]`, **RGB**, NCHW float32,
    ///   normalized `(x - 127.5) / 127.5` -> `[-1, 1]`. This matches EdgeFace's
    ///   training transform `ToTensor()` + `Normalize(mean=0.5, std=0.5)`
    ///   exactly, and happens to be identical to ArcFace preprocessing.
    /// - Output tensor `embedding`: `[1, 512]`; the backend L2-normalizes it.
    /// - Faces MUST be aligned to the canonical 5-point 112x112 template
    ///   (`RECOGNIZER_TEMPLATE_112`); EdgeFace aligns with the same
    ///   `get_reference_facial_points(default_square=True)` template ArcFace uses.
    ///   Empirically (LFW fixtures, this repo): genuine cosine ~0.5-0.65,
    ///   impostor cosine ~0.0-0.05 (clean separation; default threshold 0.4).
    pub const EDGEFACE: RecognizerConfig = RecognizerConfig {
        name: "EdgeFace-S (gamma=0.5, CC-BY-NC-SA 4.0, non-commercial)",
        model_file: "edgeface_s.onnx",
        input_size: 112,
        embedding_size: 512,
    };

    /// InsightFace buffalo_l ArcFace (ResNet50, WebFace600K), `w600k_r50.onnx`.
    ///
    /// Non-default fallback. License: MIT model zoo, but **non-commercial** weights
    /// (WebFace600K) — kept only as an optional drop-in. Same 112x112 RGB,
    /// `(x-127.5)/127.5`, 512-d, same alignment template as EdgeFace.
    pub const ARCFACE: RecognizerConfig = RecognizerConfig {
        name: "ArcFace (buffalo_l/w600k_r50)",
        model_file: "w600k_r50.onnx",
        input_size: 112,
        embedding_size: 512,
    };

    /// Canonical 5-point alignment template for a 112x112 crop. Shared by
    /// EdgeFace and ArcFace (both train on this standard ArcFace template).
    ///
    /// Order matches the detector's landmark order:
    /// right-eye, left-eye, nose, right-mouth-corner, left-mouth-corner.
    pub const RECOGNIZER_TEMPLATE_112: [(f32, f32); 5] = [
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    ];

    /// Back-compat alias for the alignment template.
    pub const ARCFACE_TEMPLATE_112: [(f32, f32); 5] = Self::RECOGNIZER_TEMPLATE_112;
}

/// Complete model configuration set
pub struct ModelSet {
    pub detector: DetectorConfig,
    pub liveness: LivenessConfig,
    pub recognizer: RecognizerConfig,
}

impl ModelSet {
    /// Default model set (YuNet + Standard Liveness + MobileFaceNet).
    ///
    /// Only the YuNet detector is wired into the active inference path right
    /// now; liveness and recognition entries are retained for the later steps.
    pub const DEFAULT: ModelSet = ModelSet {
        detector: DetectorConfig::YUNET,
        liveness: LivenessConfig::STANDARD,
        recognizer: RecognizerConfig::EDGEFACE,
    };
}
