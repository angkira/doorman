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
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(feature = "backend-ort")]
use ort::value::Value;
#[cfg(feature = "backend-ort")]
use std::path::Path;
#[cfg(feature = "backend-ort")]
use std::sync::Mutex;
#[cfg(feature = "backend-ort")]
use tracing::{info, warn};
#[cfg(feature = "backend-ort")]
use image::GenericImageView;

#[cfg(feature = "backend-ort")]
macro_rules! ort_try {
    ($expr:expr) => {
        $expr.map_err(|e| anyhow!("ORT error: {}", e))?
    };
}

#[cfg(feature = "backend-ort")]
/// ONNX Runtime backend (supports GPU via ROCm/CUDA)
/// Uses session pooling for concurrent requests
pub struct OrtBackend {
    detector_pool: Vec<Mutex<Session>>,
    /// Single MiniFASNetV2-SE liveness model session pool (may be empty: liveness
    /// is NON-FATAL and short-circuits to "pass" when unavailable).
    liveness_pool: Vec<Mutex<Session>>,
    recognizer_pool: Vec<Mutex<Session>>,
    pool_index: AtomicUsize,
}

#[cfg(feature = "backend-ort")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "backend-ort")]
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

        // Load detector pool (YuNet)
        let detector_path = models_dir.join(super::model_config::DetectorConfig::YUNET.model_file);
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

        // Load the single MiniFASNetV2-SE liveness model pool.
        // Liveness is NON-FATAL: a missing/failed model yields an empty pool and
        // check_liveness short-circuits to "pass" with a warn.
        let liveness_cfg = super::model_config::LivenessConfig::MINIFASNET;
        let liveness_path = models_dir.join(liveness_cfg.model_file);
        let mut liveness_pool = Vec::new();
        for i in 0..POOL_SIZE {
            match Self::load_model(&liveness_path, config) {
                Ok(model) => liveness_pool.push(Mutex::new(model)),
                Err(e) => warn!("✗ Failed to load liveness {} session {}: {}", liveness_cfg.model_file, i, e),
            }
        }
        if !liveness_pool.is_empty() {
            info!("✓ Loaded {} sessions for liveness model {}", liveness_pool.len(), liveness_cfg.model_file);
        } else {
            warn!("✗ Liveness model {} unavailable — liveness will be skipped", liveness_cfg.model_file);
        }

        // Load recognizer pool
        let recognizer_path = models_dir.join(super::model_config::RecognizerConfig::EDGEFACE.model_file);
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

#[cfg(feature = "backend-ort")]
impl OrtBackend {
    /// Preprocess an image into YuNet's input tensor.
    ///
    /// YuNet 2023mar has a FIXED `[1, 3, 640, 640]` input expecting **BGR**,
    /// raw float `0..255` (NO mean/std normalization), NCHW. The image is
    /// stretch-resized to the square input (aspect ratio NOT preserved); the
    /// decoder maps coordinates back to the original frame.
    fn yunet_preprocess(image: &DynamicImage, size: u32) -> Vec<f32> {
        // Stretch-resize to the square network input.
        let resized = image.resize_exact(size, size, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let n = (size * size) as usize;
        // NCHW, channel order B, G, R (YuNet/OpenCV expects BGR).
        let mut data = vec![0.0f32; 3 * n];
        let (b_off, g_off, r_off) = (0, n, 2 * n);
        for (i, px) in rgb.pixels().enumerate() {
            // px is RGB; write into BGR planes, raw 0..255.
            data[b_off + i] = px[2] as f32;
            data[g_off + i] = px[1] as f32;
            data[r_off + i] = px[0] as f32;
        }
        data
    }

    /// Fallback face crop when landmarks are unavailable: crop the (normalized)
    /// bbox from the full frame and resize to `size`x`size`. This is the legacy
    /// path and is noticeably less accurate than landmark alignment; it is only
    /// hit by warmup / landmark-less callers.
    /// Replicate facenox `crop()`: a SQUARE crop of side
    /// `max(bbox_w, bbox_h) * factor` centered on the bbox center, with any
    /// out-of-frame region filled by **reflect-101** padding (mirror without
    /// repeating the edge pixel, i.e. `cv2.BORDER_REFLECT_101`), then resized to
    /// `size`x`size` RGB. Mirrors the repo's integer arithmetic.
    fn antispoof_square_crop(
        image: &DynamicImage,
        face: &Face,
        factor: f32,
        size: u32,
    ) -> image::RgbImage {
        let src = image.to_rgb8();
        let (img_w, img_h) = (src.width() as i64, src.height() as i64);

        // bbox in source pixels (normalized [0,1] -> px), matching the repo's
        // (x, y, x+w, y+h) -> (x, y, w, h) reconstruction.
        let (nx, ny, nw, nh) = face.bbox;
        let bx = nx * img_w as f32;
        let by = ny * img_h as f32;
        let bw = (nw * img_w as f32).max(1.0);
        let bh = (nh * img_h as f32).max(1.0);

        let max_dim = bw.max(bh);
        let center_x = bx + bw / 2.0;
        let center_y = by + bh / 2.0;
        let crop_size = (max_dim * factor) as i64; // int() truncation, as in repo
        let crop_size = crop_size.max(1);
        let x0 = (center_x - max_dim * factor / 2.0) as i64; // int() truncation
        let y0 = (center_y - max_dim * factor / 2.0) as i64;

        // Reflect-101 index mapping into [0, len): mirror about the edges
        // without repeating the border pixel (matches OpenCV BORDER_REFLECT_101).
        fn reflect101(mut i: i64, len: i64) -> i64 {
            if len == 1 {
                return 0;
            }
            let period = 2 * (len - 1);
            i = ((i % period) + period) % period;
            if i >= len {
                i = period - i;
            }
            i
        }

        // Build the crop_size x crop_size square, sampling source with reflect-101.
        let mut square = image::RgbImage::new(crop_size as u32, crop_size as u32);
        for cy in 0..crop_size {
            let sy = reflect101(y0 + cy, img_h);
            for cx in 0..crop_size {
                let sx = reflect101(x0 + cx, img_w);
                let px = *src.get_pixel(sx as u32, sy as u32);
                square.put_pixel(cx as u32, cy as u32, px);
            }
        }

        // Letterbox-resize to `size` (no-op aspect change for a square crop).
        image::imageops::resize(
            &square,
            size,
            size,
            image::imageops::FilterType::Lanczos3,
        )
    }

    fn bbox_crop_resize(image: &DynamicImage, face: &Face, size: u32) -> image::RgbImage {
        let (img_w, img_h) = image.dimensions();
        let (nx, ny, nw, nh) = face.bbox;
        let x = (nx.clamp(0.0, 1.0) * img_w as f32) as u32;
        let y = (ny.clamp(0.0, 1.0) * img_h as f32) as u32;
        let w = ((nw.clamp(0.0, 1.0) * img_w as f32) as u32).max(1).min(img_w.saturating_sub(x).max(1));
        let h = ((nh.clamp(0.0, 1.0) * img_h as f32) as u32).max(1).min(img_h.saturating_sub(y).max(1));
        image
            .crop_imm(x, y, w, h)
            .resize_exact(size, size, image::imageops::FilterType::Lanczos3)
            .to_rgb8()
    }
}

#[cfg(feature = "backend-ort")]
#[async_trait]
impl MLBackend for OrtBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        use super::model_config::DetectorConfig;
        use super::yunet_decoder::{self, StrideOutputs};

        let detector = self
            .get_next_session(&self.detector_pool)
            .ok_or_else(|| anyhow!("Detector not loaded"))?;

        let cfg = DetectorConfig::YUNET;
        let (orig_width, orig_height) = image.dimensions();
        let input_size = cfg.input_width; // YuNet is square (640x640)

        // Preprocess -> BGR, raw 0..255, NCHW [1,3,640,640].
        let input_data = Self::yunet_preprocess(image, input_size);
        let input_tensor = ort_try!(Value::from_array((
            [1usize, 3, input_size as usize, input_size as usize],
            input_data
        )));

        let mut detector_lock = detector.lock().unwrap();
        let outputs = ort_try!(detector_lock.run(ort::inputs![input_tensor]));

        // Collect the per-stride output slices by name. The slices borrow from
        // `outputs`, which outlives the decode call below.
        let mut stride_views: Vec<StrideOutputs> = Vec::with_capacity(DetectorConfig::YUNET_STRIDES.len());
        for &stride in DetectorConfig::YUNET_STRIDES.iter() {
            let (_, cls) = ort_try!(outputs[format!("cls_{}", stride).as_str()].try_extract_tensor::<f32>());
            let (_, obj) = ort_try!(outputs[format!("obj_{}", stride).as_str()].try_extract_tensor::<f32>());
            let (_, bbox) = ort_try!(outputs[format!("bbox_{}", stride).as_str()].try_extract_tensor::<f32>());
            let (_, kps) = ort_try!(outputs[format!("kps_{}", stride).as_str()].try_extract_tensor::<f32>());
            stride_views.push(StrideOutputs { stride, cls, obj, bbox, kps });
        }

        let dets = yunet_decoder::decode(
            &stride_views,
            input_size,
            orig_width,
            orig_height,
            cfg.confidence_threshold,
        );
        let dets = yunet_decoder::nms(dets, cfg.iou_threshold);

        // Single-user behavior: pick the highest-scoring face.
        let best = dets
            .into_iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some(d) => {
                // Clamp the normalized bbox to the frame.
                let (mut x, mut y, w, h) = d.bbox;
                x = x.clamp(0.0, 1.0);
                y = y.clamp(0.0, 1.0);
                let w = w.clamp(0.0, 1.0 - x);
                let h = h.clamp(0.0, 1.0 - y);

                tracing::debug!(
                    "YuNet detection: score={:.3} bbox_norm=({:.3},{:.3},{:.3},{:.3}) frame={}x{}",
                    d.score, x, y, w, h, orig_width, orig_height
                );

                Ok(Some(Face {
                    bbox: (x, y, w, h),
                    confidence: d.score,
                    frame_dimensions: (orig_width, orig_height),
                    landmarks: Some(d.landmarks),
                }))
            }
            None => {
                tracing::debug!("YuNet: no face above threshold {:.2}", cfg.confidence_threshold);
                Ok(None)
            }
        }
    }

    /// MiniFASNetV2-SE anti-spoofing (facenox/face-antispoof-onnx, 128x128).
    ///
    /// Replicates the repo pipeline EXACTLY:
    /// 1. Square crop of side `max(bbox_w, bbox_h) * bbox_expansion_factor`
    ///    (default 1.5), centered on the bbox center; out-of-frame regions are
    ///    **reflect-101** padded (`cv2.BORDER_REFLECT_101`).
    /// 2. Resize to 128x128, color order **RGB**, normalize **`/255` -> [0,1]**
    ///    (NO mean/std), NCHW float32 `[1, 3, 128, 128]`.
    /// 3. Output `[1, 2]` raw logits (index 0 = real, index 1 = spoof). Decide
    ///    `is_real = (real_logit - spoof_logit) >= ln(p/(1-p))` for the configured
    ///    real-probability `p` (default 0.5 -> plain argmax).
    ///
    /// NON-FATAL: with no loaded model, returns `Ok(true)` (skip) with a warn so
    /// liveness can never block recognition.
    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        use super::model_config::LivenessConfig;

        if self.liveness_pool.is_empty() {
            warn!("Liveness skipped: MiniFASNetV2-SE model not loaded");
            return Ok(true);
        }

        let cfg = LivenessConfig::MINIFASNET;
        let size = cfg.input_size; // 128
        let session = match self.get_next_session(&self.liveness_pool) {
            Some(s) => s,
            None => {
                warn!("Liveness skipped: no MiniFASNetV2-SE session available");
                return Ok(true);
            }
        };

        // Build the reflect-101-padded square crop resized to `size`x`size`, RGB.
        let crop = Self::antispoof_square_crop(image, face, cfg.bbox_expansion_factor, size);

        // RGB, /255 -> [0,1], NCHW planar (matches the repo's `preprocess`).
        let n = (size * size) as usize;
        let mut input = vec![0.0f32; 3 * n];
        let (r_off, g_off, b_off) = (0, n, 2 * n);
        for (i, px) in crop.pixels().enumerate() {
            input[r_off + i] = px[0] as f32 / 255.0;
            input[g_off + i] = px[1] as f32 / 255.0;
            input[b_off + i] = px[2] as f32 / 255.0;
        }

        let input_tensor = ort_try!(Value::from_array((
            [1usize, 3, size as usize, size as usize],
            input
        )));
        let mut lock = session.lock().unwrap();
        let outputs = ort_try!(lock.run(ort::inputs![input_tensor]));
        let (_, logits) = ort_try!(outputs[0].try_extract_tensor::<f32>());

        if logits.len() <= cfg.real_class_index || logits.len() <= cfg.spoof_class_index {
            warn!("Liveness skipped: unexpected MiniFASNetV2-SE output len {}", logits.len());
            return Ok(true);
        }
        let real_logit = logits[cfg.real_class_index];
        let spoof_logit = logits[cfg.spoof_class_index];
        let diff = real_logit - spoof_logit;

        // logit threshold = ln(p/(1-p)); p clamped away from 0/1.
        let p = cfg.real_prob_threshold.clamp(1e-6, 1.0 - 1e-6);
        let logit_threshold = (p / (1.0 - p)).ln();
        let is_real = diff >= logit_threshold;

        // Softmax real-probability for logging only.
        let p_real = 1.0 / (1.0 + (-diff).exp());
        tracing::debug!(
            "MiniFASNetV2-SE liveness: real_logit={:.3} spoof_logit={:.3} diff={:.3} p_real={:.4} threshold_logit={:.3} -> {}",
            real_logit, spoof_logit, diff, p_real, logit_threshold,
            if is_real { "REAL" } else { "SPOOF" }
        );
        Ok(is_real)
    }

    /// Extract a 512-d face embedding using the EdgeFace-S recognizer.
    ///
    /// Preprocessing (must match `RecognizerConfig::EDGEFACE`):
    /// 1. **Align** the face to the canonical 112x112 5-point template via a
    ///    similarity transform from the detector's 5 landmarks. If the detector
    ///    provided no landmarks, fall back to a plain bbox crop+resize (degraded —
    ///    used only by warmup / landmark-less callers).
    /// 2. Color order **RGB**, normalization `(x - 127.5) / 127.5` -> [-1, 1]
    ///    (EdgeFace's `ToTensor()` + `Normalize(0.5, 0.5)`; identical to ArcFace).
    /// 3. NCHW float32 `[1, 3, 112, 112]`.
    /// 4. **L2-normalize** the 512-d output so cosine == dot product.
    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        use super::align;
        use super::model_config::RecognizerConfig;

        const SIZE: u32 = 112;
        let recognizer = self
            .get_next_session(&self.recognizer_pool)
            .ok_or_else(|| anyhow!("Recognizer not loaded"))?;

        let (img_w, img_h) = image.dimensions();

        // Build the aligned 112x112 RGB crop.
        let aligned: image::RgbImage = match face.landmarks {
            Some(landmarks_norm) => {
                // Landmarks are normalized [0,1]; convert to source pixels.
                let landmarks_px: [(f32, f32); 5] = std::array::from_fn(|i| {
                    (
                        landmarks_norm[i].0 * img_w as f32,
                        landmarks_norm[i].1 * img_h as f32,
                    )
                });
                match align::align_to_template(
                    image,
                    &landmarks_px,
                    &RecognizerConfig::RECOGNIZER_TEMPLATE_112,
                    SIZE,
                ) {
                    Some(a) => a,
                    None => Self::bbox_crop_resize(image, face, SIZE),
                }
            }
            None => Self::bbox_crop_resize(image, face, SIZE),
        };

        // Preprocess: RGB, (x - 127.5)/127.5, NCHW planar.
        let n = (SIZE * SIZE) as usize;
        let mut input = vec![0.0f32; 3 * n];
        let (r_off, g_off, b_off) = (0, n, 2 * n);
        for (i, px) in aligned.pixels().enumerate() {
            input[r_off + i] = (px[0] as f32 - 127.5) / 127.5;
            input[g_off + i] = (px[1] as f32 - 127.5) / 127.5;
            input[b_off + i] = (px[2] as f32 - 127.5) / 127.5;
        }

        let input_tensor = ort_try!(Value::from_array((
            [1usize, 3, SIZE as usize, SIZE as usize],
            input
        )));
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
        // Liveness is a non-fatal convenience check and does NOT gate readiness.
        !self.detector_pool.is_empty() && !self.recognizer_pool.is_empty()
    }

    fn name(&self) -> &'static str {
        "ONNX Runtime"
    }
}
