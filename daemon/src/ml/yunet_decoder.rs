//! YuNet (OpenCV Zoo `face_detection_yunet_2023mar.onnx`) output decoding.
//!
//! This module is intentionally free of any `ort`/ONNX types so the decode math
//! can be unit-tested with hand-built tensors. It mirrors the reference decode
//! in OpenCV's `FaceDetectorYN` / the opencv_zoo `yunet.py`:
//!
//! For every stride `s` in {8, 16, 32} the network emits, over a
//! `(IN/s) x (IN/s)` grid (row-major), four parallel tensors:
//!   * `cls`  — class score, already sigmoid-activated, shape `[N, 1]`
//!   * `obj`  — objectness, already sigmoid-activated, shape `[N, 1]`
//!   * `bbox` — `(dx, dy, dw, dh)` box deltas, shape `[N, 4]`
//!   * `kps`  — five `(dx, dy)` landmark deltas, shape `[N, 10]`
//!
//! Decode for anchor index `i` (with `row = i / cols`, `col = i % cols`):
//!   cx = (col + dx) * s        cy = (row + dy) * s
//!   w  = exp(dw) * s           h  = exp(dh) * s
//!   x  = cx - w/2              y  = cy - h/2
//!   landmark_k: lx = (col + kdx) * s, ly = (row + kdy) * s
//!   score = sqrt(cls * obj)
//!
//! All coordinates are produced in network-input (square `IN`) space, then
//! scaled back to the original frame by `(orig_w / IN, orig_h / IN)` and
//! finally normalized to `[0, 1]` so they match the rest of the pipeline
//! (the IPC layer multiplies normalized bbox by frame dimensions).

/// A single decoded detection in **normalized [0,1]** coordinates relative to
/// the original frame.
#[derive(Debug, Clone, Copy)]
pub struct YuNetDetection {
    /// (x, y, w, h) top-left + size, normalized to [0,1].
    pub bbox: (f32, f32, f32, f32),
    /// `sqrt(cls * obj)` face score.
    pub score: f32,
    /// 5 landmarks (right-eye, left-eye, nose, right-mouth, left-mouth),
    /// normalized to [0,1].
    pub landmarks: [(f32, f32); 5],
}

/// Raw per-stride output views borrowed from the ONNX session outputs.
///
/// Each slice is the flattened tensor for one stride (batch dim already
/// dropped). `cls`/`obj` have `N` elements, `bbox` has `4*N`, `kps` has `10*N`,
/// where `N = (input_size / stride)^2`.
pub struct StrideOutputs<'a> {
    pub stride: u32,
    pub cls: &'a [f32],
    pub obj: &'a [f32],
    pub bbox: &'a [f32],
    pub kps: &'a [f32],
}

/// Decode all strides into normalized detections passing the score threshold.
///
/// * `input_size` — the (square) network input edge, e.g. 640.
/// * `orig_w` / `orig_h` — original frame dimensions, used to map back from the
///   stretched square input and then normalize to [0,1].
pub fn decode(
    strides: &[StrideOutputs<'_>],
    input_size: u32,
    orig_w: u32,
    orig_h: u32,
    score_threshold: f32,
) -> Vec<YuNetDetection> {
    let mut out = Vec::new();
    let in_f = input_size as f32;
    // Map from square-input pixels -> original-frame pixels, then -> [0,1].
    // (orig_w / IN) * (1 / orig_w) == 1 / IN for x; likewise for y. So the net
    // normalization is simply division by `input_size` on each axis. We keep
    // orig dims in the signature for clarity / future letterboxing.
    let _ = (orig_w, orig_h);
    let inv_in = 1.0 / in_f;

    for s in strides {
        let stride = s.stride;
        let cols = (input_size / stride) as usize;
        let n = cols * cols;
        debug_assert_eq!(s.cls.len(), n, "cls len mismatch for stride {}", stride);
        debug_assert_eq!(s.obj.len(), n, "obj len mismatch for stride {}", stride);
        debug_assert_eq!(s.bbox.len(), 4 * n, "bbox len mismatch for stride {}", stride);
        debug_assert_eq!(s.kps.len(), 10 * n, "kps len mismatch for stride {}", stride);

        let sf = stride as f32;
        for i in 0..n {
            let cls = s.cls[i].max(0.0);
            let obj = s.obj[i].max(0.0);
            let score = (cls * obj).sqrt();
            if score < score_threshold {
                continue;
            }

            let row = (i / cols) as f32;
            let col = (i % cols) as f32;

            let b = &s.bbox[i * 4..i * 4 + 4];
            let cx = (col + b[0]) * sf;
            let cy = (row + b[1]) * sf;
            let w = b[2].exp() * sf;
            let h = b[3].exp() * sf;
            let x = cx - w / 2.0;
            let y = cy - h / 2.0;

            // Normalize to [0,1] (division by input edge maps square-input
            // pixels to fractions; identical fractions in the original frame).
            let bbox = (x * inv_in, y * inv_in, w * inv_in, h * inv_in);

            let k = &s.kps[i * 10..i * 10 + 10];
            let mut landmarks = [(0.0f32, 0.0f32); 5];
            for (j, lm) in landmarks.iter_mut().enumerate() {
                let lx = (col + k[2 * j]) * sf;
                let ly = (row + k[2 * j + 1]) * sf;
                *lm = (lx * inv_in, ly * inv_in);
            }

            out.push(YuNetDetection { bbox, score, landmarks });
        }
    }

    out
}

/// Intersection-over-Union of two normalized (x, y, w, h) boxes.
fn iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let (ax, ay, aw, ah) = a;
    let (bx, by, bw, bh) = b;
    let x1 = ax.max(bx);
    let y1 = ay.max(by);
    let x2 = (ax + aw).min(bx + bw);
    let y2 = (ay + ah).min(by + bh);
    let iw = (x2 - x1).max(0.0);
    let ih = (y2 - y1).max(0.0);
    let inter = iw * ih;
    let union = aw * ah + bw * bh - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Greedy Non-Maximum Suppression by descending score. Returns kept detections.
pub fn nms(mut dets: Vec<YuNetDetection>, iou_threshold: f32) -> Vec<YuNetDetection> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<YuNetDetection> = Vec::new();
    for d in dets {
        if keep.iter().all(|k| iou(d.bbox, k.bbox) < iou_threshold) {
            keep.push(d);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-anchor stride set that activates exactly one grid cell,
    /// so we can assert the decode math against hand-computed values.
    fn one_hot_stride(stride: u32, input_size: u32, active: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let cols = (input_size / stride) as usize;
        let n = cols * cols;
        let mut cls = vec![0.01f32; n];
        let mut obj = vec![0.0f32; n];
        let mut bbox = vec![0.0f32; 4 * n];
        let mut kps = vec![0.0f32; 10 * n];
        cls[active] = 0.81; // sqrt(0.81*1.0) = 0.9
        obj[active] = 1.0;
        // dx=dy=0.5 (center of cell), dw=dh=0 -> w=h=stride
        bbox[active * 4] = 0.5;
        bbox[active * 4 + 1] = 0.5;
        bbox[active * 4 + 2] = 0.0;
        bbox[active * 4 + 3] = 0.0;
        // landmarks at cell origin (dx=dy=0)
        (cls, obj, bbox, kps)
    }

    #[test]
    fn decode_single_anchor_math() {
        let input_size = 640u32;
        let stride = 16u32;
        let cols = (input_size / stride) as usize; // 40
        let active = 5 * cols + 7; // row 5, col 7
        let (cls, obj, bbox, kps) = one_hot_stride(stride, input_size, active);

        let strides = vec![StrideOutputs {
            stride,
            cls: &cls,
            obj: &obj,
            bbox: &bbox,
            kps: &kps,
        }];

        let dets = decode(&strides, input_size, input_size, input_size, 0.6);
        assert_eq!(dets.len(), 1, "exactly one anchor should pass threshold");
        let d = dets[0];

        // Expected in input-pixel space:
        //   cx = (7 + 0.5)*16 = 120, cy = (5 + 0.5)*16 = 88
        //   w  = exp(0)*16 = 16, h = 16
        //   x  = 120 - 8 = 112, y = 88 - 8 = 80
        // Normalized by /640:
        let inv = 1.0 / 640.0;
        let (x, y, w, h) = d.bbox;
        assert!((x - 112.0 * inv).abs() < 1e-5, "x={}", x);
        assert!((y - 80.0 * inv).abs() < 1e-5, "y={}", y);
        assert!((w - 16.0 * inv).abs() < 1e-5, "w={}", w);
        assert!((h - 16.0 * inv).abs() < 1e-5, "h={}", h);
        assert!((d.score - 0.9).abs() < 1e-4, "score={}", d.score);

        // landmark 0: (col+0)*16, (row+0)*16 = (112, 80) normalized
        assert!((d.landmarks[0].0 - 112.0 * inv).abs() < 1e-5);
        assert!((d.landmarks[0].1 - 80.0 * inv).abs() < 1e-5);
    }

    #[test]
    fn nms_suppresses_overlap() {
        let a = YuNetDetection {
            bbox: (0.1, 0.1, 0.2, 0.2),
            score: 0.9,
            landmarks: [(0.0, 0.0); 5],
        };
        // Heavily overlapping, lower score -> suppressed.
        let b = YuNetDetection {
            bbox: (0.11, 0.11, 0.2, 0.2),
            score: 0.8,
            landmarks: [(0.0, 0.0); 5],
        };
        // Far away -> kept.
        let c = YuNetDetection {
            bbox: (0.7, 0.7, 0.2, 0.2),
            score: 0.7,
            landmarks: [(0.0, 0.0); 5],
        };
        let kept = nms(vec![a, b, c], 0.3);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
        assert!((kept[1].score - 0.7).abs() < 1e-6);
    }

    #[test]
    fn below_threshold_dropped() {
        let input_size = 640u32;
        let stride = 32u32;
        let cols = (input_size / stride) as usize;
        let n = cols * cols;
        let cls = vec![0.1f32; n];
        let obj = vec![0.1f32; n]; // score = sqrt(0.01) = 0.1 < 0.6
        let bbox = vec![0.0f32; 4 * n];
        let kps = vec![0.0f32; 10 * n];
        let strides = vec![StrideOutputs { stride, cls: &cls, obj: &obj, bbox: &bbox, kps: &kps }];
        let dets = decode(&strides, input_size, input_size, input_size, 0.6);
        assert!(dets.is_empty());
    }
}
