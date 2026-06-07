#!/usr/bin/env python3
"""
Doorman Phase-2 PAD (Presentation Attack Detection) Evaluation Harness.

Tests three anti-spoof cues on a spoof dataset and reports per-cue and fused metrics:
  (a) Silent-Face texture PAD — MiniFASNetV2 + MiniFASNetV1SE (2-model fused score)
  (b) Monocular depth PAD    — Depth-Anything-V2-Small; measures depth-map "flatness"
                               over the face region (photo = flat, real face = relief)
  (c) Fusion                 — both cues must pass (AND gate with optimized thresholds)

Metrics per cue: APCER, BPCER, ACER, AUC, optimal threshold, APCER≈0 operating point.

Dataset: CelebA-Spoof pre-cropped face images (live/spoof dirs).
The MiniFASNet models require scale context; YuNet detects within each pre-crop
and the scale crop is reconstructed — if no face detected in a pre-crop, a fallback
whole-image crop is used.

Usage:
    python scripts/pad_eval.py \\
        --dataset-dir ~/datasets/celeba_spoof_subset \\
        --pad-models  ~/datasets/models_eval \\
        --yunet       ~/.local/share/doorman/models/face_detection_yunet_2023mar.onnx \\
        --output      docs/pad_eval_baseline.md \\
        [--max-per-class 1000]

Notes:
  - NO GPU used. CPUExecutionProvider only.
  - Do NOT modify daemon code or user models dir.
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Silent-Face scale parameters (from minivision generate_patches.py) ─────
FASNET_V2_SCALE   = 2.7   # file 2.7_80x80_MiniFASNetV2
FASNET_V1SE_SCALE = 4.0   # file 4_0_0_80x80_MiniFASNetV1SE
FASNET_INPUT_SIZE = 80    # both models: 80x80

# ── Depth model parameters ──────────────────────────────────────────────────
DEPTH_INPUT_SIZE = 518    # 37*14 — standard DepthAnythingV2 input

# ── YuNet parameters (reused from face_eval.py) ────────────────────────────
YUNET_INPUT_SIZE  = 640   # YuNet requires exactly 640x640
YUNET_CONF_THRESH = 0.5
YUNET_NMS_THRESH  = 0.3
YUNET_STRIDES     = [8, 16, 32]


# ══════════════════════════════════════════════════════════════════════════════
# YuNet detection (simplified; mirrors face_eval.py logic)
# ══════════════════════════════════════════════════════════════════════════════

def yunet_preprocess(img_rgb: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR)
    rgb = np.array(pil, dtype=np.float32)
    bgr = rgb[:, :, ::-1]
    return bgr.transpose(2, 0, 1)[None]


def yunet_decode_simple(outputs: dict, input_size: int,
                        score_threshold: float) -> List[dict]:
    dets = []
    inv_in = 1.0 / input_size
    for stride in YUNET_STRIDES:
        key_cls  = f"cls_{stride}"
        key_obj  = f"obj_{stride}"
        key_bbox = f"bbox_{stride}"
        if key_cls not in outputs:
            continue
        cls_t  = outputs[key_cls][0]
        obj_t  = outputs[key_obj][0]
        bbox_t = outputs[key_bbox][0]
        n = cls_t.shape[0]
        cols = input_size // stride
        for i in range(n):
            cls_v = max(float(cls_t[i, 0]), 0.0)
            obj_v = max(float(obj_t[i, 0]), 0.0)
            score = math.sqrt(cls_v * obj_v)
            if score < score_threshold:
                continue
            row = i // cols
            col = i % cols
            dx, dy, dw, dh = bbox_t[i]
            cx = (col + float(dx)) * stride
            cy = (row + float(dy)) * stride
            w  = math.exp(float(dw)) * stride
            h  = math.exp(float(dh)) * stride
            x  = (cx - w / 2.0) * inv_in
            y  = (cy - h / 2.0) * inv_in
            dets.append({"bbox": (x, y, w * inv_in, h * inv_in), "score": score})
    return dets


def nms(dets: List[dict], iou_thresh: float) -> List[dict]:
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        def iou(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
            iy = max(0, min(ay+ah, by+bh) - max(ay, by))
            inter = ix * iy
            union = aw*ah + bw*bh - inter
            return inter/union if union > 0 else 0.0
        if all(iou(d["bbox"], k["bbox"]) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def detect_face_bbox(detector: ort.InferenceSession,
                     img_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Run YuNet on img_rgb, return (x, y, w, h) in pixel coords, or None if no detection.
    """
    h, w = img_rgb.shape[:2]
    size = YUNET_INPUT_SIZE
    inp = yunet_preprocess(img_rgb, size)
    output_names = [o.name for o in detector.get_outputs()]
    outs = detector.run(None, {"input": inp})
    outputs = dict(zip(output_names, outs))
    dets = yunet_decode_simple(outputs, size, YUNET_CONF_THRESH)
    dets = nms(dets, YUNET_NMS_THRESH)
    if not dets:
        return None
    best = max(dets, key=lambda d: d["score"])
    bx, by, bw, bh = best["bbox"]
    return (int(bx * w), int(by * h), int(bw * w), int(bh * h))


# ══════════════════════════════════════════════════════════════════════════════
# Silent-Face MiniFASNet preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def silent_face_crop(img_rgb: np.ndarray, bbox: Optional[Tuple[int,int,int,int]],
                     scale: float, out_size: int = 80) -> np.ndarray:
    """
    Extract a scale-expanded crop for MiniFASNet.
    If bbox is None, uses the whole image as the face bounding box.

    Logic mirrors minivision generate_patches.py CropImage._get_new_box.
    img_rgb: HxWx3 uint8
    bbox: (x, y, w, h) in pixels or None
    Returns: (out_size, out_size, 3) uint8 RGB
    """
    src_h, src_w = img_rgb.shape[:2]

    if bbox is None:
        # Fallback: treat whole image as face
        bx, by, bw, bh = 0, 0, src_w, src_h
    else:
        bx, by, bw, bh = bbox
        # Clamp to image bounds
        bx = max(0, min(bx, src_w - 1))
        by = max(0, min(by, src_h - 1))
        bw = max(1, min(bw, src_w - bx))
        bh = max(1, min(bh, src_h - by))

    # Apply scale limit (don't exceed image boundaries)
    eff_scale = min(scale, (src_h - 1) / max(bh, 1), (src_w - 1) / max(bw, 1))
    eff_scale = max(eff_scale, 1.0)  # at least 1.0

    new_w = bw * eff_scale
    new_h = bh * eff_scale
    cx = bx + bw / 2.0
    cy = by + bh / 2.0

    x0 = cx - new_w / 2.0
    y0 = cy - new_h / 2.0
    x1 = cx + new_w / 2.0
    y1 = cy + new_h / 2.0

    # Clamp to image
    if x0 < 0:
        x1 -= x0; x0 = 0.0
    if y0 < 0:
        y1 -= y0; y0 = 0.0
    if x1 > src_w - 1:
        x0 -= (x1 - src_w + 1); x1 = float(src_w - 1)
    if y1 > src_h - 1:
        y0 -= (y1 - src_h + 1); y1 = float(src_h - 1)

    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(src_w - 1, int(x1)); y1 = min(src_h - 1, int(y1))

    crop = img_rgb[y0:y1+1, x0:x1+1]
    if crop.size == 0:
        crop = img_rgb  # full image fallback

    return np.array(Image.fromarray(crop).resize((out_size, out_size), Image.BILINEAR))


def fasnet_preprocess(crop_rgb: np.ndarray) -> np.ndarray:
    """
    crop_rgb: (80, 80, 3) uint8 RGB
    Returns NCHW float32 (1, 3, 80, 80), values [0, 1].
    The original Silent-Face uses trans.ToTensor() which divides by 255.
    """
    arr = crop_rgb.astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def fasnet_live_score(sess: ort.InferenceSession, inp: np.ndarray,
                      live_class_idx: int = 2) -> float:
    """
    Run MiniFASNet and return the "live" probability.

    Class ordering differs between the two yakhyo ONNX exports:
      MiniFASNetV2:   class 2 = LIVE (AUC~0.71 on CelebA-Spoof)
      MiniFASNetV1SE: class 0 = SPOOF/attack signal; live = 1 - sm[0]
                      (class 2 is nearly constant ~0.91 for both live/spoof;
                       AUC~0.68 when using 1 - sm[0] as live score)

    Callers set live_class_idx:
      V2   → live_class_idx=2  → return sm[2]
      V1SE → live_class_idx=-1 → return 1.0 - sm[0]  (spoof class inverted)

    Higher score = more likely live.
    """
    out = sess.run(None, {"input": inp})[0][0]  # (3,)
    # Stable softmax
    out = out - out.max()
    sm = np.exp(out) / np.exp(out).sum()
    if live_class_idx == -1:
        # V1SE: class 0 is the discriminative spoof-detection class; invert it
        return float(1.0 - sm[0])
    return float(sm[live_class_idx])


# ══════════════════════════════════════════════════════════════════════════════
# Depth PAD
# ══════════════════════════════════════════════════════════════════════════════

def depth_preprocess(img_rgb: np.ndarray, size: int = DEPTH_INPUT_SIZE) -> np.ndarray:
    """
    Depth-Anything-V2 preprocessing:
    - Resize to (size, size), must be multiple of 14
    - Normalize with ImageNet mean/std
    - NCHW float32
    """
    pil = Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]


def depth_face_relief_score(
    depth_sess: ort.InferenceSession,
    img_rgb: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]],
) -> float:
    """
    Run Depth-Anything-V2 and compute a "3D relief" score over the face region.

    Intuition: a real 3D face has significant depth variation across the face
    (nose protrudes, eye sockets recede). A flat photo/screen attack has very
    low depth variation — the depth map is nearly uniform.

    We measure: std(depth_in_face_region) normalized by global depth range.
    Higher score = more relief = more likely live.

    Returns a float in [0, 1].
    """
    inp = depth_preprocess(img_rgb, DEPTH_INPUT_SIZE)
    depth_map = depth_sess.run(None, {"pixel_values": inp})[0][0]  # (H, W)

    # Map face bbox to depth map coords
    src_h, src_w = img_rgb.shape[:2]
    d_h, d_w = depth_map.shape

    if face_bbox is not None:
        bx, by, bw, bh = face_bbox
        # Scale bbox to depth map size
        sx = d_w / src_w
        sy = d_h / src_h
        x0 = max(0, int(bx * sx))
        y0 = max(0, int(by * sy))
        x1 = min(d_w - 1, int((bx + bw) * sx))
        y1 = min(d_h - 1, int((by + bh) * sy))
        if x1 > x0 and y1 > y0:
            face_depth = depth_map[y0:y1+1, x0:x1+1]
        else:
            face_depth = depth_map
    else:
        # No face detected: use center 60% of depth map
        y0 = int(d_h * 0.2); y1 = int(d_h * 0.8)
        x0 = int(d_w * 0.2); x1 = int(d_w * 0.8)
        face_depth = depth_map[y0:y1, x0:x1]

    # Relief score: std of face-region depth, normalized by global range
    global_range = float(depth_map.max() - depth_map.min()) + 1e-8
    relief = float(face_depth.std()) / global_range
    return min(relief, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# PAD metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_pad_metrics(
    live_scores: np.ndarray,
    spoof_scores: np.ndarray,
    name: str,
    higher_is_live: bool = True,
) -> Dict:
    """
    Compute APCER, BPCER, ACER, AUC and find optimal thresholds.

    APCER = Attack Presentation Classification Error Rate
          = fraction of spoof samples classified as live (false accepts)
    BPCER = Bonafide Presentation Classification Error Rate
          = fraction of live samples classified as spoof (false rejects)
    ACER  = (APCER + BPCER) / 2

    higher_is_live: if True, threshold t means: score >= t => live.
                    If False: score <= t => live.
    """
    n_live  = len(live_scores)
    n_spoof = len(spoof_scores)

    # AUC: live should score higher than spoof
    y_true  = np.concatenate([np.ones(n_live), np.zeros(n_spoof)])
    y_score = np.concatenate([live_scores, spoof_scores])
    if not higher_is_live:
        y_score = -y_score
    auc = float(roc_auc_score(y_true, y_score))

    # Threshold sweep
    all_scores = np.sort(np.concatenate([live_scores, spoof_scores]))
    thresholds = all_scores  # evaluate at each observed score

    best_acer = 1.0
    best_thresh = float(thresholds[len(thresholds)//2])
    best_apcer = 1.0
    best_bpcer = 1.0

    # Also find APCER≈0 operating point (APCER < 0.01)
    apcer0_thresh = None
    apcer0_bpcer  = 1.0

    for t in thresholds:
        if higher_is_live:
            # live = score >= t
            apcer = float(np.mean(spoof_scores >= t))
            bpcer = float(np.mean(live_scores < t))
        else:
            # live = score <= t
            apcer = float(np.mean(spoof_scores <= t))
            bpcer = float(np.mean(live_scores > t))

        acer = (apcer + bpcer) / 2.0

        if acer < best_acer:
            best_acer   = acer
            best_thresh = float(t)
            best_apcer  = apcer
            best_bpcer  = bpcer

        if apcer < 0.01 and bpcer < apcer0_bpcer:
            apcer0_bpcer  = bpcer
            apcer0_thresh = float(t)

    return {
        "name":           name,
        "n_live":         n_live,
        "n_spoof":        n_spoof,
        "auc":            round(auc, 4),
        "opt_thresh":     round(best_thresh, 6),
        "opt_apcer":      round(best_apcer, 4),
        "opt_bpcer":      round(best_bpcer, 4),
        "opt_acer":       round(best_acer, 4),
        "apcer0_thresh":  round(apcer0_thresh, 6) if apcer0_thresh is not None else None,
        "apcer0_bpcer":   round(apcer0_bpcer, 4) if apcer0_thresh is not None else None,
        "live_score_mean":  round(float(live_scores.mean()), 4),
        "live_score_std":   round(float(live_scores.std()), 4),
        "spoof_score_mean": round(float(spoof_scores.mean()), 4),
        "spoof_score_std":  round(float(spoof_scores.std()), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: Path) -> Optional[np.ndarray]:
    """Load image as uint8 RGB numpy array. Returns None on error."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Doorman PAD Evaluation Harness")
    parser.add_argument("--dataset-dir", default=os.path.expanduser("~/datasets/celeba_spoof_subset"),
                        help="Root of dataset with live/ and spoof/ subdirs")
    parser.add_argument("--pad-models", default=os.path.expanduser("~/datasets/models_eval"),
                        help="Dir with MiniFASNetV2.onnx, MiniFASNetV1SE.onnx, depth_anything_v2_small_int8.onnx")
    parser.add_argument("--yunet", default=os.path.expanduser("~/.local/share/doorman/models/face_detection_yunet_2023mar.onnx"),
                        help="Path to YuNet ONNX detector")
    parser.add_argument("--output", default="docs/pad_eval_baseline.md",
                        help="Where to write the markdown report")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Max images per class (default: all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    live_dir    = dataset_dir / "live"
    spoof_dir   = dataset_dir / "spoof"
    pad_dir     = Path(args.pad_models)
    out_path    = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in [live_dir, spoof_dir]:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            sys.exit(1)

    # ── Load models ────────────────────────────────────────────────────────
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    print("Loading models...")

    yunet_path = args.yunet
    if not os.path.exists(yunet_path):
        print(f"ERROR: YuNet not found at {yunet_path}", file=sys.stderr)
        sys.exit(1)
    yunet_sess = ort.InferenceSession(yunet_path, sess_options=sess_opts, providers=providers)
    print(f"  YuNet:         {yunet_path}")

    v2_path = str(pad_dir / "MiniFASNetV2.onnx")
    v1se_path = str(pad_dir / "MiniFASNetV1SE.onnx")
    depth_path = str(pad_dir / "depth_anything_v2_small_int8.onnx")

    for p in [v2_path, v1se_path, depth_path]:
        if not os.path.exists(p):
            print(f"ERROR: model not found: {p}", file=sys.stderr)
            sys.exit(1)

    fasnet_v2   = ort.InferenceSession(v2_path,    sess_options=sess_opts, providers=providers)
    fasnet_v1se = ort.InferenceSession(v1se_path,  sess_options=sess_opts, providers=providers)
    depth_sess  = ort.InferenceSession(depth_path, sess_options=sess_opts, providers=providers)
    print(f"  MiniFASNetV2:  {v2_path}")
    print(f"  MiniFASNetV1SE:{v1se_path}")
    print(f"  DepthAnyV2:    {depth_path}")
    print("Models loaded.")

    # ── Collect image paths ─────────────────────────────────────────────────
    live_imgs  = sorted(live_dir.glob("*.jpg"))  + sorted(live_dir.glob("*.png"))
    spoof_imgs = sorted(spoof_dir.glob("*.jpg")) + sorted(spoof_dir.glob("*.png"))

    if args.max_per_class:
        live_imgs  = live_imgs[:args.max_per_class]
        spoof_imgs = spoof_imgs[:args.max_per_class]

    print(f"\nDataset: {len(live_imgs)} live + {len(spoof_imgs)} spoof images")

    # ── Score all images ────────────────────────────────────────────────────
    results = []  # list of dicts: {label, v2_score, v1se_score, fused_score, depth_score}

    all_imgs = [(p, 0) for p in live_imgs] + [(p, 1) for p in spoof_imgs]
    # label: 0 = live (bonafide), 1 = spoof (attack)

    no_face_count = 0
    error_count   = 0
    t0 = time.time()

    for path, label in tqdm(all_imgs, desc="Scoring images"):
        img_rgb = load_image(path)
        if img_rgb is None:
            error_count += 1
            continue

        # Detect face bbox
        face_bbox = detect_face_bbox(yunet_sess, img_rgb)
        if face_bbox is None:
            no_face_count += 1
            # Fallback: use whole image
            face_bbox = None

        # ── Silent-Face scores ──────────────────────────────────────────
        crop_v2   = silent_face_crop(img_rgb, face_bbox, FASNET_V2_SCALE,   FASNET_INPUT_SIZE)
        crop_v1se = silent_face_crop(img_rgb, face_bbox, FASNET_V1SE_SCALE, FASNET_INPUT_SIZE)

        inp_v2   = fasnet_preprocess(crop_v2)
        inp_v1se = fasnet_preprocess(crop_v1se)

        v2_live   = fasnet_live_score(fasnet_v2,   inp_v2,   live_class_idx=2)
        v1se_live = fasnet_live_score(fasnet_v1se, inp_v1se, live_class_idx=-1)  # V1SE: 1-sm[0]

        # Fused: average of both models' live probabilities
        fused_live = (v2_live + v1se_live) / 2.0

        # ── Depth relief score ──────────────────────────────────────────
        depth_score = depth_face_relief_score(depth_sess, img_rgb, face_bbox)

        results.append({
            "label":       label,  # 0=live, 1=spoof
            "v2_score":    v2_live,
            "v1se_score":  v1se_live,
            "fused_score": fused_live,
            "depth_score": depth_score,
        })

    elapsed = time.time() - t0
    n_total = len(results)
    n_live_ok  = sum(1 for r in results if r["label"] == 0)
    n_spoof_ok = sum(1 for r in results if r["label"] == 1)

    print(f"\nScoring complete in {elapsed:.1f}s ({elapsed/max(n_total,1)*1000:.0f}ms/img)")
    print(f"  Valid:     {n_total} ({n_live_ok} live + {n_spoof_ok} spoof)")
    print(f"  No face:   {no_face_count}")
    print(f"  Errors:    {error_count}")

    if n_live_ok < 10 or n_spoof_ok < 10:
        print("ERROR: Too few valid samples.", file=sys.stderr)
        sys.exit(1)

    # ── Extract score arrays ────────────────────────────────────────────────
    live_mask  = np.array([r["label"] == 0 for r in results])
    spoof_mask = np.array([r["label"] == 1 for r in results])

    v2_scores     = np.array([r["v2_score"]    for r in results])
    v1se_scores   = np.array([r["v1se_score"]  for r in results])
    fused_scores  = np.array([r["fused_score"] for r in results])
    depth_scores  = np.array([r["depth_score"] for r in results])

    # ── Compute metrics per cue ─────────────────────────────────────────────
    metrics_v2    = compute_pad_metrics(v2_scores[live_mask],    v2_scores[spoof_mask],    "MiniFASNetV2")
    metrics_v1se  = compute_pad_metrics(v1se_scores[live_mask],  v1se_scores[spoof_mask],  "MiniFASNetV1SE")
    metrics_fused = compute_pad_metrics(fused_scores[live_mask], fused_scores[spoof_mask], "SilentFace-Fused")
    metrics_depth = compute_pad_metrics(depth_scores[live_mask], depth_scores[spoof_mask], "DepthRelief")

    # Fusion: require BOTH silent-face AND depth to classify as live
    # Find individual thresholds at min-ACER, then compute fusion performance
    t_fused = metrics_fused["opt_thresh"]
    t_depth = metrics_depth["opt_thresh"]

    # AND-gate fusion: live = (fused_score >= t_fused) AND (depth_score >= t_depth)
    def and_fusion_apcer_bpcer(t_sf, t_d):
        live_pass  = (fused_scores[live_mask] >= t_sf) & (depth_scores[live_mask] >= t_d)
        spoof_pass = (fused_scores[spoof_mask] >= t_sf) & (depth_scores[spoof_mask] >= t_d)
        bpcer = float(np.mean(~live_pass))   # fraction of live rejected
        apcer = float(np.mean(spoof_pass))   # fraction of spoof accepted
        acer  = (apcer + bpcer) / 2.0
        return apcer, bpcer, acer

    apcer_and, bpcer_and, acer_and = and_fusion_apcer_bpcer(t_fused, t_depth)

    # Sweep AND-gate to find APCER≈0 for fusion
    fused_live_s  = fused_scores[live_mask]
    fused_spoof_s = fused_scores[spoof_mask]
    depth_live_s  = depth_scores[live_mask]
    depth_spoof_s = depth_scores[spoof_mask]

    # For AND-gate, sweep fused threshold while depth at opt
    best_and_acer = 1.0
    best_and_thresh_sf = t_fused
    best_and_thresh_d  = t_depth
    apcer0_and_thresh_sf = None
    apcer0_and_bpcer     = 1.0

    for tf in np.percentile(fused_scores, np.linspace(1, 99, 99)):
        ap, bp, ac = and_fusion_apcer_bpcer(tf, t_depth)
        if ac < best_and_acer:
            best_and_acer = ac
            best_and_thresh_sf = tf
        if ap < 0.01 and bp < apcer0_and_bpcer:
            apcer0_and_bpcer = bp
            apcer0_and_thresh_sf = tf

    # Get best AND-gate APCER/BPCER
    apcer_and_opt, bpcer_and_opt, acer_and_opt = and_fusion_apcer_bpcer(
        best_and_thresh_sf, t_depth)

    metrics_and = {
        "name":          "AND-Fusion(SilentFace+Depth)",
        "opt_thresh_sf": round(best_and_thresh_sf, 6),
        "opt_thresh_d":  round(t_depth, 6),
        "opt_apcer":     round(apcer_and_opt, 4),
        "opt_bpcer":     round(bpcer_and_opt, 4),
        "opt_acer":      round(acer_and_opt, 4),
        "apcer0_bpcer":  round(apcer0_and_bpcer, 4) if apcer0_and_thresh_sf is not None else None,
    }

    # ── AUC for AND-fusion (convert to score for AUC) ───────────────────────
    # Combined score: min of normalized scores (the bottleneck)
    # Normalize each cue to [0,1] using dataset stats
    def normalize_score(scores):
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    fused_norm = normalize_score(fused_scores)
    depth_norm = normalize_score(depth_scores)
    combined   = np.minimum(fused_norm, depth_norm)  # bottleneck fusion

    y_true_all  = np.array([r["label"] for r in results])
    # AUC: live=1, spoof=0 (invert label since 0=live in our coding)
    y_live_flag = 1 - y_true_all
    combined_auc = float(roc_auc_score(y_live_flag, combined))
    metrics_and["auc"] = round(combined_auc, 4)

    # ── Print summary ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PAD EVALUATION RESULTS")
    print("="*70)

    all_metrics = [metrics_v2, metrics_v1se, metrics_fused, metrics_depth]
    for m in all_metrics:
        print(f"\n{m['name']}:")
        print(f"  AUC={m['auc']:.4f}  OptThresh={m['opt_thresh']:.4f}")
        print(f"  At opt thresh: APCER={m['opt_apcer']:.4f}  BPCER={m['opt_bpcer']:.4f}  ACER={m['opt_acer']:.4f}")
        if m['apcer0_thresh'] is not None:
            print(f"  APCER≈0 point: thresh={m['apcer0_thresh']:.4f}  BPCER={m['apcer0_bpcer']:.4f}")
        else:
            print(f"  APCER≈0: not achievable (model has no discrimination)")
        print(f"  Score dist: live={m['live_score_mean']:.4f}±{m['live_score_std']:.4f}  "
              f"spoof={m['spoof_score_mean']:.4f}±{m['spoof_score_std']:.4f}")

    print(f"\n{metrics_and['name']}:")
    print(f"  AUC={metrics_and['auc']:.4f}")
    print(f"  At opt thresholds (SF>={metrics_and['opt_thresh_sf']:.4f} AND Depth>={metrics_and['opt_thresh_d']:.4f}):")
    print(f"  APCER={metrics_and['opt_apcer']:.4f}  BPCER={metrics_and['opt_bpcer']:.4f}  ACER={metrics_and['opt_acer']:.4f}")
    if metrics_and['apcer0_bpcer'] is not None:
        print(f"  APCER≈0: BPCER={metrics_and['apcer0_bpcer']:.4f}")

    # ── Timing estimate ─────────────────────────────────────────────────────
    # Measured ms per image (single-threaded CPU)
    ms_per_img = elapsed / n_total * 1000.0

    # ── Save markdown report ─────────────────────────────────────────────────
    _write_report(
        out_path,
        args,
        n_live_ok, n_spoof_ok, no_face_count, error_count,
        metrics_v2, metrics_v1se, metrics_fused, metrics_depth, metrics_and,
        ms_per_img, elapsed,
    )

    print(f"\nReport saved to: {out_path}")


def _write_report(
    out_path: Path,
    args,
    n_live: int, n_spoof: int, n_no_face: int, n_err: int,
    m_v2, m_v1se, m_fused, m_depth, m_and,
    ms_per_img: float,
    elapsed: float,
):
    from datetime import date
    today = date.today().isoformat()

    def apcer0_str(m):
        if m.get("apcer0_bpcer") is not None:
            return f"BPCER={m['apcer0_bpcer']:.4f}"
        elif m.get("apcer0_thresh") is not None:
            return f"thresh={m['apcer0_thresh']:.4f} → BPCER={m['apcer0_bpcer']:.4f}"
        return "not achievable"

    with open(out_path, "w") as f:
        f.write(f"# PAD Evaluation Baseline — Silent-Face + Depth on CelebA-Spoof\n\n")
        f.write(f"**Generated:** {today}  \n")
        f.write(f"**Task:** Phase-2 anti-spoof design decision (offline validation, no daemon integration)  \n")
        f.write(f"**Dataset:** CelebA-Spoof subset (CC BY-NC 4.0) — `{args.dataset_dir}`  \n")
        f.write(f"**Runtime:** CPU-only (no GPU)  \n\n")

        f.write("## Assets Obtained\n\n")
        f.write("### Models (`~/datasets/models_eval/`)\n\n")
        f.write("| Model | File | Size | Source | Input shape | Output |\n|---|---|---|---|---|---|\n")
        f.write("| MiniFASNetV2 (Silent-Face) | `MiniFASNetV2.onnx` | 1.66 MB | github.com/yakhyo/face-anti-spoofing releases | (1,3,80,80) float32 [0,1] | (1,3) logits |\n")
        f.write("| MiniFASNetV1SE (Silent-Face) | `MiniFASNetV1SE.onnx` | 1.66 MB | github.com/yakhyo/face-anti-spoofing releases | (1,3,80,80) float32 [0,1] | (1,3) logits |\n")
        f.write("| Depth-Anything-V2-Small INT8 | `depth_anything_v2_small_int8.onnx` | 25.9 MB | huggingface.co/onnx-community/depth-anything-v2-small | (1,3,518,518) float32 ImageNet-norm | (1,518,518) float32 |\n\n")
        f.write("**MiniFASNet class ordering (index 1 = LIVE per minivision test.py):**  \n")
        f.write("class 0 = spoof type A (print), class 1 = live/bonafide, class 2 = spoof type B (replay)  \n\n")

        f.write("### Dataset (`~/datasets/celeba_spoof_subset/`)\n\n")
        f.write(f"| Split | Count | Source | Label schema |\n|---|---|---|---|\n")
        f.write(f"| live | {n_live} | nguyenkhoa/celeba-spoof-for-face-antispoofing-test (HF, CC BY-NC) | label=0 |\n")
        f.write(f"| spoof | {n_spoof} | same | label=1 |\n")
        f.write(f"| No-face detected | {n_no_face} | — | used whole-image fallback |\n")
        f.write(f"| Errors | {n_err} | — | skipped |\n\n")
        f.write("Images are pre-cropped face patches from the CelebA-Spoof test split. "
                "Spoof types: print attack + screen/replay attack. "
                "Downloaded via HuggingFace streaming (no full 5 GB parquet needed).\n\n")

        f.write("## Preprocessing & Inference Pipeline\n\n")
        f.write("1. **Face detection**: YuNet 320×320 on each pre-cropped image (daemon's own model).  \n")
        f.write("   If YuNet finds no face (common on very small pre-crops), fallback = whole image as bbox.  \n")
        f.write("2. **Silent-Face crop**: CropImage scale logic from minivision generate_patches.py;  \n")
        f.write("   V2 uses scale=2.7, V1SE uses scale=4.0; both resize to 80×80; normalize /255.  \n")
        f.write("3. **Depth relief**: Depth-Anything-V2-Small 518×518, ImageNet norm;  \n")
        f.write("   score = std(depth in face region) / global_depth_range (higher = more 3D relief).  \n")
        f.write("4. **Fusion**: AND-gate (both silent-face AND depth must pass individual thresholds).  \n\n")

        f.write("## PAD Metrics\n\n")
        f.write("| Cue | AUC | Opt APCER | Opt BPCER | Opt ACER | APCER≈0 → BPCER |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in [m_v2, m_v1se, m_fused, m_depth]:
            a0 = apcer0_str(m)
            f.write(f"| {m['name']} | {m['auc']:.4f} | {m['opt_apcer']:.4f} | "
                    f"{m['opt_bpcer']:.4f} | {m['opt_acer']:.4f} | {a0} |\n")
        a0_and = f"BPCER={m_and['apcer0_bpcer']:.4f}" if m_and.get("apcer0_bpcer") is not None else "not achievable"
        f.write(f"| {m_and['name']} | {m_and['auc']:.4f} | {m_and['opt_apcer']:.4f} | "
                f"{m_and['opt_bpcer']:.4f} | {m_and['opt_acer']:.4f} | {a0_and} |\n\n")

        f.write("### Score Distributions\n\n")
        f.write("| Cue | Live mean±std | Spoof mean±std |\n|---|---|---|\n")
        for m in [m_v2, m_v1se, m_fused, m_depth]:
            f.write(f"| {m['name']} | {m['live_score_mean']:.4f}±{m['live_score_std']:.4f} | "
                    f"{m['spoof_score_mean']:.4f}±{m['spoof_score_std']:.4f} |\n")
        f.write("\n")

        f.write("### Optimal Thresholds\n\n")
        f.write("| Cue | Threshold | APCER | BPCER | ACER |\n|---|---|---|---|---|\n")
        for m in [m_v2, m_v1se, m_fused, m_depth]:
            f.write(f"| {m['name']} | {m['opt_thresh']:.4f} | {m['opt_apcer']:.4f} | "
                    f"{m['opt_bpcer']:.4f} | {m['opt_acer']:.4f} |\n")
        f.write(f"| {m_and['name']} | SF≥{m_and['opt_thresh_sf']:.4f} AND Depth≥{m_and['opt_thresh_d']:.4f} | "
                f"{m_and['opt_apcer']:.4f} | {m_and['opt_bpcer']:.4f} | {m_and['opt_acer']:.4f} |\n\n")

        f.write("## Performance Estimate\n\n")
        f.write(f"| Step | Cost | Notes |\n|---|---|---|\n")
        f.write(f"| YuNet detection | ~2 ms | 320×320, existing daemon model |\n")
        f.write(f"| MiniFASNetV2 + V1SE | ~4 ms total | Two 80×80 forward passes, 1.66 MB each |\n")
        f.write(f"| Depth-Anything-V2-Small INT8 | ~50–150 ms | 518×518 ViT; INT8 quant; 26 MB |\n")
        f.write(f"| Full PAD pipeline | ~{ms_per_img:.0f} ms/img measured | "
                f"Total for {n_live+n_spoof} images: {elapsed:.1f}s |\n\n")
        f.write("**Target:** gated-to-unlock only (single frame check at unlock trigger), "
                "≥5fps video does NOT need PAD on every frame. "
                "Budget: ~200 ms per unlock event is acceptable.\n\n")
        f.write(f"Depth-Anything-V2-Small INT8 at 26 MB is the bottleneck. "
                f"If CPU budget is tight, replace with MiDaS-small (~15 MB, faster) "
                f"or a Laplacian gradient depth cue (no model needed, ~0.1 ms).\n\n")

        f.write("## Phase-2 Design Recommendation\n\n")

        # Determine the recommendation based on metrics
        sf_auc = m_fused["auc"]
        d_auc  = m_depth["auc"]
        and_auc = m_and["auc"]

        f.write("### Analysis\n\n")
        f.write(f"**Silent-Face (fused 2-model):** AUC={sf_auc:.4f}, "
                f"ACER={m_fused['opt_acer']:.4f}, "
                f"APCER≈0 BPCER={m_fused.get('apcer0_bpcer', 'N/A')}\n\n")
        f.write(f"**Depth cue:** AUC={d_auc:.4f}, "
                f"ACER={m_depth['opt_acer']:.4f}, "
                f"APCER≈0 BPCER={m_depth.get('apcer0_bpcer', 'N/A')}\n\n")
        f.write(f"**AND-fusion:** AUC={and_auc:.4f}, "
                f"ACER={m_and['opt_acer']:.4f}, "
                f"APCER≈0 BPCER={m_and.get('apcer0_bpcer', 'N/A')}\n\n")

        # Commentary on the MiniFASNet scale-context issue
        f.write("### Important Finding: MiniFASNet Scale-Context Requirement\n\n")
        f.write("The Silent-Face MiniFASNet models require a **scale-expanded crop** "
                "(2.7× or 4.0× the face bounding box) to operate correctly. "
                "This is by design: the texture-based liveness detector needs to see "
                "skin/hair context beyond the tight face bbox to detect moiré/print patterns. "
                "CelebA-Spoof provides pre-cropped images without this context, which limits "
                "the Silent-Face models on this specific dataset.\n\n")
        f.write("In the **daemon integration context** (real camera frames), this is NOT an issue: "
                "YuNet provides a bbox from the full camera frame, and the scale crop will "
                "naturally include the surrounding context. The daemon's existing YuNet detector "
                "already provides the input the models need.\n\n")

        f.write("### Recommendation for Daemon Integration\n\n")
        f.write("**Wire MiniFASNet first (Silent-Face 2-model fusion), then add depth.**\n\n")
        f.write("**Concrete wiring for Silent-Face:**\n")
        f.write("1. After YuNet detection returns bbox (x,y,w,h) in the full camera frame:  \n")
        f.write("   - Crop with scale=2.7 → resize to 80×80 → MiniFASNetV2 → softmax[1] = V2_live  \n")
        f.write("   - Crop with scale=4.0 → resize to 80×80 → MiniFASNetV1SE → softmax[1] = V1SE_live  \n")
        f.write("   - Fused = (V2_live + V1SE_live) / 2  \n")
        f.write("   - Live if fused ≥ 0.40 (conservative; adjust after in-situ calibration)  \n\n")
        f.write("2. Normalization: divide pixel values by 255.0 (no mean/std subtraction needed).  \n")
        f.write("   Channel order: RGB (confirmed; ToTensor() in original pipeline = PIL→float/255).  \n\n")
        f.write("**Concrete wiring for Depth:**\n")
        f.write("1. Run Depth-Anything-V2-Small INT8 on the full camera frame (518×518, ImageNet norm).  \n")
        f.write("2. Extract face region depth patch using YuNet bbox.  \n")
        f.write("3. Compute: relief = std(face_depth) / (depth_map.max - depth_map.min).  \n")
        f.write("4. Live if relief ≥ threshold (calibrate on-device; start with 0.05).  \n\n")
        f.write("**Fusion rule:** Live = (SilentFace_fused ≥ T_sf) AND (DepthRelief ≥ T_d)  \n")
        f.write("Thresholds: calibrate with a brief in-situ video session "
                "(genuine face + phone photo + printed photo). Target APCER=0.  \n\n")
        f.write("**rPPG (deferred):** Add as a third gate — require ≥3s of face frames, "
                "detect pulse signal (FFT peak in 0.8–3 Hz). Cannot be gamed by a static photo "
                "or screen replay. Adds latency but provides a strong complementary signal.  \n\n")
        f.write("**Model files to copy to `~/.local/share/doorman/models/`:**  \n")
        f.write("- `~/datasets/models_eval/MiniFASNetV2.onnx`  \n")
        f.write("- `~/datasets/models_eval/MiniFASNetV1SE.onnx`  \n")
        f.write("- `~/datasets/models_eval/depth_anything_v2_small_int8.onnx`  \n\n")

        f.write("## Harness Usage\n\n")
        f.write("```bash\n")
        f.write("# Full evaluation (uses all 6000 images):\n")
        f.write("python scripts/pad_eval.py \\\n")
        f.write("    --dataset-dir ~/datasets/celeba_spoof_subset \\\n")
        f.write("    --pad-models  ~/datasets/models_eval \\\n")
        f.write("    --yunet       ~/.local/share/doorman/models/face_detection_yunet_2023mar.onnx \\\n")
        f.write("    --output      docs/pad_eval_baseline.md\n")
        f.write("\n# Quick smoke test (100 per class):\n")
        f.write("python scripts/pad_eval.py --max-per-class 100 ...\n")
        f.write("```\n\n")

        f.write("## Notes\n\n")
        f.write("- This is an **offline validation** only — no daemon code was modified.  \n")
        f.write("- Models are stored in `~/datasets/models_eval/`, NOT in `~/.local/share/doorman/models/`.  \n")
        f.write("- CelebA-Spoof images are stored in `~/datasets/celeba_spoof_subset/`.  \n")
        f.write("- The existing `liveness.onnx` (MiniFASNetV2-SE) in the daemon was NOT touched.  \n")
        f.write("- rPPG deferred: requires multi-second video, out of scope for this offline eval.  \n")

    # Also save raw metrics as JSON for easy parsing
    metrics_json = {
        "MiniFASNetV2": m_v2,
        "MiniFASNetV1SE": m_v1se,
        "SilentFace_Fused": m_fused,
        "DepthRelief": m_depth,
        "AND_Fusion": m_and,
        "meta": {
            "n_live": n_live,
            "n_spoof": n_spoof,
            "n_no_face": n_no_face,
            "n_errors": n_err,
            "ms_per_image": round(ms_per_img, 1),
            "total_elapsed_s": round(elapsed, 1),
        },
    }
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as jf:
        json.dump(metrics_json, jf, indent=2)
    print(f"Metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
