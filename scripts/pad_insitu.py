#!/usr/bin/env python3
"""
In-Situ PAD Calibration — real 4K camera frames, screen-replay attack.

Evaluates MiniFASNetV2, MiniFASNetV1SE, and DepthRelief on genuine full-frame
4K images vs. screen-replay attack frames captured at the actual doorbell camera.

Exhaustively tests all 3 softmax class indices for both Silent-Face models,
both scales (2.7 and 4.0), and a V2@4.0 + depth fusion.

Key contract:
  - LIVE folder:  ~/datasets/insitu/genuine/*.jpg   (label=0, bonafide)
  - SPOOF folder: ~/datasets/insitu/attack_screen/*.jpg (label=1, screen replay)
  - NO GPU — CPUExecutionProvider only
  - NO daemon / user-model changes
  - Output: docs/pad_insitu_calibration.md + docs/pad_insitu_calibration.json

Usage:
    scripts/.venv/bin/python scripts/pad_insitu.py
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

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
FASNET_INPUT_SIZE  = 80
DEPTH_INPUT_SIZE   = 518   # 37*14

YUNET_INPUT_SIZE   = 640
YUNET_CONF_THRESH  = 0.5
YUNET_NMS_THRESH   = 0.3
YUNET_STRIDES      = [8, 16, 32]

SCALES             = [2.7, 4.0]
CLASS_INDICES      = [0, 1, 2]   # exhaustive scan


# ══════════════════════════════════════════════════════════════════════════════
# YuNet detection  (shared with existing harnesses)
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
        n      = cls_t.shape[0]
        cols   = input_size // stride
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
            ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0
        if all(iou(d["bbox"], k["bbox"]) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def detect_face_bbox(
    detector: ort.InferenceSession,
    img_rgb: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    h, w = img_rgb.shape[:2]
    size  = YUNET_INPUT_SIZE
    inp   = yunet_preprocess(img_rgb, size)
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
# Silent-Face preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def silent_face_crop(
    img_rgb: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    scale: float,
    out_size: int = 80,
) -> np.ndarray:
    """
    Scale-expanded crop for MiniFASNet.  Mirrors minivision generate_patches.py.
    On a full 4K frame a 2.7x or 4.0x expansion gives substantial background context
    (screen bezel, environment texture) that the model uses for liveness discrimination.
    """
    src_h, src_w = img_rgb.shape[:2]

    if bbox is None:
        bx, by, bw, bh = 0, 0, src_w, src_h
    else:
        bx, by, bw, bh = bbox
        bx = max(0, min(bx, src_w - 1))
        by = max(0, min(by, src_h - 1))
        bw = max(1, min(bw, src_w - bx))
        bh = max(1, min(bh, src_h - by))

    new_w = bw * scale
    new_h = bh * scale
    cx = bx + bw / 2.0
    cy = by + bh / 2.0

    x0 = cx - new_w / 2.0
    y0 = cy - new_h / 2.0
    x1 = cx + new_w / 2.0
    y1 = cy + new_h / 2.0

    # Shift into image bounds; preserve real context rather than zero-padding
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

    crop = img_rgb[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        crop = img_rgb

    return np.array(Image.fromarray(crop).resize((out_size, out_size), Image.BILINEAR))


def fasnet_all_classes(
    sess: ort.InferenceSession,
    crop_rgb: np.ndarray,
) -> np.ndarray:
    """
    Run MiniFASNet and return softmax probabilities for all 3 classes.
    Returns shape (3,) float32.
    """
    arr = crop_rgb.astype(np.float32) / 255.0
    inp = arr.transpose(2, 0, 1)[None]   # NCHW
    out = sess.run(None, {"input": inp})[0][0]   # (3,)
    out = out - out.max()
    sm  = np.exp(out) / np.exp(out).sum()
    return sm.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Depth PAD
# ══════════════════════════════════════════════════════════════════════════════

def depth_preprocess(img_rgb: np.ndarray, size: int = DEPTH_INPUT_SIZE) -> np.ndarray:
    pil = Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]


def depth_face_relief(
    depth_sess: ort.InferenceSession,
    img_rgb: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]],
) -> float:
    """
    Depth relief score: std(face-region depth) / global_depth_range.
    Higher = more 3D structure = more likely live.
    On a flat screen the face has near-uniform depth; on a real face depth varies.
    """
    inp       = depth_preprocess(img_rgb, DEPTH_INPUT_SIZE)
    depth_map = depth_sess.run(None, {"pixel_values": inp})[0][0]  # (H, W)

    src_h, src_w = img_rgb.shape[:2]
    d_h, d_w     = depth_map.shape

    if face_bbox is not None:
        bx, by, bw, bh = face_bbox
        sx = d_w / src_w
        sy = d_h / src_h
        x0 = max(0, int(bx * sx))
        y0 = max(0, int(by * sy))
        x1 = min(d_w - 1, int((bx + bw) * sx))
        y1 = min(d_h - 1, int((by + bh) * sy))
        face_depth = depth_map[y0:y1 + 1, x0:x1 + 1] if (x1 > x0 and y1 > y0) else depth_map
    else:
        y0 = int(d_h * 0.2); y1 = int(d_h * 0.8)
        x0 = int(d_w * 0.2); x1 = int(d_w * 0.8)
        face_depth = depth_map[y0:y1, x0:x1]

    global_range = float(depth_map.max() - depth_map.min()) + 1e-8
    return min(float(face_depth.std()) / global_range, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    live_scores: np.ndarray,
    spoof_scores: np.ndarray,
    name: str,
    higher_is_live: bool = True,
) -> Dict:
    """
    Compute AUC, optimal-ACER operating point, and APCER=0 operating point.
    APCER = fraction of spoof samples classified as live (false accepts).
    BPCER = fraction of live samples classified as spoof (false rejects).
    """
    n_live  = len(live_scores)
    n_spoof = len(spoof_scores)

    y_true  = np.concatenate([np.ones(n_live), np.zeros(n_spoof)])
    y_score = np.concatenate([live_scores, spoof_scores])
    if not higher_is_live:
        y_score = -y_score
    auc = float(roc_auc_score(y_true, y_score))

    all_thresholds = np.sort(np.unique(np.concatenate([live_scores, spoof_scores])))

    best_acer   = 1.0
    best_thresh = float(all_thresholds[len(all_thresholds) // 2])
    best_apcer  = 1.0
    best_bpcer  = 1.0
    apcer0_thresh = None
    apcer0_bpcer  = 1.0

    for t in all_thresholds:
        if higher_is_live:
            apcer = float(np.mean(spoof_scores >= t))
            bpcer = float(np.mean(live_scores  <  t))
        else:
            apcer = float(np.mean(spoof_scores <= t))
            bpcer = float(np.mean(live_scores  >  t))
        acer = (apcer + bpcer) / 2.0

        if acer < best_acer:
            best_acer   = acer
            best_thresh = float(t)
            best_apcer  = apcer
            best_bpcer  = bpcer

        # APCER=0 strictly (no spoof passes at all)
        if apcer == 0.0 and bpcer < apcer0_bpcer:
            apcer0_bpcer  = bpcer
            apcer0_thresh = float(t)

    return {
        "name":             name,
        "n_live":           n_live,
        "n_spoof":          n_spoof,
        "auc":              round(auc, 4),
        "opt_thresh":       round(best_thresh, 6),
        "opt_apcer":        round(best_apcer, 4),
        "opt_bpcer":        round(best_bpcer, 4),
        "opt_acer":         round(best_acer, 4),
        "apcer0_thresh":    round(apcer0_thresh, 6) if apcer0_thresh is not None else None,
        "apcer0_bpcer":     round(apcer0_bpcer, 4) if apcer0_thresh is not None else None,
        "live_mean":        round(float(live_scores.mean()), 4),
        "live_std":         round(float(live_scores.std()),  4),
        "live_min":         round(float(live_scores.min()),  4),
        "live_max":         round(float(live_scores.max()),  4),
        "spoof_mean":       round(float(spoof_scores.mean()), 4),
        "spoof_std":        round(float(spoof_scores.std()),  4),
        "spoof_min":        round(float(spoof_scores.min()),  4),
        "spoof_max":        round(float(spoof_scores.max()),  4),
    }


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="In-situ PAD calibration on 4K frames")
    parser.add_argument("--genuine",    default=os.path.expanduser("~/datasets/insitu/genuine"))
    parser.add_argument("--attack",     default=os.path.expanduser("~/datasets/insitu/attack_screen"))
    parser.add_argument("--pad-models", default=os.path.expanduser("~/datasets/models_eval"))
    parser.add_argument("--yunet",      default=os.path.expanduser(
        "~/.local/share/doorman/models/face_detection_yunet_2023mar.onnx"))
    parser.add_argument("--output",     default="docs/pad_insitu_calibration.md")
    args = parser.parse_args()

    genuine_dir = Path(args.genuine)
    attack_dir  = Path(args.attack)
    pad_dir     = Path(args.pad_models)
    out_path    = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for p in [genuine_dir, attack_dir]:
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr); sys.exit(1)

    # ── Load models ────────────────────────────────────────────────────────
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    print("Loading models...")
    yunet_sess  = ort.InferenceSession(args.yunet, sess_options=sess_opts, providers=providers)
    fasnet_v2   = ort.InferenceSession(str(pad_dir / "MiniFASNetV2.onnx"),   sess_options=sess_opts, providers=providers)
    fasnet_v1se = ort.InferenceSession(str(pad_dir / "MiniFASNetV1SE.onnx"), sess_options=sess_opts, providers=providers)
    depth_sess  = ort.InferenceSession(str(pad_dir / "depth_anything_v2_small_int8.onnx"), sess_options=sess_opts, providers=providers)
    print("Models loaded.\n")

    # ── Collect images ──────────────────────────────────────────────────────
    genuine_imgs = sorted(genuine_dir.glob("*.jpg")) + sorted(genuine_dir.glob("*.png"))
    attack_imgs  = sorted(attack_dir.glob("*.jpg"))  + sorted(attack_dir.glob("*.png"))
    print(f"Dataset: {len(genuine_imgs)} genuine + {len(attack_imgs)} attack images\n")

    all_imgs = [(p, 0) for p in genuine_imgs] + [(p, 1) for p in attack_imgs]

    # Storage: per-image dict with all scores
    records         = []
    no_face_genuine = 0
    no_face_attack  = 0
    n_error         = 0
    t0              = time.time()

    for idx, (path, label) in enumerate(all_imgs):
        if idx % 20 == 0:
            print(f"  {idx}/{len(all_imgs)} ...", flush=True)

        img_rgb = load_image(path)
        if img_rgb is None:
            n_error += 1
            continue

        face_bbox = detect_face_bbox(yunet_sess, img_rgb)
        if face_bbox is None:
            if label == 0:
                no_face_genuine += 1
            else:
                no_face_attack += 1

        # ── Silent-Face: all models x scales x classes ─────────────────
        sf_scores: Dict[str, float] = {}

        for model_name, sess in [("V2", fasnet_v2), ("V1SE", fasnet_v1se)]:
            for scale in SCALES:
                crop  = silent_face_crop(img_rgb, face_bbox, scale, FASNET_INPUT_SIZE)
                probs = fasnet_all_classes(sess, crop)
                for cls_idx in CLASS_INDICES:
                    key = f"{model_name}_s{scale:.1f}_c{cls_idx}"
                    sf_scores[key] = float(probs[cls_idx])

        # ── Depth ──────────────────────────────────────────────────────
        depth_score = depth_face_relief(depth_sess, img_rgb, face_bbox)

        record = {"label": label, "path": str(path), "depth": depth_score}
        record.update(sf_scores)

        # Face fraction (diagnostic)
        if face_bbox is not None:
            bx, by, bw, bh = face_bbox
            h, w = img_rgb.shape[:2]
            record["face_frac"] = (bw * bh) / max(h * w, 1)
        else:
            record["face_frac"] = float("nan")

        records.append(record)

    elapsed   = time.time() - t0
    n_genuine = sum(1 for r in records if r["label"] == 0)
    n_attack  = sum(1 for r in records if r["label"] == 1)
    n_total   = len(records)

    print(f"\nScoring complete — {elapsed:.1f}s  ({elapsed / max(n_total, 1) * 1000:.0f} ms/img)")
    print(f"  Genuine: {n_genuine}  Attack: {n_attack}")
    print(f"  No-face: {no_face_genuine} genuine / {no_face_attack} attack")
    print(f"  Errors:  {n_error}")

    if n_genuine < 5 or n_attack < 5:
        print("ERROR: too few valid samples", file=sys.stderr); sys.exit(1)

    live_mask  = np.array([r["label"] == 0 for r in records])
    spoof_mask = np.array([r["label"] == 1 for r in records])

    # ── Compute metrics for every (model, scale, class) combo ──────────────
    all_metrics: List[Dict] = []
    score_keys = [k for k in records[0] if k not in ("label", "path", "depth", "face_frac")]

    for key in score_keys:
        scores    = np.array([r[key] for r in records])
        live_s    = scores[live_mask]
        spoof_s   = scores[spoof_mask]
        m = compute_metrics(live_s, spoof_s, key)
        all_metrics.append(m)

    # Depth
    depth_arr = np.array([r["depth"] for r in records])
    m_depth = compute_metrics(depth_arr[live_mask], depth_arr[spoof_mask], "Depth_relief")
    all_metrics.append(m_depth)

    # ── Find best single combo ──────────────────────────────────────────────
    sf_metrics  = [m for m in all_metrics if m["name"] != "Depth_relief"]
    best_single = max(sf_metrics, key=lambda m: m["auc"])

    print("\n" + "=" * 70)
    print("TOP 5 (model x scale x class) by AUC:")
    for m in sorted(sf_metrics, key=lambda m: m["auc"], reverse=True)[:5]:
        apcer0 = (f"APCER=0 @ thresh={m['apcer0_thresh']:.4f}, BPCER={m['apcer0_bpcer']:.4f}"
                  if m["apcer0_thresh"] is not None else "APCER=0 not achievable")
        print(f"  {m['name']:25s}  AUC={m['auc']:.4f}  "
              f"live={m['live_mean']:.4f}±{m['live_std']:.4f}  "
              f"spoof={m['spoof_mean']:.4f}±{m['spoof_std']:.4f}  {apcer0}")

    print(f"\nDepth relief:              AUC={m_depth['auc']:.4f}  "
          f"live={m_depth['live_mean']:.4f}±{m_depth['live_std']:.4f}  "
          f"spoof={m_depth['spoof_mean']:.4f}±{m_depth['spoof_std']:.4f}")

    # ── V2@4.0 + depth fusion (additive) ───────────────────────────────────
    # Pick the best V2 scale for fusion (whichever has higher AUC, using the best class idx)
    def _best_v2(scale: float) -> Dict:
        v2_scale_metrics = [m for m in sf_metrics
                            if m["name"].startswith(f"V2_s{scale:.1f}")]
        return max(v2_scale_metrics, key=lambda m: m["auc"])

    best_v2_27 = _best_v2(2.7)
    best_v2_40 = _best_v2(4.0)
    best_v2_for_fusion = best_v2_40 if best_v2_40["auc"] >= best_v2_27["auc"] else best_v2_27
    fusion_sf_key = best_v2_for_fusion["name"]
    fusion_sf_scale = "4.0" if "s4.0" in fusion_sf_key else "2.7"

    sf_arr = np.array([r[fusion_sf_key] for r in records])
    # Normalize both to [0,1] before averaging
    def norm01(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / (hi - lo + 1e-8)

    fused_arr = (norm01(sf_arr) + norm01(depth_arr)) / 2.0
    m_fused = compute_metrics(fused_arr[live_mask], fused_arr[spoof_mask],
                              f"Fusion({fusion_sf_key}+Depth)")
    all_metrics.append(m_fused)

    print(f"\nFusion ({fusion_sf_key} + Depth):  "
          f"AUC={m_fused['auc']:.4f}")
    if m_fused["apcer0_thresh"] is not None:
        print(f"  APCER=0 @ thresh={m_fused['apcer0_thresh']:.4f}, BPCER={m_fused['apcer0_bpcer']:.4f}")

    # ── Face fraction stats ─────────────────────────────────────────────────
    ff = np.array([r["face_frac"] for r in records])
    ff_live  = ff[live_mask  & ~np.isnan(ff)]
    ff_spoof = ff[spoof_mask & ~np.isnan(ff)]

    print(f"\nFace fraction (fraction of 4K frame area):")
    if len(ff_live)  > 0: print(f"  Genuine: mean={ff_live.mean():.4f}  median={np.median(ff_live):.4f}")
    if len(ff_spoof) > 0: print(f"  Attack:  mean={ff_spoof.mean():.4f}  median={np.median(ff_spoof):.4f}")
    no_face_rate_genuine = no_face_genuine / max(n_genuine, 1)
    no_face_rate_attack  = no_face_attack  / max(n_attack, 1)
    print(f"  No-face rate: genuine={no_face_rate_genuine:.2%}  attack={no_face_rate_attack:.2%}")

    # ── Write report and JSON ───────────────────────────────────────────────
    _write_report(
        out_path        = out_path,
        n_genuine       = n_genuine,
        n_attack        = n_attack,
        no_face_genuine = no_face_genuine,
        no_face_attack  = no_face_attack,
        n_error         = n_error,
        ff_live         = ff_live,
        ff_spoof        = ff_spoof,
        all_metrics     = all_metrics,
        best_single     = best_single,
        m_depth         = m_depth,
        m_fused         = m_fused,
        fusion_sf_key   = fusion_sf_key,
        elapsed         = elapsed,
        n_total         = n_total,
    )
    print(f"\nReport written: {out_path}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as jf:
        json.dump(
            {
                "meta": {
                    "n_genuine":        n_genuine,
                    "n_attack":         n_attack,
                    "no_face_genuine":  no_face_genuine,
                    "no_face_attack":   no_face_attack,
                    "n_errors":         n_error,
                    "elapsed_s":        round(elapsed, 1),
                    "ms_per_img":       round(elapsed / max(n_total, 1) * 1000, 1),
                },
                "metrics": {m["name"]: m for m in all_metrics},
            },
            jf,
            indent=2,
        )
    print(f"JSON written:   {json_path}")


def _write_report(
    out_path,
    n_genuine, n_attack, no_face_genuine, no_face_attack, n_error,
    ff_live, ff_spoof,
    all_metrics, best_single, m_depth, m_fused, fusion_sf_key,
    elapsed, n_total,
):
    from datetime import date

    today = date.today().isoformat()
    ms_per_img = elapsed / max(n_total, 1) * 1000.0
    no_face_rate_genuine = no_face_genuine / max(n_genuine, 1)
    no_face_rate_attack  = no_face_attack  / max(n_attack, 1)

    # Sorted ranking of SF combos
    sf_metrics   = [m for m in all_metrics if "Fusion" not in m["name"] and m["name"] != "Depth_relief"]
    ranked       = sorted(sf_metrics, key=lambda m: m["auc"], reverse=True)
    top5         = ranked[:5]

    # Go/no-go thresholds
    # Acceptable: APCER=0 achievable with BPCER ≤ 15%
    best_auc  = best_single["auc"]
    apcer0_ok = (best_single["apcer0_thresh"] is not None and
                 best_single["apcer0_bpcer"] <= 0.15)
    fused_apcer0_ok = (m_fused["apcer0_thresh"] is not None and
                       m_fused["apcer0_bpcer"] <= 0.15)

    go = apcer0_ok or fused_apcer0_ok

    def apcer0_str(m):
        if m.get("apcer0_thresh") is not None:
            return f"thresh={m['apcer0_thresh']:.4f}, BPCER={m['apcer0_bpcer']:.4f}"
        return "not achievable"

    rows_all = ""
    for m in ranked:
        rows_all += (
            f"| {m['name']} | {m['auc']:.4f} | "
            f"{m['live_mean']:.4f}±{m['live_std']:.4f} | "
            f"{m['spoof_mean']:.4f}±{m['spoof_std']:.4f} | "
            f"{m['opt_apcer']:.4f} | {m['opt_bpcer']:.4f} | "
            f"{apcer0_str(m)} |\n"
        )

    # Wiring spec (only if go)
    if go:
        best_for_wire = best_single if apcer0_ok else None
        if best_for_wire is None and fused_apcer0_ok:
            wiring_note = _fusion_wiring(fusion_sf_key, m_fused)
        else:
            wiring_note = _single_wiring(best_for_wire, m_fused, fused_apcer0_ok, fusion_sf_key)
    else:
        wiring_note = (
            "**NO WIRING — threshold criteria not met.**\n\n"
            "APCER=0 is either not achievable or requires BPCER > 15% on this dataset.\n\n"
            "Recommended next steps:\n"
            "1. Inspect images where genuine face is missed (run YuNet manually — face may be small or off-center).\n"
            "2. Capture more genuine frames at shorter distance / better lighting to raise face fraction.\n"
            "3. Test a stronger PAD model (e.g. CDCN, BCN, or a fine-tuned MobileNet-based PAD).\n"
            "4. Add print-attack frames to confirm screen-attack calibration generalises.\n"
        )

    ff_genuine_str = (f"mean={ff_live.mean():.4f}, median={np.median(ff_live):.4f}"
                      if len(ff_live) > 0 else "N/A (all no-face)")
    ff_attack_str  = (f"mean={ff_spoof.mean():.4f}, median={np.median(ff_spoof):.4f}"
                      if len(ff_spoof) > 0 else "N/A (all no-face)")

    report = f"""# In-Situ PAD Calibration — Screen-Attack on 4K Camera

**Generated:** {today}
**Purpose:** Determine whether MiniFASNet Silent-Face + DepthRelief can reject a
screen-replay attack on real 4K doorbell camera captures, and pin the wiring spec.
**Attack type tested:** Screen replay only (phone showing user's face).
**Print attack:** NOT tested in this run.
**GPU / daemon / user models:** NOT modified.

---

## 1. Dataset

| Split | Count | Resolution | Attack type |
|---|---|---|---|
| Genuine (live) | {n_genuine} | 3840×2160 | — (real face in front of camera) |
| Attack (screen replay) | {n_attack} | 3840×2160 | Phone screen showing user's face |
| Errors (unreadable) | {n_error} | — | — |

Paths: `~/datasets/insitu/genuine/` and `~/datasets/insitu/attack_screen/`

### YuNet Detection Rates

| Class | No-face count | No-face rate |
|---|---|---|
| Genuine | {no_face_genuine} | {no_face_rate_genuine:.1%} |
| Attack  | {no_face_attack}  | {no_face_rate_attack:.1%}  |

{"**FLAG:** Genuine no-face rate > 20% — some frames may have too small or off-center a face. Check capture quality before relying on these results." if no_face_rate_genuine > 0.20 else "Genuine no-face rate acceptable."}
{"**FLAG:** Attack no-face rate > 20% — many attack frames have no detected face. The screen attack may not show a full face, or the face detector is being defeated." if no_face_rate_attack > 0.20 else "Attack no-face rate acceptable."}

### Face Fraction (fraction of 4K frame area covered by YuNet bbox)

| Class | Face fraction |
|---|---|
| Genuine | {ff_genuine_str} |
| Attack  | {ff_attack_str}  |

Values near 0.002–0.01 are expected for a face at ~1 m from a 4K wide-angle camera.
The scale-expanded crop (2.7x/4.0x) will capture screen bezel/background context.

---

## 2. All (Model × Scale × Class) Results

{len(sf_metrics)} combinations evaluated (2 models × 2 scales × 3 classes):

| Config | AUC | Live mean±std | Spoof mean±std | Opt APCER | Opt BPCER | APCER=0 operating point |
|---|---|---|---|---|---|---|
{rows_all}
| Depth_relief | {m_depth['auc']:.4f} | {m_depth['live_mean']:.4f}±{m_depth['live_std']:.4f} | {m_depth['spoof_mean']:.4f}±{m_depth['spoof_std']:.4f} | {m_depth['opt_apcer']:.4f} | {m_depth['opt_bpcer']:.4f} | {apcer0_str(m_depth)} |
| {m_fused['name']} | {m_fused['auc']:.4f} | {m_fused['live_mean']:.4f}±{m_fused['live_std']:.4f} | {m_fused['spoof_mean']:.4f}±{m_fused['spoof_std']:.4f} | {m_fused['opt_apcer']:.4f} | {m_fused['opt_bpcer']:.4f} | {apcer0_str(m_fused)} |

---

## 3. Best Configuration

**Best single (by AUC): `{best_single['name']}`**

| Metric | Value |
|---|---|
| AUC | {best_single['auc']:.4f} |
| Live scores | {best_single['live_mean']:.4f} ± {best_single['live_std']:.4f} (min={best_single['live_min']:.4f}, max={best_single['live_max']:.4f}) |
| Spoof scores | {best_single['spoof_mean']:.4f} ± {best_single['spoof_std']:.4f} (min={best_single['spoof_min']:.4f}, max={best_single['spoof_max']:.4f}) |
| Optimal APCER | {best_single['opt_apcer']:.4f} |
| Optimal BPCER | {best_single['opt_bpcer']:.4f} |
| Optimal ACER  | {best_single['opt_acer']:.4f} |
| **APCER=0 threshold** | {apcer0_str(best_single)} |

**Fusion ({m_fused['name']}):**

| Metric | Value |
|---|---|
| AUC | {m_fused['auc']:.4f} |
| Live scores | {m_fused['live_mean']:.4f} ± {m_fused['live_std']:.4f} |
| Spoof scores | {m_fused['spoof_mean']:.4f} ± {m_fused['spoof_std']:.4f} |
| **APCER=0 threshold** | {apcer0_str(m_fused)} |

---

## 4. Verdict and Wiring Spec

{"### GO — Screen-attack rejection achievable" if go else "### NO-GO — Threshold criteria not met"}

{wiring_note}

---

## 5. Caveats

- **Screen attack only.** No print-attack frames were available. A threshold calibrated
  here may not generalise to a printed-photo attack.
- **Single session.** Only {n_genuine} genuine + {n_attack} attack frames, same day,
  same lighting. Production threshold should be validated across lighting conditions.
- **Class index is data-dependent.** The "live" class index identified here (on this
  specific camera/attack) may differ from the published Silent-Face convention.
  Do NOT assume index 1 = live without re-running on real data.

---

*Runtime: {ms_per_img:.0f} ms/img on CPU, {elapsed:.1f}s total for {n_total} frames.*
"""

    with open(out_path, "w") as f:
        f.write(report)


def _single_wiring(best, m_fused, fused_ok, fusion_sf_key):
    """Generate wiring spec for best single model (or fusion if single doesn't meet criteria)."""
    # Parse model / scale / class from key like "V2_s2.7_c0"
    key = best["name"]
    model_str = "MiniFASNetV2.onnx" if key.startswith("V2") else "MiniFASNetV1SE.onnx"
    scale_str = "2.7" if "s2.7" in key else "4.0"
    cls_idx   = int(key[-1])

    apcer0_note = (
        f"APCER=0 threshold: **{best['apcer0_thresh']:.4f}**  →  BPCER = {best['apcer0_bpcer']:.4f} "
        f"({best['apcer0_bpcer']*100:.0f}% genuine rejected)"
        if best["apcer0_thresh"] is not None else
        "APCER=0 not achievable with this model alone — use fusion below"
    )

    fusion_note = ""
    if fused_ok:
        fusion_note = f"""
**Optional fusion adds robustness:**
```
Combined = (norm(V2_score) + norm(depth_relief)) / 2.0
Threshold: {m_fused['apcer0_thresh']:.4f}  →  BPCER = {m_fused['apcer0_bpcer']:.4f}
```
"""

    return f"""**Best single model: `{key}`**  (AUC={best['auc']:.4f})

{apcer0_note}

### Daemon Wiring Spec

```
After YuNet detects face bbox (x, y, w, h) in the FULL 4K frame:

1. Expand crop:
   scale_expanded_crop = silent_face_crop(frame, bbox, scale={scale_str})
   # Shift-clamped, NOT zero-padded — preserve background/screen context
   # Output: 80×80 RGB uint8

2. Normalise:
   inp = crop.astype(float32) / 255.0          # [0,1], no mean/std subtraction
   inp = inp.transpose(2,0,1)[None]            # NCHW shape (1,3,80,80)

3. Inference:
   logits = ort_session.run(None, {{"input": inp}})[0][0]   # shape (3,)

4. Softmax:
   logits -= logits.max()
   sm = exp(logits) / exp(logits).sum()        # stable softmax

5. Live score:
   live_score = sm[{cls_idx}]                           # class index {cls_idx} = live on this data

6. Decision:
   is_live = (live_score >= THRESHOLD)

   Model file  : ~/datasets/models_eval/{model_str}
   Scale       : {scale_str}
   Live class  : {cls_idx}
   Threshold   : {best['apcer0_thresh']:.4f}  (APCER=0, BPCER={best['apcer0_bpcer']:.4f})
   Normalisation: /255 only, no mean/std subtraction
   Input shape : (1, 3, 80, 80) float32 [0, 1]
   Channel order: RGB (as loaded by PIL)
```
{fusion_note}
**Screen-attack caveat:** This threshold is calibrated on screen-replay only.
Re-test after capturing print-attack frames before locking this for production.
"""


def _fusion_wiring(fusion_sf_key, m_fused):
    """Generate wiring spec when fusion meets criteria but single model does not."""
    model_str = "MiniFASNetV2.onnx" if fusion_sf_key.startswith("V2") else "MiniFASNetV1SE.onnx"
    scale_str = "2.7" if "s2.7" in fusion_sf_key else "4.0"
    cls_idx   = int(fusion_sf_key[-1])

    return f"""**Best configuration: V2+Depth fusion** (AUC={m_fused['auc']:.4f})
No single model meets APCER=0 + BPCER≤15%; fusion does.

### Daemon Wiring Spec (Fusion)

```
After YuNet detects face bbox in the FULL 4K frame:

1. Silent-Face score:
   crop = silent_face_crop(frame, bbox, scale={scale_str})
   inp  = crop.float32/255.0 transposed to NCHW
   sm   = softmax(V2_model.run(inp))
   sf_score = sm[{cls_idx}]

2. Depth relief score:
   depth_map = DepthAnythingV2.run(full_frame_518x518_imagenet_norm)
   face_region = depth_map mapped to face_bbox coordinates
   depth_score = std(face_region) / (depth_map.max - depth_map.min + 1e-8)

3. Normalise both to [0,1] using calibration-set min/max,
   then average:
   combined = (norm(sf_score) + norm(depth_score)) / 2.0

4. Decision:
   is_live = (combined >= {m_fused['apcer0_thresh']:.4f})
   (APCER=0, BPCER={m_fused['apcer0_bpcer']:.4f} on this capture session)

   SF model file: ~/datasets/models_eval/{model_str}
   Scale        : {scale_str}
   Live class   : {cls_idx}
   Depth model  : ~/datasets/models_eval/depth_anything_v2_small_int8.onnx
   Fusion thresh: {m_fused['apcer0_thresh']:.4f}
```

**Screen-attack caveat:** Calibrated on screen-replay only.
Re-test with print-attack frames before production deployment.
"""


if __name__ == "__main__":
    main()
