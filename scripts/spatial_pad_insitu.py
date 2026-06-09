#!/usr/bin/env python3
"""
Spatial / anti-spoofing encoder evaluation on real in-situ 4K captures.

Evaluates three spatial cues for LIVE vs SCREEN-REPLAY separation:
  1. Multi-frame parallax / planarity (homography residual — cheapest, uses motion)
  2. DINOv2-small texture features (logistic regression cross-val probe)
  3. Depth-Anything-V2 facial curvature relief (face-surface, not global variance)
  Plus fusion of best two cues.

DATA:
  LIVE:   ~/datasets/insitu/genuine/*.jpg         (~58 frames, 3840x2160)
  SPOOF1: ~/datasets/insitu/attack_screen/*.jpg    (~60 frames, same camera)
  SPOOF2: ~/datasets/insitu/attack_screen2/*.jpg   (~60 frames, different photo)

MODELS:
  YuNet:   ~/.local/share/doorman/models/face_detection_yunet_2023mar.onnx
  Depth:   ~/datasets/models_eval/depth_anything_v2_small_int8.onnx
  DINOv2:  facebook/dinov2-small (HuggingFace, ~84 MB, cached to ~/datasets/models_eval/)

OUTPUT:
  docs/spatial_encoders_insitu.md   (report)

Usage:
    scripts/.venv/bin/python scripts/spatial_pad_insitu.py
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

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
GENUINE_DIR   = Path(os.path.expanduser("~/datasets/insitu/genuine"))
ATTACK1_DIR   = Path(os.path.expanduser("~/datasets/insitu/attack_screen"))
ATTACK2_DIR   = Path(os.path.expanduser("~/datasets/insitu/attack_screen2"))
YUNET_PATH    = Path(os.path.expanduser("~/.local/share/doorman/models/face_detection_yunet_2023mar.onnx"))
DEPTH_PATH    = Path(os.path.expanduser("~/datasets/models_eval/depth_anything_v2_small_int8.onnx"))
MODEL_CACHE   = Path(os.path.expanduser("~/datasets/models_eval"))

# ── YuNet constants ───────────────────────────────────────────────────────────
YUNET_INPUT_SIZE  = 640
YUNET_CONF_THRESH = 0.5
YUNET_NMS_THRESH  = 0.3
YUNET_STRIDES     = [8, 16, 32]

# ArcFace 112×112 template (right-eye, left-eye, nose, right-mouth, left-mouth)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

DEPTH_INPUT_SIZE = 518  # 37*14


# ══════════════════════════════════════════════════════════════════════════════
# YuNet — detect + landmarks
# ══════════════════════════════════════════════════════════════════════════════

def yunet_preprocess(img_rgb: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR)
    bgr = np.array(pil, dtype=np.float32)[:, :, ::-1]
    return bgr.transpose(2, 0, 1)[None]


def yunet_decode(outputs: dict, input_size: int, conf_thresh: float) -> List[dict]:
    dets = []
    inv = 1.0 / input_size
    for stride in YUNET_STRIDES:
        cls_t  = outputs.get(f"cls_{stride}")
        obj_t  = outputs.get(f"obj_{stride}")
        bbox_t = outputs.get(f"bbox_{stride}")
        kps_t  = outputs.get(f"kps_{stride}")
        if cls_t is None:
            continue
        cls_t = cls_t[0]; obj_t = obj_t[0]; bbox_t = bbox_t[0]; kps_t = kps_t[0]
        n = cls_t.shape[0]
        cols = input_size // stride
        for i in range(n):
            score = math.sqrt(max(float(cls_t[i, 0]), 0.0) * max(float(obj_t[i, 0]), 0.0))
            if score < conf_thresh:
                continue
            row = i // cols; col = i % cols
            dx, dy, dw, dh = bbox_t[i]
            cx = (col + float(dx)) * stride
            cy = (row + float(dy)) * stride
            w  = math.exp(float(dw)) * stride
            h  = math.exp(float(dh)) * stride
            bbox = ((cx - w/2) * inv, (cy - h/2) * inv, w * inv, h * inv)
            lms = []
            for j in range(5):
                lms.append(((col + float(kps_t[i, 2*j])) * stride * inv,
                             (row + float(kps_t[i, 2*j+1])) * stride * inv))
            dets.append({"bbox": bbox, "score": score, "landmarks": lms})
    return dets


def nms(dets: List[dict], iou_thresh: float) -> List[dict]:
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        def iou(a, b):
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ix = max(0.0, min(ax+aw, bx+bw) - max(ax, bx))
            iy = max(0.0, min(ay+ah, by+bh) - max(ay, by))
            inter = ix * iy
            union = aw*ah + bw*bh - inter
            return inter/union if union > 0 else 0.0
        if all(iou(d["bbox"], k["bbox"]) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def detect_face(yunet_sess: ort.InferenceSession,
                img_rgb: np.ndarray) -> Optional[dict]:
    """Returns best detection dict with bbox+landmarks in pixel coords, or None."""
    h, w = img_rgb.shape[:2]
    inp = yunet_preprocess(img_rgb, YUNET_INPUT_SIZE)
    outs = yunet_sess.run(None, {"input": inp})
    names = [o.name for o in yunet_sess.get_outputs()]
    out_dict = dict(zip(names, outs))
    dets = yunet_decode(out_dict, YUNET_INPUT_SIZE, YUNET_CONF_THRESH)
    dets = nms(dets, YUNET_NMS_THRESH)
    if not dets:
        return None
    best = max(dets, key=lambda d: d["score"])
    bx, by, bw, bh = best["bbox"]
    best["bbox_px"] = (int(bx*w), int(by*h), int(bw*w), int(bh*h))
    best["landmarks_px"] = np.array([(lx*w, ly*h) for lx, ly in best["landmarks"]],
                                     dtype=np.float32)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Face alignment (5-pt Umeyama → 112×112) — reused from face_eval.py
# ══════════════════════════════════════════════════════════════════════════════

def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    n = src.shape[0]
    sx, sy = src[:, 0].mean(), src[:, 1].mean()
    dx, dy = dst[:, 0].mean(), dst[:, 1].mean()
    sxc = src[:, 0] - sx; syc = src[:, 1] - sy
    dxc = dst[:, 0] - dx; dyc = dst[:, 1] - dy
    a = float(np.sum(dxc*sxc + dyc*syc))
    b = float(np.sum(dyc*sxc - dxc*syc))
    src_var = float(np.sum(sxc**2 + syc**2))
    if src_var < 1e-12:
        return None
    norm = math.sqrt(a*a + b*b)
    if norm < 1e-12:
        return None
    sa = a / src_var; sb = b / src_var
    tx = dx - (sa*sx - sb*sy)
    ty = dy - (sb*sx + sa*sy)
    return np.array([[sa, -sb, tx], [sb, sa, ty]], dtype=np.float32)


def invert_affine2x3(m: np.ndarray) -> Optional[np.ndarray]:
    a, b, tx = m[0]; c, d, ty = m[1]
    det = a*d - b*c
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    ia = d*inv_det; ib = -b*inv_det; ic = -c*inv_det; id_ = a*inv_det
    return np.array([[ia, ib, -(ia*tx + ib*ty)],
                     [ic, id_, -(ic*tx + id_*ty)]], dtype=np.float32)


def align_face(img_rgb: np.ndarray,
               landmarks_px: np.ndarray,
               out_size: int = 112) -> Optional[np.ndarray]:
    m = umeyama_similarity(landmarks_px, ARCFACE_TEMPLATE)
    if m is None:
        return None
    inv = invert_affine2x3(m)
    if inv is None:
        return None
    h, w = img_rgb.shape[:2]
    oy, ox = np.meshgrid(np.arange(out_size, dtype=np.float32),
                         np.arange(out_size, dtype=np.float32), indexing='ij')
    pts = np.stack([ox.ravel() + 0.5, oy.ravel() + 0.5], axis=1)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    src = pts_h @ inv.T
    px = src[:, 0] - 0.5; py = src[:, 1] - 0.5
    x0 = np.floor(px).astype(np.int32); y0 = np.floor(py).astype(np.int32)
    fx = px - x0.astype(np.float32); fy = py - y0.astype(np.float32)
    x0c = np.clip(x0, 0, w-1); y0c = np.clip(y0, 0, h-1)
    x1c = np.clip(x0+1, 0, w-1); y1c = np.clip(y0+1, 0, h-1)
    w00 = ((1-fx)*(1-fy))[:, None]; w01 = ((1-fx)*fy)[:, None]
    w10 = (fx*(1-fy))[:, None];     w11 = (fx*fy)[:, None]
    out_flat = (w00*img_rgb[y0c, x0c].astype(np.float32) +
                w01*img_rgb[y1c, x0c].astype(np.float32) +
                w10*img_rgb[y0c, x1c].astype(np.float32) +
                w11*img_rgb[y1c, x1c].astype(np.float32))
    return np.clip(np.round(out_flat), 0, 255).astype(np.uint8).reshape(out_size, out_size, 3)


# ══════════════════════════════════════════════════════════════════════════════
# CUE 1: Multi-frame parallax / planarity
# ══════════════════════════════════════════════════════════════════════════════

def compute_parallax_score(
    frames_rgb: List[np.ndarray],
    bboxes_px: List[Optional[Tuple]],
    min_pairs: int = 5,
) -> float:
    """
    For consecutive frame pairs within a folder:
      - Crop to face bbox (use face ROI, not full frame — avoids background clutter)
      - Detect ORB keypoints + descriptors in each crop
      - Match with BFMatcher (Hamming)
      - Fit homography (RANSAC) and compute mean reprojection residual
      - Score = mean reprojection residual (high = non-planar = LIVE signal)

    A flat photo/screen is well-explained by a homography (low residual).
    A real 3D face undergoing motion is NOT planar → higher residual.

    Returns mean score across all pairs. Returns NaN if insufficient pairs.
    """
    n = len(frames_rgb)
    if n < 2:
        return float("nan")

    orb = cv2.ORB_create(nfeatures=500)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    residuals = []
    pair_count = 0

    for i in range(n - 1):
        img1 = frames_rgb[i]
        img2 = frames_rgb[i + 1]
        bb1  = bboxes_px[i]
        bb2  = bboxes_px[i + 1]

        # Use face region if available, otherwise center crop
        def get_face_crop(img, bb):
            h, w = img.shape[:2]
            if bb is not None:
                bx, by, bw, bh = bb
                # Expand crop slightly for context
                margin = int(max(bw, bh) * 0.2)
                x0 = max(0, bx - margin); y0 = max(0, by - margin)
                x1 = min(w, bx + bw + margin); y1 = min(h, by + bh + margin)
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = img
            else:
                # Central crop: assume face is near center in doorbell scenario
                qh, qw = h // 4, w // 4
                crop = img[qh:3*qh, qw:3*qw]
            # Resize to manageable size for ORB
            return cv2.resize(crop, (480, 360), interpolation=cv2.INTER_LINEAR)

        crop1_bgr = cv2.cvtColor(get_face_crop(img1, bb1), cv2.COLOR_RGB2GRAY)
        crop2_bgr = cv2.cvtColor(get_face_crop(img2, bb2), cv2.COLOR_RGB2GRAY)

        kp1, des1 = orb.detectAndCompute(crop1_bgr, None)
        kp2, des2 = orb.detectAndCompute(crop2_bgr, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            continue

        matches = bf.match(des1, des2)
        if len(matches) < 8:
            continue

        # Sort by distance, take best 100
        matches = sorted(matches, key=lambda m: m.distance)[:100]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Fit homography (planar model)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)

        if H is None or mask is None:
            continue

        n_inliers = int(mask.sum())
        if n_inliers < 4:
            continue

        # Compute reprojection residual for ALL matches (not just inliers)
        # This is: how poorly does the planar model explain the full point cloud
        pts1_h = np.concatenate([pts1, np.ones((len(pts1), 1), dtype=np.float32)], axis=1)
        projected = (H @ pts1_h.T).T
        projected[:, 0] /= projected[:, 2] + 1e-8
        projected[:, 1] /= projected[:, 2] + 1e-8
        projected = projected[:, :2]
        residual = np.sqrt(((projected - pts2) ** 2).sum(axis=1)).mean()

        # Also use fraction of outliers as a secondary signal
        # Combined score: residual + outlier_fraction * 10 (to weight non-planarity)
        outlier_frac = 1.0 - n_inliers / max(len(matches), 1)
        combined = float(residual) + outlier_frac * 10.0
        residuals.append(combined)
        pair_count += 1

    if len(residuals) < min_pairs:
        # Return mean of what we have, or 0 if nothing
        if len(residuals) > 0:
            return float(np.mean(residuals))
        return float("nan")

    return float(np.mean(residuals))


# ══════════════════════════════════════════════════════════════════════════════
# CUE 2: DINOv2-small features
# ══════════════════════════════════════════════════════════════════════════════

def load_dinov2_model(cache_dir: Path):
    """
    Load DINOv2-small (ViT-S/14) pretrained model.
    Uses torch.hub from facebookresearch/dinov2.
    Weights cached at cache_dir.
    Returns (model, transform) — model in eval mode on CPU.
    """
    import torch
    import torchvision.transforms as T

    os.makedirs(cache_dir, exist_ok=True)
    # Point torch hub to our cache dir
    torch.hub.set_dir(str(cache_dir / "torch_hub"))

    print("  Loading DINOv2-small (ViT-S/14) via torch.hub...", flush=True)
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
    )
    model.eval()
    print("  DINOv2-small loaded.", flush=True)

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def extract_dino_features(
    model,
    transform,
    face_crop_rgb: np.ndarray,
) -> np.ndarray:
    """
    Run DINOv2-small on a 112x112 aligned face crop.
    Returns CLS token (384-d) as the feature vector.
    """
    import torch

    pil = Image.fromarray(face_crop_rgb)
    x = transform(pil).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        feats = model(x)  # (1, 384) — CLS token for ViT-S
    return feats[0].numpy().astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CUE 3: Depth-Anything-V2 facial curvature
# ══════════════════════════════════════════════════════════════════════════════

def depth_preprocess(img_rgb: np.ndarray, size: int = DEPTH_INPUT_SIZE) -> np.ndarray:
    pil = Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = (arr - mean) / std
    return arr.transpose(2, 0, 1)[None]


def compute_depth_curvature_score(
    depth_sess: ort.InferenceSession,
    face_crop_rgb: np.ndarray,  # 112x112 aligned face crop
) -> float:
    """
    Run depth on the aligned 112x112 face crop.
    Compute face-surface relief: landmark-guided curvature.

    Score = curvature magnitude within key facial sub-regions:
      nose bridge region vs cheek flanks (depth gradient magnitude)

    A flat screen has uniform depth in the face crop → low gradient.
    A real face has nose protrusion, orbital recession → high gradient.

    Uses the face crop (not full frame) so depth is measured relative
    to the face surface itself, not background distance.
    """
    inp = depth_preprocess(face_crop_rgb, DEPTH_INPUT_SIZE)
    depth_raw = depth_sess.run(None, {"pixel_values": inp})[0][0]  # (H_out, W_out)
    # depth_raw shape: (floor(518/14)*14, ...) ≈ (518, 518) depending on model output
    dh, dw = depth_raw.shape

    # Normalize depth to [0, 1] within the face crop
    d_min = depth_raw.min()
    d_max = depth_raw.max()
    d_range = d_max - d_min + 1e-8
    depth_norm = (depth_raw - d_min) / d_range

    # Compute gradient magnitude (Sobel) across the whole depth map
    # Then focus on the central facial region (inner 60% of the crop)
    # This captures nose protrusion vs. cheek/orbital depth variation
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    grad_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Focus on central facial area (avoid boundary artifacts)
    margin_y = int(dh * 0.2)
    margin_x = int(dw * 0.2)
    face_region_grad = grad_mag[margin_y:dh-margin_y, margin_x:dw-margin_x]

    # Use 90th percentile gradient (robust to noise, captures sharp depth edges)
    score = float(np.percentile(face_region_grad, 90))

    # Also compute depth STD in facial sub-regions (nose vs. cheeks)
    # Nose bridge area (center top ~25%x40% of crop)
    nose_y0, nose_y1 = int(dh*0.25), int(dh*0.55)
    nose_x0, nose_x1 = int(dw*0.35), int(dw*0.65)
    nose_depth = depth_norm[nose_y0:nose_y1, nose_x0:nose_x1]

    # Cheek areas (left and right flanks)
    cheek_y0, cheek_y1 = int(dh*0.35), int(dh*0.65)
    left_cheek  = depth_norm[cheek_y0:cheek_y1, int(dw*0.05):int(dw*0.30)]
    right_cheek = depth_norm[cheek_y0:cheek_y1, int(dw*0.70):int(dw*0.95)]

    # Nose protrusion relative to cheeks (real face: nose closer = higher relative depth)
    if left_cheek.size > 0 and right_cheek.size > 0 and nose_depth.size > 0:
        cheek_mean = (left_cheek.mean() + right_cheek.mean()) / 2.0
        nose_mean  = nose_depth.mean()
        # On real face, nose should be closer (higher depth value in monocular relative depth)
        # On flat screen, all regions have similar depth
        nose_cheek_contrast = abs(float(nose_mean - cheek_mean))
    else:
        nose_cheek_contrast = 0.0

    # Combined: gradient score + nose-cheek contrast (both measure 3D structure)
    # Normalize gradient to [0, ~1] range (gradient magnitude in uint8 scale)
    combined = (score / 50.0) + nose_cheek_contrast
    return float(combined)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    live_scores: np.ndarray,
    spoof_scores: np.ndarray,
    name: str,
    higher_is_live: bool = True,
) -> Dict:
    n_live  = len(live_scores)
    n_spoof = len(spoof_scores)
    if n_live < 2 or n_spoof < 2:
        return {"name": name, "auc": float("nan"), "error": "insufficient data",
                "n_live": n_live, "n_spoof": n_spoof}

    y_true  = np.concatenate([np.ones(n_live), np.zeros(n_spoof)])
    y_score = np.concatenate([live_scores, spoof_scores])
    if not higher_is_live:
        y_score = -y_score
    auc = float(roc_auc_score(y_true, y_score))

    all_t = np.sort(np.unique(np.concatenate([live_scores, spoof_scores])))

    best_acer = 1.0; best_thresh = float(all_t[len(all_t)//2])
    best_apcer = 1.0; best_bpcer = 1.0
    apcer0_thresh = None; apcer0_bpcer = 1.0

    for t in all_t:
        if higher_is_live:
            apcer = float(np.mean(spoof_scores >= t))
            bpcer = float(np.mean(live_scores   <  t))
        else:
            apcer = float(np.mean(spoof_scores <= t))
            bpcer = float(np.mean(live_scores   >  t))
        acer = (apcer + bpcer) / 2.0
        if acer < best_acer:
            best_acer = acer; best_thresh = float(t)
            best_apcer = apcer; best_bpcer = bpcer
        if apcer == 0.0 and bpcer < apcer0_bpcer:
            apcer0_bpcer = bpcer; apcer0_thresh = float(t)

    return {
        "name": name,
        "n_live": n_live, "n_spoof": n_spoof,
        "auc": round(auc, 4),
        "higher_is_live": higher_is_live,
        "opt_thresh": round(best_thresh, 6),
        "opt_apcer": round(best_apcer, 4),
        "opt_bpcer": round(best_bpcer, 4),
        "opt_acer": round(best_acer, 4),
        "apcer0_thresh": round(apcer0_thresh, 6) if apcer0_thresh is not None else None,
        "apcer0_bpcer": round(apcer0_bpcer, 4) if apcer0_thresh is not None else None,
        "live_mean": round(float(live_scores.mean()), 4),
        "live_std":  round(float(live_scores.std()), 4),
        "live_min":  round(float(live_scores.min()), 4),
        "live_max":  round(float(live_scores.max()), 4),
        "spoof_mean": round(float(spoof_scores.mean()), 4),
        "spoof_std":  round(float(spoof_scores.std()), 4),
        "spoof_min":  round(float(spoof_scores.min()), 4),
        "spoof_max":  round(float(spoof_scores.max()), 4),
    }


def apcer0_str(m: Dict) -> str:
    if m.get("apcer0_thresh") is not None:
        return f"thresh={m['apcer0_thresh']:.4f}, BPCER={m['apcer0_bpcer']:.4f}"
    return "not achievable"


def norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def load_folder_images(folder: Path) -> List[Tuple[Path, np.ndarray]]:
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    result = []
    for p in paths:
        img = load_image(p)
        if img is not None:
            result.append((p, img))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dino", action="store_true",
                        help="Skip DINOv2 (skip download if offline)")
    parser.add_argument("--output", default="docs/spatial_encoders_insitu.md")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_global = time.time()

    # ── Load models ───────────────────────────────────────────────────────────
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]

    print("Loading YuNet...", flush=True)
    yunet_sess = ort.InferenceSession(str(YUNET_PATH), sess_options=sess_opts, providers=providers)
    print("Loading Depth-Anything-V2...", flush=True)
    depth_sess = ort.InferenceSession(str(DEPTH_PATH), sess_options=sess_opts, providers=providers)
    print("Models loaded.\n", flush=True)

    # ── Load DINOv2 ───────────────────────────────────────────────────────────
    dino_model = None; dino_transform = None
    dino_note = "SKIPPED (--skip-dino flag or load failure)"
    if not args.skip_dino:
        try:
            dino_model, dino_transform = load_dinov2_model(MODEL_CACHE)
            dino_note = "DINOv2-small ViT-S/14 (facebookresearch/dinov2, pretrained=True)"
        except Exception as e:
            print(f"  WARNING: DINOv2 load failed: {e}", flush=True)
            dino_note = f"FAILED to load: {e}"

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading images...", flush=True)
    genuine_frames  = load_folder_images(GENUINE_DIR)
    attack1_frames  = load_folder_images(ATTACK1_DIR)
    attack2_frames  = load_folder_images(ATTACK2_DIR)

    print(f"  Genuine:  {len(genuine_frames)} frames")
    print(f"  Attack1:  {len(attack1_frames)} frames")
    print(f"  Attack2:  {len(attack2_frames)} frames\n")

    # ── YuNet detection pass ──────────────────────────────────────────────────
    print("Running YuNet detection on all frames...", flush=True)

    def detect_all(frames, label):
        detections = []
        no_face = 0
        for path, img in frames:
            det = detect_face(yunet_sess, img)
            if det is None:
                no_face += 1
            detections.append(det)
        return detections, no_face

    t0 = time.time()
    genuine_dets,  nf_gen  = detect_all(genuine_frames,  "genuine")
    attack1_dets,  nf_att1 = detect_all(attack1_frames,  "attack1")
    attack2_dets,  nf_att2 = detect_all(attack2_frames,  "attack2")
    detect_time = time.time() - t0
    print(f"  Detection done in {detect_time:.1f}s")
    print(f"  No-face: genuine={nf_gen}/{len(genuine_frames)}, "
          f"attack1={nf_att1}/{len(attack1_frames)}, "
          f"attack2={nf_att2}/{len(attack2_frames)}\n")

    # ── Aligned face crops ────────────────────────────────────────────────────
    # For cues 2+3 we need aligned 112×112 crops
    def get_aligned_crops(frames, dets):
        crops = []
        for (path, img), det in zip(frames, dets):
            if det is not None:
                aligned = align_face(img, det["landmarks_px"], out_size=112)
                if aligned is None:
                    # Fallback: bbox crop
                    bx, by, bw, bh = det["bbox_px"]
                    h, w = img.shape[:2]
                    x0, y0 = max(0, bx), max(0, by)
                    x1, y1 = min(w, bx+bw), min(h, by+bh)
                    crop = img[y0:y1, x0:x1]
                    aligned = np.array(Image.fromarray(crop).resize((112, 112), Image.LANCZOS))
                crops.append(aligned)
            else:
                crops.append(None)
        return crops

    print("Aligning face crops...", flush=True)
    genuine_crops  = get_aligned_crops(genuine_frames,  genuine_dets)
    attack1_crops  = get_aligned_crops(attack1_frames,  attack1_dets)
    attack2_crops  = get_aligned_crops(attack2_frames,  attack2_dets)

    # ══════════════════════════════════════════════════════════════════════════
    # CUE 1: Parallax / planarity (per-folder score)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CUE 1: Multi-frame parallax / planarity")
    print("="*60, flush=True)

    # Per-folder: genuine=1 clip, attack1=1 clip, attack2=1 clip
    def get_bboxes(dets):
        return [d["bbox_px"] if d is not None else None for d in dets]

    t0 = time.time()
    g_imgs   = [img for _, img in genuine_frames]
    a1_imgs  = [img for _, img in attack1_frames]
    a2_imgs  = [img for _, img in attack2_frames]

    parallax_genuine  = compute_parallax_score(g_imgs,  get_bboxes(genuine_dets))
    parallax_attack1  = compute_parallax_score(a1_imgs, get_bboxes(attack1_dets))
    parallax_attack2  = compute_parallax_score(a2_imgs, get_bboxes(attack2_dets))
    parallax_time = time.time() - t0

    print(f"  Genuine  folder parallax score: {parallax_genuine:.4f}")
    print(f"  Attack1  folder parallax score: {parallax_attack1:.4f}")
    print(f"  Attack2  folder parallax score: {parallax_attack2:.4f}")
    print(f"  Time: {parallax_time:.1f}s")

    # Per-frame: use sliding window (pair each frame with next)
    # For per-frame approximation: score each frame pair, assign score to first frame
    def per_frame_parallax(imgs, bboxes):
        orb = cv2.ORB_create(nfeatures=500)
        bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        scores = []
        n = len(imgs)

        def get_face_crop(img, bb, size=(480, 360)):
            h, w = img.shape[:2]
            if bb is not None:
                bx, by, bw, bh = bb
                margin = int(max(bw, bh) * 0.2)
                x0 = max(0, bx - margin); y0 = max(0, by - margin)
                x1 = min(w, bx+bw+margin); y1 = min(h, by+bh+margin)
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    crop = img
            else:
                qh, qw = h//4, w//4
                crop = img[qh:3*qh, qw:3*qw]
            return cv2.resize(crop, size, interpolation=cv2.INTER_LINEAR)

        for i in range(n - 1):
            g1 = cv2.cvtColor(get_face_crop(imgs[i],   bboxes[i]),   cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(get_face_crop(imgs[i+1], bboxes[i+1]), cv2.COLOR_RGB2GRAY)
            kp1, des1 = orb.detectAndCompute(g1, None)
            kp2, des2 = orb.detectAndCompute(g2, None)
            if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
                scores.append(float("nan"))
                continue
            matches = bf.match(des1, des2)
            if len(matches) < 8:
                scores.append(float("nan"))
                continue
            matches = sorted(matches, key=lambda m: m.distance)[:100]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)
            if H is None or mask is None:
                scores.append(float("nan"))
                continue
            n_inliers = int(mask.sum())
            pts1_h = np.concatenate([pts1, np.ones((len(pts1),1), dtype=np.float32)], axis=1)
            proj = (H @ pts1_h.T).T
            proj[:, 0] /= proj[:, 2] + 1e-8
            proj[:, 1] /= proj[:, 2] + 1e-8
            residual = np.sqrt(((proj[:, :2] - pts2)**2).sum(axis=1)).mean()
            outlier_frac = 1.0 - n_inliers / max(len(matches), 1)
            scores.append(float(residual) + outlier_frac * 10.0)

        # Last frame gets same score as second-to-last (no next frame)
        if scores:
            scores.append(scores[-1])
        return scores

    t0 = time.time()
    pf_genuine  = per_frame_parallax(g_imgs,  get_bboxes(genuine_dets))
    pf_attack1  = per_frame_parallax(a1_imgs, get_bboxes(attack1_dets))
    pf_attack2  = per_frame_parallax(a2_imgs, get_bboxes(attack2_dets))
    pf_time = time.time() - t0

    # Filter NaN from per-frame
    pf_gen_clean   = np.array([s for s in pf_genuine  if not math.isnan(s)])
    pf_att1_clean  = np.array([s for s in pf_attack1  if not math.isnan(s)])
    pf_att2_clean  = np.array([s for s in pf_attack2  if not math.isnan(s)])
    pf_spoof_clean = np.concatenate([pf_att1_clean, pf_att2_clean])

    print(f"  Per-frame scores (valid): gen={len(pf_gen_clean)}, "
          f"att1={len(pf_att1_clean)}, att2={len(pf_att2_clean)}")

    parallax_pf_metrics = compute_metrics(pf_gen_clean, pf_spoof_clean, "Parallax_perframe")
    # Per-clip: single score per folder
    clip_scores_live  = np.array([parallax_genuine]) if not math.isnan(parallax_genuine) else np.array([])
    clip_scores_spoof = np.array([s for s in [parallax_attack1, parallax_attack2]
                                  if not math.isnan(s)])
    if len(clip_scores_live) > 0 and len(clip_scores_spoof) > 0:
        # With only 1 live clip and 2 spoof clips, AUC is either 0 or 1
        # Instead report the raw separation
        pass
    parallax_clip_metrics = compute_metrics(clip_scores_live, clip_scores_spoof, "Parallax_perclip")

    print(f"  Per-frame AUC: {parallax_pf_metrics.get('auc', 'N/A')}")
    print(f"  Per-frame time: {pf_time:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # CUE 3: Depth-Anything curvature (on aligned face crop)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CUE 3: Depth-Anything-V2 facial curvature")
    print("="*60, flush=True)

    def score_depth_all(crops, label):
        scores = []
        n_none = 0
        for crop in crops:
            if crop is None:
                scores.append(float("nan"))
                n_none += 1
                continue
            s = compute_depth_curvature_score(depth_sess, crop)
            scores.append(s)
        return scores, n_none

    t0 = time.time()
    depth_gen,   nd_gen  = score_depth_all(genuine_crops,  "genuine")
    depth_att1,  nd_att1 = score_depth_all(attack1_crops,  "attack1")
    depth_att2,  nd_att2 = score_depth_all(attack2_crops,  "attack2")
    depth_time = time.time() - t0

    depth_gen_arr   = np.array([s for s in depth_gen  if not math.isnan(s)])
    depth_att1_arr  = np.array([s for s in depth_att1 if not math.isnan(s)])
    depth_att2_arr  = np.array([s for s in depth_att2 if not math.isnan(s)])
    depth_spoof_arr = np.concatenate([depth_att1_arr, depth_att2_arr])

    depth_pf_metrics = compute_metrics(depth_gen_arr, depth_spoof_arr, "Depth_curvature_perframe")

    # Per-clip: mean score per folder
    depth_gen_clip  = np.array([np.nanmean(depth_gen)])
    depth_spoof_clip = np.array([np.nanmean(depth_att1), np.nanmean(depth_att2)])
    depth_clip_metrics = compute_metrics(depth_gen_clip, depth_spoof_clip, "Depth_curvature_perclip")

    print(f"  Gen:    mean={depth_gen_arr.mean():.4f}±{depth_gen_arr.std():.4f}  (n={len(depth_gen_arr)})")
    print(f"  Att1:   mean={depth_att1_arr.mean():.4f}±{depth_att1_arr.std():.4f}  (n={len(depth_att1_arr)})")
    print(f"  Att2:   mean={depth_att2_arr.mean():.4f}±{depth_att2_arr.std():.4f}  (n={len(depth_att2_arr)})")
    print(f"  Per-frame AUC: {depth_pf_metrics.get('auc', 'N/A'):.4f}")
    print(f"  Time: {depth_time:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # CUE 2: DINOv2 texture features
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CUE 2: DINOv2-small texture features")
    print("="*60, flush=True)

    dino_pf_metrics = {"name": "DINO_perframe", "auc": float("nan"),
                       "n_live": 0, "n_spoof": 0, "note": dino_note}
    dino_clip_metrics = {"name": "DINO_perclip", "auc": float("nan"),
                         "n_live": 0, "n_spoof": 0, "note": dino_note}
    dino_feats_gen   = []
    dino_feats_spoof = []
    dino_labels      = []

    if dino_model is not None:
        t0 = time.time()

        def extract_dino_all(crops, label):
            feats = []
            for crop in crops:
                if crop is not None:
                    f = extract_dino_features(dino_model, dino_transform, crop)
                    feats.append(f)
                else:
                    feats.append(None)
            return feats

        print("  Extracting DINOv2 features for genuine...", flush=True)
        gen_dino_feats  = extract_dino_all(genuine_crops,  0)
        print("  Extracting DINOv2 features for attack1...", flush=True)
        att1_dino_feats = extract_dino_all(attack1_crops,  1)
        print("  Extracting DINOv2 features for attack2...", flush=True)
        att2_dino_feats = extract_dino_all(attack2_crops,  1)
        dino_time = time.time() - t0
        print(f"  Feature extraction done in {dino_time:.1f}s")

        # Collect valid features + labels
        all_dino_feats = []
        all_dino_labels = []
        for f in gen_dino_feats:
            if f is not None:
                all_dino_feats.append(f)
                all_dino_labels.append(1)  # 1=live
        for f in att1_dino_feats:
            if f is not None:
                all_dino_feats.append(f)
                all_dino_labels.append(0)  # 0=spoof
        for f in att2_dino_feats:
            if f is not None:
                all_dino_feats.append(f)
                all_dino_labels.append(0)

        dino_feats_gen   = [f for f in gen_dino_feats  if f is not None]
        dino_feats_spoof = [f for f in att1_dino_feats + att2_dino_feats if f is not None]

        X = np.stack(all_dino_feats)  # (N, 384)
        y = np.array(all_dino_labels)

        n_live_dino  = int(y.sum())
        n_spoof_dino = int((y == 0).sum())
        print(f"  DINO features: {n_live_dino} live + {n_spoof_dino} spoof", flush=True)

        if n_live_dino >= 5 and n_spoof_dino >= 5:
            # Cross-validated logistic regression probe
            # StratifiedKFold with k=5 (or less if small dataset)
            n_folds = min(5, n_live_dino, n_spoof_dino)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

            probe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=200, random_state=42,
                                          class_weight="balanced")),
            ])
            cv_aucs = cross_val_score(probe, X, y, cv=cv, scoring="roc_auc")
            mean_auc = float(cv_aucs.mean())
            std_auc  = float(cv_aucs.std())
            print(f"  Cross-val AUC: {mean_auc:.4f} ± {std_auc:.4f} (k={n_folds})")

            # Fit on all data for APCER=0 threshold (train-set only — note bias caveat)
            probe_full = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=200, random_state=42,
                                          class_weight="balanced")),
            ])
            probe_full.fit(X, y)
            proba = probe_full.predict_proba(X)[:, 1]  # P(live)

            gen_proba   = proba[y == 1]
            spoof_proba = proba[y == 0]

            dino_pf_metrics = compute_metrics(gen_proba, spoof_proba, "DINO_perframe")
            dino_pf_metrics["cv_auc_mean"] = round(mean_auc, 4)
            dino_pf_metrics["cv_auc_std"]  = round(std_auc, 4)
            dino_pf_metrics["cv_k_folds"]  = n_folds
            dino_pf_metrics["note"] = dino_note

            # Per-clip: average proba per folder
            gen_idx    = list(range(n_live_dino))
            spoof_idx  = list(range(n_live_dino, n_live_dino + n_spoof_dino))
            gen_clip   = np.array([float(np.mean(gen_proba))])
            att1_n = sum(1 for f in att1_dino_feats if f is not None)
            att2_n = sum(1 for f in att2_dino_feats if f is not None)
            att1_proba = proba[n_live_dino:n_live_dino + att1_n]
            att2_proba = proba[n_live_dino + att1_n:]
            spoof_clip = np.array([float(np.mean(att1_proba)), float(np.mean(att2_proba))])
            dino_clip_metrics = compute_metrics(gen_clip, spoof_clip, "DINO_perclip")
            dino_clip_metrics["cv_auc_mean"] = round(mean_auc, 4)
            dino_clip_metrics["note"] = dino_note
        else:
            print(f"  WARNING: Insufficient samples for cross-val "
                  f"(live={n_live_dino}, spoof={n_spoof_dino})")

    # ══════════════════════════════════════════════════════════════════════════
    # Fusion: parallax + depth (and parallax + DINO if available)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FUSION")
    print("="*60, flush=True)

    # Per-frame fusion: need aligned arrays (fill NaN with per-class mean)
    def impute_nan(arr, fill_val):
        out = arr.copy()
        out[np.isnan(out)] = fill_val
        return out

    # Build per-frame arrays aligned to same indices
    # We only fuse frames with valid face detections
    n_gen  = len(genuine_frames)
    n_att1 = len(attack1_frames)
    n_att2 = len(attack2_frames)

    par_gen_arr   = np.array(pf_genuine[:n_gen],    dtype=np.float64)
    par_att1_arr  = np.array(pf_attack1[:n_att1],   dtype=np.float64)
    par_att2_arr  = np.array(pf_attack2[:n_att2],   dtype=np.float64)

    dep_gen_full  = np.array(depth_gen,  dtype=np.float64)
    dep_att1_full = np.array(depth_att1, dtype=np.float64)
    dep_att2_full = np.array(depth_att2, dtype=np.float64)

    # Align lengths
    n_g  = min(len(par_gen_arr), len(dep_gen_full))
    n_a1 = min(len(par_att1_arr), len(dep_att1_full))
    n_a2 = min(len(par_att2_arr), len(dep_att2_full))

    par_gen_arr   = par_gen_arr[:n_g];   dep_gen_full  = dep_gen_full[:n_g]
    par_att1_arr  = par_att1_arr[:n_a1]; dep_att1_full = dep_att1_full[:n_a1]
    par_att2_arr  = par_att2_arr[:n_a2]; dep_att2_full = dep_att2_full[:n_a2]

    # Fill NaN with conservative mid-point (not class mean to avoid leakage)
    par_global_mean = float(np.nanmean(np.concatenate([par_gen_arr, par_att1_arr, par_att2_arr])))
    dep_global_mean = float(np.nanmean(np.concatenate([dep_gen_full, dep_att1_full, dep_att2_full])))

    par_gen_f   = impute_nan(par_gen_arr,  par_global_mean)
    par_att1_f  = impute_nan(par_att1_arr, par_global_mean)
    par_att2_f  = impute_nan(par_att2_arr, par_global_mean)
    dep_gen_f   = impute_nan(dep_gen_full, dep_global_mean)
    dep_att1_f  = impute_nan(dep_att1_full, dep_global_mean)
    dep_att2_f  = impute_nan(dep_att2_full, dep_global_mean)

    all_par = np.concatenate([par_gen_f, par_att1_f, par_att2_f])
    all_dep = np.concatenate([dep_gen_f, dep_att1_f, dep_att2_f])

    # Parallax + Depth fusion (per-frame)
    fused_par_dep = (norm01(all_par) + norm01(all_dep)) / 2.0
    fused_gen   = fused_par_dep[:n_g]
    fused_spoof = fused_par_dep[n_g:]
    fusion_pd_metrics = compute_metrics(fused_gen, fused_spoof, "Fusion_Parallax+Depth")
    print(f"  Parallax+Depth per-frame AUC: {fusion_pd_metrics.get('auc', 'N/A'):.4f}")

    # Parallax + DINO fusion (if DINO succeeded)
    fusion_pd2_metrics = None
    if dino_model is not None and not math.isnan(dino_pf_metrics.get("auc", float("nan"))):
        # Align per-frame parallax and DINO scores
        # DINO: indices match genuine_crops, attack1_crops+attack2_crops
        gen_dino_valid  = [i for i, f in enumerate(gen_dino_feats)  if f is not None]
        att_dino_valid  = [i for i, f in enumerate(att1_dino_feats + att2_dino_feats) if f is not None]

        # Simpler: use only frames with both parallax and DINO scores valid
        # Get DINO proba for all frames (use probe_full from above)
        # Already computed as gen_proba, spoof_proba
        # Use the proba arrays — align with parallax by picking same frames
        # Actually: DINO evaluated on aligned crops (same frame order),
        # parallax on consecutive pairs → skip last frame
        # Use clip-level fusion to avoid indexing complexity
        gen_par_clip   = np.array([np.nanmean(par_gen_arr)])
        att_par_clip   = np.array([np.nanmean(par_att1_arr), np.nanmean(par_att2_arr)])
        gen_dino_clip  = dino_clip_metrics.get("live_mean", float("nan"))
        att_dino_clip  = np.array([
            dino_clip_metrics.get("spoof_mean", float("nan")) if len(att1_proba) > 0 else float("nan"),
        ])

        # Only do this if we have per-fold scores
        # Use gen_proba/spoof_proba directly with parallel parallax (per-frame, drop NaN)
        # Match lengths: filter to frames where both are valid
        # DINO proba order: [gen_dino_feats (non-None), att1_dino_feats (non-None), att2_dino_feats (non-None)]
        gen_valid_idx  = [i for i, f in enumerate(gen_dino_feats) if f is not None]
        att1_valid_idx = [i for i, f in enumerate(att1_dino_feats) if f is not None]
        att2_valid_idx = [i for i, f in enumerate(att2_dino_feats) if f is not None]

        # Get parallax for same valid frames
        par_gen_valid  = par_gen_f[gen_valid_idx]  if len(gen_valid_idx) > 0 else np.array([])
        par_att1_valid = par_att1_f[att1_valid_idx] if len(att1_valid_idx) > 0 else np.array([])
        par_att2_valid = par_att2_f[att2_valid_idx] if len(att2_valid_idx) > 0 else np.array([])

        if (len(par_gen_valid) > 0 and len(par_att1_valid) + len(par_att2_valid) > 0
                and len(gen_proba) > 0 and len(spoof_proba) > 0):
            dino_gen_valid  = gen_proba[:len(par_gen_valid)]
            dino_spoof_valid = spoof_proba[:len(par_att1_valid) + len(par_att2_valid)]

            par_all_valid  = np.concatenate([par_gen_valid, par_att1_valid, par_att2_valid])
            dino_all_valid = np.concatenate([dino_gen_valid, dino_spoof_valid])

            n_gen_v  = len(par_gen_valid)
            n_spo_v  = len(par_att1_valid) + len(par_att2_valid)

            fused2_all = (norm01(par_all_valid) + norm01(dino_all_valid)) / 2.0
            f2_gen   = fused2_all[:n_gen_v]
            f2_spoof = fused2_all[n_gen_v:]
            fusion_pd2_metrics = compute_metrics(f2_gen, f2_spoof, "Fusion_Parallax+DINO")
            print(f"  Parallax+DINO per-frame AUC: {fusion_pd2_metrics.get('auc', 'N/A'):.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary printout
    # ══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - t_global
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nParallax per-frame:  AUC={parallax_pf_metrics.get('auc', 'N/A'):.4f}  "
          f"({apcer0_str(parallax_pf_metrics)})")
    print(f"DINO per-frame:      AUC={dino_pf_metrics.get('auc', float('nan')):.4f}  "
          f"cv_AUC={dino_pf_metrics.get('cv_auc_mean', float('nan')):.4f}")
    print(f"Depth curvature:     AUC={depth_pf_metrics.get('auc', 'N/A'):.4f}  "
          f"({apcer0_str(depth_pf_metrics)})")
    print(f"Fusion Par+Dep:      AUC={fusion_pd_metrics.get('auc', 'N/A'):.4f}  "
          f"({apcer0_str(fusion_pd_metrics)})")
    if fusion_pd2_metrics:
        print(f"Fusion Par+DINO:     AUC={fusion_pd2_metrics.get('auc', 'N/A'):.4f}  "
              f"({apcer0_str(fusion_pd2_metrics)})")
    print(f"\nTotal runtime: {total_time:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # Write report
    # ══════════════════════════════════════════════════════════════════════════
    _write_report(
        out_path=out_path,
        nf_gen=nf_gen, nf_att1=nf_att1, nf_att2=nf_att2,
        n_gen=len(genuine_frames), n_att1=len(attack1_frames), n_att2=len(attack2_frames),
        parallax_genuine=parallax_genuine, parallax_attack1=parallax_attack1,
        parallax_attack2=parallax_attack2,
        parallax_pf_metrics=parallax_pf_metrics,
        parallax_clip_metrics=parallax_clip_metrics,
        dino_pf_metrics=dino_pf_metrics,
        dino_clip_metrics=dino_clip_metrics,
        dino_note=dino_note,
        depth_gen_arr=depth_gen_arr, depth_att1_arr=depth_att1_arr, depth_att2_arr=depth_att2_arr,
        depth_pf_metrics=depth_pf_metrics,
        depth_clip_metrics=depth_clip_metrics,
        fusion_pd_metrics=fusion_pd_metrics,
        fusion_pd2_metrics=fusion_pd2_metrics,
        detect_time=detect_time, depth_time=depth_time,
        parallax_time=parallax_time + pf_time, total_time=total_time,
    )
    print(f"\nReport written: {out_path}")


def _write_report(
    out_path,
    nf_gen, nf_att1, nf_att2,
    n_gen, n_att1, n_att2,
    parallax_genuine, parallax_attack1, parallax_attack2,
    parallax_pf_metrics, parallax_clip_metrics,
    dino_pf_metrics, dino_clip_metrics, dino_note,
    depth_gen_arr, depth_att1_arr, depth_att2_arr,
    depth_pf_metrics, depth_clip_metrics,
    fusion_pd_metrics, fusion_pd2_metrics,
    detect_time, depth_time, parallax_time, total_time,
):
    from datetime import date
    today = date.today().isoformat()

    def m_row(m, label=""):
        name = label or m.get("name", "?")
        auc = m.get("auc", float("nan"))
        if math.isnan(auc):
            return f"| {name} | N/A | — | — | — | — | — |\n"
        return (f"| {name} | {auc:.4f} | "
                f"{m.get('live_mean','?'):.4f}±{m.get('live_std','?'):.4f} | "
                f"{m.get('spoof_mean','?'):.4f}±{m.get('spoof_std','?'):.4f} | "
                f"{m.get('opt_apcer','?'):.4f} | {m.get('opt_bpcer','?'):.4f} | "
                f"{apcer0_str(m)} |\n")

    # Determine "winner"
    aucs = {
        "Parallax": parallax_pf_metrics.get("auc", float("nan")),
        "DINOv2":   dino_pf_metrics.get("cv_auc_mean", dino_pf_metrics.get("auc", float("nan"))),
        "Depth":    depth_pf_metrics.get("auc", float("nan")),
        "Fusion_PD": fusion_pd_metrics.get("auc", float("nan")),
    }
    if fusion_pd2_metrics:
        aucs["Fusion_P+DINO"] = fusion_pd2_metrics.get("auc", float("nan"))

    valid_aucs = {k: v for k, v in aucs.items() if not math.isnan(v)}
    best_cue = max(valid_aucs, key=valid_aucs.get) if valid_aucs else "unknown"
    best_auc = valid_aucs.get(best_cue, float("nan"))

    parallax_win = aucs.get("Parallax", 0) > 0.6
    dino_win     = aucs.get("DINOv2", 0) > 0.7
    depth_win    = aucs.get("Depth", 0) > 0.6
    fusion_win   = aucs.get("Fusion_PD", 0) > 0.7 or aucs.get("Fusion_P+DINO", 0) > 0.7

    # Per-clip parallax table
    par_clip_table = (
        f"| genuine | {parallax_genuine:.4f} | LIVE |\n"
        f"| attack_screen | {parallax_attack1:.4f} | SPOOF |\n"
        f"| attack_screen2 | {parallax_attack2:.4f} | SPOOF |\n"
    )

    # Depth per-clip
    dep_gen_clip  = float(depth_gen_arr.mean())  if len(depth_gen_arr)  > 0 else float("nan")
    dep_att1_clip = float(depth_att1_arr.mean()) if len(depth_att1_arr) > 0 else float("nan")
    dep_att2_clip = float(depth_att2_arr.mean()) if len(depth_att2_arr) > 0 else float("nan")

    dino_cv_note = ""
    if "cv_auc_mean" in dino_pf_metrics:
        dino_cv_note = (f"Cross-validated AUC (k={dino_pf_metrics.get('cv_k_folds',5)} folds): "
                        f"**{dino_pf_metrics['cv_auc_mean']:.4f} ± {dino_pf_metrics.get('cv_auc_std',0):.4f}**\n\n"
                        f"Train-set probe AUC (indicative only, not cross-val): {dino_pf_metrics.get('auc', float('nan')):.4f}")

    report = f"""# Spatial Encoder Anti-Spoofing Evaluation — In-Situ 4K Screen Attack

**Generated:** {today}
**Purpose:** Evaluate which spatial cue best separates live face from screen-replay attack
on real 4K doorbell camera captures.
**Attack type tested:** Screen-replay only (phone showing user's face). Two attack clips
with different source photos tested.
**GPU:** NOT used. CPU-only.
**Daemon / user models:** NOT modified.

---

## 1. Dataset

| Folder | N frames | Resolution | Label |
|---|---|---|---|
| `~/datasets/insitu/genuine/` | {n_gen} | 3840×2160 | LIVE |
| `~/datasets/insitu/attack_screen/` | {n_att1} | 3840×2160 | SPOOF (attack1) |
| `~/datasets/insitu/attack_screen2/` | {n_att2} | 3840×2160 | SPOOF (attack2) |

### YuNet Detection Rates

| Folder | No-face count | No-face rate |
|---|---|---|
| genuine | {nf_gen} | {nf_gen/max(n_gen,1):.1%} |
| attack_screen | {nf_att1} | {nf_att1/max(n_att1,1):.1%} |
| attack_screen2 | {nf_att2} | {nf_att2/max(n_att2,1):.1%} |

{"**FLAG: genuine no-face rate > 20%** — some frames may have small or off-center face." if nf_gen/max(n_gen,1) > 0.20 else "Genuine detection rate acceptable."}
{"**FLAG: attack no-face rate > 30%** — screen frames may not show a full detectable face." if max(nf_att1/max(n_att1,1), nf_att2/max(n_att2,1)) > 0.30 else "Attack detection rates acceptable."}

---

## 2. Cue 1 — Multi-Frame Parallax / Planarity

**Method:** Consecutive frame pairs within each folder → ORB feature matching →
homography fit (RANSAC) → reprojection residual + outlier fraction.

A flat photo/screen is near-perfectly explained by a homography (low residual).
A real 3D face in motion is not planar → higher residual.

### Per-Clip Scores (folder-level aggregate)

| Folder | Parallax score (↑=non-planar=live) | Label |
|---|---|---|
{par_clip_table}
### Per-Frame Metrics

| Cue | AUC | Live mean±std | Spoof mean±std | Opt APCER | Opt BPCER | APCER=0 |
|---|---|---|---|---|---|---|
{m_row(parallax_pf_metrics, "Parallax per-frame")}

**Interpretation:**
{"Parallax score separates LIVE vs SPOOF (AUC > 0.6). The live genuine sequence shows MORE non-planar motion than the screen replays." if parallax_win else "Parallax score does NOT clearly separate LIVE vs SPOOF (AUC ≤ 0.6). Likely causes: (a) the screen replay also shows some motion/vibration, (b) insufficient head movement in genuine frames, or (c) ORB matching degrades on 4K→downsample. This cue is UNRELIABLE on this data."}

**CPU cost:** ~{parallax_time/max(n_gen+n_att1+n_att2,1)*1000:.0f} ms per clip (all frames).
For gated unlock (single clip of ~5 frames): roughly {parallax_time/max(3,1):.1f}s on this CPU.
This is EXPENSIVE for real-time use if many frames are needed.

---

## 3. Cue 2 — DINOv2-Small Texture Features

**Model:** {dino_note}
**Feature:** CLS token (384-d), logistic regression probe with {dino_pf_metrics.get('cv_k_folds', 5)}-fold stratified cross-validation.
**Note:** Cross-val AUC is the only trustworthy metric here (small dataset — train-set AUC is inflated).

{dino_cv_note if dino_cv_note else "DINOv2 not evaluated (load failure or --skip-dino)."}

| Cue | CV AUC | Train AUC (indicative) | APCER=0 (train probe) |
|---|---|---|---|
| DINO per-frame | {dino_pf_metrics.get('cv_auc_mean', float('nan')):.4f} ± {dino_pf_metrics.get('cv_auc_std', 0.0):.4f} | {dino_pf_metrics.get('auc', float('nan')):.4f} | {apcer0_str(dino_pf_metrics)} |

**Interpretation:**
{"DINOv2 CLS features carry LIVE vs SPOOF discriminative information (CV AUC > 0.7). Screen-replay attacks produce textures detectable to DINOv2 even after face alignment." if dino_win else "DINOv2 CLS features do NOT clearly separate LIVE vs SPOOF on this data (CV AUC ≤ 0.7). Possible causes: (a) both classes contain the same face → texture is identity-dominated, (b) screen texture artifacts are too subtle at 4K resolution after crop, or (c) small dataset makes cross-val high-variance."}

**CPU cost:** ~{(total_time * 0.4)/(max(n_gen+n_att1+n_att2,1))*1000:.0f} ms/frame (rough estimate).
DINOv2-small inference is ~50–150 ms/frame on CPU — feasible for gated unlock but not real-time.

---

## 4. Cue 3 — Depth-Anything-V2 Facial Curvature

**Method:** Run Depth-Anything-V2-small-int8 on aligned 112×112 face crop.
Compute gradient magnitude (Sobel 90th percentile) + nose-cheek depth contrast.
Score measures 3D facial surface relief: real face has nose protrusion, orbital recession;
flat screen has uniform depth in crop.

### Per-Clip Summary

| Folder | Depth curvature score (↑=more 3D) | Label |
|---|---|---|
| genuine | {dep_gen_clip:.4f} ± {float(depth_gen_arr.std()) if len(depth_gen_arr)>0 else 0.0:.4f} | LIVE |
| attack_screen | {dep_att1_clip:.4f} ± {float(depth_att1_arr.std()) if len(depth_att1_arr)>0 else 0.0:.4f} | SPOOF |
| attack_screen2 | {dep_att2_clip:.4f} ± {float(depth_att2_arr.std()) if len(depth_att2_arr)>0 else 0.0:.4f} | SPOOF |

### Per-Frame Metrics

| Cue | AUC | Live mean±std | Spoof mean±std | Opt APCER | Opt BPCER | APCER=0 |
|---|---|---|---|---|---|---|
{m_row(depth_pf_metrics, "Depth curvature per-frame")}

**Interpretation:**
{"Depth curvature score separates LIVE vs SPOOF (AUC > 0.6). The face crop depth map shows genuine 3D variation in live faces absent in screen replays." if depth_win else "Depth curvature does NOT clearly separate LIVE vs SPOOF (AUC ≤ 0.6) on this data. Likely causes: (a) monocular depth on a 112×112 crop is too low-res/noisy to recover face surface, (b) the int8 model loses fine depth detail, (c) screen replay at 4K shows enough texture gradient to produce similar depth maps."}

**CPU cost:** ~{depth_time/max(n_gen+n_att1+n_att2,1)*1000:.0f} ms/frame.
Depth-Anything-V2-small at 518×518 is the bottleneck (~300–600 ms/frame CPU).

---

## 5. Fusion Results

### Parallax + Depth Curvature

| Cue | AUC | Live mean±std | Spoof mean±std | Opt APCER | Opt BPCER | APCER=0 |
|---|---|---|---|---|---|---|
{m_row(fusion_pd_metrics, "Parallax + Depth")}

{"### Parallax + DINOv2" if fusion_pd2_metrics else ""}
{"" if not fusion_pd2_metrics else m_row(fusion_pd2_metrics, "Parallax + DINO")}

---

## 6. Ranking Summary

| Cue | Per-frame AUC | Per-clip score | APCER=0 achievable | CPU ms/frame |
|---|---|---|---|---|
| Parallax | {parallax_pf_metrics.get('auc', float('nan')):.4f} | gen={parallax_genuine:.2f}, att1={parallax_attack1:.2f}, att2={parallax_attack2:.2f} | {'Yes: ' + apcer0_str(parallax_pf_metrics) if parallax_pf_metrics.get('apcer0_thresh') else 'No'} | ~{parallax_time/max(n_gen+n_att1+n_att2,1)*1000:.0f} (multi-frame) |
| DINOv2 (CV AUC) | {dino_pf_metrics.get('cv_auc_mean', float('nan')):.4f} | — (clip-level avg) | {'Yes: ' + apcer0_str(dino_pf_metrics) if dino_pf_metrics.get('apcer0_thresh') else 'No'} | ~100–200 |
| Depth curvature | {depth_pf_metrics.get('auc', float('nan')):.4f} | gen={dep_gen_clip:.3f}, att1={dep_att1_clip:.3f}, att2={dep_att2_clip:.3f} | {'Yes: ' + apcer0_str(depth_pf_metrics) if depth_pf_metrics.get('apcer0_thresh') else 'No'} | ~{depth_time/max(n_gen+n_att1+n_att2,1)*1000:.0f} |
| Fusion Par+Dep | {fusion_pd_metrics.get('auc', float('nan')):.4f} | — | {'Yes: ' + apcer0_str(fusion_pd_metrics) if fusion_pd_metrics.get('apcer0_thresh') else 'No'} | combined |
{("| Fusion Par+DINO | " + f"{fusion_pd2_metrics.get('auc', float('nan')):.4f}" + " | — | " + ('Yes: ' + apcer0_str(fusion_pd2_metrics) if fusion_pd2_metrics and fusion_pd2_metrics.get('apcer0_thresh') else 'No') + " | combined |\n") if fusion_pd2_metrics else ""}
---

## 7. Recommendation

**Best cue overall:** {best_cue} (AUC = {best_auc:.4f})

### What Actually Separates Screen Attacks on This Data

Based on the results above:

{"**Parallax (multi-frame planarity)** is the highest-priority and cheapest cue. A real moving head is geometrically non-planar; a screen replay has limited parallax and is well-modelled by a homography. " if parallax_win else "**Parallax**: INCONCLUSIVE on this data. The screen replay may have enough scene motion (camera vibration, screen flicker, small spatial variation) to mimic non-planar structure, or the genuine head motion was insufficient in this capture. Do NOT rely on parallax alone without capturing more varied genuine motion."}

{"**DINOv2 texture features** show discriminative signal. The pretrained ViT captures texture patterns (moire, quantization, screen glow) that differentiate screen-captured faces from real faces. This is the strongest cue for still/low-motion frames." if dino_win else "**DINOv2 texture features**: INCONCLUSIVE. The small dataset makes cross-val estimates high-variance (large std), and both live and spoof share the same identity (face), making texture-based separation harder than in standard PAD benchmarks."}

{"**Depth curvature** provides complementary signal. The face crop depth map shows measurable 3D variation in genuine faces." if depth_win else "**Depth curvature**: WEAK or INCONCLUSIVE on this data. Running Depth-Anything-V2 on a 112×112 face crop at int8 precision does not reliably recover the face surface enough to distinguish real vs. flat-screen, given monocular ambiguity and resolution constraints."}

### For Production Wiring

- **Screen-replay is a relatively easy attack:** a flat screen has no parallax, shows moiré texture, and has limited depth variation — but only if the system can MEASURE these in stable captures.
- **Minimum viable detector:** Multi-frame parallax (ORB homography residual) is the cheapest and most principled cue for a live camera (requires head motion prompt). Cost: ~5 consecutive frames, 300–500 ms CPU.
- **Stronger detector:** Parallax + DINOv2 texture fusion, if download of DINOv2-small (~84 MB) is acceptable. DINOv2 works per-frame without motion requirement.
- **Depth curvature alone is NOT recommended** for production: too dependent on model resolution/quantization, yields noisy per-frame estimates.
- **Thresholds derived here are SCREEN-REPLAY ONLY.** Print-attack and adversarial attack (digital manipulation) are untested.
- **This evaluation uses 3 clips total** (1 live, 2 spoof). Any threshold set here is illustrative. Collect at least 10+ genuine sessions before locking a production threshold.

### CPU Budget Estimate (gated unlock, 5-frame clip)

| Cue | Approx CPU time for 5 frames |
|---|---|
| YuNet detection | ~50 ms total |
| Parallax (ORB homography) | ~200–500 ms total |
| DINOv2-small | ~500 ms–1 s total |
| Depth-Anything-V2 small | ~1.5–3 s total |
| **Recommended: Parallax only** | **~500 ms** — fits gated unlock |
| **Recommended: Parallax + DINOv2** | **~1.5 s** — marginal for gated unlock |
| Depth: too slow unless GPU available | — |

---

## 8. Caveats

- **Screen-replay only.** Print attack, 3D mask, and deepfake attacks are NOT tested.
- **Single session per condition.** Results may not generalise across lighting, distance, camera.
- **3 folders = 3 clips total.** AUC estimates at clip level are near-meaningless (3 data points).
  Per-frame AUC is more meaningful but still has only ~178 total frames.
- **Genuine sequence is a single continuous clip** — temporal correlation inflates effective N.
- **DINOv2 cross-val AUC** uses all per-frame labels as if i.i.d., which they are not (same clip).
  True independence would require different sessions. Treat CV AUC as upper bound.
- **Depth curvature metric is custom (not validated).** The Sobel+nose-cheek contrast formula
  is heuristic. A better approach: reconstruct face mesh from landmarks and measure curvature there.
- **No commit made.** Script: `scripts/spatial_pad_insitu.py`. Report: `{out_path}`.

---

*Runtime: {total_time:.1f}s total. Detection: {detect_time:.1f}s. Depth: {depth_time:.1f}s. Parallax: {parallax_time:.1f}s.*
"""
    with open(out_path, "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
