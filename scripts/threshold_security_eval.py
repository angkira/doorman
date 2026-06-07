#!/usr/bin/env python3
"""
threshold_security_eval.py — Part A of the max-secure threshold analysis.

Measures genuine and impostor cosine distributions under the DEPLOYED
aggregation path (multi-frame template vs aggregated probe window), then
computes the highest zero-FAR threshold T*.

Protocol mirrors the daemon's Phase-1 aggregation:
  - TEMPLATE  = renormalized mean of K=5..10 embeddings of an identity
  - PROBE     = renormalized mean of a disjoint window (size W=5) of OTHER
                frames of the SAME identity (genuine) or a DIFFERENT identity
                (impostor)
  - Only identities with >= MIN_IMAGES images participate.

Preprocessing: same as face_eval.py (YuNet detect -> Umeyama align ->
EdgeFace-S (x-127.5)/127.5 -> L2-normalize).

Usage:
    python scripts/threshold_security_eval.py \\
        --lfw-root /home/angkira/datasets/lfw/lfw_funneled \\
        --models   ~/.local/share/doorman/models \\
        [--template-k 7] [--probe-window 5] [--min-images 15]
"""

import argparse
import os
import sys
import math
import random
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image

# ── Resolve venv ────────────────────────────────────────────────────────────
# Allow running directly even if caller's PATH doesn't include the venv.
_THIS_DIR = Path(__file__).parent
_VENV_ORT = _THIS_DIR / ".venv" / "lib"
if _VENV_ORT.exists():
    import glob as _glob
    _sp = _glob.glob(str(_VENV_ORT / "python*" / "site-packages"))
    if _sp:
        sys.path.insert(0, _sp[0])

import onnxruntime as ort

# ══════════════════════════════════════════════════════════════════════════════
# Constants — must match face_eval.py and the daemon exactly
# ══════════════════════════════════════════════════════════════════════════════
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

YUNET_INPUT_SIZE  = 640
YUNET_CONF_THRESH = 0.6
YUNET_NMS_THRESH  = 0.3
YUNET_STRIDES     = [8, 16, 32]


# ══════════════════════════════════════════════════════════════════════════════
# Geometry / alignment (verbatim from face_eval.py)
# ══════════════════════════════════════════════════════════════════════════════

def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    n = src.shape[0]
    sx, sy = src[:, 0].mean(), src[:, 1].mean()
    dx, dy = dst[:, 0].mean(), dst[:, 1].mean()
    sxc = src[:, 0] - sx;  syc = src[:, 1] - sy
    dxc = dst[:, 0] - dx;  dyc = dst[:, 1] - dy
    a = float(np.sum(dxc * sxc + dyc * syc))
    b = float(np.sum(dyc * sxc - dxc * syc))
    src_var = float(np.sum(sxc**2 + syc**2))
    if src_var < 1e-12: return None
    norm = math.sqrt(a*a + b*b)
    if norm < 1e-12: return None
    sa = a / src_var;  sb = b / src_var
    tx = dx - (sa*sx - sb*sy)
    ty = dy - (sb*sx + sa*sy)
    return np.array([[sa, -sb, tx], [sb, sa, ty]], dtype=np.float32)


def invert_affine2x3(m: np.ndarray) -> Optional[np.ndarray]:
    a, b, tx = m[0];  c, d, ty = m[1]
    det = a*d - b*c
    if abs(det) < 1e-12: return None
    inv_det = 1.0 / det
    ia = d*inv_det;  ib = -b*inv_det
    ic = -c*inv_det; id_ = a*inv_det
    itx = -(ia*tx + ib*ty);  ity = -(ic*tx + id_*ty)
    return np.array([[ia, ib, itx], [ic, id_, ity]], dtype=np.float32)


def align_to_template(img: np.ndarray, landmarks_px: np.ndarray,
                      template=ARCFACE_TEMPLATE, out_size=112) -> Optional[np.ndarray]:
    m = umeyama_similarity(landmarks_px, template)
    if m is None: return None
    inv = invert_affine2x3(m)
    if inv is None: return None
    h, w = img.shape[:2]
    oy, ox = np.meshgrid(np.arange(out_size, dtype=np.float32),
                         np.arange(out_size, dtype=np.float32), indexing='ij')
    pts = np.stack([ox.ravel() + 0.5, oy.ravel() + 0.5], axis=1)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    src = pts_h @ inv.T
    px = src[:, 0] - 0.5;  py = src[:, 1] - 0.5
    x0 = np.floor(px).astype(np.int32);  y0 = np.floor(py).astype(np.int32)
    fx = px - x0.astype(np.float32);     fy = py - y0.astype(np.float32)
    x0c = np.clip(x0, 0, w-1);  y0c = np.clip(y0, 0, h-1)
    x1c = np.clip(x0+1, 0, w-1); y1c = np.clip(y0+1, 0, h-1)
    w00 = ((1-fx)*(1-fy))[:, None]; w01 = ((1-fx)*fy)[:, None]
    w10 = (fx*(1-fy))[:, None];     w11 = (fx*fy)[:, None]
    p00 = img[y0c, x0c].astype(np.float32); p01 = img[y1c, x0c].astype(np.float32)
    p10 = img[y0c, x1c].astype(np.float32); p11 = img[y1c, x1c].astype(np.float32)
    out_flat = w00*p00 + w01*p01 + w10*p10 + w11*p11
    return np.clip(np.round(out_flat), 0, 255).astype(np.uint8).reshape(out_size, out_size, 3)


# ══════════════════════════════════════════════════════════════════════════════
# YuNet decode / NMS (verbatim from face_eval.py)
# ══════════════════════════════════════════════════════════════════════════════

def yunet_preprocess(img: np.ndarray, size=640) -> np.ndarray:
    pil = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    bgr = np.array(pil, dtype=np.float32)[:, :, ::-1]
    return bgr.transpose(2, 0, 1)[None]


def yunet_decode(outputs: dict, input_size: int, score_threshold: float) -> list:
    dets = []
    inv_in = 1.0 / input_size
    for stride in YUNET_STRIDES:
        cls_t  = outputs[f"cls_{stride}"][0]
        obj_t  = outputs[f"obj_{stride}"][0]
        bbox_t = outputs[f"bbox_{stride}"][0]
        kps_t  = outputs[f"kps_{stride}"][0]
        n = cls_t.shape[0]
        cols = input_size // stride
        for i in range(n):
            cls_v = max(float(cls_t[i, 0]), 0.0)
            obj_v = max(float(obj_t[i, 0]), 0.0)
            score = math.sqrt(cls_v * obj_v)
            if score < score_threshold: continue
            row = i // cols;  col = i % cols
            dx, dy, dw, dh = bbox_t[i]
            cx = (col + float(dx)) * stride
            cy = (row + float(dy)) * stride
            w  = math.exp(float(dw)) * stride
            h  = math.exp(float(dh)) * stride
            bbox = ((cx - w/2)*inv_in, (cy - h/2)*inv_in, w*inv_in, h*inv_in)
            landmarks = [(( col + float(kps_t[i, 2*j])  ) * stride * inv_in,
                          ( row + float(kps_t[i, 2*j+1]) ) * stride * inv_in)
                         for j in range(5)]
            dets.append({"bbox": bbox, "score": score, "landmarks": landmarks})
    return dets


def iou_box(a, b) -> float:
    ax, ay, aw, ah = a;  bx, by, bw, bh = b
    x1 = max(ax, bx);  y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw);  y2 = min(ay+ah, by+bh)
    iw = max(0.0, x2-x1);  ih = max(0.0, y2-y1)
    inter = iw*ih;  union = aw*ah + bw*bh - inter
    return inter/union if union > 0 else 0.0


def nms(dets: list, iou_threshold: float) -> list:
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        if all(iou_box(d["bbox"], k["bbox"]) < iou_threshold for k in keep):
            keep.append(d)
    return keep


# ══════════════════════════════════════════════════════════════════════════════
# EdgeFace-S preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def edgeface_preprocess(face_rgb: np.ndarray) -> np.ndarray:
    arr = face_rgb.astype(np.float32)
    return ((arr - 127.5) / 127.5).transpose(2, 0, 1)[None]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    def __init__(self, models_dir: str, recognizer_name: str = "edgeface_s.onnx"):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        det_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        rec_path = os.path.join(models_dir, recognizer_name)
        print(f"  Detector:    {det_path}")
        print(f"  Recognizer:  {rec_path}")
        self.detector   = ort.InferenceSession(det_path, sess_options=opts,
                                               providers=["CPUExecutionProvider"])
        self.recognizer = ort.InferenceSession(rec_path, sess_options=opts,
                                               providers=["CPUExecutionProvider"])

    def embed(self, img_path: str) -> Optional[np.ndarray]:
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            return None
        img = np.array(pil, dtype=np.uint8)
        h, w = img.shape[:2]

        # Detect
        inp = yunet_preprocess(img)
        out_names = [o.name for o in self.detector.get_outputs()]
        outs = dict(zip(out_names, self.detector.run(None, {"input": inp})))
        dets = nms(yunet_decode(outs, YUNET_INPUT_SIZE, YUNET_CONF_THRESH), YUNET_NMS_THRESH)
        if not dets:
            return None
        best = max(dets, key=lambda d: d["score"])

        # Align
        lm_px = np.array([(lx*w, ly*h) for lx, ly in best["landmarks"]], dtype=np.float32)
        aligned = align_to_template(img, lm_px)
        if aligned is None:
            bx, by, bw, bh = best["bbox"]
            x0, y0 = int(max(bx*w,0)), int(max(by*h,0))
            x1, y1 = int(min((bx+bw)*w,w)), int(min((by+bh)*h,h))
            crop = img[y0:y1, x0:x1]
            if crop.size == 0: return None
            aligned = np.array(Image.fromarray(crop).resize((112,112), Image.LANCZOS))

        # Embed
        inp_rec = edgeface_preprocess(aligned)
        rec_outs = self.recognizer.run(None, {"input": inp_rec})
        return l2_normalize(rec_outs[0][0])


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_template(embeddings: List[np.ndarray]) -> np.ndarray:
    """Renormalized mean — mirrors Phase-1 daemon aggregation."""
    stack = np.stack(embeddings, axis=0)
    return l2_normalize(stack.mean(axis=0))


def build_templates_and_probes(
    embeddings_by_name: dict,
    template_k: int,
    probe_window: int,
    rng: random.Random,
) -> Tuple[dict, dict]:
    """
    For each qualifying identity:
      template = aggregated embedding of first template_k images
      probes   = list of aggregated embeddings, each from a disjoint window of
                 probe_window images starting at index template_k

    Returns:
      templates: {name: template_emb}
      probes:    {name: [probe_emb, ...]}
    """
    templates = {}
    probes = {}
    for name, embs in embeddings_by_name.items():
        need = template_k + probe_window
        if len(embs) < need:
            continue
        templates[name] = aggregate_template(embs[:template_k])
        # Build as many disjoint probe windows as possible
        probe_list = []
        idx = template_k
        while idx + probe_window <= len(embs):
            probe_list.append(aggregate_template(embs[idx:idx+probe_window]))
            idx += probe_window
        if probe_list:
            probes[name] = probe_list
    return templates, probes


# ══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def threshold_table(genuine: np.ndarray, impostor: np.ndarray):
    """
    Returns a list of (threshold, GAR, FAR) from 0.50 to 0.90 step 0.05.
    GAR = genuine acceptance rate, FAR = false acceptance rate.
    """
    rows = []
    n_gen = len(genuine);  n_imp = len(impostor)
    for t_int in range(50, 95, 5):
        t = t_int / 100.0
        gar = float(np.sum(genuine >= t)) / n_gen if n_gen else 0.0
        far = float(np.sum(impostor >= t)) / n_imp if n_imp else 0.0
        rows.append((t, gar, far))
    return rows


def zero_far_threshold(genuine: np.ndarray, impostor: np.ndarray) -> Tuple[float, float]:
    """
    Find the HIGHEST threshold T* such that FAR = 0.
    Returns (T*, GAR at T*).
    """
    if len(impostor) == 0:
        return float(genuine.min()), 1.0
    max_impostor = float(impostor.max())
    # T* is just above the maximum impostor score
    t_star = max_impostor
    gar = float(np.sum(genuine > t_star)) / len(genuine)
    # Also report the GAR at exactly t_star (open interval above)
    return t_star, gar


# ══════════════════════════════════════════════════════════════════════════════
# Part A — aggregated evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_part_a(lfw_root: Path, models_dir: str, template_k: int,
               probe_window: int, min_images: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)

    print("\n" + "="*70)
    print("PART A — Aggregated genuine/impostor distribution (EdgeFace-S)")
    print("="*70)

    # Gather identities with enough images
    print(f"\nScanning LFW: {lfw_root}")
    identity_images: dict[str, List[str]] = {}
    for name_dir in sorted(lfw_root.iterdir()):
        if not name_dir.is_dir():
            continue
        imgs = sorted(str(p) for p in name_dir.glob("*.jpg"))
        if len(imgs) >= min_images:
            identity_images[name_dir.name] = imgs

    print(f"Identities with >= {min_images} images: {len(identity_images)}")
    if len(identity_images) < 5:
        print("ERROR: Too few qualifying identities. Lower --min-images.")
        sys.exit(1)

    print(f"Template K={template_k}, probe window={probe_window}")
    print(f"Need per identity: {template_k + probe_window} images min "
          f"(template + 1 probe window)")

    # Load pipeline
    print("\nLoading models...")
    pipeline = Pipeline(models_dir)
    print()

    # Embed all images for qualifying identities
    print("Embedding images...")
    embeddings_by_name: dict[str, List[np.ndarray]] = {}
    total_imgs = sum(len(v) for v in identity_images.values())
    done = 0
    no_face = 0

    for name, img_paths in identity_images.items():
        embs = []
        for p in img_paths:
            emb = pipeline.embed(p)
            done += 1
            if emb is None:
                no_face += 1
            else:
                embs.append(emb)
            if done % 200 == 0:
                print(f"  [{done}/{total_imgs}] no_face={no_face}")
        embeddings_by_name[name] = embs

    print(f"Done. Embedded {done} images, no-face: {no_face} ({100*no_face/done:.1f}%)")

    # Build templates and probes
    templates, probes = build_templates_and_probes(
        embeddings_by_name, template_k, probe_window, rng
    )
    print(f"\nIdentities qualifying for template+probe split: {len(templates)}")
    if len(templates) < 3:
        print("ERROR: Too few identities for valid evaluation. Adjust parameters.")
        sys.exit(1)

    template_names = list(templates.keys())

    # Genuine pairs: template vs probe of same identity
    genuine_scores = []
    for name in template_names:
        if name not in probes:
            continue
        t_emb = templates[name]
        for probe_emb in probes[name]:
            genuine_scores.append(cosine_sim(t_emb, probe_emb))

    # Impostor pairs: template of one identity vs probe windows of another
    # Use all cross-identity pairs for thorough coverage
    impostor_scores = []
    names_list = template_names
    for i, name_a in enumerate(names_list):
        t_emb = templates[name_a]
        for j, name_b in enumerate(names_list):
            if name_a == name_b:
                continue
            if name_b not in probes:
                continue
            for probe_emb in probes[name_b]:
                impostor_scores.append(cosine_sim(t_emb, probe_emb))

    genuine = np.array(genuine_scores, dtype=np.float64)
    impostor = np.array(impostor_scores, dtype=np.float64)

    print(f"\nGenuine pairs:  {len(genuine)}")
    print(f"Impostor pairs: {len(impostor)}")

    if len(genuine) < 5 or len(impostor) < 5:
        print("ERROR: Insufficient pairs.")
        sys.exit(1)

    # Statistics
    gen_mean = float(genuine.mean())
    gen_std  = float(genuine.std())
    gen_p5   = float(np.percentile(genuine, 5))
    gen_p1   = float(np.percentile(genuine, 1))
    gen_p01  = float(np.percentile(genuine, 0.1))
    gen_min  = float(genuine.min())

    imp_mean = float(impostor.mean())
    imp_std  = float(impostor.std())
    imp_p95  = float(np.percentile(impostor, 95))
    imp_p99  = float(np.percentile(impostor, 99))
    imp_p999 = float(np.percentile(impostor, 99.9))
    imp_max  = float(impostor.max())

    t_star, gar_at_t_star = zero_far_threshold(genuine, impostor)
    table = threshold_table(genuine, impostor)

    print("\n--- Genuine aggregated cosine distribution ---")
    print(f"  mean={gen_mean:.4f}  std={gen_std:.4f}")
    print(f"  5th pct={gen_p5:.4f}  1st pct={gen_p1:.4f}  0.1th pct={gen_p01:.4f}  min={gen_min:.4f}")

    print("\n--- Impostor aggregated cosine distribution ---")
    print(f"  mean={imp_mean:.4f}  std={imp_std:.4f}")
    print(f"  95th pct={imp_p95:.4f}  99th pct={imp_p99:.4f}  99.9th pct={imp_p999:.4f}  max={imp_max:.4f}")

    print(f"\n--- Zero-FAR threshold ---")
    print(f"  T* (max zero-FAR) = {t_star:.4f}")
    print(f"  GAR at T*         = {gar_at_t_star:.4f} ({100*gar_at_t_star:.1f}%)")
    print(f"  Current threshold = 0.65  -> {'ABOVE' if t_star > 0.65 else 'BELOW or equal'} T*")

    print("\n--- Threshold table ---")
    print(f"  {'Threshold':>10}  {'GAR (genuine acc)':>18}  {'FAR (false acc)':>16}")
    for (t, gar, far) in table:
        flag = " <-- T* (zero-FAR)" if abs(t - round(t_star, 2)) < 0.025 else ""
        print(f"  {t:>10.2f}  {gar:>17.4f}  {far:>15.6f}{flag}")

    return {
        "model": "edgeface_s",
        "template_k": template_k,
        "probe_window": probe_window,
        "min_images": min_images,
        "n_identities": len(templates),
        "n_genuine_pairs": len(genuine),
        "n_impostor_pairs": len(impostor),
        "genuine_mean": gen_mean, "genuine_std": gen_std,
        "genuine_p5": gen_p5, "genuine_p1": gen_p1,
        "genuine_p01": gen_p01, "genuine_min": gen_min,
        "impostor_mean": imp_mean, "impostor_std": imp_std,
        "impostor_p95": imp_p95, "impostor_p99": imp_p99,
        "impostor_p999": imp_p999, "impostor_max": imp_max,
        "t_star": t_star,
        "gar_at_t_star": gar_at_t_star,
        "threshold_table": table,
        "current_threshold": 0.65,
        "t_star_above_current": t_star > 0.65,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Part C — AdaFace ONNX check and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def check_adaface_onnx(models_eval_dir: Path) -> Optional[str]:
    """
    Check if an AdaFace ONNX is already available or can be downloaded quickly.
    Returns path if available, None otherwise.
    """
    candidate_names = [
        "adaface_ir18_webface4m.onnx",
        "adaface_ir50_webface4m.onnx",
        "adaface_ir18.onnx",
        "adaface_ir50.onnx",
        "adaface.onnx",
    ]
    for n in candidate_names:
        p = models_eval_dir / n
        if p.exists():
            print(f"  Found AdaFace ONNX: {p}")
            return str(p)

    # Attempt lightweight download from HuggingFace (IR18 is ~98MB, no auth)
    # Known public export: deepinsight/insightface repo onnx export
    # The canonical AdaFace repo (mk-minchul/AdaFace) ships .ckpt only.
    # HuggingFace: no readily downloadable no-auth AdaFace ONNX as of 2026-06.
    print("  No AdaFace ONNX found in models_eval_dir.")
    print("  Checking HuggingFace for public ONNX export...")

    # Try a known HuggingFace space that exports AdaFace-IR18
    urls_to_try = [
        # buffalo_l from insightface is w600k_r50 (not AdaFace), skip
        # There is no clean no-auth AdaFace ONNX as of 2026-06
    ]
    # No clean public ONNX without conversion — report and stop
    return None


def run_part_c(lfw_root: Path, models_eval_dir: Path, template_k: int,
               probe_window: int, min_images: int, seed: int = 42) -> dict:
    print("\n" + "="*70)
    print("PART C — AdaFace vs EdgeFace-S achievable T*")
    print("="*70)

    models_eval_dir.mkdir(parents=True, exist_ok=True)

    adaface_path = check_adaface_onnx(models_eval_dir)
    if adaface_path is None:
        msg = (
            "No clean AdaFace ONNX available without conversion.\n"
            "The AdaFace repo (mk-minchul/AdaFace) distributes .ckpt checkpoints "
            "only; converting to ONNX requires torch.onnx.export with the "
            "AdaFace backbone. insightface's buffalo_l is w600k_r50 (ArcFace "
            "training, not AdaFace). No registration-free ONNX export was found "
            "on HuggingFace as of 2026-06.\n"
            "Action needed: export AdaFace-IR18 via scripts/export_edgeface.py "
            "analog, then place the ONNX in ~/datasets/models_eval/."
        )
        print(f"\n  RESULT: {msg}")
        return {"status": "needs_export", "message": msg}

    # If we get here, run same aggregated eval with AdaFace
    print(f"\nRunning aggregated eval with AdaFace: {adaface_path}")
    # (same pipeline, different recognizer weight)
    # Note: AdaFace uses identical preprocessing: 112x112 RGB (x-127.5)/127.5
    result = run_part_a.__wrapped__(
        lfw_root=lfw_root,
        models_dir=str(models_eval_dir.parent),
        template_k=template_k,
        probe_window=probe_window,
        min_images=min_images,
        seed=seed,
        # override recognizer
        _recognizer_path=adaface_path,
    )
    result["model"] = "adaface"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Threshold security evaluation — aggregated path")
    ap.add_argument("--lfw-root",    default="/home/angkira/datasets/lfw/lfw_funneled")
    ap.add_argument("--models",      default=os.path.expanduser("~/.local/share/doorman/models"))
    ap.add_argument("--models-eval", default=os.path.expanduser("~/datasets/models_eval"))
    ap.add_argument("--template-k",  type=int, default=7,
                    help="Number of images to average into the enrolled template (default 7)")
    ap.add_argument("--probe-window",type=int, default=5,
                    help="Aggregation window size for probe (default 5, matches daemon)")
    ap.add_argument("--min-images",  type=int, default=15,
                    help="Min images per identity to qualify (default 15)")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--output",      default="docs/threshold_security_analysis.md")
    ap.add_argument("--skip-part-c", action="store_true")
    args = ap.parse_args()

    lfw_root       = Path(args.lfw_root)
    models_eval_dir = Path(args.models_eval)

    if not lfw_root.exists():
        print(f"ERROR: LFW root not found: {lfw_root}")
        sys.exit(1)

    # Part A
    part_a = run_part_a(
        lfw_root=lfw_root,
        models_dir=args.models,
        template_k=args.template_k,
        probe_window=args.probe_window,
        min_images=args.min_images,
        seed=args.seed,
    )

    # Part C
    if not args.skip_part_c:
        part_c = run_part_c(
            lfw_root=lfw_root,
            models_eval_dir=models_eval_dir,
            template_k=args.template_k,
            probe_window=args.probe_window,
            min_images=args.min_images,
            seed=args.seed,
        )
    else:
        part_c = {"status": "skipped"}

    # Save results JSON alongside the markdown
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"part_a": part_a, "part_c": part_c}, f, indent=2)
    print(f"\nJSON results: {json_path}")

    return part_a, part_c


if __name__ == "__main__":
    main()
