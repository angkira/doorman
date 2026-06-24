#!/usr/bin/env python3
"""Offline Doorman R&D dataset/probe runner.

This script intentionally runs outside the daemon. It never touches the live
gallery and never promotes a model. It reads opt-in capture packs plus the JSONL
attempt/label logs, writes an experiment report, and can run a simple image
feature baseline when enough labeled captures exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


POSITIVE = {"ConfirmedGenuine", "FalseNegative", "ProactiveUnlocked"}
NEGATIVE = {"Rejected", "SuspectedSpoof"}


@dataclass
class FrameSample:
    attempt_id: str
    label: str
    image_path: Path
    manifest_path: Path
    source_index: int
    bbox_norm: list[float] | None
    sharpness: float | None
    face_area_frac: float | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(dataset_dir: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for path in sorted(dataset_dir.glob("labels-*.jsonl")):
        for row in read_jsonl(path):
            labels[row["attempt_id"]] = row["outcome"]
    return labels


def load_attempt_outcomes(dataset_dir: Path) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for path in sorted(dataset_dir.glob("attempts-*.jsonl")):
        for row in read_jsonl(path):
            outcomes[row["attempt_id"]] = row["outcome"]
    return outcomes


def collect_samples(dataset_dir: Path) -> list[FrameSample]:
    labels = load_labels(dataset_dir)
    attempts = load_attempt_outcomes(dataset_dir)
    samples: list[FrameSample] = []
    for manifest_path in sorted(dataset_dir.glob("captures/*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        attempt_id = manifest["attempt_id"]
        label = labels.get(attempt_id, manifest.get("outcome") or attempts.get(attempt_id, "Unknown"))
        for frame in manifest.get("frames", []):
            image_path = manifest_path.parent / frame["filename"]
            if image_path.exists():
                conditions = frame.get("conditions") or {}
                samples.append(
                    FrameSample(
                        attempt_id=attempt_id,
                        label=label,
                        image_path=image_path,
                        manifest_path=manifest_path,
                        source_index=int(frame.get("source_index", 0)),
                        bbox_norm=frame.get("bbox_norm"),
                        sharpness=conditions.get("sharpness"),
                        face_area_frac=conditions.get("face_area_frac"),
                    )
                )
    return samples


def inventory_report(dataset_dir: Path) -> dict[str, Any]:
    attempts = []
    labels = []
    for path in sorted(dataset_dir.glob("attempts-*.jsonl")):
        attempts.extend(read_jsonl(path))
    for path in sorted(dataset_dir.glob("labels-*.jsonl")):
        labels.extend(read_jsonl(path))
    samples = collect_samples(dataset_dir)
    return {
        "schema_version": 1,
        "dataset_dir": str(dataset_dir),
        "attempts": len(attempts),
        "labels": len(labels),
        "capture_frames": len(samples),
        "capture_attempts": len({s.attempt_id for s in samples}),
        "attempt_outcomes": dict(Counter(row.get("outcome", "Unknown") for row in attempts)),
        "label_outcomes": dict(Counter(row.get("outcome", "Unknown") for row in labels)),
        "capture_labels": dict(Counter(s.label for s in samples)),
    }


def extract_timm_features(samples: list[FrameSample], model_name: str, batch_size: int) -> np.ndarray:
    from PIL import Image
    import timm
    import torch
    from timm.data import create_transform, resolve_model_data_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    transform = create_transform(**resolve_model_data_config(model))
    feats: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            imgs = []
            for sample in batch:
                img = Image.open(sample.image_path).convert("RGB")
                imgs.append(transform(img))
            x = torch.stack(imgs).to(device)
            y = model(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            y = y.flatten(1).detach().cpu().float().numpy()
            feats.append(y)
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 0), dtype=np.float32)


def _install_da3_optional_dependency_stubs() -> None:
    """DA3 imports optional exporters/pose tools even when depth-only inference does not use them."""

    export_mod = types.ModuleType("depth_anything_3.utils.export")
    export_mod.export = lambda *args, **kwargs: None
    sys.modules.setdefault("depth_anything_3.utils.export", export_mod)

    pose_mod = types.ModuleType("depth_anything_3.utils.pose_align")
    pose_mod.align_poses_umeyama = lambda pred, target: pred
    sys.modules.setdefault("depth_anything_3.utils.pose_align", pose_mod)


def _bbox_to_slice(bbox_norm: list[float] | None, h: int, w: int) -> tuple[slice, slice]:
    if not bbox_norm or len(bbox_norm) != 4:
        return slice(0, h), slice(0, w)
    x, y, bw, bh = [float(v) for v in bbox_norm]
    x0 = max(0, min(w - 1, int(round(x * w))))
    y0 = max(0, min(h - 1, int(round(y * h))))
    x1 = max(x0 + 1, min(w, int(round((x + bw) * w))))
    y1 = max(y0 + 1, min(h, int(round((y + bh) * h))))
    return slice(y0, y1), slice(x0, x1)


def _depth_fingerprint(depth: np.ndarray, conf: np.ndarray | None, sample: FrameSample, patch_size: int) -> np.ndarray:
    from PIL import Image

    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros((patch_size * patch_size + 12,), dtype=np.float32)

    valid_depth = depth[finite]
    d_min = float(valid_depth.min())
    d_range = float(valid_depth.max() - d_min)
    if d_range < 1e-6:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = (depth - d_min) / d_range
    depth_norm = np.nan_to_num(depth_norm, nan=0.0, posinf=1.0, neginf=0.0)

    ys, xs = _bbox_to_slice(sample.bbox_norm, depth.shape[0], depth.shape[1])
    face = depth_norm[ys, xs]
    if face.size == 0:
        face = depth_norm

    patch_img = Image.fromarray(np.clip(face * 255.0, 0, 255).astype(np.uint8))
    patch = np.asarray(
        patch_img.resize((patch_size, patch_size), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    ).reshape(-1) / 255.0

    face_conf = np.asarray(conf[ys, xs], dtype=np.float32) if conf is not None else np.empty((0,), dtype=np.float32)
    all_conf = np.asarray(conf, dtype=np.float32) if conf is not None else np.empty((0,), dtype=np.float32)
    stats = np.asarray(
        [
            float(face.mean()),
            float(face.std()),
            float(np.percentile(face, 10)),
            float(np.percentile(face, 50)),
            float(np.percentile(face, 90)),
            float(depth_norm.mean()),
            float(depth_norm.std()),
            float(face.std() / (depth_norm.max() - depth_norm.min() + 1e-6)),
            float(face_conf.mean()) if face_conf.size else 0.0,
            float(all_conf.mean()) if all_conf.size else 0.0,
            float(sample.sharpness or 0.0),
            float(sample.face_area_frac or 0.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, patch], axis=0)


def extract_da3_depth_features(
    samples: list[FrameSample],
    model_dir: str,
    process_res: int,
    patch_size: int,
) -> np.ndarray:
    from PIL import Image
    import torch

    _install_da3_optional_dependency_stubs()
    from depth_anything_3.api import DepthAnything3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(model_dir).to(device).eval()

    features_by_key: dict[tuple[Path, int], np.ndarray] = {}
    grouped: dict[Path, list[FrameSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.manifest_path, []).append(sample)

    for manifest_path in sorted(grouped):
        group = sorted(grouped[manifest_path], key=lambda s: s.source_index)
        images = [Image.open(sample.image_path).convert("RGB") for sample in group]
        prediction = model.inference(
            images,
            process_res=process_res,
            export_dir=None,
            export_format="mini_npz",
        )
        depth = np.asarray(prediction.depth, dtype=np.float32)
        conf = np.asarray(prediction.conf, dtype=np.float32) if hasattr(prediction, "conf") else None
        for idx, sample in enumerate(group):
            sample_conf = conf[idx] if conf is not None else None
            features_by_key[(sample.manifest_path, sample.source_index)] = _depth_fingerprint(
                depth[idx],
                sample_conf,
                sample,
                patch_size,
            )

    return np.stack([features_by_key[(s.manifest_path, s.source_index)] for s in samples], axis=0)


def run_linear_probe(features: np.ndarray, labels: list[int]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(labels, dtype=np.int64)
    counts = Counter(y.tolist())
    min_class = min(counts.values()) if counts else 0
    if len(counts) < 2 or min_class < 2:
        return {
            "status": "not_enough_classes",
            "class_counts": {str(k): v for k, v in counts.items()},
        }
    folds = min(5, min_class)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    prob = cross_val_predict(clf, features, y, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "status": "ok",
        "folds": folds,
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "class_counts": {str(k): v for k, v in counts.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/data/doorman-dataset")
    parser.add_argument("--out-dir", default="/models/rnd")
    parser.add_argument("--encoder", default="inventory", help="inventory, da3-base, or timm:<model-name>")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--da3-model-dir", default="/mnt/datasets/models/depth-anything/DA3-BASE")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-patch-size", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=2000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = inventory_report(dataset_dir)
    report["encoder"] = args.encoder

    samples = collect_samples(dataset_dir)
    if args.max_frames > 0:
        samples = samples[: args.max_frames]

    if args.encoder != "inventory":
        usable = [s for s in samples if s.label in POSITIVE or s.label in NEGATIVE]
        labels = [1 if s.label in POSITIVE else 0 for s in usable]
        report["used_frames"] = len(usable)
        if not usable:
            report["probe"] = {"status": "no_labeled_capture_frames"}
        elif args.encoder in {"da3-base", "depthanything3-base", "depth-anything-3-base"}:
            features = extract_da3_depth_features(
                usable,
                args.da3_model_dir,
                args.da3_process_res,
                args.da3_patch_size,
            )
            np.save(out_dir / "features.npy", features)
            (out_dir / "samples.json").write_text(
                json.dumps(
                    [
                        s.__dict__ | {"image_path": str(s.image_path), "manifest_path": str(s.manifest_path)}
                        for s in usable
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
            report["model_dir"] = args.da3_model_dir
            report["feature_kind"] = "da3_depth_face_patch_and_stats"
            report["feature_shape"] = list(features.shape)
            report["probe"] = run_linear_probe(features, labels)
        elif args.encoder.startswith("timm:"):
            model_name = args.encoder.split(":", 1)[1]
            features = extract_timm_features(usable, model_name, args.batch_size)
            np.save(out_dir / "features.npy", features)
            (out_dir / "samples.json").write_text(
                json.dumps([s.__dict__ | {"image_path": str(s.image_path), "manifest_path": str(s.manifest_path)} for s in usable], indent=2),
                encoding="utf-8",
            )
            report["feature_shape"] = list(features.shape)
            report["probe"] = run_linear_probe(features, labels)
        else:
            report["probe"] = {"status": "unsupported_encoder", "encoder": args.encoder}

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
