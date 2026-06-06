# ONNX Models Guide for doorman

doorman runs three ONNX models in-process via ONNX Runtime (the `ort` crate).
They are not bundled in git (the `.onnx` weights are gitignored); fetch them with
the provided script.

## Required models

Written to both the repo copy `data/models/` and your runtime models dir
(`$XDG_DATA_HOME/doorman/models`, i.e. `~/.local/share/doorman/models` for
`--user`; `/var/lib/doorman/models` for the system service):

| Role        | File                                   | Model            | Input        | Output        | License |
|-------------|----------------------------------------|------------------|--------------|---------------|---------|
| Detection   | `face_detection_yunet_2023mar.onnx`    | YuNet (2023mar)  | 640×640 BGR  | boxes + 5 landmarks | **MIT** |
| Recognition | `edgeface_s.onnx`                      | EdgeFace-S (γ=0.5) | 112×112 RGB | 512-d embedding | **CC-BY-NC-SA 4.0 (non-commercial)** |
| Liveness    | `minifasnet_v2se.onnx`                 | MiniFASNetV2-SE  | 128×128 RGB  | `[1,2]` real/spoof logits | **Apache-2.0** |

An optional INT8 liveness variant (`minifasnet_v2se_int8.onnx`, ~600 KB) is also
fetched but not used by default.

## Fetching

```bash
./scripts/fetch_models.sh        # or: make models
```

This:

- **downloads** YuNet (OpenCV Zoo, pinned commit) and MiniFASNetV2-SE
  (facenox release v1.0.0), each verified by SHA-256;
- **verifies/exports** EdgeFace-S: the weights are produced once from the
  upstream PyTorch checkpoint by `scripts/export_edgeface.py`. If
  `data/models/edgeface_s.onnx` is missing and a torch venv exists at
  `/tmp/edgeface-venv` (override with `EDGEFACE_VENV`), the script exports it
  automatically; otherwise create it once in a torch env:

  ```bash
  python -m venv /tmp/edgeface-venv
  /tmp/edgeface-venv/bin/pip install torch torchvision timm onnx
  /tmp/edgeface-venv/bin/python scripts/export_edgeface.py --out data/models/edgeface_s.onnx
  ```

All models are then mirrored into the runtime models dir.

## Model details

### YuNet (detection)

Fixed `[1, 3, 640, 640]` **BGR** input, raw float `0..255` (no mean/std). Twelve
outputs — a `cls_/obj_/bbox_/kps_` group per stride {8, 16, 32}. The daemon
decodes boxes + 5 facial landmarks, scores as `sqrt(cls·obj)`, and runs NMS. See
`daemon/src/ml/yunet_decoder.rs` and `daemon/src/ml/model_config.rs`.

### EdgeFace-S (recognition)

`[1, 3, 112, 112]` **RGB**, normalized `(x − 127.5) / 127.5` → `[-1, 1]`. Faces
are aligned to the canonical 5-point 112×112 ArcFace template
(`daemon/src/ml/align.rs`) before embedding. Output is a 512-d vector that the
daemon L2-normalizes; matching is cosine similarity (default threshold 0.4).

> **License:** EdgeFace-S weights are **CC-BY-NC-SA 4.0 — non-commercial only**.
> (The "BSD-3" sometimes cited refers to the Idiap `bob` framework *code*, not
> these weights.) For commercial use, swap in **AuraFace-v1** (fal) — native
> ONNX, commercial-OK, same 112×112 aligned-RGB input / 512-d output.

### MiniFASNetV2-SE (liveness)

A single model fed a `128×128` **RGB** crop (face bbox → square `max(w,h)·1.5`,
reflect-101 padded), normalized `/255` → `[0,1]`, NCHW. Output `[1, 2]` raw
logits: index 0 = real, index 1 = spoof. Decision:
`is_real = (real_logit − spoof_logit) ≥ ln(p/(1−p))` (default `p = 0.5`, i.e.
argmax). Liveness is a convenience deterrent, non-fatal on failure, and can be
disabled with `authentication.liveness_enabled = false`.

## Verification

```bash
doorman status        # shows model / camera / daemon health
```

When models are absent, model-dependent tests skip rather than fail, so a fresh
clone still builds and tests.

## Sources

- YuNet: https://github.com/opencv/opencv_zoo
- EdgeFace: https://github.com/otroshi/edgeface
- AuraFace-v1 (commercial alternative): https://huggingface.co/fal/AuraFace-v1
- MiniFASNetV2-SE: https://github.com/facenox/face-antispoof-onnx

## License notes

- **YuNet** detector (OpenCV Zoo): **MIT**.
- **EdgeFace-S** recognizer (Idiap, γ=0.5): weights **CC-BY-NC-SA 4.0 —
  NON-COMMERCIAL**. For commercial use, use **AuraFace-v1** (fal).
- **MiniFASNetV2-SE** liveness (facenox/face-antispoof-onnx, v1.0.0):
  **Apache-2.0**. Architecture derives from the Silent-Face / MiniFASNet project
  by Minivision AI (also Apache-2.0).

Ensure you comply with each model's license for your use case. In particular,
**the default EdgeFace-S recognizer is non-commercial**; swap to AuraFace-v1 for
commercial deployments.
