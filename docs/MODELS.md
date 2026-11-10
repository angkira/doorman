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

Depth-Anything-V2-Small (depth liveness, optional) is documented under License notes below.

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
> (The BSD-3 LICENSE in `otroshi/edgeface` covers the *code*; the weights licence is
> unverified from the primary card, see License notes.) For commercial use, swap in **AuraFace-v1** (fal) — native
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
doorman models verify # SHA-256 of the loaded model files vs the manifest (read-only)
```

### Integrity manifest

`doormand` hashes the model files it loads (YuNet detector, EdgeFace-S
recognizer, MiniFASNet liveness, and the Depth-Anything depth model while
`authentication.liveness_enabled` is on) at start and compares them with a
manifest in `sha256sum` format. Files that merely sit in the models directory
(legacy models, `minifasnet_v2se_int8.onnx`) are not checked.

* The shipped manifest is `packaging/models.sha256`, compiled into the binary.
  It lists the hashes pinned in `scripts/fetch_models.sh` (a test keeps the two
  in sync). The depth model has no pinned upstream hash in this repository, so it
  is **not** listed: it is reported as unverified at start, never refused.
* A loaded file that is listed and does not match makes the daemon refuse to
  start (fail closed), with a log line naming the file. `systemd` will keep
  retrying; the password path is unaffected. `[models] allow_unverified_models =
  true` accepts the mismatch (a warning is logged each start); use it only while
  you replace a model on purpose.
* To pin a model the package does not know (the depth model, a replaced
  recognizer): generate lines on a machine you trust with
  `doorman models verify --print-manifest`, put them in a root-owned file (mode
  0644, not group/world writable) and set `[models] manifest_path` to it. Its
  entries override the built-in ones. Check with `doorman models verify`
  (exit 0 all verified, 1 mismatch or error, 2 only unlisted/missing files).
* Before upgrading a host whose model files were not produced by
  `fetch_models.sh` (a locally exported recognizer, for example), run
  `doorman models verify` first: a differing recognizer would stop the daemon.
* With a GPU depth sidecar (`doorman-depthd`) the file in `models_dir` is
  verified, but the sidecar opens its own path; keep them the same file.

When models are absent, model-dependent tests skip rather than fail, so a fresh
clone still builds and tests.

## Sources

- YuNet: https://github.com/opencv/opencv_zoo
- EdgeFace: https://github.com/otroshi/edgeface
- AuraFace-v1 (commercial alternative): https://huggingface.co/fal/AuraFace-v1
- MiniFASNetV2-SE: https://github.com/facenox/face-antispoof-onnx
- Depth-Anything-V2: https://github.com/DepthAnything/Depth-Anything-V2

## License notes

Status as of 2026-11-01. "Verified" means read from the primary source named; the
Hugging Face host was not reachable from the research environment, so HF model cards
were not read directly.

| Model | Code licence | Weights licence | Status | Source |
|---|---|---|---|---|
| YuNet (OpenCV Zoo, `face_detection_yunet_2023mar.onnx`) | MIT | MIT ("All files in this directory are licensed under MIT License"; LICENSE: MIT, (c) 2020 Shiqi Yu) | verified | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet , https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE |
| EdgeFace-S (Idiap, `edgeface_s_gamma_05`) | BSD-3-Clause (repo LICENSE, (c) 2024 Idiap) | **CC BY-NC-SA 4.0, non-commercial** per Idiap HF model cards of other EdgeFace variants (XXS, Base) and a third-party conversion of the S weights; the S-GAMMA card itself was not read. One third-party mirror claims BSD-3 for the weights and is contradicted by Idiap's own cards. | code verified; weights **[unverified]** (treat as non-commercial) | https://github.com/otroshi/edgeface (LICENSE raw: https://raw.githubusercontent.com/otroshi/edgeface/main/LICENSE), https://huggingface.co/Idiap/EdgeFace-Base , https://huggingface.co/Idiap/EdgeFace-XXS |
| Depth-Anything-V2-Small | Apache-2.0 (repo) | Small: Apache-2.0. Base/Large: CC-BY-NC-4.0 (Giant unreleased at time of reading) | verified from the upstream README license section; HF card not read | https://github.com/DepthAnything/Depth-Anything-V2 |
| MiniFASNetV2-SE (ONNX export, facenox) | Apache-2.0 (LICENSE of face-antispoof-onnx) | Apache-2.0 for the facenox ONNX release; the original Minivision caffemodels carry no stated licence in their README | facenox verified; upstream weights **[unverified]** | https://github.com/facenox/face-antispoof-onnx , https://github.com/minivision-ai/Silent-Face-Anti-Spoofing (repo licence Apache-2.0, per GitHub sidebar and LICENSE file) |

Notes:

- The default EdgeFace-S recognizer is to be treated as **non-commercial**; swap to
  AuraFace-v1 for commercial deployments.
- Depth-Anything-V2-Small is used by the depth-based liveness check but is not fetched
  by `scripts/fetch_models.sh`. Do not substitute Base/Large without accepting
  CC-BY-NC-4.0.
- Ensure you comply with each model's licence for your use case.

## Licence table for every model file seen on the host

Status as of 2026-11-01. "Redistribute" means: may be shipped inside the `.deb`.
Sources read in this round are cited; Hugging Face (`huggingface.co`) was not
reachable from the session, so the EdgeFace-S-GAMMA card could not be read.

| File | Model | Weights licence | Redistribute in package | Status |
|---|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet (OpenCV Zoo) | MIT | yes | verified (table above) |
| `edgeface_s.onnx` | EdgeFace-S (gamma 0.5), Idiap | CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution) per Idiap cards of sibling variants; the Idiap/EdgeFace-S-GAMMA card was **not read** | **[unverified]**: redistribution is only plausible under CC BY-NC-SA terms | **[unverified]** |
| `minifasnet_v2se.onnx`, `minifasnet_v2se_int8.onnx` | MiniFASNetV2-SE (facenox ONNX export) | Apache-2.0 (facenox); upstream Minivision weights unstated | yes for the facenox export | facenox verified, upstream **[unverified]** |
| `depth_anything_v2_small_fp32.onnx`, `depth.ncnn.{param,bin}` | Depth-Anything-V2-Small (and its ncnn conversion) | Apache-2.0 (Small) | yes (not packaged today, see below) | verified from upstream README |
| `mobilefacenet.onnx`, `w600k_mbf.onnx` | insightface-style MobileFaceNet (WebFace600K) | **non-commercial research only** | **no** | verified, see below |
| `w600k_r50.onnx` (not on the host; named in code) | insightface buffalo_l ArcFace R50 | **non-commercial research only** | **no** | verified, see below |
| `scrfd_500m.onnx` | insightface SCRFD | insightface model zoo terms: non-commercial research only | **no** | verified (zoo statement) |
| `blazeface.onnx(.bak)`, `face_detection_short_range.tflite` | MediaPipe BlazeFace | Apache-2.0 expected | not packaged | [unverified], not read |
| `ultraface-rfb-320.onnx`, `ultraface-slim-320.onnx` | Ultra-Light-Fast-Generic-Face-Detector | MIT expected | not packaged | [unverified], not read |
| `liveness.onnx(.data)`, `liveness.pt` | unknown origin | unknown | no | [unverified], provenance unknown |

insightface evidence (read 2026-11-01):

- https://github.com/deepinsight/insightface (README): "The code of InsightFace is
  released under the MIT License", but the training data and the models trained on
  it (manually or automatically downloaded) "are available for non-commercial
  research purposes only"; for the buffalo_l package a licensing contact is given.
- https://github.com/deepinsight/insightface/tree/master/model_zoo : "ALL models are
  available for non-commercial research purposes only." The zoo lists buffalo_sc
  (SCRFD-500MF + MBF@WebFace600K) and a MobileFaceNet WebFace600K entry, which is the
  family `w600k_mbf.onnx` and (by name) `mobilefacenet.onnx` belong to. The exact
  origin of the 261 MB `mobilefacenet.onnx` is not known; treat it the same way.

### Packaging rule

The package ships only models with a **verified** redistribution licence. A model
whose licence is non-commercial, share-alike, or unverified must not be added to
`runtime_models` in `scripts/build_deb.sh` without the owner's decision.

Current `scripts/build_deb.sh` (`runtime_models`, lines 99-104) ships:

| Packaged file | Redistribution licence | Flag |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | MIT, verified | ok |
| `minifasnet_v2se.onnx`, `minifasnet_v2se_int8.onnx` | Apache-2.0 (facenox), verified | ok (upstream weights unstated) |
| `edgeface_s.onnx` | CC BY-NC-SA 4.0 expected, **unverified** | **FLAG**: shipped without a verified redistribution licence; non-commercial and share-alike terms (attribution notice, same licence) apply if the expectation is right. Not removed; owner decides. |

`depth_anything_v2_small_fp32.onnx` (Apache-2.0) is not packaged, which is why a
package-only install has no depth liveness; the installer copies it from the
repository `data/models` (`daemon/src/install.rs:1520-1545`).

## Host model directory: used by HEAD vs legacy

`/var/lib/doorman/models` listing from `.plans/2026-11-01_doorman-hardening-latency/HOST-FACTS.md`
section 8. "Used" means a code path at HEAD opens the file for the default
`[ml] backend = "ort"`. Nothing here is to be deleted by tooling; the owner decides.

| File | Status at HEAD | Evidence |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | used (detector) | `daemon/src/ml/ort_backend.rs:231`, `model_config.rs:52` |
| `edgeface_s.onnx` | used (recognizer) | `ort_backend.rs:349`, `model_config.rs:195` |
| `minifasnet_v2se.onnx` | used (legacy MiniFASNet liveness session, loaded when present; the fatal gate is depth) | `ort_backend.rs:250`, `model_config.rs:114` |
| `depth_anything_v2_small_fp32.onnx` | used (depth gate, in-process CPU) | `ort_backend.rs:315-344`, `model_config.rs:160` |
| `depth.ncnn.param`, `depth.ncnn.bin` | used only if `ml.backend = "ncnn"` in a `backend-ncnn` build | `daemon/src/ml/ncnn_hybrid.rs:34-35` |
| `minifasnet_v2se_int8.onnx` | legacy: packaged, never opened by code | no reference in `daemon/src` or `shared/src` |
| `w600k_mbf.onnx`, `mobilefacenet.onnx`, `blazeface.onnx`, `blazeface.onnx.bak`, `face_detection_short_range.tflite`, `scrfd_500m.onnx`, `ultraface-rfb-320.onnx`, `ultraface-slim-320.onnx`, `liveness.onnx`, `liveness.onnx.data`, `liveness.pt` | legacy: no file name reference in HEAD code (only a `w600k_r50.onnx` alias, itself unused, `model_config.rs:207`) | grep over `daemon shared bench pam_module` |

The unused files that are insightface-derived (`w600k_mbf.onnx`, `mobilefacenet.onnx`,
`scrfd_500m.onnx`) are non-commercial research only; keeping them on a machine is
a licensing matter for the owner, not a runtime one.
