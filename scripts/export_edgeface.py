#!/usr/bin/env python3
"""Export EdgeFace-S (gamma=0.5) to ONNX for the doorman recognizer.

EdgeFace (https://github.com/otroshi/edgeface) is a lightweight edge
face-recognition model. NOTE: the EdgeFace-S **weights are CC-BY-NC-SA 4.0
(NON-COMMERCIAL)** — the "BSD-3" sometimes cited is the `bob` framework code,
not these weights. For commercial use, swap in AuraFace-v1 (fal, native ONNX). EdgeFace-S has ~3.65M params; the checkpoint
(edgeface_s_gamma_05.pt) is ~15 MB and exports to a ~15 MB ONNX, satisfying the
"lightweight" requirement (vs the 174 MB ArcFace w600k_r50).

I/O contract (verified against the repo's inference code):
  - Input  "input": [1, 3, 112, 112], float32, NCHW, **RGB**.
  - Preprocessing: torchvision ToTensor() (-> [0,1], RGB, CHW) then
    Normalize(mean=0.5, std=0.5) == (x - 0.5) / 0.5, i.e. mapped to [-1, 1].
    In 0..255 pixel terms this is (pixel - 127.5) / 127.5 — identical to
    InsightFace ArcFace preprocessing, so the Rust backend reuses it.
  - Alignment: faces are aligned to the canonical 5-point 112x112 template
    (get_reference_facial_points(default_square=True)), the same template the
    doorman backend already uses (RecognizerConfig::EDGEFACE_TEMPLATE_112).
  - Output "embedding": [1, 512] float32 (the daemon L2-normalizes it).

Usage:
    python scripts/export_edgeface.py \
        --repo /tmp/edgeface-repo \
        --checkpoint /tmp/edgeface-repo/checkpoints/edgeface_s_gamma_05.pt \
        --out data/models/edgeface_s.onnx

Requires a torch-capable env (see scripts/fetch_models.sh / the project README):
    uv venv --python 3.11 /tmp/edgeface-venv
    uv pip install --python /tmp/edgeface-venv torch torchvision onnx \
        onnxruntime timm huggingface_hub numpy pillow
"""
import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        default="/tmp/edgeface-repo",
        help="Path to a clone of https://github.com/otroshi/edgeface",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to edgeface_s_gamma_05.pt (defaults to <repo>/checkpoints/edgeface_s_gamma_05.pt)",
    )
    ap.add_argument(
        "--out",
        default="data/models/edgeface_s.onnx",
        help="Output ONNX path",
    )
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / "backbones").is_dir():
        print(f"ERROR: {repo} does not look like the edgeface repo (no backbones/)", file=sys.stderr)
        return 2
    sys.path.insert(0, str(repo))

    checkpoint = Path(args.checkpoint) if args.checkpoint else repo / "checkpoints" / "edgeface_s_gamma_05.pt"
    if not checkpoint.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    from backbones import get_model  # noqa: E402  (path injected above)

    model_name = "edgeface_s_gamma_05"
    print(f"Building {model_name} ...")
    model = get_model(model_name)
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    with torch.no_grad():
        ref = model(dummy)
    print(f"PyTorch output shape: {tuple(ref.shape)}")
    assert ref.shape == (1, 512), f"expected [1,512], got {tuple(ref.shape)}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use the legacy TorchScript-based exporter (dynamo=False): it emits a single
    # self-contained .onnx file with weights inlined (the new dynamo exporter
    # splits weights into a fragile .onnx.data sidecar). A single file is far
    # easier to ship + checksum in fetch_models.sh.
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,  # fixed [1,3,112,112] -> [1,512]
        dynamo=False,
    )
    # Defensively remove any stale external-data sidecar from a prior dynamo run.
    sidecar = out.with_name(out.name + ".data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"Exported ONNX -> {out} ({out.stat().st_size / 1e6:.2f} MB)")

    # ---- Verify the ONNX loads + matches torch within tolerance ----
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(out)))
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    outp = sess.get_outputs()[0]
    print(f"ONNX input : name={inp.name!r} shape={inp.shape} type={inp.type}")
    print(f"ONNX output: name={outp.name!r} shape={outp.shape} type={outp.type}")

    onnx_out = sess.run(None, {inp.name: dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(onnx_out - ref.numpy())))
    print(f"max |onnx - torch| = {max_abs:.6e}")
    assert max_abs < 1e-3, f"ONNX/torch mismatch too large: {max_abs}"
    print("OK: ONNX verified against PyTorch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
