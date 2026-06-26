import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rnd_depth_fusion_experiment.py"
SPEC = importlib.util.spec_from_file_location("rnd_depth_fusion_experiment", SCRIPT_PATH)
rnd = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = rnd
SPEC.loader.exec_module(rnd)


def write_capture(dataset_dir: Path, attempt_id: str, outcome: str, frames: list[dict]) -> None:
    capture_dir = dataset_dir / "captures" / "20260626" / attempt_id
    capture_dir.mkdir(parents=True)
    manifest_frames = []
    for index, frame in enumerate(frames):
        filename = f"frame-{index:03d}.jpg"
        (capture_dir / filename).write_bytes(b"stub")
        manifest_frames.append({"filename": filename, "source_index": index, **frame})
    (capture_dir / "manifest.json").write_text(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "outcome": outcome,
                "frame_count": len(frames),
                "frames": manifest_frames,
            }
        ),
        encoding="utf-8",
    )


def test_inventory_report_includes_capture_quality_by_label(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "labels-20260626.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"attempt_id": "ok", "outcome": "ConfirmedGenuine"}),
                json.dumps({"attempt_id": "fn", "outcome": "FalseNegative"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_capture(
        dataset_dir,
        "ok",
        "ConfirmedGenuine",
        [
            {
                "bbox_norm": [0.1, 0.2, 0.3, 0.4],
                "conditions": {"sharpness": 100.0, "face_area_frac": 0.12},
            },
            {
                "bbox_norm": [0.1, 0.2, 0.2, 0.3],
                "conditions": {"sharpness": 80.0, "face_area_frac": 0.06},
            },
        ],
    )
    write_capture(
        dataset_dir,
        "fn",
        "FalseNegative",
        [
            {"bbox_norm": None, "conditions": {"sharpness": 50.0, "face_area_frac": None}},
            {
                "bbox_norm": [0.1, 0.2, 0.01, 0.01],
                "conditions": {"sharpness": 40.0, "face_area_frac": 0.0001},
            },
        ],
    )

    report = rnd.inventory_report(dataset_dir)

    quality = report["capture_quality"]["labels"]
    assert quality["ConfirmedGenuine"]["frames"] == 2
    assert quality["ConfirmedGenuine"]["missing_bbox"] == 0
    assert quality["ConfirmedGenuine"]["face_area_frac"]["p50"] == 0.09
    assert quality["FalseNegative"]["frames"] == 2
    assert quality["FalseNegative"]["missing_bbox"] == 1
    assert quality["FalseNegative"]["face_area_frac"]["p50"] == 0.0001
