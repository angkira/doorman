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


def test_calibration_report_selects_threshold_with_no_false_accepts_when_possible():
    errors = [0.1, 0.2, 0.3, 0.55, 0.7]
    labels = [1, 1, 1, 0, 0]

    report = rnd.calibrate_one_class_errors(errors, labels)

    assert report["status"] == "ok"
    assert report["threshold_source"] == "max_positive_below_min_negative"
    assert report["threshold"] == 0.3
    assert report["false_accept_rate"] == 0.0
    assert report["false_reject_rate"] == 0.0


def test_deployment_manifest_records_auxiliary_scorer_contract(tmp_path):
    one_class = {
        "status": "ok",
        "model_path": str(tmp_path / "one_class_autoencoder.pt"),
        "reconstruction_error_p95": 0.4,
    }
    calibration = {
        "status": "ok",
        "threshold": 0.3,
        "false_accept_rate": 0.0,
        "false_reject_rate": 0.1,
    }

    manifest = rnd.write_deployment_manifest(
        out_dir=tmp_path,
        encoder="da3-base",
        feature_kind="da3_depth_face_patch_and_stats",
        model_dir="/mnt/datasets/models/depth-anything/DA3-BASE",
        feature_shape=[10, 1036],
        one_class=one_class,
        calibration=calibration,
        probe={"status": "ok", "roc_auc": 0.99},
        process_res=504,
        patch_size=32,
    )

    assert manifest["scorer_kind"] == "da3_one_class_reconstruction"
    assert manifest["decision"]["threshold"] == 0.3
    assert manifest["decision"]["lower_is_more_genuine"] is True
    assert manifest["runtime"]["integration_status"] == "candidate_not_hot_path"
    assert (tmp_path / "deployment_manifest.json").exists()
