"""The postinst warns about GPU grants in operator-owned systemd files.

Only the marked scan block of packaging/deb/postinst is executed, against a
temporary directory given by DOORMAN_SYSTEMD_ETC; /etc is never read.

Run with: PYTHONPATH=src pytest tests/test_postinst_gpu_scan.py
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
POSTINST = REPO / "packaging" / "deb" / "postinst"
CLEAN_UNIT = "[Service]\nExecStart=/usr/bin/doormand\nDevicePolicy=closed\n"


def scan_block() -> str:
    lines, in_block = [], False
    for line in POSTINST.read_text().splitlines():
        if line.strip() == "# gpu-scan-end":
            in_block = False
        if in_block:
            lines.append(line)
        if line.strip() == "# gpu-scan-begin":
            in_block = True
    assert lines, "scan block not found in postinst"
    return "\n".join(lines)


def run_scan(etc: Path) -> subprocess.CompletedProcess:
    script = scan_block() + "\nwarn_gpu_grants\n"
    return subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "DOORMAN_SYSTEMD_ETC": str(etc)},
        capture_output=True,
        text=True,
        check=False,
    )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_clean_setup_prints_nothing(tmp_path: Path) -> None:
    write(tmp_path / "doormand.service", CLEAN_UNIT)
    write(tmp_path / "doormand.service.d" / "10-limits.conf", "[Service]\nNice=5\n")
    result = run_scan(tmp_path)
    assert result.returncode == 0
    assert result.stderr == "" and result.stdout == ""


def test_missing_directory_is_fine(tmp_path: Path) -> None:
    result = run_scan(tmp_path / "does-not-exist")
    assert result.returncode == 0 and result.stderr == ""


@pytest.mark.parametrize(
    "line",
    [
        "DeviceAllow=char-drm rw",
        "DeviceAllow=/dev/dri/renderD128 rw",
        "DeviceAllow=/dev/kfd rw",
        "SupplementaryGroups=video render",
        "SupplementaryGroups=render",
        "Group=render",
    ],
)
def test_offending_unit_line_is_reported_with_file_and_line(
    tmp_path: Path, line: str
) -> None:
    unit = write(tmp_path / "doormand.service", CLEAN_UNIT + line + "\n")
    result = run_scan(tmp_path)
    assert result.returncode == 0
    assert f"{unit}:4" in result.stderr
    assert line in result.stderr
    assert "daemon-reload" in result.stderr
    assert result.stdout == ""


def test_offending_drop_in_is_reported(tmp_path: Path) -> None:
    write(tmp_path / "doormand.service", CLEAN_UNIT)
    drop_in = write(
        tmp_path / "doormand.service.d" / "10-drm.conf",
        "[Service]\nDeviceAllow=char-drm rw\nDeviceAllow=/dev/dri rw\n",
    )
    result = run_scan(tmp_path)
    assert f"{drop_in}:2" in result.stderr
    assert f"{drop_in}:3" in result.stderr
    assert result.stderr.count("WARNING") == 2


def test_comment_only_mentions_are_not_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "doormand.service",
        "# DeviceAllow=char-drm rw\n  ; /dev/kfd is not granted\n"
        "[Service]\n# SupplementaryGroups=render\nExecStart=/usr/bin/doormand\n",
    )
    result = run_scan(tmp_path)
    assert result.stderr == ""


def test_words_containing_render_are_not_flagged(tmp_path: Path) -> None:
    write(
        tmp_path / "doormand.service",
        CLEAN_UNIT + "SupplementaryGroups=video rendering renderer\n",
    )
    assert run_scan(tmp_path).stderr == ""


def test_scan_never_modifies_the_files(tmp_path: Path) -> None:
    unit = write(tmp_path / "doormand.service", CLEAN_UNIT + "DeviceAllow=/dev/kfd rw\n")
    drop_in = write(
        tmp_path / "doormand.service.d" / "10-drm.conf", "[Service]\nGroup=render\n"
    )
    before = (unit.read_bytes(), drop_in.read_bytes())
    run_scan(tmp_path)
    assert (unit.read_bytes(), drop_in.read_bytes()) == before
