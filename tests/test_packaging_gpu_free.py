"""The shipped unit and postinst must not grant GPU access.

Inference is CPU-only in the packaged service. GPU device nodes and the
`render` group are granted only by the generated depthd unit when the owner
opts into `--ep rocm` (see docs/systemd-hardening.md). Comments are ignored so
the files may state the prohibition in prose.

Run with: PYTHONPATH=src pytest tests/test_packaging_gpu_free.py
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FILES = [
    REPO / "packaging" / "systemd" / "doormand.service",
    REPO / "packaging" / "deb" / "postinst",
]
FORBIDDEN = {
    "char-drm": re.compile(r"char-drm"),
    "/dev/dri": re.compile(r"/dev/dri"),
    "/dev/kfd": re.compile(r"/dev/kfd"),
    # `render` as a whole word; "rendered" and "renderer" do not match.
    "render group": re.compile(r"\brender\b"),
}


def active_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("name", sorted(FORBIDDEN))
def test_packaged_files_grant_no_gpu_access(path: Path, name: str) -> None:
    lines = active_lines(path)
    assert lines, f"{path} has no active lines; wrong path?"
    hits = [line for line in lines if FORBIDDEN[name].search(line)]
    assert not hits, f"{path.name} contains {name}: {hits}"
