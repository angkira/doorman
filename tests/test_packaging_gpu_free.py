"""The shipped unit and postinst must not grant GPU access.

Inference is CPU-only in the packaged service. GPU device nodes and the
`render` group are granted only by the generated depthd unit when the owner
opts into `--ep rocm` (see docs/systemd-hardening.md). Comments are ignored so
the files may state the prohibition in prose.

The postinst also WARNS about GPU grants in the operator's own systemd files, so
its scan pattern has to name the forbidden strings. That read-only block sits
between the `# gpu-scan-begin` / `# gpu-scan-end` markers; it alone is excluded
here, and `test_scan_block_is_read_only` pins that it cannot grant anything.

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


SCAN_BEGIN = "# gpu-scan-begin"
SCAN_END = "# gpu-scan-end"
# Commands that change users, groups or files; none may appear in the scan block.
MUTATING = re.compile(
    r"\b(usermod|adduser|addgroup|useradd|groupadd|install|tee|rm|mv|cp|chmod|chown|systemctl)\b"
    r"|sed\s+-i|(^|[^>&0-9])>(?!&)"
)


def split_scan_block(text: str) -> tuple[list[str], list[str]]:
    """Return (lines outside the scan block, lines inside it)."""
    outside, inside, in_block = [], [], False
    for line in text.splitlines():
        if line.strip() == SCAN_BEGIN:
            in_block = True
        elif line.strip() == SCAN_END:
            in_block = False
        else:
            (inside if in_block else outside).append(line)
    return outside, inside


def active_lines(path: Path) -> list[str]:
    lines = []
    outside, _ = split_scan_block(path.read_text())
    for raw in outside:
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


def test_scan_block_is_read_only() -> None:
    text = (REPO / "packaging" / "deb" / "postinst").read_text()
    assert text.count(SCAN_BEGIN) == 1 and text.count(SCAN_END) == 1
    _, inside = split_scan_block(text)
    code = [
        line.strip()
        for line in inside
        if line.strip() and not line.strip().startswith("#")
    ]
    assert code, "scan block is empty"
    # Output goes to stderr via `>&2` only; no file, user or group changes.
    for line in code:
        # Quoted text is prose or a pattern, not a command.
        stripped = re.sub(r"\"[^\"]*\"|'[^']*'", "", line)
        stripped = stripped.replace(">&2", "").replace("<<<", "")
        assert not MUTATING.search(stripped), f"scan block must be read-only: {line}"
