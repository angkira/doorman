"""The sway lock adapter must not log to a predictable /tmp path."""

import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "packaging" / "adapters" / "sway" / "doorman-sway-lock"
LEGACY_LOG = Path("/tmp/doorman-sway-lock.log")
DIR_MODE = 0o700


def _run(tmp_path, runtime_dir):
    """Run the adapter with stub locker and watcher."""
    stub = tmp_path / "stub"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "USER": "tester",
        "GTKLOCK": str(stub),
        "DOORMAN_SWAY_WATCH": str(stub),
        "DOORMAN_LOCK_BACKEND": "gtklock",
        "DOORMAN_GTKLOCK_CONFIG": str(tmp_path / "none"),
    }
    if runtime_dir is not None:
        env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    return subprocess.run(
        ["sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=20
    )


def _legacy_state():
    return LEGACY_LOG.stat().st_mtime_ns if LEGACY_LOG.exists() else None


def test_log_goes_to_a_private_runtime_dir(tmp_path):
    runtime = tmp_path / "run"
    runtime.mkdir()
    before = _legacy_state()
    proc = _run(tmp_path, runtime)
    assert proc.returncode == 0, proc.stderr
    log_dir = runtime / "doorman"
    assert stat.S_IMODE(log_dir.stat().st_mode) == DIR_MODE
    assert "lock user=tester" in (log_dir / "sway-lock.log").read_text()
    assert _legacy_state() == before


def test_unset_runtime_dir_fails_safe_without_writing_to_tmp(tmp_path):
    before = _legacy_state()
    proc = _run(tmp_path, None)
    assert proc.returncode == 0, proc.stderr
    assert _legacy_state() == before


def test_symlinked_log_dir_is_not_followed(tmp_path):
    runtime = tmp_path / "run"
    runtime.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (runtime / "doorman").symlink_to(target)
    proc = _run(tmp_path, runtime)
    assert proc.returncode == 0, proc.stderr
    assert not (target / "sway-lock.log").exists()
