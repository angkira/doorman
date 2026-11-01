"""Tests for the niri lock scripts: syntax, shellcheck, and behaviour with stubs.

Run with: PYTHONPATH=src pytest tests/test_niri_lock_scripts.py
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "packaging" / "adapters" / "niri" / "bin"
LOCK = BIN / "doorman-lock"
POLL = BIN / "doorman-lock-watch-poll"
SCRIPTS = [LOCK, POLL]

# Longest any scenario may take.
TIMEOUT = 15.0

LOCKER_STUB = """#!/bin/sh
# Fake locker: records its pid and arguments, exits on SIGTERM.
printf '%s\\n' "$$" > "$STUB_DIR/{name}.pid"
printf '%s\\n' "$@" > "$STUB_DIR/{name}.args"
trap 'exit 0' TERM
while :; do sleep 0.05; done
"""

EXITING_LOCKER_STUB = """#!/bin/sh
# Fake locker refused because the session is already locked.
echo "{name}: already locked" >&2
exit 1
"""

RECORDER_STUB = """#!/bin/sh
printf '%s\\n' "$@" > "$STUB_DIR/{name}.args"
"""


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shellcheck_clean(script: Path) -> None:
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    subprocess.run(["shellcheck", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_strict_mode_and_no_personal_paths(script: Path) -> None:
    text = script.read_text()
    assert "set -euo pipefail" in text
    assert "/home/" not in text
    assert "pkill" not in text or script == POLL  # no kill-by-name outside the fallback
    assert os.access(script, os.X_OK)


def test_event_mode_has_no_daemonize_and_no_name_kill() -> None:
    text = LOCK.read_text()
    assert "--daemonize" not in text
    assert "pkill" not in text
    assert "unlock-watch" in text


class Env:
    """Isolated environment with stub executables first on PATH."""

    def __init__(self, tmp_path: Path) -> None:
        self.stub_dir = tmp_path / "stubs"
        self.stub_dir.mkdir()
        self.runtime = tmp_path / "run"
        self.runtime.mkdir()
        self.config = tmp_path / "config"
        self.config.mkdir()
        self.env = {
            "PATH": f"{self.stub_dir}:{os.environ['PATH']}",
            "HOME": str(tmp_path / "home"),
            "XDG_RUNTIME_DIR": str(self.runtime),
            "XDG_CONFIG_HOME": str(self.config),
            "STUB_DIR": str(self.stub_dir),
            "DOORMAN_USER": "alice",
            "DOORMAN_SOCKET": str(tmp_path / "doorman.sock"),
            "DOORMAN_LOCKER_SETTLE": "0.2",
            # Never pick up a real locker or doorman from the host.
            "GTKLOCK": "gtklock",
            "SWAYLOCK": "swaylock",
        }

    def stub(self, name: str, template: str) -> None:
        path = self.stub_dir / name
        path.write_text(template.format(name=name))
        path.chmod(0o755)

    @property
    def log(self) -> Path:
        return self.runtime / "doorman" / "lock.log"

    def read_args(self, name: str) -> list[str]:
        return (self.stub_dir / f"{name}.args").read_text().splitlines()

    def wait_for(self, name: str) -> None:
        deadline = time.monotonic() + TIMEOUT
        target = self.stub_dir / name
        while not target.exists() or target.stat().st_size == 0:
            assert time.monotonic() < deadline, f"{name} never appeared"
            time.sleep(0.02)

    def run(self, extra_env: dict[str, str] | None = None, args: list[str] | None = None):
        env = {**self.env, **(extra_env or {})}
        return subprocess.Popen(
            [str(LOCK), *(args or [])],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def stop_locker(env: Env, name: str, proc: subprocess.Popen) -> None:
    """End the fake lock like a password unlock would, then reap the script."""
    pid = int((env.stub_dir / f"{name}.pid").read_text())
    os.kill(pid, signal.SIGTERM)
    proc.wait(timeout=TIMEOUT)


def test_event_mode_starts_watcher_with_the_real_locker_pid(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.config / "gtklock").mkdir()
    (env.config / "gtklock" / "config.ini").write_text("[main]\n")

    proc = env.run(args=["--extra-locker-arg"])
    env.wait_for("doorman.args")
    locker_pid = (env.stub_dir / "gtklock.pid").read_text().strip()

    watch = env.read_args("doorman")
    assert watch[:3] == ["--socket", env.env["DOORMAN_SOCKET"], "unlock-watch"]
    assert watch[watch.index("--user") + 1] == "alice"
    assert watch[watch.index("--locker-pid") + 1] == locker_pid
    assert watch[watch.index("--locker-name") + 1] == "gtklock"

    locker_args = env.read_args("gtklock")
    assert "--daemonize" not in locker_args
    assert locker_args[:2] == ["--config", str(env.config / "gtklock" / "config.ini")]
    assert "--extra-locker-arg" in locker_args

    stop_locker(env, "gtklock", proc)
    assert proc.returncode == 0
    log = env.log.read_text()
    assert f"pid={locker_pid}" in log and "exited" in log
    assert oct(env.log.parent.stat().st_mode & 0o777) == "0o700"


def test_locker_exiting_at_once_means_already_locked(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)

    proc = env.run()
    proc.wait(timeout=TIMEOUT)

    assert proc.returncode == 0
    assert not (env.stub_dir / "doorman.args").exists(), "no second watcher"
    assert "already locked" in env.log.read_text()


def test_falls_back_to_swaylock_without_gtklock(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.stub("swaylock", LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)

    proc = env.run({"GTKLOCK": "gtklock-not-installed"})
    env.wait_for("doorman.args")
    watch = env.read_args("doorman")
    assert watch[watch.index("--locker-name") + 1] == "swaylock"
    stop_locker(env, "swaylock", proc)
    assert proc.returncode == 0


def test_no_locker_installed_is_an_error(tmp_path: Path) -> None:
    env = Env(tmp_path)
    proc = env.run({"GTKLOCK": "gtklock-not-installed", "SWAYLOCK": "swaylock-not-installed"})
    proc.wait(timeout=TIMEOUT)
    assert proc.returncode == 127
    assert "no screen locker" in proc.stderr.read()


def test_poll_mode_uses_the_old_watcher(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    env.stub("poll-watcher", RECORDER_STUB)

    proc = env.run(
        {
            "DOORMAN_UNLOCK_MODE": "poll",
            "DOORMAN_LOCK_WATCH_POLL": str(env.stub_dir / "poll-watcher"),
        }
    )
    env.wait_for("poll-watcher.args")
    assert env.read_args("poll-watcher") == [
        "alice",
        env.env["DOORMAN_SOCKET"],
        "gtklock",
    ]
    assert not (env.stub_dir / "doorman.args").exists(), "event helper not started"
    stop_locker(env, "gtklock", proc)


def test_unknown_unlock_mode_leaves_password_unlock_working(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)

    proc = env.run({"DOORMAN_UNLOCK_MODE": "bogus"})
    env.wait_for("gtklock.pid")
    time.sleep(0.5)
    assert proc.poll() is None, "locker must stay up"
    assert not (env.stub_dir / "doorman.args").exists()
    stop_locker(env, "gtklock", proc)
    assert "unsupported DOORMAN_UNLOCK_MODE" in env.log.read_text()


def test_niri_installer_installs_the_lock_scripts() -> None:
    installer = (REPO / "scripts" / "install_local_niri_desktop.sh").read_text()
    assert "packaging/adapters/niri/bin/doorman-lock\"" in installer
    assert "packaging/adapters/niri/bin/doorman-lock-watch-poll\"" in installer
    kdl = (REPO / "packaging" / "adapters" / "niri" / "config.kdl").read_text()
    assert "/home/" not in kdl.split("doorman-lock")[0].splitlines()[-1]
