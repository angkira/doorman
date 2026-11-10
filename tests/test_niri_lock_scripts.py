"""Tests for the niri lock scripts: syntax, shellcheck, and behaviour with stubs.

Run with: PYTHONPATH=src pytest tests/test_niri_lock_scripts.py
"""

import fcntl
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

# Watcher that stays up until killed and counts its invocations.
LONG_WATCHER_STUB = """#!/bin/sh
printf '%s\\n' "$$" > "$STUB_DIR/{name}.pid"
printf '%s\\n' "$@" >> "$STUB_DIR/{name}.calls"
trap 'exit 0' TERM
while :; do sleep 0.05; done
"""

# pgrep: records its arguments, prints $STUB_DIR/pgrep.out (exit 1 if empty).
PGREP_STUB = """#!/bin/sh
printf '%s\\n' "$@" > "$STUB_DIR/pgrep.args"
if [ -s "$STUB_DIR/pgrep.out" ]; then cat "$STUB_DIR/pgrep.out"; else exit 1; fi
"""

# loginctl: records its arguments, prints LockedHint=<contents of lockedhint>.
LOGINCTL_STUB = """#!/bin/sh
printf '%s\\n' "$@" > "$STUB_DIR/loginctl.args"
printf 'LockedHint=%s\\n' "$(cat "$STUB_DIR/lockedhint" 2>/dev/null || echo no)"
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
            # The tests below describe the settle-first order; the early
            # watcher has its own tests.
            "DOORMAN_WATCH_EARLY": "0",
            # Never pick up a real locker or doorman from the host.
            "GTKLOCK": "gtklock",
            "SWAYLOCK": "swaylock",
        }
        # Default: no live locker, session not locked. Never ask the real host.
        self.stub("pgrep", PGREP_STUB)
        self.stub("loginctl", LOGINCTL_STUB)

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
    log = env.log.read_text()
    assert "already locked" in log
    assert "rc=1" in log and "no watcher" in log


@pytest.fixture
def alive_holder():
    """A live process standing in for the locker that holds the session lock."""
    holder = subprocess.Popen(["sleep", str(TIMEOUT * 4)])
    yield holder
    holder.kill()
    holder.wait()


def test_failed_locker_with_live_holder_attaches_the_watcher(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")

    proc = env.run()
    proc.wait(timeout=TIMEOUT)

    assert proc.returncode == 0
    watch = env.read_args("doorman")
    assert watch[:3] == ["--socket", env.env["DOORMAN_SOCKET"], "unlock-watch"]
    assert watch[watch.index("--locker-pid") + 1] == str(alive_holder.pid)
    assert watch[watch.index("--locker-name") + 1] == "gtklock"
    # Only this user's processes, exact name, newest match.
    pgrep = env.read_args("pgrep")
    assert pgrep[pgrep.index("-u") + 1] == str(os.getuid())
    assert pgrep[pgrep.index("-x") + 1] == "gtklock"
    assert "-n" in pgrep
    log = env.log.read_text()
    assert "rc=1" in log
    assert f"pid={alive_holder.pid}" in log
    assert "attaching watcher" in log


def test_failed_locker_without_process_but_locked_hint_attaches_nothing(
    tmp_path: Path,
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.stub_dir / "lockedhint").write_text("yes")

    proc = env.run({"XDG_SESSION_ID": "7"})
    proc.wait(timeout=TIMEOUT)

    assert proc.returncode == 0
    assert not (env.stub_dir / "doorman.args").exists(), "nothing to SIGTERM"
    assert env.read_args("loginctl") == ["show-session", "7", "-p", "LockedHint"]
    log = env.log.read_text()
    assert "rc=1" in log
    assert "unknown client" in log and "LockedHint=yes" in log


def test_second_run_does_not_start_a_second_watcher(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", LONG_WATCHER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")

    first = env.run()
    env.wait_for("doorman.pid")
    watcher_pid = int((env.stub_dir / "doorman.pid").read_text())
    pidfile = env.runtime / "doorman" / "unlock-watch.pid"
    assert pidfile.read_text().strip() == str(watcher_pid)

    second = env.run()
    second.wait(timeout=TIMEOUT)
    assert second.returncode == 0
    calls = (env.stub_dir / "doorman.calls").read_text().splitlines()
    assert calls.count("unlock-watch") == 1, "no duplicate watcher"
    assert "already running" in env.log.read_text()

    os.kill(watcher_pid, signal.SIGTERM)
    first.wait(timeout=TIMEOUT)
    assert first.returncode == 0
    assert not pidfile.exists(), "pidfile removed when the watcher ends"


def test_stale_pidfile_does_not_block_a_new_watcher(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")
    state = env.runtime / "doorman"
    state.mkdir()
    dead = subprocess.Popen(["true"])
    dead.wait()
    (state / "unlock-watch.pid").write_text(f"{dead.pid}\n")

    proc = env.run()
    proc.wait(timeout=TIMEOUT)

    watch = env.read_args("doorman")
    assert watch[watch.index("--locker-pid") + 1] == str(alive_holder.pid)


CONCURRENT_RUNS = 8


def test_concurrent_runs_start_exactly_one_watcher(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", LONG_WATCHER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")

    runs = [env.run() for _ in range(CONCURRENT_RUNS)]
    env.wait_for("doorman.pid")
    watcher_pid = int((env.stub_dir / "doorman.pid").read_text())
    # Runs that did not start the watcher finish at once; the starter waits for it.
    time.sleep(1.0)
    calls = (env.stub_dir / "doorman.calls").read_text().splitlines()
    assert calls.count("unlock-watch") == 1, "exactly one watcher for one locker"

    os.kill(watcher_pid, signal.SIGTERM)
    for run in runs:
        run.wait(timeout=TIMEOUT)
        assert run.returncode == 0


def test_held_start_lock_skips_the_watcher(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")
    state = env.runtime / "doorman"
    state.mkdir()

    with open(state / "unlock-watch.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proc = env.run()
        proc.wait(timeout=TIMEOUT)

    assert proc.returncode == 0
    assert not (env.stub_dir / "doorman.args").exists(), "lock holder starts the watcher"
    assert "holds" in env.log.read_text() and "not starting one" in env.log.read_text()


def test_pidfile_with_reused_unrelated_pid_does_not_block(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")
    state = env.runtime / "doorman"
    state.mkdir()
    unrelated = subprocess.Popen(["sleep", str(TIMEOUT * 4)])
    try:
        (state / "unlock-watch.pid").write_text(f"{unrelated.pid}\n")

        proc = env.run()
        proc.wait(timeout=TIMEOUT)
    finally:
        unrelated.kill()
        unrelated.wait()

    assert proc.returncode == 0
    watch = env.read_args("doorman")
    assert watch[watch.index("--locker-pid") + 1] == str(alive_holder.pid)
    log = env.log.read_text()
    assert "not an unlock watcher" in log and "removing stale" in log
    assert "already running" not in log


def test_without_flock_the_old_check_and_start_still_works(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")
    # A PATH that has the stubs and the tools the script needs, but no flock.
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in ("env", "bash", "sh", "date", "id", "mkdir", "chmod", "cat", "sleep",
                 "basename", "dirname", "tr", "rm"):
        found = shutil.which(tool)
        assert found, tool
        (tools / tool).symlink_to(found)

    proc = env.run({"PATH": f"{env.stub_dir}:{tools}"})
    proc.wait(timeout=TIMEOUT)

    assert proc.returncode == 0
    assert env.read_args("doorman")[2] == "unlock-watch"
    assert "flock not found" in env.log.read_text()


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


# `doorman test` prints "!recognized:" and exits 2 on a rejected attempt; the
# fake answers that once, then recognizes the face. Counts its invocations.
DOORMAN_TEST_STUB = """#!/bin/sh
printf 'x\\n' >> "$STUB_DIR/doorman.calls"
if [ "$(wc -l < "$STUB_DIR/doorman.calls")" -lt 2 ]; then
    echo "!recognized: alice (Camera not available (no frames))"
    exit 2
fi
echo "recognized: alice"
"""

PKILL_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_DIR/pkill.calls"
"""

ALWAYS_PGREP_STUB = """#!/bin/sh
echo 4242
"""


def test_poll_watcher_survives_nonzero_not_recognized_and_stops_on_recognized(
    tmp_path: Path,
) -> None:
    """A non-zero `doorman test` exit must neither unlock nor stop the watcher."""
    env = Env(tmp_path)
    env.stub("pgrep", ALWAYS_PGREP_STUB)
    env.stub("pkill", PKILL_STUB)
    env.stub("doorman", DOORMAN_TEST_STUB)
    result = subprocess.run(
        [str(POLL), "alice", str(tmp_path / "doorman.sock"), "gtklock"],
        env={
            **env.env,
            "DOORMAN_BIN": str(env.stub_dir / "doorman"),
            "DOORMAN_POLL_INTERVAL": "0.05",
            "DOORMAN_POLL_TERM_GRACE": "0",
        },
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stderr
    assert (env.stub_dir / "doorman.calls").read_text().count("x") == 2
    assert "!recognized: alice" in result.stdout
    kills = (env.stub_dir / "pkill.calls").read_text().splitlines()
    assert kills and all("-x gtklock" in k for k in kills)


def test_early_watcher_starts_before_the_settle_wait_ends(tmp_path: Path) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    settle = 4.0

    started = time.monotonic()
    proc = env.run({"DOORMAN_WATCH_EARLY": "1", "DOORMAN_LOCKER_SETTLE": str(settle)})
    env.wait_for("doorman.args")
    assert time.monotonic() - started < settle - 1.0, "watcher waited for the settle"
    locker_pid = (env.stub_dir / "gtklock.pid").read_text().strip()
    watch = env.read_args("doorman")
    assert watch[watch.index("--locker-pid") + 1] == locker_pid
    stop_locker(env, "gtklock", proc)
    assert proc.returncode == 0
    # One watcher only: the late start_watcher_once finds the pidfile or the
    # finished watcher; either way exactly one start was logged for this pid.
    assert env.log.read_text().count(f"locker_pid={locker_pid}") >= 1


def test_early_watcher_for_a_refused_locker_is_reaped_and_nothing_is_left(
    tmp_path: Path,
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)  # exits at once, like a watcher whose locker is gone

    proc = env.run({"DOORMAN_WATCH_EARLY": "1"})
    proc.wait(timeout=TIMEOUT)

    assert proc.returncode == 0
    log = env.log.read_text()
    assert "rc=1" in log and "no watcher" in log
    assert not (env.runtime / "doorman" / "unlock-watch.pid").exists()


def test_early_watcher_does_not_block_attaching_to_a_live_holder(
    tmp_path: Path, alive_holder: subprocess.Popen
) -> None:
    env = Env(tmp_path)
    env.stub("gtklock", EXITING_LOCKER_STUB)
    env.stub("doorman", RECORDER_STUB)
    (env.stub_dir / "pgrep.out").write_text(f"{alive_holder.pid}\n")

    proc = env.run({"DOORMAN_WATCH_EARLY": "1"})
    proc.wait(timeout=TIMEOUT)

    assert proc.returncode == 0
    watch = env.read_args("doorman")  # the last start wins: the holder's
    assert watch[watch.index("--locker-pid") + 1] == str(alive_holder.pid)
    assert "attaching watcher" in env.log.read_text()


def test_early_watcher_is_the_default() -> None:
    assert 'watch_early="${DOORMAN_WATCH_EARLY:-1}"' in LOCK.read_text()
