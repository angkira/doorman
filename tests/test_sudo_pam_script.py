"""Tests for scripts/doorman-sudo-pam.sh against a temporary PAM directory.

Run with: PYTHONPATH=src pytest tests/test_sudo_pam_script.py
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "doorman-sudo-pam.sh"
ACK = "--i-have-a-root-shell-open"
MODULE_LINE = "auth sufficient pam_doorman.so"

DEBIAN_SUDO = """#%PAM-1.0

session    required   pam_env.so readenv=1 user_readenv=0
@include common-auth
@include common-account
@include common-session-noninteractive
"""

FEDORA_SUDO = """#%PAM-1.0
auth       include      system-auth
account    include      system-auth
session    include      system-auth
"""

# Exit codes of the script.
REFUSED = 2
FAILED = 1
USAGE = 64


@pytest.fixture
def env(tmp_path):
    pam_dir = tmp_path / "pam.d"
    pam_dir.mkdir()
    module_dir = tmp_path / "security"
    module_dir.mkdir()
    (module_dir / "pam_doorman.so").write_bytes(b"")
    return {
        **os.environ,
        "PAM_DIR": str(pam_dir),
        "PAM_MODULE_DIR": str(module_dir),
    }


def run(env, *args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True
    )


def sudo_file(env) -> Path:
    return Path(env["PAM_DIR"]) / "sudo"


def backups(env):
    return sorted(Path(env["PAM_DIR"]).glob("sudo.doorman-backup.*"))


def test_script_is_shellcheck_clean():
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize("original", [DEBIAN_SUDO, FEDORA_SUDO])
def test_enable_inserts_before_first_auth_and_backs_up(env, original):
    sudo_file(env).write_text(original)
    result = run(env, "enable", ACK)
    assert result.returncode == 0, result.stderr
    lines = sudo_file(env).read_text().splitlines()
    idx = lines.index(MODULE_LINE)
    assert lines.count(MODULE_LINE) == 1
    first_auth = next(
        i
        for i, line in enumerate(lines)
        if line.startswith(("@include", "auth")) and i != idx
    )
    assert idx == first_auth - 1, "inserted right before the first auth line/include"
    assert [b.read_text() for b in backups(env)] == [original]
    assert "recovery: cp -p" in result.stderr


def test_enable_is_idempotent(env):
    sudo_file(env).write_text(DEBIAN_SUDO)
    assert run(env, "enable", ACK).returncode == 0
    after_first = sudo_file(env).read_text()
    second = run(env, "enable", ACK)
    assert second.returncode == 0
    assert "already enabled" in second.stdout
    assert sudo_file(env).read_text() == after_first
    assert len(backups(env)) == 1, "no extra backup when nothing changes"


def test_disable_restores_the_original(env):
    sudo_file(env).write_text(DEBIAN_SUDO)
    run(env, "enable", ACK)
    result = run(env, "disable", ACK)
    assert result.returncode == 0, result.stderr
    assert sudo_file(env).read_text() == DEBIAN_SUDO
    again = run(env, "disable", ACK)
    assert again.returncode == 0 and "already disabled" in again.stdout


def test_backups_never_overwrite_each_other(env):
    sudo_file(env).write_text(DEBIAN_SUDO)
    run(env, "enable", ACK)
    run(env, "disable", ACK)
    run(env, "enable", ACK)
    assert len(backups(env)) == 3
    assert len({b.name for b in backups(env)}) == 3


def test_file_mode_is_preserved(env):
    sudo_file(env).write_text(DEBIAN_SUDO)
    sudo_file(env).chmod(0o640)
    run(env, "enable", ACK)
    assert sudo_file(env).stat().st_mode & 0o777 == 0o640


def test_status(env):
    sudo_file(env).write_text(DEBIAN_SUDO)
    assert "disabled" in run(env, "status").stdout
    run(env, "enable", ACK)
    assert run(env, "status").stdout.startswith("enabled")
    # status needs no acknowledgement and changes nothing
    assert len(backups(env)) == 1


@pytest.mark.parametrize("action", ["enable", "disable"])
def test_refuses_without_the_root_shell_flag(env, action):
    sudo_file(env).write_text(DEBIAN_SUDO)
    result = run(env, action)
    assert result.returncode == REFUSED
    assert "WARNING" in result.stderr and "cp -p" in result.stderr
    assert sudo_file(env).read_text() == DEBIAN_SUDO
    assert backups(env) == []


def test_refuses_when_no_auth_path_to_insert_before(env):
    original = "#%PAM-1.0\naccount include system-auth\nsession include system-auth\n"
    sudo_file(env).write_text(original)
    result = run(env, "enable", ACK)
    assert result.returncode == FAILED
    assert sudo_file(env).read_text() == original
    assert backups(env) == []


def test_refuses_when_module_is_not_installed(env):
    sudo_file(env).write_text(DEBIAN_SUDO)
    (Path(env["PAM_MODULE_DIR"]) / "pam_doorman.so").unlink()
    result = run(env, "enable", ACK)
    assert result.returncode == FAILED
    assert "pam_doorman.so not found" in result.stderr
    assert sudo_file(env).read_text() == DEBIAN_SUDO


def test_disable_refuses_if_it_would_leave_no_password_path(env):
    only_doorman = f"#%PAM-1.0\n{MODULE_LINE}\naccount include system-auth\n"
    sudo_file(env).write_text(only_doorman)
    result = run(env, "disable", ACK)
    assert result.returncode == FAILED
    assert sudo_file(env).read_text() == only_doorman


def test_missing_file_and_bad_usage(env):
    assert run(env, "status").returncode == FAILED
    sudo_file(env).write_text(DEBIAN_SUDO)
    assert run(env).returncode == USAGE
    assert run(env, "frobnicate").returncode == USAGE
    assert run(env, "enable", ACK, "extra").returncode == USAGE
    assert sudo_file(env).read_text() == DEBIAN_SUDO


def test_leaves_no_temp_files_behind(env):
    sudo_file(env).write_text(DEBIAN_SUDO)
    run(env, "enable", ACK)
    run(env, "disable", ACK)
    leftovers = [p.name for p in Path(env["PAM_DIR"]).iterdir() if p.name.startswith(".sudo.")]
    assert leftovers == []
