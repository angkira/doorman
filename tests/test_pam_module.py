"""pam_doorman.so driven by pamtester against a fake daemon socket.

Builds the real module, writes a PAM service file that loads it with the
`socket=` module argument (pam_module/src/lib.rs), and answers on a Python UNIX
socket speaking the daemon's JSON-line protocol.

libpam reads service files only from /etc/pam.d, so pamtester runs inside a
private mount namespace (`unshare -m`) in which a temp directory is bind-mounted
over /etc/pam.d. The host's /etc is never modified. That needs root (or
`sudo -n`); without it, or without pamtester, the tests skip with the reason.

Run with: PYTHONPATH=src python3 -m pytest tests/test_pam_module.py
"""

import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="PAM is Linux-only"
)

TEST_USER = "nobody"
SERVICE = "doorman-test"
# pam_strerror() texts, as printed by pamtester.
MSG_AUTH_ERR = "Authentication failure"
MSG_AUTHINFO_UNAVAIL = "cannot retrieve authentication info"
# The module's reply budget for a non-sudo service is AUTH_TIMEOUT_SECS (3) in
# shared/src/lib.rs, plus 1 s deadline slack. Keep the bound loose for CI.
TIMEOUT_CASE_MAX_S = 6.0
# Spec: a down daemon costs less than this on top of the PAM stack itself.
DOWN_OVERHEAD_MAX_S = 0.100
OVERHEAD_RUNS = 7


def _skip(reason):
    """Skip, or fail when REQUIRE_PAM_TESTS=1 (CI: a skip proves nothing)."""
    if os.environ.get("REQUIRE_PAM_TESTS") == "1":
        pytest.fail(f"required but unavailable: {reason}")
    pytest.skip(reason)


def _privileged_prefix():
    """Command prefix that gives a root process, or None."""
    if os.geteuid() == 0:
        return []
    if shutil.which("sudo") and subprocess.run(
        ["sudo", "-n", "true"], capture_output=True
    ).returncode == 0:
        return ["sudo", "-n"]
    return None


@pytest.fixture(scope="session")
def tools():
    pamtester = shutil.which("pamtester")
    if not pamtester:
        _skip("pamtester is not installed (apt install pamtester)")
    unshare = shutil.which("unshare")
    if not unshare:
        _skip("unshare (util-linux) is not installed")
    prefix = _privileged_prefix()
    if prefix is None:
        _skip("needs root or passwordless sudo to bind-mount over /etc/pam.d")
    probe = subprocess.run(
        prefix + [unshare, "-m", "true"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        _skip(f"cannot create a mount namespace: {probe.stderr.strip()}")
    return {"pamtester": pamtester, "unshare": unshare, "prefix": prefix}


@pytest.fixture(scope="session")
def module_path():
    if not shutil.which("cargo"):
        _skip("cargo is not installed")
    out = subprocess.run(
        ["cargo", "build", "--locked", "-p", "pam_doorman", "--message-format=json"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    for line in out.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact" and msg["target"]["name"] == "pam_doorman":
            for f in msg["filenames"]:
                if f.endswith(".so"):
                    return Path(f)
    pytest.fail("cargo did not report a pam_doorman .so artifact")


class FakeDaemon:
    """One JSON-line reply per connection; records the request lines."""

    def __init__(self, path, reply):
        self.path = path
        self.reply = reply  # dict, or None to hold the connection without answering
        self.requests = []
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(path))
        os.chmod(path, 0o666)
        self._srv.listen(8)
        self._srv.settimeout(0.1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        with conn:
            conn.settimeout(10)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
            self.requests.append(json.loads(buf))
            if self.reply is None:
                self._stop.wait(10)  # silent daemon
                return
            conn.sendall(json.dumps(self.reply).encode() + b"\n")

    def close(self):
        self._stop.set()
        self._srv.close()


class Env:
    def __init__(self, tools, module, root):
        self.tools = tools
        self.module = module
        self.root = Path(root)
        self.pam_d = self.root / "pam.d"
        self.pam_d.mkdir()
        self.sock = self.root / "doorman.sock"
        self.script = self.root / "run.sh"
        self.script.write_text(
            "#!/bin/sh\n"
            f'mount --bind "{self.pam_d}" /etc/pam.d || exit 90\n'
            'exec "$@"\n'
        )

    def service(self, stack, name=SERVICE):
        (self.pam_d / name).write_text(stack + "\n")
        return name

    def doorman_line(self, control="required"):
        return f"auth {control} {self.module} socket={self.sock}"

    def run(self, service):
        cmd = self.tools["prefix"] + [
            self.tools["unshare"],
            "-m",
            "sh",
            str(self.script),
            self.tools["pamtester"],
            service,
            TEST_USER,
            "authenticate",
        ]
        started = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return proc, time.monotonic() - started


@pytest.fixture()
def env(tools, module_path):
    with tempfile.TemporaryDirectory(prefix="pam-doorman-") as root:
        os.chmod(root, 0o755)
        yield Env(tools, module_path, root)


@pytest.fixture()
def daemon(env):
    started = []

    def start(reply):
        d = FakeDaemon(env.sock, reply)
        started.append(d)
        return d

    yield start
    for d in started:
        d.close()


def output(proc):
    return proc.stdout + proc.stderr


def test_success_reply_is_pam_success(env, daemon):
    d = daemon({"status": "success"})
    proc, _ = env.run(env.service(env.doorman_line()))
    assert proc.returncode == 0, output(proc)
    assert "successfully authenticated" in output(proc)
    (req,) = d.requests
    assert req["type"] == "authenticate"
    assert req["username"] == TEST_USER
    assert req["service"] == SERVICE
    assert req["protocol_version"] == 1


def test_failure_reply_is_auth_err(env, daemon):
    daemon({"status": "failure", "reason": "no match"})
    proc, _ = env.run(env.service(env.doorman_line()))
    assert proc.returncode != 0
    assert MSG_AUTH_ERR in output(proc)


@pytest.mark.parametrize(
    "reply",
    [
        {"status": "denied", "reason": "not permitted"},
        {"status": "rate_limited", "retry_after_s": 5},
        {"status": "unsupported_protocol", "reason": "x", "supported_version": 1},
    ],
    ids=["denied", "rate_limited", "unsupported_protocol"],
)
def test_refusals_are_authinfo_unavail(env, daemon, reply):
    daemon(reply)
    proc, _ = env.run(env.service(env.doorman_line()))
    assert proc.returncode != 0
    assert MSG_AUTHINFO_UNAVAIL in output(proc)


def test_refusal_falls_through_to_the_next_module(env, daemon):
    daemon({"status": "denied", "reason": "not permitted"})
    stack = env.doorman_line("sufficient") + "\nauth required pam_permit.so"
    proc, _ = env.run(env.service(stack))
    assert proc.returncode == 0, output(proc)


def test_success_short_circuits_a_sufficient_stack(env, daemon):
    daemon({"status": "success"})
    stack = env.doorman_line("sufficient") + "\nauth required pam_deny.so"
    proc, _ = env.run(env.service(stack))
    assert proc.returncode == 0, output(proc)


def test_failure_does_not_skip_the_password_modules(env, daemon):
    daemon({"status": "failure", "reason": "no match"})
    stack = env.doorman_line("sufficient") + "\nauth required pam_deny.so"
    proc, _ = env.run(env.service(stack))
    assert proc.returncode != 0


def test_daemon_down_falls_through_with_under_100ms_overhead(env):
    assert not env.sock.exists()
    baseline = env.service("auth required pam_deny.so", name="doorman-baseline")
    with_module = env.service(env.doorman_line(), name=SERVICE)
    base, mod = [], []
    for _ in range(OVERHEAD_RUNS):
        proc, t = env.run(baseline)
        assert proc.returncode != 0
        base.append(t)
        proc, t = env.run(with_module)
        assert MSG_AUTHINFO_UNAVAIL in output(proc), output(proc)
        mod.append(t)
    overhead = statistics.median(mod) - statistics.median(base)
    print(f"daemon-down median overhead: {overhead * 1000:.1f} ms")
    assert overhead < DOWN_OVERHEAD_MAX_S, f"median overhead {overhead * 1000:.0f} ms"


def test_silent_daemon_times_out_and_falls_through(env, daemon):
    daemon(None)
    stack = env.doorman_line("sufficient") + "\nauth required pam_permit.so"
    proc, elapsed = env.run(env.service(stack))
    assert proc.returncode == 0, output(proc)  # fell through to pam_permit
    assert elapsed < TIMEOUT_CASE_MAX_S, f"{elapsed:.1f} s"
    assert elapsed >= 2.5, f"returned before the reply budget: {elapsed:.1f} s"


def test_root_never_uses_the_daemon(env, daemon):
    d = daemon({"status": "success"})
    env.service(env.doorman_line())
    cmd = env.tools["prefix"] + [
        env.tools["unshare"], "-m", "sh", str(env.script),
        env.tools["pamtester"], SERVICE, "root", "authenticate",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert MSG_AUTHINFO_UNAVAIL in output(proc)
    assert d.requests == []
