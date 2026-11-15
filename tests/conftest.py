"""Suite-wide guard: no stub process started by a test may outlive the run.

The niri lock-script tests start fake lockers and watchers (shell loops). A test
that failed or was interrupted used to leave them running, and a leaked
`gtklock` stub then fooled `pgrep -x gtklock` on the host for 40+ minutes.
The stubs now end themselves and the tests reap them; this hook fails the run if
any process whose environment names THIS pytest process as STUB_OWNER is still
alive when the session ends.
"""

import os
import time
from pathlib import Path

LEAK_GRACE_SECS = 2.0


def _leaked_stub_processes() -> list[str]:
    me = str(os.getpid())
    leaked = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or entry.name == me:
            continue
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
            if f"STUB_OWNER={me}".encode() not in environ:
                continue
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue  # exited meanwhile, or not ours to read
        leaked.append(f"{entry.name}: {cmdline.strip()}")
    return leaked


def pytest_sessionfinish(session, exitstatus):
    deadline = time.monotonic() + LEAK_GRACE_SECS
    leaked = _leaked_stub_processes()
    while leaked and time.monotonic() < deadline:
        time.sleep(0.1)
        leaked = _leaked_stub_processes()
    if leaked:
        print("\nLEAKED STUB PROCESSES after the test session:")
        for line in leaked:
            print(f"  {line}")
        session.exitstatus = 1
