"""Tests for scripts/pad_protocol.py: statistics and the session against a stub `doorman`.

Run with: PYTHONPATH=src pytest tests/test_pad_protocol.py
The stub prints canned `doorman --json test` answers; no camera, daemon or
frame is involved.
"""

import datetime as dt
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pad_protocol.py"
spec = importlib.util.spec_from_file_location("pad_protocol", SCRIPT)
pp = importlib.util.module_from_spec(spec)
sys.modules["pad_protocol"] = pp
spec.loader.exec_module(pp)

STUB = """#!{python}
import json, os, sys
d = os.environ["STUB_DIR"]
with open(d + "/argv.log", "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
queue = json.load(open(d + "/queue.json"))
counter = d + "/n"
i = int(open(counter).read()) if os.path.exists(counter) else 0
open(counter, "w").write(str(i + 1))
entry = queue[min(i, len(queue) - 1)]
sys.stdout.write(entry["stdout"])
sys.exit(entry["exit"])
"""


def answer(decision, code, exit_code, bucket=None, liveness=None, **extra):
    body = {"decision": decision, "reason_code": code, "similarity_bucket": bucket,
            "liveness_ok": liveness, "frames_judged": None, "duration_ms": 1234}
    body.update(extra)
    return {"stdout": json.dumps(body), "exit": exit_code}


RECOGNIZED = answer("recognized", "matched", 0, 0.8, True)
LIVENESS_REJECT = answer("not_recognized", "liveness_rejected", 2, None, False)
NOT_MATCHED = answer("not_recognized", "not_matched", 2, 0.25, True)
NO_FACE = answer("not_recognized", "no_clear_face", 2)
ERROR = {"stdout": "", "exit": 1}


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def clock(self):
        return self.now


@pytest.fixture
def stub(tmp_path):
    directory = tmp_path / "stub"
    directory.mkdir()
    exe = directory / "doorman"
    exe.write_text(STUB.format(python=sys.executable))
    exe.chmod(0o755)
    os.environ["STUB_DIR"] = str(directory)

    def program(queue):
        (directory / "queue.json").write_text(json.dumps(queue))
        (directory / "n").unlink(missing_ok=True)
        (directory / "argv.log").unlink(missing_ok=True)

    program.exe = exe
    program.dir = directory
    program.calls = lambda: [json.loads(l) for l in (directory / "argv.log").read_text().splitlines()]
    yield program
    os.environ.pop("STUB_DIR", None)


def run_main(stub, tmp_path, extra, queue, time_=None):
    stub(queue)
    clock = time_ or FakeTime()
    out_lines = []
    args = ["--user", "alice", "--doorman", str(stub.exe), "--no-prompt", "--countdown", "0",
            "--pause", "0", "--shadow-log", "", "--out", str(tmp_path / "res"), *extra]
    code = pp.main(args, session_factory=lambda a: pp.Session(
        a, sleep=clock.sleep, clock=clock.clock, out=out_lines.append))
    return code, clock, out_lines


def load_summary(tmp_path):
    data = json.loads((tmp_path / "res.json").read_text())
    return {row["class"]: row for row in data["summary"]}, data


# ---- statistics -------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [(10, 0.2589), (20, 0.1391), (30, 0.0950), (100, 0.0295)])
def test_zero_event_upper_bound_is_one_minus_alpha_to_the_one_over_n(n, expected):
    assert pp.clopper_pearson_upper(0, n) == pytest.approx(expected, abs=5e-4)
    assert pp.clopper_pearson_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-9)


def test_two_sided_zero_of_100_is_the_3_6_percent_figure():
    assert pp.clopper_pearson_upper(0, 100, two_sided=True) == pytest.approx(0.0362, abs=5e-4)


@pytest.mark.parametrize("k,n,upper,lower", [(1, 10, 0.3942, 0.0051), (5, 10, 0.7776, 0.2224),
                                              (3, 20, 0.3437, 0.0422)])
def test_low_counts_match_known_one_sided_bounds(k, n, upper, lower):
    assert pp.clopper_pearson_upper(k, n) == pytest.approx(upper, abs=1e-3)
    assert pp.clopper_pearson_lower(k, n) == pytest.approx(lower, abs=1e-3)


def test_bounds_are_ordered_monotone_and_edge_cases_hold():
    assert pp.clopper_pearson_upper(10, 10) == 1.0
    assert pp.clopper_pearson_upper(0, 0) == 1.0
    assert pp.clopper_pearson_lower(0, 10) == 0.0
    uppers = [pp.clopper_pearson_upper(k, 30) for k in range(0, 31)]
    assert uppers == sorted(uppers)
    for k in range(0, 31):
        assert pp.clopper_pearson_lower(k, 30) <= k / 30 <= pp.clopper_pearson_upper(k, 30)
    with pytest.raises(ValueError):
        pp.clopper_pearson_upper(11, 10)


def test_binomial_cdf_known_values():
    assert pp.binomial_cdf(0, 10, 0.5) == pytest.approx(1 / 1024)
    assert pp.binomial_cdf(5, 10, 0.5) == pytest.approx(0.623046875)
    assert pp.binomial_cdf(10, 10, 0.3) == 1.0


# ---- interpretation of `doorman --json test` --------------------------------------

def test_parse_each_documented_outcome():
    ok = pp.parse_doorman_output(RECOGNIZED["stdout"], 0)
    assert (ok["decision"], ok["reason_code"], ok["similarity_bucket"]) == ("recognized", "matched", 0.8)
    rej = pp.parse_doorman_output(LIVENESS_REJECT["stdout"], 2)
    assert (rej["decision"], rej["liveness_ok"]) == ("not_recognized", False)
    sus = pp.parse_doorman_output(
        answer("suspended", "face_auth_suspended", 2, suspended_until="2026-01-01T00:00:30Z")["stdout"], 2)
    assert sus["decision"] == "suspended" and sus["suspended_until"] == "2026-01-01T00:00:30Z"


@pytest.mark.parametrize("stdout,code", [("", 1), ("not json", 2), ('{"decision": "weird"}', 0), ("[]", 0)])
def test_errors_and_garbage_are_infrastructure_never_a_decision(stdout, code):
    assert pp.parse_doorman_output(stdout, code)["decision"] == "error"


def test_seconds_until():
    now = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    assert pp.seconds_until("2026-01-01T00:01:30Z", now) == 90
    assert pp.seconds_until(None, now) is None and pp.seconds_until("garbage", now) is None


# ---- the session against the stub -------------------------------------------------

def test_a_full_run_counts_apcer_and_bpcer_and_writes_private_files(stub, tmp_path):
    queue = [RECOGNIZED] * 4 + [LIVENESS_REJECT] * 3 + [RECOGNIZED] + [LIVENESS_REJECT] * 2
    code, _, _ = run_main(stub, tmp_path, ["--classes", "genuine,phone_photo", "--genuine", "4",
                                           "--attempts", "6"], queue)
    assert code == 0
    summary, data = load_summary(tmp_path)
    genuine, photo = summary["genuine"], summary["phone_photo"]
    assert (genuine["accepted"], genuine["rejected"], genuine["rate"]) == (4, 0, 0.0)
    assert genuine["upper_95"] == pytest.approx(1 - 0.05 ** (1 / 4))
    assert (photo["n_responded"], photo["accepted"], photo["rejected"]) == (6, 1, 5)
    assert photo["rate"] == pytest.approx(1 / 6)
    assert photo["rate_name"] == "APCER" and genuine["rate_name"] == "BPCER"
    assert photo["reasons"] == {"matched": 1, "liveness_rejected": 5}
    assert "SCREENING, NOT CERTIFICATION" in data["headline"]
    assert stub.calls()[0] == ["--json", "test", "alice"]
    for name in ("res.json", "res.csv", "res-bench.csv"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def test_zero_accepted_attacks_still_report_the_upper_bound(stub, tmp_path):
    code, *_ = run_main(stub, tmp_path, ["--classes", "print_matte", "--attempts", "10"], [LIVENESS_REJECT])
    assert code == 0
    row = load_summary(tmp_path)[0]["print_matte"]
    assert (row["accepted"], row["rate"]) == (0, 0.0)
    assert row["upper_95"] == pytest.approx(0.2589, abs=5e-4)


def test_csv_columns_and_no_frame_data(stub, tmp_path):
    run_main(stub, tmp_path, ["--classes", "genuine", "--genuine", "2"], [RECOGNIZED])
    lines = (tmp_path / "res.csv").read_text().splitlines()
    assert lines[0].split(",")[:9] == ["class", "kind", "index", "decision", "reason_code",
                                       "similarity_bucket", "liveness_ok", "duration_ms", "timestamp"]
    assert lines[1].startswith("genuine,genuine,1,recognized,matched,0.8,true,1234,")
    text = (tmp_path / "res.json").read_text().lower()
    for forbidden in ("alice", "frame", "embedding", "image"):
        assert forbidden not in text.replace("frames_judged", "")


def test_no_clear_face_is_a_non_response_and_errors_are_excluded(stub, tmp_path):
    queue = [NO_FACE, ERROR, RECOGNIZED, LIVENESS_REJECT]
    code, *_ = run_main(stub, tmp_path, ["--classes", "phone_photo", "--attempts", "3"], queue)
    assert code == 0
    row = load_summary(tmp_path)[0]["phone_photo"]
    assert (row["non_response"], row["infra_excluded"], row["accepted"], row["rejected"]) == (1, 1, 1, 1)
    assert row["n_responded"] == 2
    # the errored attempt was retried, so the class still got its 3 valid presentations
    assert row["n_presented"] - row["infra_excluded"] == 3


def test_three_errors_in_a_row_stop_the_run_and_still_write_results(stub, tmp_path):
    code, *_ = run_main(stub, tmp_path, ["--classes", "genuine", "--genuine", "5"], [ERROR])
    assert code == 1
    summary, data = load_summary(tmp_path)
    assert summary["genuine"]["infra_excluded"] == 3 and summary["genuine"]["n_responded"] == 0
    assert data["meta"]["completed"] is False
    assert summary["genuine"]["rate"] is None and summary["genuine"]["upper_95"] is None


def test_suspended_waits_out_the_cooldown_then_repeats_the_attempt(stub, tmp_path):
    until = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    suspended = answer("suspended", "face_auth_suspended", 2, suspended_until=until)
    code, clock, out = run_main(stub, tmp_path, ["--classes", "genuine", "--genuine", "1"],
                                [suspended, RECOGNIZED])
    assert code == 0
    assert sum(clock.slept) >= 88, clock.slept  # the cooldown, not a retry storm
    assert len(stub.calls()) == 2
    assert any("suspended by the failure budget" in line for line in out)
    summary, data = load_summary(tmp_path)
    assert summary["genuine"]["n_presented"] == 1, "the refused run is not an attempt"
    assert len(data["meta"]["suspensions"]) == 1


def test_budget_pacing_keeps_judged_failures_below_the_limit(stub, tmp_path):
    code, clock, out = run_main(
        stub, tmp_path,
        ["--classes", "print_matte", "--attempts", "8", "--budget-max", "5", "--budget-window", "300"],
        [LIVENESS_REJECT])
    assert code == 0
    assert any("Pacing" in line for line in out)
    # never more than 4 judged failures inside any 300 s window of the fake clock
    assert sum(clock.slept) >= 300
    assert len(stub.calls()) == 8


def test_without_pacing_nothing_waits(stub, tmp_path):
    code, clock, _ = run_main(stub, tmp_path, ["--classes", "print_matte", "--attempts", "8"],
                              [LIVENESS_REJECT])
    assert code == 0 and sum(clock.slept) == 0


def test_bench_csv_feeds_pad_report(stub, tmp_path):
    queue = [RECOGNIZED, NO_FACE, LIVENESS_REJECT]
    run_main(stub, tmp_path, ["--classes", "genuine,phone_photo", "--genuine", "1", "--attempts", "2"],
             queue)
    rows = (tmp_path / "res-bench.csv").read_text().splitlines()
    assert rows[0] == "label,species,score,response,session,subject"
    assert rows[1].startswith("bona_fide,none,1.0,ok,")
    assert rows[2].startswith("attack,phone_photo,0.0,no_face,")
    assert rows[3].startswith("attack,phone_photo,0.0,ok,")


def test_shadow_log_records_are_correlated_when_readable(stub, tmp_path):
    log = tmp_path / "shadow.jsonl"
    log.write_text("")
    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    record = {"attempt_id": "a1", "timestamp_unix_ms": now_ms, "frames": {"judged": 9},
              "similarity": {"per_frame": {"median": 0.77, "min": 0.7}},
              "depth": {"relief": {"min": 0.11}, "margin_min": 0.06},
              "minifasnet": {"state": "scored", "logit_diff": 7.5}}
    stub([RECOGNIZED])
    log.write_text(json.dumps(record) + "\n")  # appended after the tail was opened? no: before.
    # The tail only reads what is appended after it starts, so append from the stub side.
    tail = pp.ShadowTail(str(log))
    with open(log, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    found = pp.match_shadow(tail.new_records(), now_ms - 10, now_ms + 10)
    assert found == {"attempt_id": "a1", "frames_judged": 9, "similarity_median": 0.77,
                     "similarity_min": 0.7, "depth_relief_min": 0.11, "depth_margin_min": 0.06,
                     "minifasnet_state": "scored", "minifasnet_logit_diff": 7.5}
    assert pp.match_shadow([record], now_ms + 60000, now_ms + 61000) is None


def test_unreadable_shadow_log_is_silently_skipped(tmp_path):
    tail = pp.ShadowTail(str(tmp_path / "missing.jsonl"))
    assert not tail.available and tail.new_records() == []


def test_help_and_unknown_class(capsys):
    with pytest.raises(SystemExit) as info:
        pp.main(["--user", "x", "--classes", "nope", "--no-prompt"])
    assert "unknown class" in str(info.value)


def test_default_output_path_is_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pp.default_prefix(dt.date(2026, 11, 10)) == tmp_path / "doorman-pad-2026-11-10"
