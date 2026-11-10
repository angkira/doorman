#!/usr/bin/env python3
"""DEPRECATED: use `doorman pad run` (native, calibration-aware; docs/PAD-PROTOCOL.md).

Presentation-attack screening protocol for doorman (interactive, stdlib only).

You run it, in front of the camera, with a phone and a printed photo of yourself.
It walks through genuine and attack classes, runs `doorman --json test <user>`
once per attempt, and records only the decision, reason code, similarity bucket,
liveness flag, duration and time. It never captures, stores or shows a frame.

At the end it prints and writes (mode 0600) JSON, CSV and a CSV in the format of
`doorman-bench pad-report`, with per-class counts, APCER / BPCER and the
one-sided 95 % Clopper-Pearson upper bound. This is SCREENING, not certification:
a handful of attempts per class gives wide bounds (0 of 10 accepted still allows
a true rate up to 25.9 %).

Failure budget: every `doorman test` counts toward `[ipc] face_max_judged_failures`
(default 5 judged failures in 5 minutes, then a 30 s / 2 min / 10 min lockout),
and attack attempts are SUPPOSED to fail. For the session, raise the limit in
doorman.toml (0 disables it) and restart the daemon, then put it back. Without
that the script still works: it waits out every cooldown (see --budget-max for
pacing that avoids tripping it) and says so.

Usage:  python3 scripts/pad_protocol.py --user NAME
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL_VERSION = 1
CONFIDENCE = 0.95
DEFAULT_GENUINE = 10
DEFAULT_ATTACK = 10
DEFAULT_CONDITION = 5
DEFAULT_COUNTDOWN_S = 3.0
DEFAULT_PAUSE_S = 2.0
DEFAULT_TEST_TIMEOUT_S = 45.0
DEFAULT_SHADOW_LOG = "/run/doorman/shadow-pad.jsonl"
SHADOW_WAIT_S = 1.5
SHADOW_MATCH_BEFORE_MS = 1000
SHADOW_MATCH_AFTER_MS = 4000
COOLDOWN_MARGIN_S = 2.0
MAX_CONSECUTIVE_INFRA = 3
BUDGET_WINDOW_S = 300.0

EXIT_RECOGNIZED = 0
EXIT_ERROR = 1
EXIT_NOT_RECOGNIZED = 2

# `reason_code`s that count as a judged failure toward the daemon's budget.
JUDGED_FAILURE_CODES = {"not_matched", "liveness_rejected", "identity_floor", "ambiguous_faces"}
# The system found nothing to judge: reported apart from accept / reject.
NON_RESPONSE_CODES = {"no_clear_face", "no_frames"}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * log_p + (n - i) * log_q)
        total += math.exp(log_term)
    return min(1.0, total)


def _bisect(f, lo: float = 0.0, hi: float = 1.0, iterations: int = 200) -> float:
    """Root of a function that is decreasing in p and crosses zero in [lo, hi]."""
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_upper(k: int, n: int, confidence: float = CONFIDENCE,
                          two_sided: bool = False) -> float:
    """Exact upper confidence bound of a binomial proportion (k events in n trials).

    One-sided by default: the p for which P(X <= k | n, p) = 1 - confidence.
    `two_sided=True` gives the upper end of the two-sided interval instead
    (alpha / 2 in the tail), which is wider. Found by bisection on the binomial
    CDF, no libraries. n = 0 has no information: the bound is 1.
    """
    if n <= 0:
        return 1.0
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside 0..{n}")
    if k >= n:
        return 1.0
    alpha = (1.0 - confidence) / (2.0 if two_sided else 1.0)
    return _bisect(lambda p: binomial_cdf(k, n, p) - alpha)


def clopper_pearson_lower(k: int, n: int, confidence: float = CONFIDENCE,
                          two_sided: bool = False) -> float:
    """Exact lower bound: the p for which P(X >= k | n, p) = 1 - confidence."""
    if n <= 0 or k <= 0:
        return 0.0
    alpha = (1.0 - confidence) / (2.0 if two_sided else 1.0)
    # P(X >= k) = 1 - P(X <= k-1) is increasing in p; bisect on its negation.
    return _bisect(lambda p: alpha - (1.0 - binomial_cdf(k - 1, n, p)))


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AttackClass:
    name: str
    kind: str  # "genuine" (bona fide presentation) or "attack"
    title: str
    instruction: str
    group: str  # "baseline", "attack" or "condition"


CLASSES: list[AttackClass] = [
    AttackClass("genuine", "genuine", "Genuine owner, baseline",
                "Look at the camera normally, at your usual distance, good light, no glasses change.",
                "baseline"),
    AttackClass("phone_photo", "attack", "Phone: still photo of the owner",
                "Show a photo of YOUR face on the phone screen, filling the view, brightness at maximum. "
                "No live face in view.", "attack"),
    AttackClass("phone_replay", "attack", "Phone: video replay of the owner",
                "Play a video of YOUR face on the phone (blinking, moving), screen to the camera, "
                "maximum brightness. No live face in view.", "attack"),
    AttackClass("print_matte", "attack", "Printed photo of the owner",
                "Hold a printed photo of YOUR face (face about 8-10 cm wide) in front of the camera. "
                "Move it slightly. No live face in view.", "attack"),
    AttackClass("mixed_track", "attack", "Photo next to a live face",
                "Show your LIVE face and, beside it in the same view, a phone or print with a face "
                "(yours or a lookalike).", "attack"),
    AttackClass("genuine_dark", "genuine", "Genuine: dark room",
                "Switch the room light off or dim it a lot; look at the camera.", "condition"),
    AttackClass("genuine_backlight", "genuine", "Genuine: backlight",
                "Sit with a bright window or lamp behind you; look at the camera.", "condition"),
    AttackClass("genuine_glasses_on", "genuine", "Genuine: glasses on",
                "Wear your glasses (if you have none, skip this class with --classes).", "condition"),
    AttackClass("genuine_glasses_off", "genuine", "Genuine: glasses off",
                "Take your glasses off (skip if you have none).", "condition"),
    AttackClass("genuine_yaw_left", "genuine", "Genuine: head turned about 20 degrees left",
                "Turn your head about 20 degrees to the left, eyes on the camera.", "condition"),
    AttackClass("genuine_yaw_right", "genuine", "Genuine: head turned about 20 degrees right",
                "Turn your head about 20 degrees to the right, eyes on the camera.", "condition"),
    AttackClass("genuine_near", "genuine", "Genuine: near",
                "Sit close: about 30 cm from the camera.", "condition"),
    AttackClass("genuine_far", "genuine", "Genuine: far",
                "Sit far: about 1 m from the camera.", "condition"),
]
CLASS_BY_NAME = {c.name: c for c in CLASSES}


def build_plan(classes: list[AttackClass], genuine_n: int, attack_n: int,
               condition_n: int) -> list[tuple[AttackClass, int]]:
    """(class, 1-based index) for every attempt, class by class."""
    counts = {"baseline": genuine_n, "attack": attack_n, "condition": condition_n}
    return [(c, i) for c in classes for i in range(1, counts[c.group] + 1)]


# --------------------------------------------------------------------------
# Running `doorman --json test`
# --------------------------------------------------------------------------

@dataclass
class Attempt:
    cls: str
    kind: str
    index: int
    decision: str
    reason_code: str
    similarity_bucket: float | None
    liveness_ok: bool | None
    duration_ms: int | None
    timestamp: str
    exit_code: int | None
    suspended_until: str | None = None
    shadow: dict | None = None
    note: str = ""


def parse_doorman_output(stdout: str, exit_code: int) -> dict:
    """Interpret one `doorman --json test` run. Never raises.

    Returns the report fields; an unreadable answer or an exit code of 1 is an
    `error` (infrastructure), kept apart from a rejection.
    """
    report: dict = {}
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            report = parsed
    except (json.JSONDecodeError, ValueError):
        pass
    decision = report.get("decision")
    reason = report.get("reason_code")
    if exit_code == EXIT_ERROR or decision not in {"recognized", "not_recognized", "suspended", "error"}:
        return {"decision": "error", "reason_code": reason or "unparsable_output",
                "similarity_bucket": None, "liveness_ok": None,
                "duration_ms": report.get("duration_ms"), "suspended_until": None}
    return {"decision": decision, "reason_code": reason or "unknown",
            "similarity_bucket": report.get("similarity_bucket"),
            "liveness_ok": report.get("liveness_ok"),
            "duration_ms": report.get("duration_ms"),
            "suspended_until": report.get("suspended_until")}


def run_doorman(doorman: str, user: str, timeout_s: float) -> dict:
    """One `doorman --json test USER`; returns the parsed fields plus exit_code."""
    started = time.monotonic()
    try:
        done = subprocess.run([doorman, "--json", "test", user], capture_output=True,
                              text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"decision": "error", "reason_code": "timeout", "similarity_bucket": None,
                "liveness_ok": None, "duration_ms": int((time.monotonic() - started) * 1000),
                "suspended_until": None, "exit_code": None}
    except OSError as exc:
        return {"decision": "error", "reason_code": f"cannot_run:{type(exc).__name__}",
                "similarity_bucket": None, "liveness_ok": None, "duration_ms": None,
                "suspended_until": None, "exit_code": None}
    fields = parse_doorman_output(done.stdout, done.returncode)
    fields["exit_code"] = done.returncode
    return fields


def seconds_until(rfc3339: str | None, now: dt.datetime | None = None) -> float | None:
    """Seconds from `now` to an RFC 3339 UTC time (`2026-01-01T00:00:00Z`), or None."""
    if not rfc3339:
        return None
    try:
        until = dt.datetime.fromisoformat(rfc3339.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = now or dt.datetime.now(dt.timezone.utc)
    return (until - now).total_seconds()


# --------------------------------------------------------------------------
# Optional correlation with the daemon's shadow PAD log
# --------------------------------------------------------------------------

class ShadowTail:
    """Reads records appended to the shadow log after construction. Silent when
    the file is missing or unreadable (the daemon runs as another user)."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.offset = self._size()
        self.available = self.offset is not None

    def _size(self) -> int | None:
        if not self.path:
            return None
        try:
            return os.stat(self.path).st_size
        except OSError:
            return None

    def new_records(self) -> list[dict]:
        if not self.path or self.offset is None:
            return []
        try:
            with open(self.path, "rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                if size < self.offset:  # rotated
                    self.offset = 0
                handle.seek(self.offset)
                data = handle.read()
                self.offset += len(data)
        except OSError:
            return []
        records = []
        for line in data.decode("utf-8", "replace").splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records


def summarize_shadow(record: dict) -> dict:
    """The fields worth keeping next to an attempt (aggregates only)."""
    sim = (record.get("similarity") or {}).get("per_frame") or {}
    depth = record.get("depth") or {}
    relief = depth.get("relief") or {}
    mini = record.get("minifasnet") or {}
    return {
        "attempt_id": record.get("attempt_id"),
        "frames_judged": (record.get("frames") or {}).get("judged"),
        "similarity_median": sim.get("median"),
        "similarity_min": sim.get("min"),
        "depth_relief_min": relief.get("min"),
        "depth_margin_min": depth.get("margin_min"),
        "minifasnet_state": mini.get("state"),
        "minifasnet_logit_diff": mini.get("logit_diff"),
    }


def match_shadow(records: list[dict], started_ms: int, ended_ms: int) -> dict | None:
    """The record whose timestamp falls inside the attempt (with some slack)."""
    lo, hi = started_ms - SHADOW_MATCH_BEFORE_MS, ended_ms + SHADOW_MATCH_AFTER_MS
    inside = [r for r in records
              if isinstance(r.get("timestamp_unix_ms"), int) and lo <= r["timestamp_unix_ms"] <= hi]
    return summarize_shadow(inside[-1]) if inside else None


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def summarize(attempts: list[Attempt]) -> list[dict]:
    """Per-class counts and rates.

    accepted / rejected / non_response are the valid presentations; infra
    (errors) is excluded and listed. APCER = accepted / responded for an attack
    class; BPCER = rejected / responded for a genuine class (`bpcer_with_nr`
    also counts the non-responses as failures). The upper bound is the one-sided
    95 % Clopper-Pearson bound of that rate.
    """
    rows = []
    for name in dict.fromkeys(a.cls for a in attempts):
        group = [a for a in attempts if a.cls == name]
        kind = group[0].kind
        accepted = sum(1 for a in group if a.decision == "recognized")
        rejected = sum(1 for a in group if a.decision == "not_recognized"
                       and a.reason_code not in NON_RESPONSE_CODES)
        non_response = sum(1 for a in group if a.decision == "not_recognized"
                           and a.reason_code in NON_RESPONSE_CODES)
        infra = len(group) - accepted - rejected - non_response
        responded = accepted + rejected
        reasons: dict[str, int] = {}
        for a in group:
            reasons[a.reason_code] = reasons.get(a.reason_code, 0) + 1
        errors = accepted if kind == "attack" else rejected
        rate = errors / responded if responded else None
        upper = clopper_pearson_upper(errors, responded) if responded else None
        row = {
            "class": name, "kind": kind, "n_presented": len(group), "n_responded": responded,
            "accepted": accepted, "rejected": rejected, "non_response": non_response,
            "infra_excluded": infra,
            "rate_name": "APCER" if kind == "attack" else "BPCER",
            "rate": rate, "upper_95": upper, "reasons": reasons,
        }
        if kind == "genuine":
            with_nr = responded + non_response
            row["bpcer_with_nr"] = (rejected + non_response) / with_nr if with_nr else None
        rows.append(row)
    return rows


def headline(rows: list[dict]) -> str:
    attacks = [r for r in rows if r["kind"] == "attack" and r["n_responded"]]
    genuine = [r for r in rows if r["kind"] == "genuine" and r["n_responded"]]
    parts = []
    if attacks:
        worst = max(attacks, key=lambda r: (r["upper_95"], r["rate"]))
        parts.append(f"worst attack class: {worst['class']} accepted {worst['accepted']}/"
                     f"{worst['n_responded']} (APCER {pct(worst['rate'])}, 95% upper bound "
                     f"{pct(worst['upper_95'])})")
    if genuine:
        base = next((r for r in genuine if r["class"] == "genuine"), genuine[0])
        parts.append(f"genuine baseline: rejected {base['rejected']}/{base['n_responded']} "
                     f"(BPCER {pct(base['rate'])}, 95% upper bound {pct(base['upper_95'])})")
    parts.append("SCREENING, NOT CERTIFICATION: small samples give wide bounds, one phone and "
                 "one lighting setup are not the world")
    return "; ".join(parts)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def format_table(rows: list[dict]) -> str:
    head = f"{'class':<22}{'kind':<8}{'n':>4}{'resp':>6}{'acc':>5}{'rej':>5}{'nr':>4}{'infra':>6}" \
           f"{'rate':>8}{'95% upper':>11}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['class']:<22}{r['kind']:<8}{r['n_presented']:>4}{r['n_responded']:>6}"
                     f"{r['accepted']:>5}{r['rejected']:>5}{r['non_response']:>4}{r['infra_excluded']:>6}"
                     f"{pct(r['rate']):>8}{pct(r['upper_95']):>11}")
    lines.append("rate = APCER (attack accepted) or BPCER (genuine rejected); nr = no face / no frames; "
                 "infra = errors, excluded")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------

CSV_COLUMNS = ["class", "kind", "index", "decision", "reason_code", "similarity_bucket",
               "liveness_ok", "duration_ms", "timestamp", "shadow_depth_margin_min",
               "shadow_minifasnet_logit_diff", "shadow_similarity_median"]
BENCH_COLUMNS = ["label", "species", "score", "response", "session", "subject"]


def write_private(path: Path, text: str) -> None:
    """Write `text` to `path` with mode 0600 (also when the file existed)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", newline="") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def attempts_csv(attempts: list[Attempt]) -> str:
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(CSV_COLUMNS)
    for a in attempts:
        shadow = a.shadow or {}
        writer.writerow([a.cls, a.kind, a.index, a.decision, a.reason_code,
                         "" if a.similarity_bucket is None else a.similarity_bucket,
                         "" if a.liveness_ok is None else str(a.liveness_ok).lower(),
                         "" if a.duration_ms is None else a.duration_ms, a.timestamp,
                         shadow.get("depth_margin_min", ""), shadow.get("minifasnet_logit_diff", ""),
                         shadow.get("similarity_median", "")])
    return buffer.getvalue()


def bench_csv(attempts: list[Attempt], session: str) -> str:
    """Input for `doorman-bench pad-report --threshold 0.5`: score 1.0 = accepted,
    0.0 = rejected; no-face / no-frames rows are non-responses; errors are left out."""
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(BENCH_COLUMNS)
    for a in attempts:
        if a.decision == "recognized":
            response, score = "ok", 1.0
        elif a.decision == "not_recognized" and a.reason_code in NON_RESPONSE_CODES:
            response, score = "no_face", 0.0
        elif a.decision == "not_recognized":
            response, score = "ok", 0.0
        else:
            continue
        label = "attack" if a.kind == "attack" else "bona_fide"
        species = a.cls if a.kind == "attack" else "none"
        writer.writerow([label, species, score, response, session, "owner"])
    return buffer.getvalue()


def results_json(attempts: list[Attempt], rows: list[dict], meta: dict, line: str) -> str:
    return json.dumps({
        "protocol_version": PROTOCOL_VERSION,
        "meta": meta,
        "summary": rows,
        "headline": line,
        "attempts": [a.__dict__ for a in attempts],
    }, indent=2) + "\n"


def default_prefix(today: dt.date | None = None) -> Path:
    today = today or dt.date.today()
    return Path.home() / f"doorman-pad-{today.isoformat()}"


# --------------------------------------------------------------------------
# The interactive session
# --------------------------------------------------------------------------

@dataclass
class Session:
    args: argparse.Namespace
    sleep: callable = time.sleep
    clock: callable = time.monotonic
    ask: callable = input
    out: callable = print
    now: callable = lambda: dt.datetime.now(dt.timezone.utc)
    attempts: list[Attempt] = field(default_factory=list)
    suspensions: list[dict] = field(default_factory=list)
    judged_failures: list[float] = field(default_factory=list)
    tail: ShadowTail | None = None

    def prompt(self, text: str) -> None:
        if not self.args.no_prompt:
            self.ask(text)

    def wait_out_cooldown(self, until: str | None, why: str) -> None:
        remaining = seconds_until(until, self.now())
        wait = max(remaining if remaining is not None else 30.0, 1.0) + COOLDOWN_MARGIN_S
        self.out(f"  Face auth is suspended by the failure budget ({why}); waiting {wait:.0f} s. "
                 "This is the budget doing its job. Raise [ipc] face_max_judged_failures for the "
                 "session to avoid it.")
        self.suspensions.append({"at": self.now().isoformat(), "wait_s": round(wait, 1), "why": why})
        left = wait
        while left > 0:
            step = min(left, 15.0)
            self.sleep(step)
            left -= step
            if left > 0 and wait > 30:
                self.out(f"  ... {left:.0f} s left")

    def pace_for_budget(self) -> None:
        """With --budget-max N: never have more than N-1 judged failures inside the
        budget window, by waiting for the oldest one to age out."""
        limit = self.args.budget_max
        if limit <= 0:
            return
        window = self.args.budget_window
        while True:
            now = self.clock()
            self.judged_failures = [t for t in self.judged_failures if now - t < window]
            if len(self.judged_failures) < max(limit - 1, 1):
                return
            wait = window - (now - self.judged_failures[0]) + 1.0
            self.out(f"  Pacing: {len(self.judged_failures)} judged failures in the last "
                     f"{window:.0f} s; waiting {wait:.0f} s to stay under the budget of {limit}.")
            self.sleep(wait)

    def one_attempt(self, cls: AttackClass, index: int, total_in_class: int) -> Attempt | None:
        consecutive_infra = 0
        while True:
            self.pace_for_budget()
            self.prompt(f"  [{cls.name} {index}/{total_in_class}] Prepare, then press Enter ... ")
            if self.args.countdown > 0:
                self.out(f"  starting in {self.args.countdown:g} s ...")
                self.sleep(self.args.countdown)
            started = self.now()
            fields = run_doorman(self.args.doorman, self.args.user, self.args.timeout)
            ended = self.now()
            if fields["decision"] == "suspended":
                self.wait_out_cooldown(fields.get("suspended_until"), "doorman test was refused")
                continue  # the refused run did not look at a face: do it again
            if fields["reason_code"] in JUDGED_FAILURE_CODES:
                self.judged_failures.append(self.clock())
            shadow = None
            if self.tail and self.tail.available:
                deadline = self.clock() + SHADOW_WAIT_S
                records: list[dict] = []
                while True:
                    records += self.tail.new_records()
                    shadow = match_shadow(records, int(started.timestamp() * 1000),
                                          int(ended.timestamp() * 1000))
                    if shadow or self.clock() >= deadline:
                        break
                    self.sleep(0.1)
            attempt = Attempt(
                cls=cls.name, kind=cls.kind, index=index, decision=fields["decision"],
                reason_code=fields["reason_code"], similarity_bucket=fields["similarity_bucket"],
                liveness_ok=fields["liveness_ok"], duration_ms=fields["duration_ms"],
                timestamp=started.isoformat(timespec="seconds"), exit_code=fields.get("exit_code"),
                suspended_until=fields.get("suspended_until"), shadow=shadow)
            label = {"recognized": "ACCEPTED", "not_recognized": "rejected"}.get(
                attempt.decision, attempt.decision.upper())
            self.out(f"  -> {label} ({attempt.reason_code})")
            if attempt.decision == "error":
                consecutive_infra += 1
                if consecutive_infra >= MAX_CONSECUTIVE_INFRA:
                    self.attempts.append(attempt)
                    self.out(f"  {MAX_CONSECUTIVE_INFRA} errors in a row: is the daemon running "
                             "and the user enrolled? Stopping.")
                    return None
                self.attempts.append(attempt)
                self.out("  (error: recorded as infrastructure, excluded from the rates; retrying)")
                continue
            self.attempts.append(attempt)
            self.sleep(self.args.pause)
            return attempt

    def preflight(self) -> bool:
        self.out("doorman presentation-attack screening protocol")
        self.out("  Nothing is saved except decisions: no frames, no screenshots.")
        self.out("  Every attempt runs `doorman --json test`, which counts toward the failure budget")
        self.out("  ([ipc] face_max_judged_failures, default 5 in 5 min, then 30 s / 2 min / 10 min).")
        self.out("  Recommended: set face_max_judged_failures = 0 in doorman.toml for this session,")
        self.out("  restart doormand, and restore it afterwards. Otherwise this script waits out")
        self.out("  every cooldown (use --budget-max 5 to pace the attempts and avoid most of them).")
        self.out("  Also set [dataset] keep_raw_frames = false for the session, or the daemon stores")
        self.out("  the attack frames and your own.")
        if self.args.no_prompt:
            return True
        answer = self.ask("Continue? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    def run(self, plan: list[tuple[AttackClass, int]]) -> bool:
        totals: dict[str, int] = {}
        for cls, _ in plan:
            totals[cls.name] = totals.get(cls.name, 0) + 1
        current = None
        for cls, index in plan:
            if cls.name != current:
                current = cls.name
                self.out("")
                self.out(f"== {cls.title} ({totals[cls.name]} attempts) ==")
                self.out(f"   {cls.instruction}")
            if self.one_attempt(cls, index, totals[cls.name]) is None:
                return False
        return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True, help="enrolled user to test (the owner)")
    p.add_argument("--doorman", default="doorman", help="doorman CLI (default: doorman on PATH)")
    p.add_argument("--genuine", type=int, default=DEFAULT_GENUINE, help="baseline genuine attempts")
    p.add_argument("--attempts", type=int, default=DEFAULT_ATTACK, help="attempts per attack class")
    p.add_argument("--condition-attempts", type=int, default=DEFAULT_CONDITION,
                   help="attempts per genuine-condition class (dark, backlight, glasses, yaw, distance)")
    p.add_argument("--classes", help="comma separated subset of: " + ", ".join(CLASS_BY_NAME))
    p.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN_S,
                   help="seconds to wait after Enter before the attempt")
    p.add_argument("--pause", type=float, default=DEFAULT_PAUSE_S, help="seconds between attempts")
    p.add_argument("--timeout", type=float, default=DEFAULT_TEST_TIMEOUT_S,
                   help="give up on one `doorman test` after this many seconds")
    p.add_argument("--budget-max", type=int, default=0,
                   help="your [ipc] face_max_judged_failures: pace attempts so fewer than this many "
                        "judged failures fall inside --budget-window (0 = no pacing; cooldowns are "
                        "still waited out)")
    p.add_argument("--budget-window", type=float, default=BUDGET_WINDOW_S,
                   help="your [ipc] face_failure_window_secs")
    p.add_argument("--shadow-log", default=DEFAULT_SHADOW_LOG,
                   help="daemon shadow PAD log to correlate with, if readable ('' to skip)")
    p.add_argument("--out", help="output path prefix (default ~/doorman-pad-<date>); writes "
                                 ".json, .csv and -bench.csv with mode 0600")
    p.add_argument("--no-prompt", action="store_true", help="never wait for Enter (tests, scripting)")
    args = p.parse_args(argv)
    for name in ("genuine", "attempts", "condition_attempts"):
        if getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0")
    return args


def select_classes(spec: str | None) -> list[AttackClass]:
    if not spec:
        return list(CLASSES)
    chosen = []
    for name in (s.strip() for s in spec.split(",") if s.strip()):
        if name not in CLASS_BY_NAME:
            raise SystemExit(f"unknown class {name!r}; choose from: {', '.join(CLASS_BY_NAME)}")
        chosen.append(CLASS_BY_NAME[name])
    return chosen


def main(argv: list[str] | None = None, session_factory=Session) -> int:
    args = parse_args(argv)
    classes = select_classes(args.classes)
    plan = build_plan(classes, args.genuine, args.attempts, args.condition_attempts)
    session = session_factory(args)
    session.tail = ShadowTail(args.shadow_log) if args.shadow_log else None
    if not session.preflight():
        print("aborted")
        return 1
    if session.tail and not session.tail.available:
        print(f"(shadow log {args.shadow_log} not readable: no correlation; "
              "enable [liveness] shadow_pad_logging to add it)")
    print(f"{len(plan)} attempts planned.")
    started = session.now()
    completed = session.run(plan)
    rows = summarize(session.attempts)
    line = headline(rows)
    print("\n" + format_table(rows))
    print("\n" + line)
    prefix = Path(args.out).expanduser() if args.out else default_prefix()
    meta = {"started": started.isoformat(timespec="seconds"), "completed": completed,
            "planned_attempts": len(plan), "confidence": CONFIDENCE, "bound": "one-sided Clopper-Pearson",
            "suspensions": session.suspensions, "shadow_log_used": bool(session.tail and session.tail.available),
            "user_recorded": False}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    session_id = started.strftime("%Y%m%dT%H%M%S")
    write_private(prefix.with_suffix(".json") if prefix.suffix else Path(f"{prefix}.json"),
                  results_json(session.attempts, rows, meta, line))
    write_private(Path(f"{prefix}.csv"), attempts_csv(session.attempts))
    write_private(Path(f"{prefix}-bench.csv"), bench_csv(session.attempts, session_id))
    print(f"\nwritten (mode 0600): {prefix}.json, {prefix}.csv, {prefix}-bench.csv")
    print(f"cross-check: doorman-bench pad-report --scores {prefix}-bench.csv --threshold 0.5")
    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())
