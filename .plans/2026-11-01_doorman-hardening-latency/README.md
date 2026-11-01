# Doorman: hardening + time-to-unlock — work pack for the cloud session

Date: 2026-11-01. Owner: angkira. Executor: Claude Code cloud session / developer.
Repo: this one (GitHub `angkira/doorman`, branch `main`).

## Why this pack exists
Face unlock works (recognition similarity 0.78–0.82 vs threshold 0.4) but:
1. It is **slow**: measured lock→unlock ≈ **7.5 s** (3.6 s best case with no liveness false-reject). Most of the time is plumbing, not models. See `00-context.md`.
2. It has **security gaps** that block using it for `sudo`: world-writable socket (0666) with a trusted `username` field, no peer-credential check, weak single-frame liveness.
3. The **unlock path** is a polling shell script outside the repo.
4. **Repo/packaging hygiene** drift (binary older than HEAD, stale files, config backups).

## Workstreams (one branch + one PR each; do NOT push to main, do NOT release)
| ID | File | Priority | Effort | Depends on |
|---|---|---|---|---|
| WS1 | `01-security-ipc.md` — SO_PEERCRED, socket perms, admin ops root-only | P0 | S | — |
| WS2 | `02-unlock-path.md` — event-driven unlock, bring lock scripts into repo | P0 | M | WS1 (peer check) |
| WS3 | `03-liveness.md` — fix false rejects, multi-frame, calibration | P1 | M | WS6 data |
| WS4 | `04-camera-latency.md` — warm start, MJPEG, fix V4L2 open, resolution | P1 | M | — |
| WS5 | `05-decision-pipeline.md` — early exit N-of-M, multi-template, quality gate | P1 | S–M | WS6 |
| WS6 | `06-bench-harness.md` — benchmark/eval harness + model experiment matrix | P0 (start first) | M | — |
| WS7 | `07-sudo-pam.md` — opt-in sudo via PAM with strict policy | P2 | M | WS1, WS3 |
| WS8 | `08-hygiene-release.md` — packaging, config schema, privacy defaults, cleanup | P2 | S | — |

Suggested order: WS6 and WS1 in parallel on day 1 → WS2 → WS4/WS5/WS3 in parallel → WS7 → WS8. Each workstream file states scope, concrete tasks, acceptance criteria and verification commands.

## Targets (proposed; the harness in WS6 must confirm or revise them)
- Time-to-unlock from lock event: **p50 ≤ 2.0 s, p95 ≤ 3.5 s** (camera cold, user already facing the screen).
- Liveness false-reject on genuine user: **≤ 1%** over ≥ 200 genuine attempts in mixed lighting.
- FAR at the chosen threshold: report at 1e-3 and 1e-4 on an impostor set; no regression vs current.
- No unauthenticated local user can trigger or modify anything for another user (WS1).

## Ground rules for the executor
- Minimal diffs; no drive-by refactors; one logical change per commit; imperative English commit messages.
- Fail fast, explicit errors, no magic values (put thresholds in config with defaults + docs).
- Every change needs tests. Run `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test --workspace`, and the Python tests in `tests/` before opening a PR. Show command output in the PR description.
- Do not enable anything new by default if it changes auth behavior: new behavior is behind a config flag, default = current behavior, unless a workstream says otherwise (WS1 is the exception: it is a fix).
- Never log or commit face frames, embeddings, secrets, API keys. No personal paths in committed files (use `$HOME`, `%h`, `XDG_*`).
- The cloud environment has **no camera, no niri/Wayland session, no PAM stack you can lock yourself out of**. Anything that needs the real camera/lock screen goes into a "Host validation checklist" in the PR for the human to run; do not claim it verified.
- When a statement below is marked **[hypothesis]** or **[unverified]**, verify it in code first and record the result in the PR.

## Definition of done (per workstream)
Code + tests green in CI, PR description with: what changed, why, evidence (commands + trimmed output), config flags added, migration notes, host validation checklist, known limitations.

## Files in this pack
`00-context.md` (read first), `01`…`08` (workstreams), `HANDOFF-PROMPT.md` (paste into the cloud session).
