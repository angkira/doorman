# Baseline (from host audit 2026-11-01, not reproduced by the harness)

These numbers come from a read-only audit of the owner's installation on
2026-11-01 (journal and logs), recorded in
`.plans/2026-11-01_doorman-hardening-latency/00-context.md`. They were NOT
produced by `doorman-bench` and have not been re-measured. Treat them as the
target to confirm or correct with the commands in `README.md`.

Host: AMD Ryzen 7 8700G, CPU inference (`--device cpu`), UGREEN UVC webcam
(RGB only), ORT `ort 2.0.0-rc.12`, installed daemon older than repo HEAD.

## Pipeline numbers (from host audit 2026-11-01)

| Item | Value |
|---|---|
| Capture config | 1280x720@30, effective about 15 fps |
| Detection (YuNet 2023mar, 640x640) | about 7.6 fps |
| Recognition similarity (genuine) | 0.78 to 0.82 vs threshold 0.4 |
| Liveness (Depth-Anything-V2-Small fp32, 518x518, CPU) | relief threshold 0.05; genuine calibration range 0.068 to 0.245; one rejection at 0.0491 |
| Auth frames | 10 quality embeddings, mean, cosine >= 0.4 |

## Lock to unlock timeline (lock at 13:12, from host audit 2026-11-01)

| Phase | Delta |
|---|---|
| lock start to armed | about 0.2 s |
| ffmpeg spawn to first frame (cold camera) | 1.3 s |
| attempt 1: liveness rejected genuine user | about 1.3 s lost |
| watcher `sleep 2` | 2.0 s |
| attempt 2: about 10 quality frames + depth pass | 1.25 s |
| recognised to locker gone | about 0.5 s |
| Total | about 7.5 s (about 3.6 s if liveness passes first time) |

An earlier run (11:03) recognised the user 1.65 s after arming.

## Targets (proposed, to be confirmed by the harness)

Time-to-unlock p50 <= 2.0 s, p95 <= 3.5 s; liveness false reject on the genuine
user <= 1% over >= 200 attempts; FAR reported at 1e-3 and 1e-4.

## Still unmeasured

Where the 1.25 s of attempt 2 splits between detection, embedding and depth
(open question 3 in the context file). Enable `[telemetry] timing_log = true`
and run `doorman-bench latency` on the host to answer it; paste the results as
a new, dated file next to this one.
