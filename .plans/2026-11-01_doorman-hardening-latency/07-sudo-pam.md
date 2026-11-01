# WS7 — Opt-in face authentication for `sudo` via PAM (P2, effort M; depends on WS1 and WS3)

## Goal and honest threat model
The owner wants to use doorman for `sudo`. A webcam-only (RGB) face check is a **convenience factor, not a strong security boundary**: liveness is a single/few-frame monocular-depth heuristic; photo/screen/mask attacks are not ruled out; a local process can trigger the camera. Therefore: **opt-in, off by default, stricter policy than the lock screen, password always remains available**. Do not market it as strong authentication in docs.

Already known facts: the PAM module (`pam_module/src/lib.rs`) is a thin socket client returning PAM_SUCCESS / PAM_AUTH_ERR / PAM_AUTHINFO_UNAVAIL (daemon down, timeout, root, no username). The owner's sudo is **sudo-rs**, which honours PAM. Quick test used by the README: `sudo -k; sudo true`.

## Policy design (config `[pam.sudo]`, all explicit, defaults = disabled)
- `enabled = false`.
- `service_names = ["sudo"]` — apply stricter policy when the PAM service is sudo (the module receives `PAM_SERVICE`); `sudo-i`, `su`, `polkit-1`, `login`, display managers are **separate opt-ins**, default not allowed.
- Stricter thresholds for this service: `similarity_threshold` override (≥ lock-screen value, set from WS6 calibration), liveness **required** (never "advisory"), `n_of_m` stricter (e.g. 4 of 5), shorter total timeout (≤ 4 s), then fall through to password quietly.
- Freshness: face match must come from frames captured **after** the PAM request starts (no reuse of the lock-screen recognition loop result).
- No-presence short circuit: if the screen is locked or the session is not the active seat session, deny face auth (PAM then asks for password).
- Per-user attempt limiting and exponential backoff using WS1's rate limiter; after K consecutive face failures in a window, face auth is disabled for that user for T minutes (password still works).
- Audit log: one syslog/journal line per attempt (service, uid, result, reason code, similarity bucket) — never frames/embeddings.
- Deliberate non-goal: the module cannot see *which command* sudo will run, so per-command policy is not possible in PAM. Document mitigations: keep `timestamp_timeout` short, don't enable the module for `su`/`sudo -i`/polkit initially, keep password for sensitive classes via separate sudoers rules that require a fresh authentication.

## Tasks
1. Implement the stricter profile in daemon `Authenticate` path: request carries `service` and `rhost/tty` hints from PAM; daemon selects the policy profile; unknown service → default (lock-screen) profile **only if** explicitly allowed, else `Denied`.
2. PAM module: pass service name, handle `PAM_TTY`, return `PAM_IGNORE`/`AUTHINFO_UNAVAIL` quickly if face unavailable (so password prompt appears with minimal delay, ≤ 100 ms overhead when daemon is down).
3. Packaging: ship `pam-configs/doorman-sudo` (for `pam-auth-update`) or a documented one-liner and a **safe installer script** that edits `/etc/pam.d/sudo` with a backup, validates the result, and refuses to run unless a root shell is open elsewhere (print a prominent warning). Never auto-modify PAM during package install.
4. Docs `docs/SUDO.md`: threat model, how to enable/disable, recovery (how to restore `/etc/pam.d/sudo` from the backup when locked out; recommend keeping a root shell or TTY open during setup), comparison to alternatives (FIDO2 key / fingerprint reader are stronger for sudo).
5. Optional stretch (flag, default off): screen-flash/head-pose challenge from WS3 required for sudo profile.

## Tests
- Daemon: policy-profile selection table; stricter thresholds applied only for sudo service; freshness (stale-frame rejection) test with injected frame timestamps; rate-limit/backoff; liveness-required.
- PAM module: unit tests with a mock daemon over a temp socket for each return-code path; timeout behaviour; no panic across FFI.
- Installer script: run against a copy of a temp `pam.d` directory; verify backup + idempotency + refusal paths.
- E2E (Rust/Python, no real PAM): drive the daemon with `service="sudo"` requests.

## Acceptance
- With `enabled=false`, behaviour is byte-for-byte unchanged; the installer is the only thing that wires sudo.
- With sudo profile on and the mock daemon down, `sudo` falls through to password within the PAM module's bounded delay.
- Docs include the recovery procedure and the threat model; PR description lists the limitations above verbatim.

## Host validation checklist (human) — do in this order, keep a second root shell open
`sudo -k; sudo true` with face → success; with sunglasses/dark room → falls to password; photo of the owner on a phone held to the camera → must be rejected (record outcome honestly, including if it is NOT rejected); after 3 failures face is suspended for the configured time; `journalctl -t doorman` shows audit lines.
