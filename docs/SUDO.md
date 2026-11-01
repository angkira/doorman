# Face authentication for sudo (opt-in)

Off by default. Nothing in a package install touches PAM; you enable it in two
independent places (the daemon config and the PAM file) and either one alone
leaves `sudo` exactly as it was.

## Threat model, honestly

A webcam-only (RGB) face check is a **convenience factor, not a strong security
boundary**: liveness is a single/few-frame monocular-depth heuristic; photo/screen/mask
attacks are not ruled out; a local process can trigger the camera.

What this means in practice:

* Someone who holds a good photo or video of you, a screen replay or a mask may
  get through the depth check. The liveness gate is a deterrent, not proof.
  Calibration numbers are in `docs/decision-and-liveness.md`.
* Any process running as you (or as root) can make the daemon open the camera
  for a `sudo` attempt. It still needs your face in front of the lens, but it
  needs nothing else.
* A FIDO2 security key (`pam_u2f`) or a fingerprint reader are stronger and
  cheaper to reason about for sudo. Prefer them where you have the hardware;
  this exists for machines where you do not.
* The password always stays available: the module is `sufficient`, and every
  failure, refusal and timeout falls through to the password prompt.

### Limitations

* Deliberate non-goal: the module cannot see *which command* sudo will run, so per-command policy is not possible in PAM.
* A webcam-only (RGB) face check is a **convenience factor, not a strong security boundary**: liveness is a single/few-frame monocular-depth heuristic; photo/screen/mask attacks are not ruled out; a local process can trigger the camera.
* The optional screen-flash / head-pose challenge is not implemented. The
  `LivenessSignal` interface (`daemon/src/liveness.rs`) is where one would
  plug in; `[liveness] signals` rejects every entry until then.
* Presence is judged from the lock state of the one logind session the daemon
  resolved at startup (`lock_supervisor`). If you move to another seat session,
  restart the daemon.
* Suspension state is in memory and resets when the daemon restarts.

Mitigations for the first point, in `sudoers`:

* Keep `Defaults timestamp_timeout` short (for example `1`), so one face match
  does not authorize sudo for 15 minutes.
* Do not enable the module for `su`, `sudo -i`, polkit (`polkit-1`) or display
  managers. They are separate opt-ins (`service_names`) and off by default.
* Keep sensitive command classes behind a rule that needs a fresh password.

## What the strict profile enforces

An `Authenticate` whose `service` (the PAM module sends `PAM_SERVICE`) is in
`[pam.sudo] service_names` gets this profile; any other service, and clients
that send none (the CLI, old PAM modules, the lock screen), take the unchanged
default path.

| Rule | Behaviour |
|---|---|
| Master switch | `enabled = false`: the request is answered `Denied` ("sudo face auth disabled"); the PAM module falls through to the password. |
| Decision | Always `n_of_m`, default 4 of 5 frames (`accept_n`, `window_m`), at `similarity_threshold` (default: the recognition threshold; if set it must be at least that). |
| Liveness | Required, never advisory: `authentication.liveness_enabled` must be `true` when sudo is enabled, and at least 3 frames (`liveness_frames`, odd). A liveness error fails closed. |
| Time | `timeout_ms` (default and maximum 4000), counted from the moment the request arrived. |
| Freshness | Only frames captured at or after the request arrival are used; the frame already on screen when the request came in is skipped. |
| Presence | Denied when the tracked session is locked, or when no session is tracked (no lock supervisor, `--user` mode): fail closed. |
| Suspension | `max_failures` (3) consecutive judged failures of one user within `suspend_secs` (600) suspend face auth for that user for `suspend_secs`; requests are then answered `Denied` at once, without opening the camera. A success resets the count. Camera or enrollment problems do not count. |
| Audit | One line per attempt, see below. |

The peer uid is deliberately not part of profile selection. PAM runs as root,
but a non-root process that claims a sudo service only gets the stricter
profile (or the disabled denial), which is harmless; claiming another service
gets the default profile, which that process could already get by sending no
service. The claim therefore needs no authorization.

### Audit log

Each strict attempt (and each refusal before the camera) writes one line to the
tracing target `doorman::audit`, with `service`, `uid` (of the peer), `user`,
`result`, `reason` (a stable code), and `similarity_bucket` (rounded to 0.05, or
`none`). Frames and embeddings are never logged.

    journalctl -u doormand | grep doorman::audit

Reason codes: `matched`, `not_matched`, `no_clear_face`, `liveness_rejected`,
`liveness_unavailable`, `no_frames`, `camera_unavailable`, `not_enrolled`,
`session_locked`, `no_session`, `suspended`, `disabled`. The default log filter
includes the target; if you set `RUST_LOG` yourself, keep `doorman::audit=info`
(plain `RUST_LOG=info`, as in the shipped unit, already does).

## Which PAM service names

sudo and sudo-rs authenticate with the PAM service `sudo`. When built with its
`pam-login` feature (the usual choice on Debian and Fedora) sudo-rs uses the
service `sudo-i` for `-i/--login`. Only `sudo` is in `service_names` by default;
add `sudo-i` yourself if you want it. Source: the sudo-rs 0.2.12 README,
<https://docs.rs/crate/sudo-rs/0.2.12/source/README.md> ("Sudo-rs will use the
`sudo` and `sudo-i` service configuration").

## Configuration

```toml
[pam.sudo]
enabled = false                 # master switch
service_names = ["sudo"]        # PAM_SERVICE values that get the strict profile
# similarity_threshold = 0.5    # optional; >= authentication.similarity_threshold
accept_n = 4
window_m = 5
liveness_frames = 3             # odd, 3..=9
timeout_ms = 4000               # 1..=4000
max_failures = 3
suspend_secs = 600
```

Invalid values stop the daemon at startup with an explicit message (for
example a threshold below the recognition threshold, `timeout_ms` above 4000,
fewer than 3 liveness frames, or sudo enabled with liveness switched off).

## Enable

Keep a second root shell open (another terminal, a TTY login as root) for the
whole procedure.

1. Enroll as usual and check `doorman test <user>` works.
2. Set `enabled = true` under `[pam.sudo]` in `/etc/doorman/doorman.toml` and
   restart `doormand`. At this point `sudo` is unchanged: nothing calls the
   daemon with a sudo service yet.
3. Wire the PAM file (as root, from the shell you are keeping open):

       sudo scripts/doorman-sudo-pam.sh status
       sudo scripts/doorman-sudo-pam.sh enable --i-acknowledge-root-shell-open

   It makes a backup `/etc/pam.d/sudo.doorman-backup.<timestamp>`, inserts
   `auth sufficient pam_doorman.so` before the first `auth` line or `@include`,
   refuses if no password path would remain or the module is not installed, and
   prints the recovery command. Set `PAM_DIR` to try it on a copy first.
4. In another terminal: `sudo -k; sudo true`. Test with your face, then with
   sunglasses or a dark room (it must fall through to the password), then with
   a photo of yourself on a phone (record honestly whether it is rejected).

## Disable

    sudo scripts/doorman-sudo-pam.sh disable --i-acknowledge-root-shell-open

removes the line again (after taking a backup). Setting `enabled = false` in
the daemon config is enough to stop face auth for sudo even with the PAM line
still in place: the daemon answers `Denied` and the password prompt follows.

## Recovery if sudo is broken

From the root shell you kept open:

    cp -p /etc/pam.d/sudo.doorman-backup.<timestamp> /etc/pam.d/sudo

Without a root shell: log in as root on a text console (`Ctrl+Alt+F3`), or
`su -` with the root password, or boot into recovery mode, and run the same
command (`ls /etc/pam.d/sudo.doorman-backup.*` lists the backups). If the root
account has no password and you have no other root path, only recovery mode or
a live system helps, which is why the script insists on the open root shell.

## Failure behaviour of the PAM module

* Daemon not running, socket missing or backlog full: the connect is
  non-blocking and fails at once, so the password prompt appears without
  perceptible delay (tested: well under 100 ms).
* No answer: the module gives up after the daemon's budget plus a margin
  (about 5 s for sudo services, 4 s otherwise).
* Return codes: match is `PAM_SUCCESS`, a judged non-match is `PAM_AUTH_ERR`,
  everything else (refusal, rate limit, protocol mismatch, daemon down,
  timeout, panic) is `PAM_AUTHINFO_UNAVAIL`. `PAM_AUTHINFO_UNAVAIL` is used
  instead of `PAM_IGNORE` because under `sufficient` both continue to the
  password, while under a mistaken `required` UNAVAIL fails closed and IGNORE
  would silently skip the module.
* A panic never unwinds into libpam (`catch_unwind` at both entry points).
