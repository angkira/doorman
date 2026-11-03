# Threat model

Status: written 2026-11-03 from the code at the commit that introduced this
file. Nothing here was measured on a hostile test bench. "Mitigated" means a
control exists in code and is covered by a test or by reading the code;
it does not mean the risk is zero. Where a claim is unverified it says so.

doorman is a **convenience factor**. A webcam-only face check does not
protect against a motivated attacker who has your photo, video or physical
presence, and it must not be the only barrier in front of anything valuable
(see [SUDO.md](SUDO.md), "Threat model, honestly"). This document lists what
the software does and does not defend against so that deployment choices are
informed.

## Assets and attackers

Assets, most sensitive first:

1. The authentication decision (`Success` to the PAM module): it grants a login,
   a screen unlock or, if opted in, `sudo`.
2. Face templates (`embeddings.bin`) and, when enabled, captured raw frames
   (`dataset/`): biometrics cannot be rotated like a password.
3. The camera stream (privacy: LED on, frames in memory).
4. Availability of the password fallback (the module must never lock a user out).

Attackers considered:

| Attacker | Capability |
|---|---|
| A: passer-by | physical access to the locked machine, a photo/video/screen/mask of the owner |
| B: local unprivileged user | runs code as a different uid, can connect to world-writable sockets |
| C: same-uid malware | runs as the owner (can already read most of the owner's data, kill the locker, use the camera device if it has `video` access) |
| D: daemon compromise | code execution as user `doorman` (e.g. through a bug in image/model parsing) |
| E: root | everything; out of scope except where a control claims otherwise |
| F: supply chain | malicious crate, model file, release artifact or CI change |

## Trust boundaries and data flow

```
 [camera] -> doormand (user doorman, sandboxed) <-socket 0666-> PAM module (root, in login/sudo/greeter)
                |  \__ /var/lib/doorman (0700): embeddings.bin, dataset/, models/       CLI (user / sudo)
                |  \__ /etc/doorman/doorman.toml (root, 0644)                           lock helpers (user)
                \__ stream sockets (frames/debug, 0600)
```

The daemon's `Success` reply is the root of trust: the PAM module obeys it
without further checks, and PAM runs inside privileged programs. **A compromised
daemon (attacker D) can approve any login that has `auth sufficient
pam_doorman.so`** (lock screen, console login, and `sudo` if opted in). The
sandbox ([systemd-hardening.md](systemd-hardening.md)) limits what D can do
besides that, but cannot remove this property. It is inherent in the design
(face match runs outside the privileged process, on purpose).

## STRIDE per boundary

Status values: **mitigated**, **partial**, **accepted** (known, deliberately
left), **open** (should be fixed or measured).

### 1. Command socket (`/run/doorman/doorman.sock`, mode 0666)

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| S | Peer claims to be another user in `username` | uid taken from `SO_PEERCRED`, not from the request; `authorize()` in `daemon/src/ipc_policy.rs`; fails closed if credentials cannot be read | none known | mitigated |
| S | Fake daemon answers `Success` to PAM | socket lives in `/run/doorman`, created by systemd `RuntimeDirectory=` owned by `doorman`; only root/`doorman` can bind there | the PAM module (`pam_module/src/lib.rs`) does not check the server's uid, so it trusts the directory permissions alone. `socket=` module arg is root-controlled (PAM file) and no env var is read. | partial |
| S | Downgrade by omitting `protocol_version` | `effective_protocol_version` rejects omission once v2 exists | none | mitigated |
| T | Malformed or oversized request | 64 KiB line cap read with `take`, UTF-8 and JSON parse errors drop the connection; exhaustive `match` in `authorize()`; fuzzed (`fuzz/`, `daemon/tests/fuzz_smoke.rs`: 10k+ inputs/target, no panic, never over-grants to an unprivileged peer) | fuzzing was short (minutes); no long campaign | mitigated |
| R | Denial of having tried | `Authenticate` outcomes logged at info/warn with uid/pid; sudo profile writes a structured `doorman::audit` line per attempt | the default lock-screen path has no per-attempt audit record beyond the log line; logs are local and root-editable | partial |
| I | User enumeration | unknown user gets the same `Denied` text as another user's account; `Status` has no names | timing difference between known and unknown user is not measured | accepted |
| I | `Status` readable by anyone | returns no biometric data or user names | reveals that doorman runs and its mode | accepted |
| D | Connection flood / camera abuse | 64 global and 8 per-uid connections, per-uid rate limit (20/min, root keyed per target user), 10 s request read timeout | any local user may make the daemon open the camera (LED on, CPU) for their own username up to the rate limit; a busy camera delays a legitimate unlock | accepted |
| E | Unprivileged user runs enroll/remove/shutdown | admin requests need uid 0 or `admin_uids`; compile-time exhaustive classification | `trusted_uids`/`admin_uids` misconfiguration is operator error | mitigated |

Stream sockets (frames/debug): created 0600, peer must be root/`admin_uids`
(or the file mode is the gate with `stream_socket_group`). They carry face
frames; **mitigated**. If an operator sets a group, every member can read
frames: **accepted**.

### 2. PAM module <-> daemon

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| S/E | Attacker redirects the module to a socket they control | socket path is a compile-time constant; only override is the `socket=` argument from `/etc/pam.d` (root-writable); no environment variable, because PAM runs inside setuid programs whose environment the caller controls | whoever can edit `/etc/pam.d` is root | mitigated |
| D | Daemon hangs and the login hangs | non-blocking connect (down daemon costs <1 ms, tested in `tests/test_pam_module.py`), read timeout 3 s (sudo: up to 4.5 s), hard deadline thread, panic guard | a wedged but accepting daemon costs the user up to 4 s per attempt | mitigated |
| E | Mis-ordered PAM stack turns a refusal into a lock-out or a bypass | every refusal/error returns `PAM_AUTHINFO_UNAVAIL`; only a daemon `Success` returns `PAM_SUCCESS`; root user never goes to the daemon | `sufficient` + weak daemon = face is a full authenticator for that service | accepted (documented in INSTALL/SUDO) |
| T | A compromised daemon approves any user | none specific; see "daemon's Success is the root of trust" above | complete bypass for services using the module | accepted (design) |
| I | Syslog leaks daemon-supplied text | message passed through a fixed `"%s"` format | none known | mitigated |

### 3. Camera device (`/dev/video*`)

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| S | **Presentation attack** (photo, replayed video on a screen, mask) | monocular-depth liveness (MiniFASNet / depth relief, `docs/decision-and-liveness.md`), N-of-M frame decision, sudo profile requires liveness | an RGB webcam heuristic; "photos, screens and masks are not ruled out" (README, SUDO.md). Attack success rate is **not measured** in this repo | open |
| S | **Virtual camera / v4l2loopback replay**: a process or module feeds prerecorded or synthetic frames into a `/dev/video*` node | `DevicePolicy=closed` limits the daemon to video4linux char devices; loopback nodes are video4linux devices, so they are **allowed** | The daemon selects the camera by `device_index` (default 0) and never checks the V4L2 driver, card name, bus info or capabilities (grep of `daemon/src/camera/` finds no loopback/driver check). A loopback device loaded before the real camera enumerates can take index 0. Creating one needs root (module load) or an already-loaded `v4l2loopback` with permissive node access, so this matters mostly for an attacker who already has local code execution plus a loaded module. **The daemon cannot detect or refuse a virtual camera.** | open |
| S | A USB device posing as a camera | none (any V4L2 capture node is accepted by index) | physical attacker with a USB port; also open-ended device re-enumeration changes `device_index` meaning | open |
| I | Camera used without the owner noticing | LED is hardware-controlled; the daemon opens the camera only for requests that pass the gate | a permitted request (own username) turns the camera on | accepted |
| T | Replay by tampering with frames in the daemon | frames live in daemon memory in a separate uid with `NoNewPrivileges` | root or same-uid ptrace is out of scope | accepted |
| D | Covering the lens | falls through to the password | unlock slower | accepted |

### 4. Config file (`/etc/doorman/doorman.toml`)

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| T | Edit threshold, sudo settings, `admin_uids` | file is root-owned; `Config::validate` rejects out-of-range values and sudo-without-liveness; `doorman config check` diffs unknown keys; parser fuzzed (no panic, deep nesting handled) | needs root to edit | mitigated |
| T | Typo in a security key is ignored | `doorman config check` reports unknown keys | the daemon itself does not reject unknown keys (no `deny_unknown_fields`) | partial |
| T | **A bad config silently weakens the daemon**: with `--config` or `DOORMAN_CONFIG`, a parse or validation error logs a warning and runs with `Config::default()` (`daemon/src/main.rs` lines ~263-277) | defaults are not insecure (sudo off), but a stricter threshold set by the admin is lost without the daemon stopping | silent fallback to default threshold 0.4 and default `admin_uids` | open |
| E | A root-run CLI reads an attacker-chosen config (formerly `./doorman.toml` for `sudo doorman dataset purge`) | `Config::load_with` (`shared/src/config.rs`, `shared/src/config_trust.rs`): with euid 0 the working directory and user config dir are never read; only `--config`, `DOORMAN_CONFIG` or `/etc/doorman/doorman.toml`, and the file and its directory must be root-owned and not group/world-writable, else an explicit error. Tests: `config_trust::tests::*` (injectable metadata provider). The purge also refuses symlinks and paths outside the dataset directory | the check is on the final path component and its directory only (parent directories further up are not walked); TOCTOU between check and read needs a root-owned writable location, i.e. root | mitigated |
| I | Config is world-readable (0644) | contains no secrets | reveals uids and thresholds | accepted |

### 5. Embeddings and dataset files (`/var/lib/doorman`)

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| I | Template theft | directory 0700, files 0600 created with that mode (`daemon/src/storage.rs`), owner `doorman`; dataset dirs 0700/files 0600 | stored as plain `f32` vectors, no encryption; readable by root, by the `doorman` uid, and by any backup that copies `/var/lib/doorman` | partial (see ISO/IEC 24745 section) |
| T | Replace a template with the attacker's (persistent backdoor) | atomic write (tmp + fsync + rename), private modes | no integrity tag; anyone with write access as `doorman` or root can silently enroll a face | accepted (needs daemon compromise or root) |
| T | Malformed `embeddings.bin` crashes or hangs the daemon | decoder returns explicit errors for bad magic/version/truncation; fuzzed v1 and v2 (no panic or unbounded allocation in the short runs); NaN vectors compare false against the threshold | a corrupt file stops the daemon at startup (fail closed; password still works) | mitigated |
| I | Raw frames stored for R&D | dataset capture is off by default, retention + `doorman dataset purge`, 0700/0600 | when enabled, raw face images sit unencrypted next to the templates | partial |
| R | Who enrolled whom | root-only `Enroll`; logged | no signed enrollment record | accepted |

### 6. Models directory

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| T | Swap an ONNX model (e.g. a liveness model that always says "live") | `/var/lib/doorman/models` is owned by `doorman`, not writable by users; `scripts/fetch_models.sh` verifies SHA-256 on download | the daemon does **not** verify a hash at load time, so a changed file under `/var/lib/doorman/models` is used as is; the deb postinst copies models without checking them. Writing there needs `doorman` or root. | partial |
| T | Malicious model exploits the ONNX runtime parser | none beyond the sandbox (`MemoryDenyWriteExecute=false`, no seccomp filter) | code execution as `doorman` = attacker D | open |

### 7. Lock-screen helper scripts and adapters

`packaging/adapters/{niri,sway}`, `scripts/doorman-sudo-pam.sh`,
`doorman unlock-watch`. They run as the desktop user and call
`doorman`/`WaitForMatch`, then SIGTERM the locker on `Matched`.

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| S/E | Same-uid malware unlocks the screen (fake daemon via `DOORMAN_SOCKET`, or just `pkill gtklock`) | none; a screen locker is not a boundary against code running as the same user | attacker C already holds the session's data; lock screens defend against attacker A | accepted |
| T | Predictable log/pid files in `/tmp` (`/tmp/doorman-sway-lock.log` in the sway adapter) | the niri adapter keeps its pidfile under `$XDG_RUNTIME_DIR` | `/tmp` log: symlink/overwrite by another local user, minor | open |
| E | `doorman-sudo-pam.sh` edits `/etc/pam.d/sudo` | requires an explicit acknowledgement flag and a second root shell; tested in `tests/test_sudo_pam_script.py` | a bad edit can lock out sudo | mitigated |

### 8. Sudo profile (opt-in)

Stricter than the lock screen: liveness mandatory, 4-of-5 frames, 4 s limit,
fresh frames only, refused when the session is locked, suspension after
repeated failures, per-attempt audit line
([SUDO.md](SUDO.md), `daemon/src/sudo_policy.rs`). Residuals: the face is
still `sufficient`, so a successful presentation attack gives root; the
sudoers timestamp (default 15 min) extends one match; any process of the
user can trigger an attempt (needs the face in front of the lens). **partial**;
the document itself advises FIDO2 or a fingerprint reader.

### 9. Installer, packages and postinst

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| E | Installer leaves `/var/lib/doorman` or the socket directory too open | `StateDirectoryMode=0700`; postinst `install -d -m 0700`; systemd `RuntimeDirectory=` | `install.sh` (user-mode installer) has a `TODO: Implement model download` and different paths; not reviewed here | partial |
| E | Unit hardening regresses | `systemd-analyze security` score gated in CI (currently 5.2 MEDIUM, `scripts/check_systemd_security.sh`); GPU-grant tests | the score is a heuristic; no `SystemCallFilter=` | partial |
| T | postinst copies models and (re)starts the service as root | models come from the package's own `/usr/share/doorman/models` | package is unsigned (no signing/provenance in CI); trust is whatever the user's download path gives | open |
| E | Dev unit `doormand-user.service` has only `NoNewPrivileges=true` | documented as a dev unit | running it in production loses the sandbox | accepted |

### 10. CI and supply chain

| STRIDE | Threat | Existing mitigation | Residual risk | Status |
|---|---|---|---|---|
| T | Vulnerable or yanked dependency | `cargo deny` job (advisories, yanked, licenses, sources=crates.io), `Cargo.lock` committed and `--locked` everywhere; ignores carry reason and review date in `deny.toml` | 9 ignored advisories (unmaintained or GUI-only, listed in `deny.toml`), including `bincode 1.x` in the daemon itself | partial |
| T | Malicious crate / typosquat | lock file + `cargo deny` sources; no build scripts audited | no `cargo vet`, no vendoring, large GUI tree (eframe) in the default build | open |
| T | Compromised GitHub Action | `permissions: contents: read`; actions pinned to major tags (`@v4`, `@v2`, `@stable`) | tags are mutable; not pinned to commit SHAs | open |
| T | Tampered release artifact / models | `fetch_models.sh` checks SHA-256 | no signed releases or SBOM | open |
| T | Parser bugs in untrusted-input paths | fuzz targets and seeded tests for IPC lines, config, embeddings (this change) | no continuous fuzzing beyond a 60 s CI run; image/ONNX/camera decoders are not fuzzed | partial |

## Biometric-specific threats

| Threat | What the code does | Status |
|---|---|---|
| Presentation attack (print, screen, mask) | single/few-frame depth liveness, N-of-M decision; heuristic only | open: no PAD evaluation per ISO/IEC 30107-3 style testing in this repo (that standard's procedures are not reproduced here) |
| Camera feed injection (v4l2loopback, virtual cameras, OBS, hostile USB camera) | none; device chosen by index; loopback devices are allowed by `DeviceAllow=char-video4linux` | open: the daemon can neither detect nor refuse a virtual camera. A possible mitigation is to read `VIDIOC_QUERYCAP` (driver name `v4l2 loopback`, bus_info `platform:v4l2loopback-*`) and refuse such nodes, and to select the device by stable `by-id`/`by-path` instead of index. Not implemented. |
| Template theft | 0600/0700, plain vectors | partial, see next section |
| Hill-climbing / template inversion | attacker needs access to scores or the template; scores are not returned to unprivileged peers (only success/failure; `Matched.similarity` goes to the authorized caller of `WaitForMatch`) | accepted; embeddings are 512-d model outputs, not invertible by design but model inversion attacks exist in the literature, unquantified here |
| False accept rate | decision threshold: cosine 0.4 (`default_similarity_threshold`), N-of-M frames | open: **the false accept rate at 0.4 is unknown**. `scripts/threshold_security_eval.py` computes an impostor distribution on LFW-like data, but no result is recorded in `docs/`. A near-twin, sibling or a demographic group with a worse model can have a much higher FAR than a pooled number. |
| Replay of a captured successful session | each attempt captures fresh frames; the sudo profile ignores frames older than the request | mitigated at the IPC level; not a defense against feed injection (above) |
| Enrollment poisoning | enroll is root-only | mitigated |
| Biometric data retention | dataset off by default, purge command | partial |

## Summary of open items

1. Virtual-camera (v4l2loopback) and USB camera spoofing: no detection; device chosen by index.
2. Presentation-attack resistance and FAR at threshold 0.4 not measured.
3. Daemon silently falls back to default config on a bad `--config`/`DOORMAN_CONFIG`.
4. ~~Root-run CLI loads `./doorman.toml` from the working directory (`dataset purge`).~~ Closed: root only accepts a trusted config path.
5. Models are not integrity-checked at load; no `SystemCallFilter=`; ONNX/image decoding runs inside the daemon that is the root of trust.
6. Daemon compromise = authentication bypass (design property; sandbox only limits collateral damage).
7. Release/CI hardening: actions not SHA-pinned, no signed artifacts or SBOM, no `cargo vet`, 9 documented advisory ignores.
8. Sway adapter writes a log to a predictable `/tmp` path.
9. Template protection beyond file permissions (next section).

---

## Template protection and ISO/IEC 24745

This section is analysis only. No cryptography is implemented.

### What the standard asks (and what is verified here)

ISO/IEC 24745 ("Biometric information protection") is paywalled and was **not
read** for this document. The following is from its public descriptions and
general literature, so treat the wording as unverified until someone with the
text checks it:

- It addresses the protection of stored biometric references against
  disclosure, tampering and misuse, and the privacy of the data subject.
- Properties usually cited from it: **confidentiality** and **integrity** of
  the stored reference (security), **irreversibility** (the stored form does
  not allow recovering the biometric sample) and **unlinkability** (references
  from different applications or databases cannot be linked), and
  **renewability/revocability** (a compromised reference can be replaced by a
  new one derived from the same biometric).
- It describes a reference as split into a pseudonymous identifier and
  auxiliary data, with protection mechanisms such as encryption and
  cancelable-biometrics / helper-data schemes. The exact normative
  requirements and their numbering are not quoted here.

### Current state

| Property | doorman today |
|---|---|
| Confidentiality | file permissions only: `embeddings.bin` 0600 in `/var/lib/doorman` 0700, owner `doorman` (`daemon/src/storage.rs`). Plain 512-d `f32` vectors, bincode. No encryption. |
| Integrity | none beyond atomic replace (tmp file, fsync, rename) and decode errors on malformed files. No MAC; a modified file with valid structure loads. |
| Irreversibility | no protection: the stored vector is the model's embedding. Whether a face-like image can be reconstructed from an EdgeFace embedding is not assessed here; the literature says it can for some models, so treat templates as sensitive personal data. |
| Unlinkability | no: the same person enrolled on two hosts with the same model yields comparable vectors. |
| Renewability | manual only: `doorman remove` + re-enroll. The face itself cannot be renewed; only the stored template. |
| Format versioning | v1 (headerless) and v2 (magic `DRMNEMB\0` + version) loaders exist and are fuzzed; a v1 file is backed up before upgrade (`embeddings.bin.v1.bak`, same 0600 mode). |

### Who the protections would stop

The attacker with **root on the running host** (E) defeats every option below:
root can read the daemon's memory, ask systemd for the credential, call
`Enroll`, or edit the template. The daemon must also hold plaintext vectors in
memory to match, so attacker D (code execution as `doorman`) sees them too.
Encryption therefore does not improve security against a live compromised
host. It does help against:

1. offline disclosure: stolen or discarded disk, cloud snapshot, backup copy,
   support bundle, `rsync` of `/var/lib`;
2. copying a template to another machine (host-bound key);
3. accidental exposure through a wrong mode or a misconfigured share;
4. offline tampering (an edited or swapped file in a backup being restored).

These are real but narrower than "template protection" suggests.

### Options

**A. AEAD-sealed file with a systemd credential key.** Store the key with
`systemd-creds encrypt` (host key, optionally TPM2-bound) and hand it to the
daemon with `LoadCredentialEncrypted=` in `doormand.service`; the daemon
decrypts at start into `$CREDENTIALS_DIRECTORY` (private tmpfs) and seals
`embeddings.bin` with AES-256-GCM or ChaCha20-Poly1305 (per-write random
nonce, header magic/version/key id as associated data). Covers confidentiality
and integrity (the tag) for threats 1 to 4. Needs systemd with credential
support (introduced around v250; TPM2 sealing later; **version requirements
unverified**, check the target distributions). Cost: key lifecycle (lost or
reset TPM means re-enrolling, acceptable because the password fallback
exists), a new v3 format, migration from v1/v2 including the `.bak` file,
a new fuzz target. A TPM2 key bound to PCRs can break after firmware updates;
bind to a stable PCR set (for example only Secure Boot state) or use the host
key.

**B. HMAC only.** Authenticates the file with a key from the same credential
but leaves vectors readable. Gives integrity (threat 4) without
confidentiality. Strictly dominated by A, because AEAD gives both for the same
key handling. Only attractive if encryption were rejected for performance or
debuggability reasons, which does not apply to a few hundred KB.

**C. Per-host salt / random projection (cancelable biometrics).** Multiply
each vector by a key-seeded random orthogonal matrix before storing, and do the
same to the probe. Orthogonal maps preserve cosine similarity exactly, so the
threshold and the N-of-M logic stay unchanged. Gives unlinkability across hosts
and renewability (a new key yields a new transformed template). Limits:
the key lives on the same host, so against E and D it adds nothing; as
confidentiality it is weaker than A (a linear map with a known structure can be
solved from enough known vector pairs, and it is invertible by anyone with the
key); renewal still needs the original sample or a re-enrollment, because the
raw vector is not kept. Its real advantage, matching without ever decrypting,
is irrelevant when the same process holds the key.

**D. Hardware matching / secure element.** Out of scope for a webcam
convenience tool; mentioned for completeness.

### Recommendation

1. Do not implement the random projection (C) or a separate HMAC (B).
2. If the project wants to claim any alignment with 24745, implement **A** as
   format v3: AEAD with a systemd-credential key, header authenticated as
   associated data, migration from v1/v2 that also seals or removes
   `embeddings.bin.v1.bak`, a v3 fuzz seed, and a documented recovery step
   (`doorman remove` + re-enroll after key loss). Fall back to the current plain
   v2 (with a warning) only when no credential is configured.
3. Before that, cheap and independent of crypto: refuse to start when
   `embeddings.bin` or the state directory is group/world-accessible (the
   storage code sets modes on creation but does not check existing files), and
   document that backups of `/var/lib/doorman` must be encrypted or excluded.
4. Treat the optional raw-frame dataset as a bigger privacy risk than the
   templates: it stores face images. Keep it off by default (it is), purge on a
   schedule, and encrypt it with the same mechanism if it is ever used outside
   a lab machine.
5. State plainly in user docs that after a template leak the template can be
   revoked but the face cannot, so a false-accept path based on a stolen
   template plus a presentation attack remains.

Honest value: A turns "copy the file and use it" into "copy the file and also
break the host key or the TPM" for offline attackers. It does nothing against
anyone who already controls the running machine. It is worth doing before any
claim of template protection, but not before the open items in the summary above
that affect a live attacker (virtual-camera spoofing, silent config fallback).
