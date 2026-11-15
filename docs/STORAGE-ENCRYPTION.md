# Storage encryption (sealed embeddings, ROUND 10)

`embeddings.bin` holds 512-d face templates. Permissions (0600 file in a 0700
directory) stop unprivileged local users, but not an offline disk read, a
live-USB, or copying the file to another machine. This document describes the
opt-in sealed store that closes those vectors. See `docs/THREAT-MODEL.md`
(template-protection section) for the analysis this implements (option A, with
the deviations noted below).

## Threat model

In scope: stolen disk / offline read; unprivileged malware reading the file;
copying `embeddings.bin` to another machine (must not decrypt there); silent
substitution of the file (detected — the daemon fails closed, never falls back
to an empty store or a different file).

Out of scope (documented, not claimed): confidentiality against a live root
(RAM dump, ptrace of the running daemon — mitigated only by hygiene, see
below); binding the key to the exact daemon binary (no IMA/EVM on this host —
the TPM attests the boot chain, not `/usr/bin/doormand`).

Rejected: password-only encryption. Face unlock at the login screen happens
before any password is typed, so a password-derived key would kill the primary
use case. Two modes instead (§2).

## Layout: v4 sealed envelope

Magic `DRMNEMB\0`, `u32` version `4`, 12-byte random nonce, then ChaCha20-Poly1305
ciphertext of a whole version-3 file image. v1/v2/v3 decode paths are
byte-for-byte untouched; the v3 bincode is the AEAD plaintext, so decoding is
reused after opening. AAD is `doorman-embeddings-v4` (domain separation from
the PIN wrap). A version byte flipped to 3, a spliced nonce, or a wrong key all
fail authentication — never silent plaintext.

Backups reuse the crash-safe scheme (`daemon/src/storage.rs::backup_old_layout`,
0600, `create_new`, fsynced rename): the first sealed write copies the current
file to `embeddings.bin.v3.bak` (likewise `.v1.bak` / `.v2.bak` when sealing an
older file), never overwritten. An older daemon cannot read v4: the `.bak`
chain is the rollback path. There is no downgrade: saving over a sealed file
without the key is refused instead of writing plaintext.

## Modes

- **TPM-only** (recommended): `sudo doorman key enroll [--tpm-pcrs 7]` generates
  a 256-bit key, seals it with `systemd-creds encrypt --with-key=tpm2`, and
  installs `/etc/systemd/system/doormand.service.d/10-storage-key.conf`
  (`SetCredentialEncrypted`). systemd decrypts it into `$CREDENTIALS_DIRECTORY`
  (root-owned tmpfs) before the service starts; the daemon loads it into
  `mlock`'d, zeroized-on-drop RAM and never logs or exposes it. Face unlock at
  GDM keeps working. Offline reads and cross-machine copies are useless.
- **PIN recovery**: `--pin` (enroll and rotate) wraps the data key with
  Argon2id(password) into `embeddings.bin.pin` (0600, root-owned sidecar).
  `sudo doorman key unlock` unwraps it locally and hands the data key to the
  running daemon over the root-only IPC — no restart. Until the first unlock
  after boot, a sealed store answers `storage_locked` (fail closed, including
  at GDM — accepted tradeoff, documented here).
- **Rotation**: `sudo doorman key rotate [--pin] [--tpm-pcrs 7]` re-seals the
  store with a fresh key and, when a TPM drop-in already exists, re-seals the
  boot credential too (PIN-only mode has no drop-in and stays that way).
  Refuses when the file is sealed and no PIN recovery exists (a lost new key
  would strand the store). `key enroll` refuses on a live sealed/keyed store —
  enroll is first-time setup only, rotate is the live-store operation.

Deviations from `docs/THREAT-MODEL.md` option A: format is v4 (v3 was taken by
template metadata in round 9b); no plaintext fallback when no credential is
configured — an unsealed store keeps working as before, but a sealed store
without a key fails closed instead of warning; Argon2id (not in the doc) for
the PIN wrap; `SetCredentialEncrypted` inline blob instead of a key file.

## Fragility and recovery

PCR-bound keys break when the measured boot chain changes (firmware, kernel,
bootloader updates). Default `--tpm-pcrs 7` (SecureBoot state) minimizes this;
wider sets are possible and more brittle. If unseal breaks: unlock via the PIN
blob (if enrolled), else restore `.v3.bak` (or older) and re-enroll. Backups of
`/var/lib/doorman` must themselves be encrypted or excluded — a backup of a
sealed file is safe to store, a backup of the PIN sidecar is a second key copy.

## Hygiene (live-root mitigation only)

Key pages are `mlock`'d (best effort: containers may forbid it — warned, not
fatal) and zeroized on drop; the PIN and key live only in root-owned processes
and the daemon's RAM. The unit already denies coredumps of secrets only
insofar as the kernel default holds — verify `CoredumpFilter` on hosts where
that matters. None of this stops a live root; it raises the cost for
less-privileged malware and accidents.

## Operator commands

- `doorman key status` — sealed? loaded? PIN enrolled? Any user, no secrets.
- `sudo doorman key enroll [--tpm-pcrs 7] [--pin]` — then
  `sudo systemctl restart doormand`. The store seals on the first write after
  the restart (watch for `embeddings.bin.v3.bak`).
- `sudo doorman key unlock` — PIN mode, no restart.
- `sudo doorman key rotate [--pin]` — then re-run `key enroll`.
- Rollback to an unsealed store: stop the daemon, restore the `.bak` that
  matches the layout the old build reads, remove the drop-in, restart.
