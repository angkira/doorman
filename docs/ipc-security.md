# IPC security

The daemon exposes three Unix sockets. This page records how each is protected.

## Command socket (`doorman.sock`)

Mode `0666`: any local process can connect. This is deliberate. PAM runs as
root, while the CLI (`doorman test`) and the lock watcher run as the user, so a
single group would need every desktop user enrolled in a `doorman` group.
Group management is too invasive for a desktop package.

Security therefore comes entirely from `SO_PEERCRED`: the daemon reads the
uid/gid/pid the kernel reports for each connection and authorizes every
request (`daemon/src/ipc_policy.rs`). If the credentials cannot be read, the
connection is dropped (fail closed).

| Request | Allowed for |
|---|---|
| `Authenticate{username}` | peer uid == uid of `username`, root, or `trusted_uids` |
| `WaitForMatch{username, timeout_ms}` | same as `Authenticate` (it yields the same decision) |
| `Enroll`, `RemoveUser`, `ListUsers`, `DetectAndRecognize`, `GetLatestDetection`, `Shutdown` | root or `admin_uids` |
| `Status` | anyone (no biometric data, no user names) |

`Authenticate` may also carry `service` and `tty` (the PAM module's
`PAM_SERVICE` / `PAM_TTY`; additive, protocol version stays 1). A service listed
in `[pam.sudo] service_names` selects the strict sudo profile, see
`docs/SUDO.md`; other services are unaffected.

Every request variant is classified in an exhaustive `match`; adding a variant
does not compile until it is classified. An unknown username gets the same
`Denied` message as another user's account, so user existence is not leaked.

Consequence: `doorman enroll` / `remove` / `list` now require root (`sudo`).

Refusals are explicit responses: `Denied{reason}`, `RateLimited{retry_after_s}`
and `UnsupportedProtocol{reason, supported_version}`. The PAM module maps all of
them to `PAM_AUTHINFO_UNAVAIL` (fall through to the password) and logs to
syslog; the CLI prints the reason and exits non-zero.

### Rate limiting

Authenticate and WaitForMatch attempts (one per call, however long the wait)
are limited per peer uid over a sliding 60 s window
(`max_auth_attempts_per_minute`, default 20, `0` = off). Root is limited too,
to bound brute force through PAM. Every PAM caller runs as uid 0, so for root
the bucket is keyed by (uid 0, target username): attempts against one account
cannot exhaust the budget of the others. Non-root callers are keyed by uid
alone. State is in memory and resets on restart.

### Connection and request-size limits

The socket is world-connectable, so before any authorization:

- A request line is capped at 64 KiB (`MAX_REQUEST_LINE_BYTES`). The read is
  bounded with `take`, so a peer that never sends a newline cannot make the
  daemon buffer more; the connection is dropped without a reply.
- At most 64 connections are open at once (global), and at most
  `[ipc] max_connections_per_uid` (default 8, `0` = off) per peer uid; extra
  connections are dropped. A long `WaitForMatch` holds one slot per watcher.

### Logging

Each Authenticate logs peer uid/pid and outcome at info; denials and rate
limiting log at warn. Frames and embeddings are never logged.

### Protocol version

Requests may carry `"protocol_version": N` next to the usual fields
(`{"type":"authenticate","username":"u","protocol_version":1}`). Absent means
`UNVERSIONED_PROTOCOL_VERSION` (1), so old PAM modules and clients keep working
while the current version is 1. Any other value is answered with
`UnsupportedProtocol`. The PAM module and CLI send `1`.

Once a protocol v2 exists, a missing `protocol_version` is rejected with
`UnsupportedProtocol` ("protocol_version is required"): the gate only treats it
as v1 while `PROTOCOL_VERSION <= UNVERSIONED_PROTOCOL_VERSION`
(`effective_protocol_version`, covered by
`missing_protocol_version_is_rejected_once_a_newer_version_exists`). Bumping
`PROTOCOL_VERSION` therefore cannot leave a downgrade-by-omission path.

## Frame and debug stream sockets

They carry face frames and enrollment progress, so they are created `0600`
(owner/root only). Set `[ipc] stream_socket_group = "name"` to chown them to
that group and use `0660`. On accept the peer is also checked: root or
`admin_uids` (when a group is configured the file mode is the gate, since
supplementary groups are not visible via `SO_PEERCRED`); others are dropped.

The live progress bar of `doorman enroll` reads the debug socket; for a
non-root user it is skipped silently unless the group option is used.

## `[ipc]` configuration

```toml
[ipc]
trusted_uids = []                 # may Authenticate for any user
admin_uids = []                   # may run admin operations
max_auth_attempts_per_minute = 20 # 0 disables
max_connections_per_uid = 8       # 0 disables; global cap is 64
wait_for_match_max_ms = 60000     # cap for one WaitForMatch, see docs/unlock-path.md
prearm_camera = true              # score frames while waiting for LockedHint; timing only
prearm_max_hold_secs = 5          # 1..=30; camera released after this without a lock
# stream_socket_group = "doorman"
```

All keys are optional.
