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
| `Enroll`, `RemoveUser`, `ListUsers`, `DetectAndRecognize`, `GetLatestDetection`, `Shutdown` | root or `admin_uids` |
| `Status` | anyone (no biometric data, no user names) |

Every request variant is classified in an exhaustive `match`; adding a variant
does not compile until it is classified. An unknown username gets the same
`Denied` message as another user's account, so user existence is not leaked.

Consequence: `doorman enroll` / `remove` / `list` now require root (`sudo`).

Refusals are explicit responses: `Denied{reason}`, `RateLimited{retry_after_s}`
and `UnsupportedProtocol{reason, supported_version}`. The PAM module maps all of
them to `PAM_AUTHINFO_UNAVAIL` (fall through to the password) and logs to
syslog; the CLI prints the reason and exits non-zero.

### Rate limiting

Authenticate attempts are limited per peer uid over a sliding 60 s window
(`max_auth_attempts_per_minute`, default 20, `0` = off). Root is limited too,
to bound brute force through PAM. State is in memory and resets on restart.

### Logging

Each Authenticate logs peer uid/pid and outcome at info; denials and rate
limiting log at warn. Frames and embeddings are never logged.

### Protocol version

Requests may carry `"protocol_version": N` next to the usual fields
(`{"type":"authenticate","username":"u","protocol_version":1}`). Absent means 1,
so old PAM modules and clients keep working. Any other value is answered with
`UnsupportedProtocol`. The PAM module and CLI send `1`.

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
# stream_socket_group = "doorman"
```

All keys are optional.
