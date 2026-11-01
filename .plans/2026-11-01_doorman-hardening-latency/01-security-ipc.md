# WS1 — IPC security: peer credentials, socket permissions, admin ops (P0, effort S)

## Problem
`/run/doorman/doorman.sock` is mode 0666 and requests carry a plain `username`. Any local process can trigger an authentication attempt (and camera use) as any enrolled user, and can call `Enroll` / `RemoveUser` / debug endpoints. The same applies to `doorman-frames.sock` and `doorman-debug.sock` (all 0666 on the host).

## Tasks
1. In the accept path of `daemon/src/ipc.rs`, read peer credentials with `SO_PEERCRED` (`UnixStream::peer_cred()` or `getsockopt`) and attach `PeerCred{uid,gid,pid}` to the request context.
2. Authorization policy (put in one small module, e.g. `daemon/src/ipc/authz.rs`, pure function + table-driven tests):
   - `Authenticate{username}`: allowed if peer uid == uid of `username` (resolve via `getpwnam_r`/`nix::unistd::User`), **or** peer uid == 0 (PAM runs as root for sudo/login/sddm), **or** peer is in an explicit allow-list in config (`[ipc] trusted_uids = []`, default empty).
   - `Enroll`, `RemoveUser`, `ListUsers`, any dataset/export/debug request: **root only** (or `[ipc] admin_uids`).
   - Everything else: deny by default with an explicit error variant (`Response::Denied{reason}`); never silently ignore.
3. Socket permissions: create the socket directory `0750 root:doorman`? Decide and document one of:
   - (preferred) socket `0666` is **removed**; sockets `0660`, group `doorman`, and users who need unlock are in that group — **or**
   - keep world-connectable (the user-space watcher and PAM run as different uids) **but** rely entirely on SO_PEERCRED policy above.
   Pick the second if group management is too invasive for a desktop package; record the choice and its rationale in `docs/`. The frames/debug sockets must not be world-readable in either case (they leak face frames).
4. Rate limiting: per-peer-uid limit on Authenticate attempts (config `[ipc] max_auth_attempts_per_minute`, default e.g. 20) + lockout response `Response::RateLimited{retry_after_s}`; log (without biometric data) denied/limited requests at warn.
5. Logging: log peer uid/pid and decision for every Authenticate (success/failure/denied) at info; never log frames or embeddings.
6. Protocol version field: add `protocol_version` to the request, reject unknown major versions explicitly (needed by WS2 subscription).

## Tests
- Unit: policy table (self allowed, root allowed, other user denied, admin ops root-only, trusted_uids honoured, unknown username).
- Integration (Rust, in `daemon/tests/`): spawn the IPC server on a temp socket in-process; connect as the same uid (allowed); simulate other uid by injecting a fake `PeerCred` through a test seam (do not require running as root); assert `Denied`.
- Socket mode test: created sockets have the documented mode.

## Acceptance
- A process running as uid A cannot authenticate, enroll, or remove for uid B (shown by tests).
- PAM module as root still works (existing e2e test updated).
- No new default breaks `doorman test <user>` run by the same user.
- `cargo test --workspace`, clippy, fmt green.

## Host validation checklist (human)
- `doorman test $USER` works as the user; as another local user it prints a denial.
- `sudo true` path (after WS7) still reaches the daemon as root.
- `ls -l /run/doorman/` matches the documented modes.
