//! IPC peer-credential policy (SO_PEERCRED), rate limiting and socket creation.
//!
//! The command socket stays world-connectable (PAM runs as root, the CLI and
//! lock watcher run as the user); every request is therefore authorized here
//! against the kernel-reported peer credentials. See `docs/ipc-security.md`.
//!
//! Everything in this module is independent of the camera and ML models so it
//! can be tested at the handler level with an injected [`PeerCred`].

use anyhow::{bail, Context, Result};
use doorman_shared::{IpcConfig, Request, RequestEnvelope, Response, PROTOCOL_VERSION};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

/// Mode of the command socket: world-connectable, security is SO_PEERCRED policy.
pub const COMMAND_SOCKET_MODE: u32 = 0o666;
/// Default mode of the frame/debug stream sockets (they leak face frames).
pub const STREAM_SOCKET_MODE: u32 = 0o600;
/// Stream socket mode when `[ipc] stream_socket_group` is set.
pub const STREAM_SOCKET_GROUP_MODE: u32 = 0o660;
/// Sliding window for the per-uid Authenticate rate limit.
pub const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(60);

/// Upper bound on simultaneously open command connections; further ones are
/// dropped. Connections are served concurrently, so this bounds task/fd use.
pub const MAX_CONCURRENT_CONNECTIONS: usize = 64;
/// How long a connected peer may take to send its request line.
pub const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// Reason for any refusal to authenticate as a user. Deliberately identical for
/// "other uid" and "unknown username" so user existence is not leaked.
pub const DENIED_AUTHENTICATE: &str = "not permitted to authenticate as this user";
/// Reason for refusing an administrative request.
pub const DENIED_ADMIN: &str = "administrative operation requires root";

/// Kernel-reported credentials of the connecting process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerCred {
    pub uid: u32,
    pub gid: u32,
    pub pid: Option<i32>,
}

/// Outcome of [`authorize`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Denied { reason: &'static str },
}

fn is_admin(peer: &PeerCred, cfg: &IpcConfig) -> bool {
    peer.uid == 0 || cfg.admin_uids.contains(&peer.uid)
}

/// Pure authorization decision. The `match` is exhaustive on purpose: adding a
/// `Request` variant fails to compile until it is classified here.
pub fn authorize(
    peer: &PeerCred,
    req: &Request,
    cfg: &IpcConfig,
    resolve_uid: impl Fn(&str) -> Option<u32>,
) -> Decision {
    match req {
        // Self, root, or an explicitly trusted uid. WaitForMatch yields the same
        // decision as Authenticate, so it needs exactly the same authority.
        Request::Authenticate { username, .. } | Request::WaitForMatch { username, .. } => {
            let allowed = peer.uid == 0
                || cfg.trusted_uids.contains(&peer.uid)
                || resolve_uid(username) == Some(peer.uid);
            if allowed {
                Decision::Allow
            } else {
                Decision::Denied {
                    reason: DENIED_AUTHENTICATE,
                }
            }
        }
        // Read-only, no biometric data and no user names.
        Request::Status => Decision::Allow,
        // Administrative: enrollment data, user enumeration, frames, shutdown.
        Request::Enroll { .. }
        | Request::RemoveUser { .. }
        | Request::ListUsers
        | Request::DetectAndRecognize
        | Request::GetLatestDetection
        | Request::Shutdown => {
            if is_admin(peer, cfg) {
                Decision::Allow
            } else {
                Decision::Denied {
                    reason: DENIED_ADMIN,
                }
            }
        }
    }
}

/// Per-uid sliding-window limiter. The clock is injected by the caller (`now`).
pub struct RateLimiter {
    max_per_window: u32,
    window: Duration,
    hits: Mutex<HashMap<u32, VecDeque<Instant>>>,
}

impl RateLimiter {
    /// `max_per_window == 0` disables limiting.
    pub fn new(max_per_window: u32, window: Duration) -> Self {
        Self {
            max_per_window,
            window,
            hits: Mutex::new(HashMap::new()),
        }
    }

    /// Record an attempt by `uid` at `now`. `Err(retry_after_s)` when over budget
    /// (the attempt is then not recorded).
    pub fn check(&self, uid: u32, now: Instant) -> Result<(), u64> {
        if self.max_per_window == 0 {
            return Ok(());
        }
        let mut map = self.hits.lock().unwrap_or_else(|e| e.into_inner());
        let q = map.entry(uid).or_default();
        while q
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) >= self.window)
        {
            q.pop_front();
        }
        if q.len() >= self.max_per_window as usize {
            let oldest = *q.front().expect("non-empty when at limit");
            let wait = self.window - now.saturating_duration_since(oldest);
            return Err(wait.as_secs() + u64::from(wait.subsec_nanos() > 0));
        }
        q.push_back(now);
        Ok(())
    }
}

type UidResolver = Box<dyn Fn(&str) -> Option<u32> + Send + Sync>;
type Clock = Box<dyn Fn() -> Instant + Send + Sync>;

/// Version check + authorization + rate limiting for one request.
pub struct Gatekeeper {
    cfg: IpcConfig,
    limiter: RateLimiter,
    resolve_uid: UidResolver,
    clock: Clock,
}

/// Resolve a username to a uid via the system user database.
pub fn system_resolve_uid(username: &str) -> Option<u32> {
    nix::unistd::User::from_name(username)
        .ok()
        .flatten()
        .map(|u| u.uid.as_raw())
}

impl Gatekeeper {
    pub fn new(cfg: IpcConfig) -> Self {
        Self::with_parts(cfg, Box::new(system_resolve_uid), Box::new(Instant::now))
    }

    /// Test seam: inject the uid resolver and the clock.
    pub fn with_parts(cfg: IpcConfig, resolve_uid: UidResolver, clock: Clock) -> Self {
        let limiter = RateLimiter::new(cfg.max_auth_attempts_per_minute, RATE_LIMIT_WINDOW);
        Self {
            cfg,
            limiter,
            resolve_uid,
            clock,
        }
    }

    pub fn config(&self) -> &IpcConfig {
        &self.cfg
    }

    /// `Ok(())` to proceed, `Err(response)` to send instead of handling.
    pub fn check(&self, peer: &PeerCred, env: &RequestEnvelope) -> Result<(), Response> {
        let version = env.protocol_version.unwrap_or(PROTOCOL_VERSION);
        if version != PROTOCOL_VERSION {
            warn!(
                peer_uid = peer.uid,
                peer_pid = ?peer.pid,
                version,
                "rejected request with unsupported protocol version"
            );
            return Err(Response::UnsupportedProtocol {
                reason: format!("unsupported protocol_version {version}"),
                supported_version: PROTOCOL_VERSION,
            });
        }

        if let Decision::Denied { reason } =
            authorize(peer, &env.request, &self.cfg, &self.resolve_uid)
        {
            warn!(
                peer_uid = peer.uid,
                peer_pid = ?peer.pid,
                request = request_name(&env.request),
                "denied IPC request: {reason}"
            );
            return Err(Response::Denied {
                reason: reason.to_string(),
            });
        }

        if matches!(
            env.request,
            Request::Authenticate { .. } | Request::WaitForMatch { .. }
        ) {
            if let Err(retry_after_s) = self.limiter.check(peer.uid, (self.clock)()) {
                warn!(
                    peer_uid = peer.uid,
                    peer_pid = ?peer.pid,
                    retry_after_s,
                    "rate limited Authenticate"
                );
                return Err(Response::RateLimited { retry_after_s });
            }
        }
        Ok(())
    }
}

/// Variant name only; never includes request payload.
pub fn request_name(req: &Request) -> &'static str {
    match req {
        Request::Authenticate { .. } => "authenticate",
        Request::Enroll { .. } => "enroll",
        Request::ListUsers => "list_users",
        Request::RemoveUser { .. } => "remove_user",
        Request::Status => "status",
        Request::Shutdown => "shutdown",
        Request::DetectAndRecognize => "detect_and_recognize",
        Request::GetLatestDetection => "get_latest_detection",
        Request::WaitForMatch { .. } => "wait_for_match",
    }
}

/// Read peer credentials from the connected socket. Callers must fail closed.
pub fn peer_from_stream(stream: &UnixStream) -> Result<PeerCred> {
    let c = stream.peer_cred().context("SO_PEERCRED lookup failed")?;
    Ok(PeerCred {
        uid: c.uid(),
        gid: c.gid(),
        pid: c.pid(),
    })
}

/// A request that passed the gate, or the response to send instead.
pub enum Gated {
    Allowed(Request),
    Rejected(Response),
}

/// Read one JSON request line and run it through the gate.
pub async fn read_gated_request<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    peer: &PeerCred,
    gate: &Gatekeeper,
) -> Result<Gated> {
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let env: RequestEnvelope =
        serde_json::from_str(line.trim()).context("Failed to parse request")?;
    Ok(match gate.check(peer, &env) {
        Ok(()) => Gated::Allowed(env.request),
        Err(resp) => Gated::Rejected(resp),
    })
}

/// Write one JSON-line response and flush.
pub async fn write_response(stream: &mut UnixStream, response: &Response) -> Result<()> {
    let response_json = serde_json::to_string(response)?;
    stream.write_all(response_json.as_bytes()).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;
    Ok(())
}

/// Serve one connection: peer credentials (fail closed), gate, `handler`, reply.
async fn serve_connection<H, Fut>(stream: UnixStream, gate: &Gatekeeper, handler: &H) -> Result<()>
where
    H: Fn(PeerCred, Request) -> Fut,
    Fut: Future<Output = Response>,
{
    let peer = match peer_from_stream(&stream) {
        Ok(p) => p,
        Err(e) => {
            warn!("Rejecting connection: {e:#}");
            return Ok(());
        }
    };
    let mut reader = BufReader::new(stream);
    let gated = tokio::time::timeout(
        REQUEST_READ_TIMEOUT,
        read_gated_request(&mut reader, &peer, gate),
    )
    .await
    .context("timed out waiting for the request line")??;
    let response = match gated {
        // A WaitForMatch can run for a minute and holds the camera on; its
        // client cancels it by closing the connection, so stop working then.
        Gated::Allowed(request @ Request::WaitForMatch { .. }) => {
            tokio::select! {
                response = handler(peer, request) => response,
                () = peer_closed(&mut reader) => {
                    info!(peer_uid = peer.uid, "WaitForMatch cancelled: client disconnected");
                    return Ok(());
                }
            }
        }
        Gated::Allowed(request) => handler(peer, request).await,
        Gated::Rejected(response) => response,
    };
    write_response(reader.get_mut(), &response).await
}

/// Resolves when the peer closes (or breaks) the connection; stray bytes after
/// the request line are discarded.
async fn peer_closed(reader: &mut BufReader<UnixStream>) {
    loop {
        match reader.fill_buf().await {
            Ok([]) | Err(_) => return,
            Ok(stray) => {
                let n = stray.len();
                reader.consume(n);
            }
        }
    }
}

/// Accept loop for the command socket. Every connection is served in its own
/// task, so a long request (Authenticate, WaitForMatch, Enroll) never blocks
/// others such as Status. Shared state belongs in `handler`'s captures.
pub async fn serve<H, Fut>(listener: UnixListener, gate: Arc<Gatekeeper>, handler: Arc<H>)
where
    H: Fn(PeerCred, Request) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Response> + Send + 'static,
{
    let slots = Arc::new(Semaphore::new(MAX_CONCURRENT_CONNECTIONS));
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            Err(e) => {
                error!("Accept error: {}", e);
                continue;
            }
        };
        let Ok(slot) = slots.clone().try_acquire_owned() else {
            warn!("Dropping connection: {MAX_CONCURRENT_CONNECTIONS} already open");
            continue;
        };
        let gate = gate.clone();
        let handler = handler.clone();
        tokio::spawn(async move {
            let _slot = slot;
            if let Err(e) = serve_connection(stream, &gate, &*handler).await {
                error!("Connection error: {:#}", e);
            }
        });
    }
}

/// Whether `peer` may stay connected to a frame/debug stream socket. With a
/// configured stream group, filesystem permissions are the gate (supplementary
/// groups are not visible through SO_PEERCRED); otherwise root/admin only.
pub fn stream_peer_allowed(peer: &PeerCred, cfg: &IpcConfig) -> bool {
    cfg.stream_socket_group.is_some() || is_admin(peer, cfg)
}

/// Bind a unix socket at `path` with exactly `mode`, optionally chowned to `group`.
pub fn bind_socket(path: &str, mode: u32, group: Option<&str>) -> Result<UnixListener> {
    let _ = std::fs::remove_file(path);
    let listener =
        UnixListener::bind(path).with_context(|| format!("Failed to bind UNIX socket {path}"))?;
    if let Some(name) = group {
        let gid = nix::unistd::Group::from_name(name)
            .with_context(|| format!("group lookup failed for '{name}'"))?
            .map(|g| g.gid);
        let Some(gid) = gid else {
            bail!("[ipc] stream_socket_group '{name}' does not exist");
        };
        nix::unistd::chown(Path::new(path), None, Some(gid))
            .with_context(|| format!("Failed to chown {path} to group '{name}'"))?;
    }
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
        .with_context(|| format!("Failed to set mode {mode:o} on {path}"))?;
    Ok(listener)
}

/// Bind a frame/debug stream socket per `[ipc] stream_socket_group`.
pub fn bind_stream_socket(path: &str, cfg: &IpcConfig) -> Result<UnixListener> {
    let group = cfg.stream_socket_group.as_deref();
    let mode = if group.is_some() {
        STREAM_SOCKET_GROUP_MODE
    } else {
        STREAM_SOCKET_MODE
    };
    let l = bind_socket(path, mode, group)?;
    info!("stream socket {path} mode {mode:o} group {group:?}");
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALICE: u32 = 1000;
    const BOB: u32 = 1001;

    fn peer(uid: u32) -> PeerCred {
        PeerCred {
            uid,
            gid: uid,
            pid: Some(1),
        }
    }

    fn resolve(name: &str) -> Option<u32> {
        match name {
            "alice" => Some(ALICE),
            "bob" => Some(BOB),
            _ => None,
        }
    }

    fn auth(name: &str) -> Request {
        Request::Authenticate {
            username: name.into(),
            service: None,
            tty: None,
        }
    }

    fn wait(name: &str) -> Request {
        Request::WaitForMatch {
            username: name.into(),
            timeout_ms: 1000,
        }
    }

    #[test]
    fn policy_table() {
        let mut trusted = IpcConfig::default();
        trusted.trusted_uids = vec![2000];
        let mut admin = IpcConfig::default();
        admin.admin_uids = vec![3000];
        let def = IpcConfig::default();
        let allow = Decision::Allow;
        let deny_a = Decision::Denied {
            reason: DENIED_AUTHENTICATE,
        };
        let deny_adm = Decision::Denied {
            reason: DENIED_ADMIN,
        };

        let cases: Vec<(&str, PeerCred, Request, &IpcConfig, Decision)> = vec![
            ("self", peer(ALICE), auth("alice"), &def, allow.clone()),
            ("root for user", peer(0), auth("alice"), &def, allow.clone()),
            ("other uid", peer(BOB), auth("alice"), &def, deny_a.clone()),
            (
                "unknown user as user",
                peer(BOB),
                auth("ghost"),
                &def,
                deny_a.clone(),
            ),
            (
                "unknown user as root",
                peer(0),
                auth("ghost"),
                &def,
                allow.clone(),
            ),
            (
                "trusted uid",
                peer(2000),
                auth("alice"),
                &trusted,
                allow.clone(),
            ),
            (
                "trusted uid not in cfg",
                peer(2000),
                auth("alice"),
                &def,
                deny_a.clone(),
            ),
            ("wait self", peer(ALICE), wait("alice"), &def, allow.clone()),
            ("wait root", peer(0), wait("alice"), &def, allow.clone()),
            (
                "wait other uid",
                peer(BOB),
                wait("alice"),
                &def,
                deny_a.clone(),
            ),
            (
                "wait unknown user as user",
                peer(BOB),
                wait("ghost"),
                &def,
                deny_a.clone(),
            ),
            (
                "wait trusted uid",
                peer(2000),
                wait("alice"),
                &trusted,
                allow.clone(),
            ),
            (
                "admin_uid cannot auth for others",
                peer(3000),
                auth("alice"),
                &admin,
                deny_a,
            ),
            (
                "status open",
                peer(BOB),
                Request::Status,
                &def,
                allow.clone(),
            ),
            (
                "enroll root",
                peer(0),
                Request::Enroll {
                    username: "alice".into(),
                    add: false,
                },
                &def,
                allow.clone(),
            ),
            (
                "enroll self denied",
                peer(ALICE),
                Request::Enroll {
                    username: "alice".into(),
                    add: false,
                },
                &def,
                deny_adm.clone(),
            ),
            (
                "remove other denied",
                peer(BOB),
                Request::RemoveUser {
                    username: "alice".into(),
                },
                &def,
                deny_adm.clone(),
            ),
            (
                "list denied",
                peer(ALICE),
                Request::ListUsers,
                &def,
                deny_adm.clone(),
            ),
            (
                "list admin_uid",
                peer(3000),
                Request::ListUsers,
                &admin,
                allow.clone(),
            ),
            (
                "trusted is not admin",
                peer(2000),
                Request::ListUsers,
                &trusted,
                deny_adm.clone(),
            ),
            (
                "detect denied",
                peer(ALICE),
                Request::DetectAndRecognize,
                &def,
                deny_adm.clone(),
            ),
            (
                "latest denied",
                peer(ALICE),
                Request::GetLatestDetection,
                &def,
                deny_adm.clone(),
            ),
            (
                "shutdown denied",
                peer(ALICE),
                Request::Shutdown,
                &def,
                deny_adm,
            ),
            ("shutdown root", peer(0), Request::Shutdown, &def, allow),
        ];
        for (name, p, req, cfg, want) in cases {
            assert_eq!(authorize(&p, &req, cfg, resolve), want, "case: {name}");
        }
    }

    #[test]
    fn rate_limiter_sliding_window() {
        let rl = RateLimiter::new(3, Duration::from_secs(60));
        let t0 = Instant::now();
        for i in 0..3 {
            assert!(rl.check(ALICE, t0 + Duration::from_secs(i)).is_ok());
        }
        // Other uid has its own budget.
        assert!(rl.check(BOB, t0).is_ok());
        // Over budget: retry when the oldest hit (t0) leaves the window.
        assert_eq!(rl.check(ALICE, t0 + Duration::from_secs(10)), Err(50));
        // Rejected attempts are not recorded; after the oldest expires one slot opens.
        assert!(rl.check(ALICE, t0 + Duration::from_secs(60)).is_ok());
        assert!(rl.check(ALICE, t0 + Duration::from_secs(60)).is_err());
    }

    #[test]
    fn rate_limiter_zero_disables() {
        let rl = RateLimiter::new(0, Duration::from_secs(60));
        let t0 = Instant::now();
        for _ in 0..100 {
            assert!(rl.check(0, t0).is_ok());
        }
    }

    #[test]
    fn gatekeeper_version_handling() {
        let gate = Gatekeeper::new(IpcConfig::default());
        let mk = |v| RequestEnvelope {
            protocol_version: v,
            request: Request::Status,
        };
        assert!(gate.check(&peer(ALICE), &mk(None)).is_ok());
        assert!(gate
            .check(&peer(ALICE), &mk(Some(PROTOCOL_VERSION)))
            .is_ok());
        assert!(matches!(
            gate.check(&peer(ALICE), &mk(Some(PROTOCOL_VERSION + 1))),
            Err(Response::UnsupportedProtocol { .. })
        ));
    }
}
