//! systemd-logind lock supervisor (SYSTEM bus).
//!
//! Resolves the active graphical session on seat0 and watches its lock / idle
//! state, exposing a single derived gate:
//!
//! ```text
//! armed = LockedHint && !IdleHint
//! ```
//!
//! `armed` drives the detection/recognition pipeline in `main.rs` exactly like
//! the preview-subscriber gate: the pipeline runs while the screen is LOCKED and
//! the display is NOT idle (so we only burn the iGPU when there is actually a
//! locked screen a face could unlock), and stops otherwise.
//!
//! Robustness is mandatory: a missing session, an unavailable system bus, or a
//! session that disappears must NEVER panic or kill the daemon — every failure
//! path logs a warning and emits `armed = false` (fail-safe: no spurious
//! pipeline, no proactive unlock).
//!
//! This module also exposes the resolved session path + a cheap [`Unlocker`]
//! handle so the proactive-unlock path (see `recognition_pipeline`) can call
//! `Session.Unlock()`, plus a monotonically increasing "lock generation"
//! counter that ticks on every `Lock` signal so the one-shot unlock logic can
//! re-arm after a fresh lock.

use std::sync::Arc;

use tokio::sync::watch;
use tracing::{info, warn};
use zbus::zvariant::OwnedObjectPath;
use zbus::Connection;

/// logind Manager proxy: enumerate sessions.
#[zbus::proxy(
    interface = "org.freedesktop.login1.Manager",
    default_service = "org.freedesktop.login1",
    default_path = "/org/freedesktop/login1"
)]
trait LoginManager {
    /// Returns (session_id, uid, user_name, seat_id, object_path).
    fn list_sessions(&self) -> zbus::Result<Vec<(String, u32, String, String, OwnedObjectPath)>>;
}

/// logind Session proxy: lock/unlock signals + the hint properties we gate on.
#[zbus::proxy(
    interface = "org.freedesktop.login1.Session",
    default_service = "org.freedesktop.login1"
)]
trait LoginSession {
    fn unlock(&self) -> zbus::Result<()>;

    #[zbus(property)]
    fn locked_hint(&self) -> zbus::Result<bool>;

    #[zbus(property)]
    fn idle_hint(&self) -> zbus::Result<bool>;

    #[zbus(property)]
    fn active(&self) -> zbus::Result<bool>;

    #[zbus(property)]
    fn type_(&self) -> zbus::Result<String>;

    #[zbus(property)]
    fn seat(&self) -> zbus::Result<(String, OwnedObjectPath)>;

    #[zbus(signal)]
    fn lock(&self) -> zbus::Result<()>;

    #[zbus(signal)]
    fn unlock(&self) -> zbus::Result<()>;
}

/// Cheap clonable handle that performs `Session.Unlock()` on the resolved
/// session. Holds the shared zbus connection + the session object path. Calling
/// `unlock()` when the daemon's `doorman` user lacks the
/// `org.freedesktop.login1.lock-sessions` polkit grant returns an
/// "not authorized" error, which is logged and otherwise harmless (the polkit
/// rule is installed by the separate installer stage).
#[derive(Clone)]
pub struct Unlocker {
    conn: Connection,
    session_path: OwnedObjectPath,
}

impl Unlocker {
    /// Attempt to unlock the resolved logind session. Returns Err on D-Bus /
    /// polkit failure; the caller decides how to log it (never fatal).
    pub async fn unlock(&self) -> zbus::Result<()> {
        let session = LoginSessionProxy::builder(&self.conn)
            .path(self.session_path.clone())?
            .build()
            .await?;
        session.unlock().await
    }

    /// The resolved logind session object path. Exposed per the lock-supervisor
    /// contract (callers may want `Manager.UnlockSession(id)` or diagnostics);
    /// the in-tree unlock path uses `unlock()` directly, so this is currently
    /// unused but intentionally part of the public handle.
    #[allow(dead_code)]
    pub fn session_path(&self) -> &OwnedObjectPath {
        &self.session_path
    }
}

/// Handles + receivers handed back to `main.rs` / the recognition path.
pub struct LockSupervisor {
    /// `armed = LockedHint && !IdleHint`. Pipeline gate.
    pub armed_rx: watch::Receiver<bool>,
    /// Monotonic counter bumped on every `Lock` signal. The proactive-unlock
    /// one-shot watches this to re-arm after a fresh lock.
    pub lock_gen_rx: watch::Receiver<u64>,
    /// `Some` once a session was resolved; lets the unlock path call Unlock().
    /// `None` if no session/bus was available at startup.
    pub unlocker: Option<Unlocker>,
}

/// Connect to the SYSTEM bus, resolve the active graphical seat0 session, and
/// spawn a background task that keeps `armed` / `lock_gen` up to date.
///
/// On ANY failure to reach the bus or find a session this still returns a valid
/// `LockSupervisor` whose `armed_rx` is pinned to `false` (and `unlocker` is
/// `None`); it logs a warning and never errors out, so the daemon always starts.
pub async fn spawn() -> LockSupervisor {
    let (armed_tx, armed_rx) = watch::channel(false);
    let (lock_gen_tx, lock_gen_rx) = watch::channel(0u64);

    let conn = match Connection::system().await {
        Ok(c) => c,
        Err(e) => {
            warn!(
                "lock-supervisor: cannot connect to system D-Bus ({e}); pipeline lock-gating disabled (armed=false)"
            );
            return LockSupervisor { armed_rx, lock_gen_rx, unlocker: None };
        }
    };

    let session_path = match resolve_graphical_session(&conn).await {
        Some(p) => p,
        None => {
            warn!(
                "lock-supervisor: no active graphical seat0 session found; pipeline lock-gating disabled (armed=false)"
            );
            return LockSupervisor { armed_rx, lock_gen_rx, unlocker: None };
        }
    };

    info!("lock-supervisor: tracking logind session {}", session_path.as_str());

    let unlocker = Some(Unlocker { conn: conn.clone(), session_path: session_path.clone() });

    // Background watcher. Robust: if it can't build the session proxy it just
    // emits armed=false and exits (daemon keeps running, pipeline stays gated
    // off / preview-only).
    let armed_tx = Arc::new(armed_tx);
    let lock_gen_tx = Arc::new(lock_gen_tx);
    tokio::spawn(run_watcher(conn, session_path, armed_tx, lock_gen_tx));

    LockSupervisor { armed_rx, lock_gen_rx, unlocker }
}

async fn build_session_proxy(
    conn: &Connection,
    path: OwnedObjectPath,
) -> zbus::Result<LoginSessionProxy<'static>> {
    LoginSessionProxy::builder(conn).path(path)?.build().await
}

/// Pick the active graphical session on seat0. Preference order:
///   1. seat0 + graphical (wayland/x11) + Active=true
///   2. seat0 + graphical (first match)
/// Falls back to `None` if none qualifies.
async fn resolve_graphical_session(conn: &Connection) -> Option<OwnedObjectPath> {
    let manager = match LoginManagerProxy::new(conn).await {
        Ok(m) => m,
        Err(e) => {
            warn!("lock-supervisor: cannot build logind Manager proxy: {e}");
            return None;
        }
    };
    let sessions = match manager.list_sessions().await {
        Ok(s) => s,
        Err(e) => {
            warn!("lock-supervisor: ListSessions failed: {e}");
            return None;
        }
    };

    let mut fallback: Option<OwnedObjectPath> = None;
    for (id, _uid, _user, seat, path) in sessions {
        if seat != "seat0" {
            continue;
        }
        let session = match build_session_proxy(conn, path.clone()).await {
            Ok(s) => s,
            Err(e) => {
                warn!("lock-supervisor: cannot build session proxy for {id}: {e}");
                continue;
            }
        };
        let stype = session.type_().await.unwrap_or_default();
        let is_graphical = stype == "wayland" || stype == "x11" || stype == "mir";
        if !is_graphical {
            continue;
        }
        if session.active().await.unwrap_or(false) {
            return Some(path);
        }
        if fallback.is_none() {
            fallback = Some(path);
        }
    }
    fallback
}

/// Compute and publish `armed`, deduping repeated identical values.
fn publish_armed(armed_tx: &watch::Sender<bool>, locked: bool, idle: bool) {
    let armed = locked && !idle;
    if *armed_tx.borrow() != armed {
        info!("lock-supervisor: armed={armed} (LockedHint={locked}, IdleHint={idle})");
        let _ = armed_tx.send(armed);
    }
}

async fn run_watcher(
    conn: Connection,
    session_path: OwnedObjectPath,
    armed_tx: Arc<watch::Sender<bool>>,
    lock_gen_tx: Arc<watch::Sender<u64>>,
) {
    use futures::StreamExt;

    let session = match build_session_proxy(&conn, session_path.clone()).await {
        Ok(s) => s,
        Err(e) => {
            warn!("lock-supervisor: cannot build session proxy ({e}); armed pinned false");
            let _ = armed_tx.send(false);
            return;
        }
    };

    // Seed initial state.
    let mut locked = session.locked_hint().await.unwrap_or(false);
    let mut idle = session.idle_hint().await.unwrap_or(false);
    publish_armed(&armed_tx, locked, idle);

    // Subscribe to the lock/unlock signals and the relevant property changes.
    let mut lock_stream = match session.receive_lock().await {
        Ok(s) => s,
        Err(e) => {
            warn!("lock-supervisor: cannot subscribe Lock signal ({e}); armed pinned false");
            let _ = armed_tx.send(false);
            return;
        }
    };
    let mut unlock_stream = match session.receive_unlock().await {
        Ok(s) => s,
        Err(e) => {
            warn!("lock-supervisor: cannot subscribe Unlock signal ({e})");
            let _ = armed_tx.send(false);
            return;
        }
    };
    let mut locked_changed = session.receive_locked_hint_changed().await;
    let mut idle_changed = session.receive_idle_hint_changed().await;

    info!("lock-supervisor: watching Lock/Unlock + LockedHint/IdleHint");

    loop {
        tokio::select! {
            // Explicit Lock signal: the screen locker just locked. Treat as
            // LockedHint=true immediately and bump the lock generation so the
            // proactive-unlock one-shot re-arms.
            Some(_) = lock_stream.next() => {
                locked = true;
                let next = lock_gen_tx.borrow().wrapping_add(1);
                let _ = lock_gen_tx.send(next);
                info!("lock-supervisor: Lock signal (gen={next})");
                publish_armed(&armed_tx, locked, idle);
            }
            // Explicit Unlock signal (PAM/our own unlock succeeded).
            Some(_) = unlock_stream.next() => {
                locked = false;
                info!("lock-supervisor: Unlock signal");
                publish_armed(&armed_tx, locked, idle);
            }
            // LockedHint property changed (covers lockers that only set the hint).
            Some(change) = locked_changed.next() => {
                if let Ok(v) = change.get().await {
                    locked = v;
                    publish_armed(&armed_tx, locked, idle);
                }
            }
            // IdleHint property changed (display blank/idle => disarm).
            Some(change) = idle_changed.next() => {
                if let Ok(v) = change.get().await {
                    idle = v;
                    publish_armed(&armed_tx, locked, idle);
                }
            }
            else => {
                // All streams ended (session went away). Fail safe.
                warn!("lock-supervisor: signal streams closed (session gone?); armed=false");
                let _ = armed_tx.send(false);
                break;
            }
        }
    }
}
