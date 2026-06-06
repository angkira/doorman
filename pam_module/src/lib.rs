//! Doorman PAM module.
//!
//! The PAM integration links against libpam and is therefore Linux-only. On
//! other platforms (e.g. macOS dev machines) this crate compiles to an empty
//! cdylib so the workspace still builds; the PAM symbols are simply absent.
//!
//! Deployment model
//! ----------------
//! `pam_doorman` is a thin **IPC client**. All face capture / recognition lives
//! in the always-running daemon (`doormand`, a systemd service). When PAM calls
//! `pam_sm_authenticate`, this module connects to the daemon's UNIX socket and
//! sends an `Authenticate { username }` request; the daemon performs a *fresh*
//! camera capture + match and replies `Success`/`Failure` within
//! `AUTH_TIMEOUT_SECS`.
//!
//! Recommended PAM line (see INSTALL.md) — `auth sufficient pam_doorman.so`.
//! "sufficient" means: success here is enough to authenticate, but anything
//! else (face not matched, daemon down, timeout) falls through to the next
//! module (normally `pam_unix.so` password). This module therefore MUST NEVER
//! hang the login and MUST fail soft when it cannot reach the daemon.
//!
//! Return-code policy
//! ------------------
//! * match              -> `PAM_SUCCESS`        (login proceeds)
//! * no match           -> `PAM_AUTH_ERR`       (try next module / password)
//! * daemon unreachable -> `PAM_AUTHINFO_UNAVAIL` (auth source down; fall through)
//! * timeout / IO error -> `PAM_AUTHINFO_UNAVAIL` (never block the greeter)
//! * root / no username -> `PAM_AUTHINFO_UNAVAIL` (let password handle it)
#![cfg(target_os = "linux")]

use doorman_shared::{Request, Response, AUTH_TIMEOUT_SECS, SOCKET_PATH};
use std::ffi::CStr;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

// PAM return codes — canonical Linux-PAM values from <security/_pam_types.h>.
// These are a stable ABI; do not change them.
const PAM_SUCCESS: libc::c_int = 0;
const PAM_AUTH_ERR: libc::c_int = 7;
const PAM_AUTHINFO_UNAVAIL: libc::c_int = 9;

/// Hard wall-clock ceiling for the whole authenticate attempt.
///
/// The daemon is bounded by `AUTH_TIMEOUT_SECS` (its per-socket read timeout),
/// but a wedged daemon, a half-open socket, or a slow connect could still stall
/// the greeter. We run the IPC on a worker thread and give up after this
/// deadline no matter what — the greeter must never freeze. We allow one extra
/// second of slack over the socket read timeout for connect + serialization.
fn hard_deadline() -> Duration {
    Duration::from_secs(AUTH_TIMEOUT_SECS + 1)
}

/// Outcome of an authentication attempt, mapped to a PAM code by the caller.
enum AuthOutcome {
    /// Daemon replied: face matched.
    Matched,
    /// Daemon replied: face not matched (or user not enrolled).
    NotMatched,
    /// Could not reach / get a usable reply from the daemon.
    Unavailable,
}

/// Connect to the daemon and run one authenticate exchange.
///
/// Returns `Matched` / `NotMatched` only when the daemon actually answered;
/// any connection or protocol failure yields `Unavailable` so PAM falls through
/// to the password prompt instead of hard-failing.
fn authenticate_ipc(username: &str) -> AuthOutcome {
    // System-mode socket path. PAM runs as root at the greeter, so the daemon
    // (a system service) owns /run/doorman.sock. The user-mode XDG socket is
    // only used by the dev preview/CLI, never by the greeter.
    let mut stream = match UnixStream::connect(SOCKET_PATH) {
        Ok(s) => s,
        Err(_) => return AuthOutcome::Unavailable, // daemon not running -> fall through
    };

    // Bound every blocking syscall. The hard-deadline thread below is the
    // backstop; these keep individual reads/writes from blocking forever.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(AUTH_TIMEOUT_SECS)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));

    let request = Request::Authenticate {
        username: username.to_string(),
    };
    let request_json = match serde_json::to_string(&request) {
        Ok(j) => j,
        Err(_) => return AuthOutcome::Unavailable,
    };

    if writeln!(stream, "{}", request_json).is_err() || stream.flush().is_err() {
        return AuthOutcome::Unavailable;
    }

    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    if reader.read_line(&mut response_line).is_err() || response_line.is_empty() {
        return AuthOutcome::Unavailable; // timeout or daemon closed the socket
    }

    match serde_json::from_str::<Response>(&response_line) {
        Ok(Response::Success { .. }) => AuthOutcome::Matched,
        Ok(Response::Failure { .. }) => AuthOutcome::NotMatched,
        // Progress or anything unexpected: treat as "couldn't decide".
        Ok(_) => AuthOutcome::Unavailable,
        Err(_) => AuthOutcome::Unavailable,
    }
}

/// Run `authenticate_ipc` on a worker thread with a hard wall-clock deadline so
/// the greeter can never be frozen by a stuck daemon or socket.
fn authenticate_with_deadline(username: &str) -> AuthOutcome {
    let (tx, rx) = mpsc::channel();
    let user = username.to_string();
    // Detached worker: if it overruns the deadline we abandon it. The send may
    // fail (receiver gone) — that's fine, the thread just exits.
    thread::spawn(move || {
        let _ = tx.send(authenticate_ipc(&user));
    });

    match rx.recv_timeout(hard_deadline()) {
        Ok(outcome) => outcome,
        // Worker overran the deadline: do not block login, fall through to password.
        Err(_) => AuthOutcome::Unavailable,
    }
}

#[no_mangle]
pub extern "C" fn pam_sm_authenticate(
    pamh: *mut pam_sys::PamHandle,
    _flags: libc::c_int,
    _argc: libc::c_int,
    _argv: *const *const libc::c_char,
) -> libc::c_int {
    unsafe {
        let mut user: *const libc::c_char = std::ptr::null();
        let ret = pam_sys::raw::pam_get_user(pamh, &mut user, std::ptr::null());

        // If PAM can't give us a username, this is not a face-auth failure —
        // let the rest of the stack (password) handle it.
        if ret != PAM_SUCCESS || user.is_null() {
            return PAM_AUTHINFO_UNAVAIL;
        }

        let username = match CStr::from_ptr(user).to_str() {
            Ok(s) => s,
            Err(_) => return PAM_AUTHINFO_UNAVAIL,
        };

        // Never attempt face unlock for root; defer to password.
        if username == "root" || username.is_empty() {
            return PAM_AUTHINFO_UNAVAIL;
        }

        match authenticate_with_deadline(username) {
            AuthOutcome::Matched => PAM_SUCCESS,
            AuthOutcome::NotMatched => PAM_AUTH_ERR,
            AuthOutcome::Unavailable => PAM_AUTHINFO_UNAVAIL,
        }
    }
}

#[no_mangle]
pub extern "C" fn pam_sm_setcred(
    _pamh: *mut pam_sys::PamHandle,
    _flags: libc::c_int,
    _argc: libc::c_int,
    _argv: *const *const libc::c_char,
) -> libc::c_int {
    PAM_SUCCESS
}
