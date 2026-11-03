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
//! sends an `Authenticate { username, service, tty }` request (service and tty
//! are `PAM_SERVICE` / `PAM_TTY`); the daemon performs a *fresh*
//! camera capture + match and replies `Success`/`Failure` within
//! `AUTH_TIMEOUT_SECS`.
//!
//! Recommended PAM line (see INSTALL.md) — `auth sufficient pam_doorman.so`.
//! The only module argument is `socket=/abs/path` (tests; see `SOCKET_ARG_PREFIX`).
//! "sufficient" means: success here is enough to authenticate, but anything
//! else (face not matched, daemon down, timeout) falls through to the next
//! module (normally `pam_unix.so` password). This module therefore MUST NEVER
//! hang the login and MUST fail soft when it cannot reach the daemon.
//!
//! Return-code policy
//! ------------------
//! * match              -> `PAM_SUCCESS`        (login proceeds)
//! * no match           -> `PAM_AUTH_ERR`       (try next module / password)
//! * denied / rate-limited / protocol mismatch -> `PAM_AUTHINFO_UNAVAIL` (logged to syslog)
//! * daemon unreachable -> `PAM_AUTHINFO_UNAVAIL` (auth source down; fall through)
//! * timeout / IO error -> `PAM_AUTHINFO_UNAVAIL` (never block the greeter)
//! * root / no username -> `PAM_AUTHINFO_UNAVAIL` (let password handle it)
//! * a panic anywhere in an entry point -> `PAM_AUTHINFO_UNAVAIL` (never unwinds into libpam)
//!
//! `PAM_AUTHINFO_UNAVAIL` rather than `PAM_IGNORE` for "no answer": under
//! `sufficient` both continue to the password module, but under a mistaken
//! `required` UNAVAIL fails the stack (closed) where IGNORE would silently skip
//! the module, and the code stays the one this module has always returned.
//!
//! Latency when the daemon is down: connecting uses a non-blocking socket, so
//! a missing or busy socket fails immediately (well under 100 ms) and the
//! password prompt appears without delay.
#![cfg(target_os = "linux")]

use doorman_shared::config::MAX_SUDO_TIMEOUT_MS;
use doorman_shared::{Request, RequestEnvelope, Response, AUTH_TIMEOUT_SECS, SOCKET_PATH};
use std::ffi::CStr;
use std::io::{BufRead, BufReader, Write};
use std::os::fd::{FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixStream;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

// PAM return codes — canonical Linux-PAM values from <security/_pam_types.h>.
// These are a stable ABI; do not change them.
const PAM_SUCCESS: libc::c_int = 0;
const PAM_AUTH_ERR: libc::c_int = 7;
const PAM_AUTHINFO_UNAVAIL: libc::c_int = 9;

/// Extra time over the socket read timeout before the worker is abandoned.
const DEADLINE_SLACK: Duration = Duration::from_secs(1);
/// Margin over the daemon's own sudo timeout for connect, scheduling and the reply.
const SUDO_REPLY_MARGIN: Duration = Duration::from_millis(500);
/// Write timeout for the request line.
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);
/// PAM service names (`PAM_SERVICE`) that get the longer sudo wait: `sudo`,
/// and `sudo-i` where a distribution configures it.
const SUDO_SERVICE_PREFIX: &str = "sudo";

/// PAM module argument that overrides the daemon socket path
/// (`auth sufficient pam_doorman.so socket=/run/x/doorman.sock`). It exists for
/// the pamtester integration test (`tests/test_pam_module.py`). PAM arguments
/// come from the service file under `/etc/pam.d`, which only root can write, so
/// an unprivileged user cannot redirect the module. There is deliberately no
/// environment variable: PAM runs inside setuid programs (sudo, su) whose
/// environment the caller controls.
const SOCKET_ARG_PREFIX: &str = "socket=";

/// The socket path from the module arguments, else the system default. Only an
/// absolute path is accepted; anything else is ignored.
fn socket_from_args<'a>(args: impl IntoIterator<Item = &'a str>) -> PathBuf {
    args.into_iter()
        .filter_map(|a| a.strip_prefix(SOCKET_ARG_PREFIX))
        .map(PathBuf::from)
        .find(|p| p.is_absolute())
        .unwrap_or_else(|| PathBuf::from(SOCKET_PATH))
}

/// The module arguments as UTF-8 strings (non-UTF-8 entries are skipped).
///
/// # Safety
/// `argv` must point to `argc` valid C strings, as libpam passes them.
unsafe fn module_args<'a>(argc: libc::c_int, argv: *const *const libc::c_char) -> Vec<&'a str> {
    if argv.is_null() || argc <= 0 {
        return Vec::new();
    }
    (0..argc as usize)
        .filter_map(|i| {
            let p = *argv.add(i);
            if p.is_null() {
                None
            } else {
                CStr::from_ptr(p).to_str().ok()
            }
        })
        .collect()
}

/// What is sent to the daemon for one attempt.
struct AuthTarget<'a> {
    username: &'a str,
    service: Option<&'a str>,
    tty: Option<&'a str>,
}

/// How long the daemon is given to answer. The strict sudo profile has a
/// longer budget (up to `MAX_SUDO_TIMEOUT_MS`) than the lock screen.
fn read_timeout(service: Option<&str>) -> Duration {
    if service.is_some_and(|s| s.starts_with(SUDO_SERVICE_PREFIX)) {
        Duration::from_millis(MAX_SUDO_TIMEOUT_MS) + SUDO_REPLY_MARGIN
    } else {
        Duration::from_secs(AUTH_TIMEOUT_SECS)
    }
}

/// Outcome of an authentication attempt, mapped to a PAM code by the caller.
#[derive(Debug, PartialEq, Eq)]
enum AuthOutcome {
    /// Daemon replied: face matched.
    Matched,
    /// Daemon replied: face not matched (or user not enrolled).
    NotMatched,
    /// Daemon refused to decide (peer policy denied, or rate limited). Fall
    /// through to the next PAM module like an unreachable daemon.
    Refused,
    /// Could not reach / get a usable reply from the daemon.
    Unavailable,
}

fn outcome_to_pam(outcome: AuthOutcome) -> libc::c_int {
    match outcome {
        AuthOutcome::Matched => PAM_SUCCESS,
        AuthOutcome::NotMatched => PAM_AUTH_ERR,
        AuthOutcome::Refused | AuthOutcome::Unavailable => PAM_AUTHINFO_UNAVAIL,
    }
}

/// Log to syslog (AUTHPRIV). The message is a fixed format string argument so
/// daemon-supplied text can never be interpreted as a format string.
fn log_syslog(msg: &str) {
    if let Ok(c) = std::ffi::CString::new(msg.replace('\0', " ")) {
        unsafe {
            libc::syslog(
                libc::LOG_AUTHPRIV | libc::LOG_WARNING,
                c"%s".as_ptr(),
                c.as_ptr(),
            )
        };
    }
}

/// Connect to a Unix socket without ever blocking: the socket is non-blocking,
/// and a Unix `connect` completes or fails at once (`ENOENT`/`ECONNREFUSED`
/// when no daemon listens, `EAGAIN` when its backlog is full), so a down or
/// wedged daemon costs microseconds, not a timeout.
fn connect_nonblocking(path: &Path) -> std::io::Result<UnixStream> {
    let fd = unsafe {
        libc::socket(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
            0,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Owns the fd from here on, so every early return closes it.
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };

    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_os_str().as_bytes();
    if bytes.len() >= addr.sun_path.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "socket path too long",
        ));
    }
    for (dst, src) in addr.sun_path.iter_mut().zip(bytes) {
        *dst = *src as libc::c_char;
    }
    let rc = unsafe {
        libc::connect(
            fd,
            &addr as *const libc::sockaddr_un as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let stream = UnixStream::from(owned);
    stream.set_nonblocking(false)?;
    Ok(stream)
}

/// Connect to the daemon and run one authenticate exchange.
///
/// Returns `Matched` / `NotMatched` only when the daemon actually answered;
/// any connection or protocol failure yields `Unavailable` so PAM falls through
/// to the password prompt instead of hard-failing.
fn authenticate_ipc(socket: &Path, target: &AuthTarget, read_timeout: Duration) -> AuthOutcome {
    // System-mode socket path. PAM runs as root at the greeter, so the daemon
    // (a system service) owns /run/doorman/doorman.sock. The user-mode XDG socket is
    // only used by the dev preview/CLI, never by the greeter.
    let mut stream = match connect_nonblocking(socket) {
        Ok(s) => s,
        Err(_) => return AuthOutcome::Unavailable, // daemon not running -> fall through
    };

    // Bound every blocking syscall. The hard-deadline thread below is the
    // backstop; these keep individual reads/writes from blocking forever.
    let _ = stream.set_read_timeout(Some(read_timeout));
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));

    let request = RequestEnvelope::new(Request::Authenticate {
        username: target.username.to_string(),
        service: target.service.map(str::to_string),
        tty: target.tty.map(str::to_string),
    });
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
        Ok(Response::Denied { reason }) => {
            log_syslog(&format!("pam_doorman: daemon denied request: {reason}"));
            AuthOutcome::Refused
        }
        Ok(Response::RateLimited { retry_after_s }) => {
            log_syslog(&format!(
                "pam_doorman: daemon rate limited, retry in {retry_after_s}s"
            ));
            AuthOutcome::Refused
        }
        Ok(Response::UnsupportedProtocol { reason, .. }) => {
            log_syslog(&format!("pam_doorman: daemon protocol mismatch: {reason}"));
            AuthOutcome::Refused
        }
        // Progress or anything unexpected: treat as "couldn't decide".
        Ok(_) => AuthOutcome::Unavailable,
        Err(_) => AuthOutcome::Unavailable,
    }
}

/// Run `authenticate_ipc` on a worker thread with a hard wall-clock deadline so
/// the greeter can never be frozen by a stuck daemon or socket.
fn authenticate_with_deadline(socket: &Path, target: &AuthTarget) -> AuthOutcome {
    let read_timeout = read_timeout(target.service);
    authenticate_with_timeouts(socket, target, read_timeout, read_timeout + DEADLINE_SLACK)
}

fn authenticate_with_timeouts(
    socket: &Path,
    target: &AuthTarget,
    read_timeout: Duration,
    deadline: Duration,
) -> AuthOutcome {
    let (tx, rx) = mpsc::channel();
    let socket = socket.to_path_buf();
    let username = target.username.to_string();
    let service = target.service.map(str::to_string);
    let tty = target.tty.map(str::to_string);
    // Detached worker: if it overruns the deadline we abandon it. The send may
    // fail (receiver gone) — that's fine, the thread just exits.
    thread::spawn(move || {
        let target = AuthTarget {
            username: &username,
            service: service.as_deref(),
            tty: tty.as_deref(),
        };
        let _ = tx.send(authenticate_ipc(&socket, &target, read_timeout));
    });

    match rx.recv_timeout(deadline) {
        Ok(outcome) => outcome,
        // Worker overran the deadline (or panicked): do not block login, fall
        // through to password.
        Err(_) => AuthOutcome::Unavailable,
    }
}

/// Run an entry point body so that a panic never unwinds into libpam.
fn guarded(body: impl FnOnce() -> libc::c_int) -> libc::c_int {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(PAM_AUTHINFO_UNAVAIL)
}

/// A string PAM item (`PAM_SERVICE`, `PAM_TTY`), or `None` when unset or not UTF-8.
unsafe fn pam_item_string(
    pamh: *mut pam_sys::PamHandle,
    item: pam_sys::PamItemType,
) -> Option<String> {
    let mut ptr: *const libc::c_void = std::ptr::null();
    let ret = pam_sys::raw::pam_get_item(pamh, item as libc::c_int, &mut ptr);
    if ret != PAM_SUCCESS || ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr as *const libc::c_char)
        .to_str()
        .ok()
        .map(str::to_string)
        .filter(|s| !s.is_empty())
}

// `pamh` is an opaque handle that is only handed back to libpam, never dereferenced here.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn pam_sm_authenticate(
    pamh: *mut pam_sys::PamHandle,
    _flags: libc::c_int,
    argc: libc::c_int,
    argv: *const *const libc::c_char,
) -> libc::c_int {
    guarded(|| unsafe {
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

        let service = pam_item_string(pamh, pam_sys::PamItemType::SERVICE);
        let tty = pam_item_string(pamh, pam_sys::PamItemType::TTY);
        let target = AuthTarget {
            username,
            service: service.as_deref(),
            tty: tty.as_deref(),
        };
        let socket = socket_from_args(module_args(argc, argv));
        outcome_to_pam(authenticate_with_deadline(&socket, &target))
    })
}

#[no_mangle]
pub extern "C" fn pam_sm_setcred(
    _pamh: *mut pam_sys::PamHandle,
    _flags: libc::c_int,
    _argc: libc::c_int,
    _argv: *const *const libc::c_char,
) -> libc::c_int {
    guarded(|| PAM_SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::os::unix::net::UnixListener;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Instant;

    /// Bound for "fails at once" (the spec allows 100 ms of overhead).
    const FAST: Duration = Duration::from_millis(100);
    const TEST_READ_TIMEOUT: Duration = Duration::from_millis(300);
    const TEST_DEADLINE: Duration = Duration::from_millis(600);

    static NEXT: AtomicU32 = AtomicU32::new(0);

    fn temp_socket() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pam_doorman_test_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("d.sock")
    }

    /// Mock daemon: answers one connection with `reply` (nothing for `None`,
    /// holding the connection open), and returns the request line it received.
    fn mock_daemon(path: &Path, reply: Option<&'static str>) -> thread::JoinHandle<String> {
        let listener = UnixListener::bind(path).unwrap();
        thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            match reply {
                Some(r) => {
                    let mut stream = reader.into_inner();
                    writeln!(stream, "{r}").unwrap();
                }
                None => {
                    // Hold the connection open without answering.
                    let mut sink = Vec::new();
                    let _ = reader.into_inner().read_to_end(&mut sink);
                }
            }
            line
        })
    }

    fn run(path: &Path, service: Option<&str>) -> libc::c_int {
        let target = AuthTarget {
            username: "alice",
            service,
            tty: Some("/dev/pts/3"),
        };
        outcome_to_pam(authenticate_with_timeouts(
            path,
            &target,
            TEST_READ_TIMEOUT,
            TEST_DEADLINE,
        ))
    }

    #[test]
    fn return_codes_for_each_daemon_reply() {
        let cases: [(&str, &'static str, libc::c_int); 7] = [
            ("success", r#"{"status":"success"}"#, PAM_SUCCESS),
            (
                "failure",
                r#"{"status":"failure","reason":"x"}"#,
                PAM_AUTH_ERR,
            ),
            (
                "denied",
                r#"{"status":"denied","reason":"x"}"#,
                PAM_AUTHINFO_UNAVAIL,
            ),
            (
                "rate limited",
                r#"{"status":"rate_limited","retry_after_s":5}"#,
                PAM_AUTHINFO_UNAVAIL,
            ),
            (
                "unsupported protocol",
                r#"{"status":"unsupported_protocol","reason":"x","supported_version":1}"#,
                PAM_AUTHINFO_UNAVAIL,
            ),
            (
                "progress",
                r#"{"status":"progress","message":"m","current":1,"total":2}"#,
                PAM_AUTHINFO_UNAVAIL,
            ),
            ("garbage", "not json", PAM_AUTHINFO_UNAVAIL),
        ];
        for (name, reply, want) in cases {
            let path = temp_socket();
            let daemon = mock_daemon(&path, Some(reply));
            assert_eq!(run(&path, Some("sudo")), want, "{name}");
            daemon.join().unwrap();
        }
    }

    #[test]
    fn request_carries_service_and_tty() {
        let path = temp_socket();
        let daemon = mock_daemon(&path, Some(r#"{"status":"success"}"#));
        assert_eq!(run(&path, Some("sudo")), PAM_SUCCESS);
        let line = daemon.join().unwrap();
        let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(v["type"], "authenticate");
        assert_eq!(v["username"], "alice");
        assert_eq!(v["service"], "sudo");
        assert_eq!(v["tty"], "/dev/pts/3");
        assert_eq!(v["protocol_version"], 1);
    }

    #[test]
    fn missing_hints_stay_off_the_wire() {
        let path = temp_socket();
        let daemon = mock_daemon(&path, Some(r#"{"status":"success"}"#));
        let target = AuthTarget {
            username: "alice",
            service: None,
            tty: None,
        };
        authenticate_with_timeouts(&path, &target, TEST_READ_TIMEOUT, TEST_DEADLINE);
        let v: serde_json::Value = serde_json::from_str(daemon.join().unwrap().trim()).unwrap();
        assert!(v.get("service").is_none() && v.get("tty").is_none());
    }

    #[test]
    fn daemon_down_falls_through_at_once() {
        let path = temp_socket(); // nothing listens
        let started = Instant::now();
        assert_eq!(run(&path, Some("sudo")), PAM_AUTHINFO_UNAVAIL);
        assert!(started.elapsed() < FAST, "took {:?}", started.elapsed());
    }

    #[test]
    fn stale_socket_file_without_listener_falls_through_at_once() {
        let path = temp_socket();
        drop(UnixListener::bind(&path).unwrap()); // leaves the file, refuses connects
        let started = Instant::now();
        assert_eq!(run(&path, Some("sudo")), PAM_AUTHINFO_UNAVAIL);
        assert!(started.elapsed() < FAST, "took {:?}", started.elapsed());
    }

    #[test]
    fn silent_daemon_times_out_within_the_deadline() {
        let path = temp_socket();
        let daemon = mock_daemon(&path, None);
        let started = Instant::now();
        assert_eq!(run(&path, Some("sudo")), PAM_AUTHINFO_UNAVAIL);
        assert!(
            started.elapsed() < TEST_DEADLINE + FAST,
            "took {:?}",
            started.elapsed()
        );
        daemon.join().unwrap();
    }

    #[test]
    fn sudo_gets_the_longer_reply_budget() {
        assert_eq!(
            read_timeout(Some("login")),
            Duration::from_secs(AUTH_TIMEOUT_SECS)
        );
        assert_eq!(read_timeout(None), Duration::from_secs(AUTH_TIMEOUT_SECS));
        assert!(read_timeout(Some("sudo")) > Duration::from_millis(MAX_SUDO_TIMEOUT_MS));
        assert_eq!(read_timeout(Some("sudo-i")), read_timeout(Some("sudo")));
    }

    #[test]
    fn overlong_socket_path_is_an_error_not_a_panic() {
        let long = std::path::PathBuf::from("/".to_string() + &"a".repeat(300));
        assert!(connect_nonblocking(&long).is_err());
    }

    #[test]
    fn socket_argument_overrides_only_with_an_absolute_path() {
        let default = PathBuf::from(SOCKET_PATH);
        assert_eq!(socket_from_args([]), default);
        assert_eq!(socket_from_args(["debug", "use_first_pass"]), default);
        assert_eq!(
            socket_from_args(["socket=/tmp/t/d.sock"]),
            PathBuf::from("/tmp/t/d.sock")
        );
        assert_eq!(socket_from_args(["socket=relative.sock"]), default);
        assert_eq!(socket_from_args(["socket="]), default);
    }

    #[test]
    fn a_panic_never_crosses_the_ffi_boundary() {
        assert_eq!(guarded(|| panic!("boom")), PAM_AUTHINFO_UNAVAIL);
        assert_eq!(guarded(|| PAM_SUCCESS), PAM_SUCCESS);
    }
}
