//! `doorman unlock-watch`: event-driven face unlock for a screen locker.
//!
//! Holds a `WaitForMatch` open on the daemon socket. When the daemon reports a
//! match for the user, asks the locker process the lock script started to quit
//! (SIGTERM, then SIGKILL after a grace period). Never signals by name, never
//! signals a PID that no longer looks like the expected locker, and never
//! touches the locker when the daemon is unreachable or refuses the request.
//! Pure std plus `nix` for `kill`; links no ML or camera code.
//!
//! Exit codes are part of the interface (`docs/unlock-path.md`): see [`Exit`].

use doorman_shared::{NoMatchReason, Request, RequestEnvelope, Response};
use nix::errno::Errno;
use nix::sys::signal::{kill, Signal};
use nix::unistd::Pid;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Default length of one `WaitForMatch` call; the daemon caps it at
/// `[ipc] wait_for_match_max_ms`. The helper re-issues calls while the locker lives.
pub const DEFAULT_WAIT_TIMEOUT_MS: u64 = 30_000;
/// Default time the locker gets to exit after SIGTERM before SIGKILL.
pub const DEFAULT_TERM_GRACE_MS: u64 = 2_000;
/// How often locker liveness is checked while blocked on the daemon socket.
const LOCKER_POLL: Duration = Duration::from_millis(200);
/// Liveness check interval while waiting for the locker to exit after a signal.
const EXIT_POLL: Duration = Duration::from_millis(25);
/// Time the locker gets to disappear after SIGKILL.
const KILL_WAIT: Duration = Duration::from_millis(1_000);
/// Minimum time between `WaitForMatch` calls, so a daemon that answers
/// instantly can never turn the helper into a busy loop.
const MIN_CALL_INTERVAL: Duration = Duration::from_millis(500);
/// How long past the requested timeout a daemon may stay silent before the
/// call is abandoned and re-issued.
const RESPONSE_SLACK: Duration = Duration::from_secs(10);
/// Write timeout for the request line.
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);
/// `/proc/<pid>/comm` holds at most this many bytes of the process name.
const COMM_MAX_LEN: usize = 15;

/// Process exit codes of `doorman unlock-watch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// Matched; the locker exited after SIGTERM.
    Unlocked = 0,
    /// The locker exited by itself (password, or another unlock) before a match.
    LockerGone = 10,
    /// The daemon socket could not be reached; the locker was not touched.
    DaemonUnreachable = 11,
    /// The daemon denied the request; the locker was not touched.
    Denied = 12,
    /// The daemon rate-limited the request; the locker was not touched.
    RateLimited = 13,
    /// The daemon does not speak this protocol version; locker not touched.
    UnsupportedProtocol = 14,
    /// The PID is invalid or no longer the expected process; nothing was signalled.
    LockerMismatch = 15,
    /// The daemon answered something unexpected; the locker was not touched.
    ProtocolError = 16,
    /// Matched; SIGTERM was ignored and SIGKILL removed the locker.
    Killed = 17,
    /// Matched, but the locker could not be signalled or survived SIGKILL.
    SignalFailed = 18,
    /// The user has no enrollment; waiting cannot help.
    NotEnrolled = 19,
}

impl Exit {
    pub fn code(self) -> i32 {
        self as i32
    }
}

pub struct WatchArgs {
    pub socket: PathBuf,
    pub user: String,
    pub locker_pid: i32,
    /// Expected `/proc/<pid>/comm`; guards against PID reuse. `None` skips the check.
    pub locker_name: Option<String>,
    pub timeout_ms: u64,
    pub term_grace: Duration,
}

macro_rules! log {
    ($($arg:tt)*) => {
        eprintln!(
            "{} unlock-watch: {}",
            chrono::Local::now().format("%Y-%m-%dT%H:%M:%S%.3f%:z"),
            format_args!($($arg)*)
        )
    };
}

/// Whether `pid` is a live process (a zombie counts as gone: it can no longer lock anything).
pub fn process_alive(pid: i32) -> bool {
    match kill(Pid::from_raw(pid), None) {
        Ok(()) | Err(Errno::EPERM) => {}
        Err(_) => return false,
    }
    !matches!(proc_state(pid), Some('Z' | 'X'))
}

/// State letter from `/proc/<pid>/stat`, `None` when unreadable.
fn proc_state(pid: i32) -> Option<char> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // "pid (comm) S ...": comm may itself contain ')', so cut at the last one.
    stat[stat.rfind(')')? + 1..].trim_start().chars().next()
}

/// `/proc/<pid>/comm`, `None` when unreadable.
fn proc_comm(pid: i32) -> Option<String> {
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
    Some(comm.trim_end_matches('\n').to_string())
}

/// Kernel comm names are truncated to 15 bytes; compare accordingly.
pub fn comm_matches(actual: &str, expected: &str) -> bool {
    let cut = |s: &str| s.bytes().take(COMM_MAX_LEN).collect::<Vec<u8>>();
    cut(actual) == cut(expected)
}

/// `Err(reason)` when `pid` must not be signalled.
fn check_identity(pid: i32, expected: Option<&str>) -> Result<(), String> {
    let Some(expected) = expected else {
        return Ok(());
    };
    match proc_comm(pid) {
        Some(actual) if comm_matches(&actual, expected) => Ok(()),
        Some(actual) => Err(format!(
            "pid {pid} is '{actual}', expected '{expected}' (pid reused?)"
        )),
        None => Err(format!("cannot read the name of pid {pid}")),
    }
}

enum Call {
    Response(Response),
    /// Daemon closed or stalled without a usable answer; try again.
    NoAnswer,
    LockerGone,
}

/// One `WaitForMatch` round-trip. `Err` means the daemon could not be reached.
fn wait_for_match_once(args: &WatchArgs) -> Result<Call, String> {
    let mut stream = UnixStream::connect(&args.socket)
        .map_err(|e| format!("cannot connect to {} ({e})", args.socket.display()))?;
    stream
        .set_write_timeout(Some(WRITE_TIMEOUT))
        .and_then(|()| stream.set_read_timeout(Some(LOCKER_POLL)))
        .map_err(|e| format!("socket setup failed ({e})"))?;
    let request = RequestEnvelope::new(Request::WaitForMatch {
        username: args.user.clone(),
        timeout_ms: args.timeout_ms,
    });
    let line = serde_json::to_string(&request).map_err(|e| format!("encode failed ({e})"))?;
    writeln!(stream, "{line}")
        .and_then(|()| stream.flush())
        .map_err(|e| format!("cannot send request ({e})"))?;

    let give_up = Instant::now() + Duration::from_millis(args.timeout_ms) + RESPONSE_SLACK;
    let mut reader = BufReader::new(stream);
    let mut buf = Vec::new();
    loop {
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => return Ok(Call::NoAnswer),
            Ok(_) if buf.ends_with(b"\n") => break,
            // Partial line then EOF.
            Ok(_) => return Ok(Call::NoAnswer),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if !process_alive(args.locker_pid) {
                    return Ok(Call::LockerGone);
                }
                if Instant::now() >= give_up {
                    log!("daemon silent past the call timeout; re-issuing");
                    return Ok(Call::NoAnswer);
                }
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => {
                log!("read from daemon failed ({e})");
                return Ok(Call::NoAnswer);
            }
        }
    }
    match serde_json::from_slice::<Response>(&buf) {
        Ok(r) => Ok(Call::Response(r)),
        Err(e) => {
            log!("unparsable daemon response ({e})");
            Ok(Call::Response(Response::Failure {
                reason: "unparsable response".into(),
            }))
        }
    }
}

/// Poll until `pid` is gone or `limit` passes. `true` when it is gone.
fn wait_gone(pid: i32, limit: Duration) -> bool {
    let deadline = Instant::now() + limit;
    loop {
        if !process_alive(pid) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(EXIT_POLL);
    }
}

/// Ask the matched locker to quit: SIGTERM, then SIGKILL after the grace period.
fn unlock(args: &WatchArgs) -> Exit {
    let pid = args.locker_pid;
    if let Err(why) = check_identity(pid, args.locker_name.as_deref()) {
        log!("not signalling: {why}");
        return Exit::LockerMismatch;
    }
    match kill(Pid::from_raw(pid), Signal::SIGTERM) {
        Ok(()) => log!("sent SIGTERM to locker pid {pid}"),
        Err(Errno::ESRCH) => {
            log!("locker pid {pid} already gone");
            return Exit::Unlocked;
        }
        Err(e) => {
            log!("SIGTERM to pid {pid} failed ({e})");
            return Exit::SignalFailed;
        }
    }
    if wait_gone(pid, args.term_grace) {
        log!("locker exited after SIGTERM");
        return Exit::Unlocked;
    }
    // The locker ignored SIGTERM. A hard kill can leave the session locked
    // without a client, so this is a warning-level last resort.
    if let Err(why) = check_identity(pid, args.locker_name.as_deref()) {
        log!("not escalating: {why}");
        return Exit::LockerMismatch;
    }
    log!(
        "WARNING: locker pid {pid} ignored SIGTERM for {} ms; sending SIGKILL",
        args.term_grace.as_millis()
    );
    match kill(Pid::from_raw(pid), Signal::SIGKILL) {
        Ok(()) | Err(Errno::ESRCH) => {}
        Err(e) => {
            log!("SIGKILL to pid {pid} failed ({e})");
            return Exit::SignalFailed;
        }
    }
    if wait_gone(pid, KILL_WAIT) {
        log!("locker exited after SIGKILL");
        Exit::Killed
    } else {
        log!("locker pid {pid} survived SIGKILL");
        Exit::SignalFailed
    }
}

/// Sleep out the rest of `MIN_CALL_INTERVAL`, returning early if the locker dies.
fn throttle(started: Instant, pid: i32) {
    while started.elapsed() < MIN_CALL_INTERVAL && process_alive(pid) {
        std::thread::sleep(EXIT_POLL);
    }
}

pub fn run(args: &WatchArgs) -> Exit {
    let pid = args.locker_pid;
    // pid <= 1 would address a process group or init; never signal ourselves.
    if pid <= 1 || pid == std::process::id() as i32 {
        log!("refusing locker pid {pid}");
        return Exit::LockerMismatch;
    }
    if !process_alive(pid) {
        log!("locker pid {pid} is not running");
        return Exit::LockerGone;
    }
    if let Err(why) = check_identity(pid, args.locker_name.as_deref()) {
        log!("{why}");
        return Exit::LockerMismatch;
    }
    log!(
        "watching locker pid {pid} ({}) for user {}, socket {}",
        args.locker_name.as_deref().unwrap_or("name unchecked"),
        args.user,
        args.socket.display()
    );

    loop {
        let started = Instant::now();
        match wait_for_match_once(args) {
            Err(why) => {
                log!("daemon unreachable: {why}; leaving the locker alone");
                return Exit::DaemonUnreachable;
            }
            Ok(Call::LockerGone) => {
                log!("locker pid {pid} exited before any match");
                return Exit::LockerGone;
            }
            Ok(Call::NoAnswer) => log!("no answer from daemon; retrying"),
            Ok(Call::Response(response)) => match response {
                Response::Matched {
                    username,
                    similarity,
                    ts_unix_ms,
                } if username == args.user => {
                    log!("matched user {username} similarity {similarity:.3} (daemon ts {ts_unix_ms} ms)");
                    return unlock(args);
                }
                Response::Matched { username, .. } => {
                    log!("ignoring match for a different user ({username})");
                }
                Response::NoMatch {
                    reason: NoMatchReason::Timeout | NoMatchReason::NotArmed,
                } => {}
                Response::NoMatch {
                    reason: NoMatchReason::NotEnrolled,
                } => {
                    log!("user {} is not enrolled; nothing to wait for", args.user);
                    return Exit::NotEnrolled;
                }
                Response::Denied { reason } => {
                    log!("daemon denied the request: {reason}");
                    return Exit::Denied;
                }
                Response::RateLimited { retry_after_s } => {
                    log!("daemon rate limited the request (retry in {retry_after_s} s)");
                    return Exit::RateLimited;
                }
                Response::UnsupportedProtocol {
                    reason,
                    supported_version,
                } => {
                    log!("daemon speaks protocol {supported_version}: {reason}");
                    return Exit::UnsupportedProtocol;
                }
                other => {
                    log!("unexpected daemon response: {other:?}");
                    return Exit::ProtocolError;
                }
            },
        }
        throttle(started, pid);
        if !process_alive(pid) {
            log!("locker pid {pid} exited before any match");
            return Exit::LockerGone;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comm_is_compared_like_the_kernel_stores_it() {
        assert!(comm_matches("gtklock", "gtklock"));
        assert!(!comm_matches("gtklock", "swaylock"));
        assert!(comm_matches("a-very-long-locker-name", "a-very-long-lock"));
        assert!(!comm_matches("gtklock2", "gtklock"));
    }

    #[test]
    fn current_process_is_alive_and_a_bogus_pid_is_not() {
        assert!(process_alive(std::process::id() as i32));
        assert!(!process_alive(i32::MAX));
    }

    #[test]
    fn refuses_dangerous_pids_without_touching_anything() {
        for pid in [-1, 0, 1, std::process::id() as i32] {
            let args = WatchArgs {
                socket: PathBuf::from("/nonexistent"),
                user: "u".into(),
                locker_pid: pid,
                locker_name: None,
                timeout_ms: 1,
                term_grace: Duration::from_millis(1),
            };
            assert_eq!(run(&args), Exit::LockerMismatch, "pid {pid}");
        }
    }
}
