//! `doorman unlock-watch` against a fake daemon (JSON-line protocol on a unix
//! socket) and a fake locker (a shell that traps or ignores SIGTERM).
//! No Wayland, camera, models or daemon needed.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DOORMAN: &str = env!("CARGO_BIN_EXE_doorman");
/// Name the kernel reports for the fake locker (`sh`).
const LOCKER_COMM: &str = "sh";
const TERM_GRACE_MS: &str = "300";
/// Longest any scenario may take.
const DEADLINE: Duration = Duration::from_secs(20);

const EXIT_UNLOCKED: i32 = 0;
const EXIT_LOCKER_GONE: i32 = 10;
const EXIT_DAEMON_UNREACHABLE: i32 = 11;
const EXIT_DENIED: i32 = 12;
const EXIT_RATE_LIMITED: i32 = 13;
const EXIT_LOCKER_MISMATCH: i32 = 15;
const EXIT_KILLED: i32 = 17;
const EXIT_NOT_ENROLLED: i32 = 19;

/// Fake locker: exits on SIGTERM, or ignores it.
struct Locker(Child);

impl Locker {
    fn spawn(ignore_term: bool) -> Self {
        let trap = if ignore_term {
            r#"trap "" TERM"#
        } else {
            r#"trap "exit 0" TERM"#
        };
        let script = format!("{trap}; while :; do sleep 0.05; done");
        let child = Command::new("sh")
            .args(["-c", &script])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        // Let the shell install its trap before anything signals it.
        std::thread::sleep(Duration::from_millis(200));
        Locker(child)
    }

    fn pid(&self) -> u32 {
        self.0.id()
    }

    /// Reaps the child, so a zombie never counts as alive.
    fn exited(&mut self) -> bool {
        self.0.try_wait().unwrap().is_some()
    }

    fn wait_exit(&mut self, limit: Duration) -> bool {
        let end = Instant::now() + limit;
        while Instant::now() < end {
            if self.exited() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }
}

impl Drop for Locker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// What the fake daemon does with one connection.
#[derive(Clone)]
enum Step {
    /// Send this JSON line, then close.
    Reply(String),
    /// Read the request and keep the connection open until the client leaves.
    Hang,
}

struct FakeDaemon {
    socket: PathBuf,
    requests: Arc<Mutex<Vec<String>>>,
    _dir: tempfile::TempDir,
}

impl FakeDaemon {
    /// Serves `steps` in order, one per connection; later connections hang.
    fn start(steps: Vec<Step>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("d.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        std::thread::spawn(move || {
            let mut steps = steps.into_iter();
            for conn in listener.incoming() {
                let Ok(conn) = conn else { return };
                let step = steps.next().unwrap_or(Step::Hang);
                let seen = seen.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(conn);
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    seen.lock().unwrap().push(line.trim().to_string());
                    match step {
                        Step::Reply(json) => {
                            let mut conn = reader.into_inner();
                            let _ = writeln!(conn, "{json}");
                        }
                        Step::Hang => {
                            // Returns when the client closes the connection.
                            let mut sink = Vec::new();
                            let _ = reader.read_to_end(&mut sink);
                        }
                    }
                });
            }
        });
        FakeDaemon {
            socket,
            requests,
            _dir: dir,
        }
    }
}

fn matched(user: &str) -> Step {
    Step::Reply(format!(
        r#"{{"status":"matched","username":"{user}","similarity":0.81,"ts_unix_ms":1}}"#
    ))
}

fn no_match(reason: &str) -> Step {
    Step::Reply(format!(r#"{{"status":"no_match","reason":"{reason}"}}"#))
}

fn run_watch(socket: &std::path::Path, pid: u32, extra: &[&str]) -> Output {
    let child = Command::new(DOORMAN)
        .arg("--socket")
        .arg(socket)
        .args(["unlock-watch", "--user", "alice", "--locker-pid"])
        .arg(pid.to_string())
        .args(["--term-grace-ms", TERM_GRACE_MS])
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    wait_with_deadline(child)
}

fn wait_with_deadline(mut child: Child) -> Output {
    let end = Instant::now() + DEADLINE;
    while Instant::now() < end {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    panic!("doorman unlock-watch did not finish in {DEADLINE:?}");
}

fn code(out: &Output) -> i32 {
    out.status.code().expect("exited normally")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn match_sends_sigterm_and_locker_exits() {
    let mut locker = Locker::spawn(false);
    let daemon = FakeDaemon::start(vec![matched("alice")]);
    let out = run_watch(
        &daemon.socket,
        locker.pid(),
        &["--locker-name", LOCKER_COMM],
    );
    assert_eq!(code(&out), EXIT_UNLOCKED, "{}", stderr(&out));
    assert!(
        locker.wait_exit(Duration::from_secs(2)),
        "locker still alive"
    );
    let log = stderr(&out);
    assert!(log.contains("sent SIGTERM"), "{log}");
    assert!(!log.contains("SIGKILL"), "{log}");
    // Timestamped, one line per event.
    assert!(log.lines().all(|l| l.contains("unlock-watch:")), "{log}");

    let requests = daemon.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let req: serde_json::Value = serde_json::from_str(&requests[0]).unwrap();
    assert_eq!(req["type"], "wait_for_match");
    assert_eq!(req["username"], "alice");
    assert_eq!(req["protocol_version"], 1);
}

#[test]
fn locker_ignoring_sigterm_is_killed_with_a_warning() {
    let mut locker = Locker::spawn(true);
    let daemon = FakeDaemon::start(vec![matched("alice")]);
    let out = run_watch(
        &daemon.socket,
        locker.pid(),
        &["--locker-name", LOCKER_COMM],
    );
    assert_eq!(code(&out), EXIT_KILLED, "{}", stderr(&out));
    assert!(
        locker.wait_exit(Duration::from_secs(2)),
        "locker still alive"
    );
    let log = stderr(&out);
    assert!(log.contains("WARNING"), "{log}");
    assert!(log.contains("SIGKILL"), "{log}");
}

#[test]
fn locker_exiting_first_ends_the_helper() {
    let mut locker = Locker::spawn(false);
    let daemon = FakeDaemon::start(vec![Step::Hang]);
    let child = Command::new(DOORMAN)
        .arg("--socket")
        .arg(&daemon.socket)
        .args(["unlock-watch", "--user", "alice", "--locker-pid"])
        .arg(locker.pid().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // The user types the password: the locker goes away on its own.
    std::thread::sleep(Duration::from_millis(500));
    let _ = Command::new("kill")
        .args(["-TERM", &locker.pid().to_string()])
        .status();
    assert!(locker.wait_exit(Duration::from_secs(2)));
    let out = wait_with_deadline(child);
    assert_eq!(code(&out), EXIT_LOCKER_GONE, "{}", stderr(&out));
}

#[test]
fn unreachable_daemon_leaves_the_locker_alone() {
    let mut locker = Locker::spawn(false);
    let out = run_watch(
        std::path::Path::new("/nonexistent/doorman.sock"),
        locker.pid(),
        &[],
    );
    assert_eq!(code(&out), EXIT_DAEMON_UNREACHABLE, "{}", stderr(&out));
    assert!(!locker.exited(), "locker must not be touched");
}

#[test]
fn refusals_leave_the_locker_alone() {
    for (reply, want) in [
        (r#"{"status":"denied","reason":"no"}"#, EXIT_DENIED),
        (
            r#"{"status":"rate_limited","retry_after_s":30}"#,
            EXIT_RATE_LIMITED,
        ),
    ] {
        let mut locker = Locker::spawn(false);
        let daemon = FakeDaemon::start(vec![Step::Reply(reply.into())]);
        let out = run_watch(&daemon.socket, locker.pid(), &[]);
        assert_eq!(code(&out), want, "{reply}: {}", stderr(&out));
        assert!(!locker.exited(), "locker must not be touched for {reply}");
    }
}

#[test]
fn not_enrolled_ends_without_touching_the_locker() {
    let mut locker = Locker::spawn(false);
    let daemon = FakeDaemon::start(vec![no_match("not_enrolled")]);
    let out = run_watch(&daemon.socket, locker.pid(), &[]);
    assert_eq!(code(&out), EXIT_NOT_ENROLLED, "{}", stderr(&out));
    assert!(!locker.exited());
}

#[test]
fn match_for_another_user_is_ignored_and_timeouts_are_reissued() {
    let mut locker = Locker::spawn(false);
    let daemon = FakeDaemon::start(vec![
        no_match("timeout"),
        no_match("not_armed"),
        matched("mallory"),
        matched("alice"),
    ]);
    let out = run_watch(
        &daemon.socket,
        locker.pid(),
        &["--locker-name", LOCKER_COMM],
    );
    assert_eq!(code(&out), EXIT_UNLOCKED, "{}", stderr(&out));
    assert!(locker.wait_exit(Duration::from_secs(2)));
    assert_eq!(daemon.requests.lock().unwrap().len(), 4);
    assert!(stderr(&out).contains("different user"), "{}", stderr(&out));
}

#[test]
fn pid_that_is_not_the_expected_locker_is_never_signalled() {
    let mut locker = Locker::spawn(false);
    let daemon = FakeDaemon::start(vec![matched("alice")]);
    let out = run_watch(&daemon.socket, locker.pid(), &["--locker-name", "gtklock"]);
    assert_eq!(code(&out), EXIT_LOCKER_MISMATCH, "{}", stderr(&out));
    assert!(!locker.exited(), "a reused pid must not be signalled");
    assert!(
        daemon.requests.lock().unwrap().is_empty(),
        "must refuse before asking the daemon"
    );
}
