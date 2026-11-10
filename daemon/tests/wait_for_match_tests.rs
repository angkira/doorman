//! Handler-level `WaitForMatch` tests: real unix socket, the real accept loop,
//! gate and wait logic, with a fake decision source (no camera, no models).

use async_trait::async_trait;
use doorman_shared::{IpcConfig, NoMatchReason, Request, Response};
use doormand::ipc_policy::{self, Gated, Gatekeeper, PeerCred};
use doormand::unlock_wait::{self, AttemptOutcome, DecisionSource};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

/// Longest any of these tests may wait for the daemon side to react.
const PROMPT: Duration = Duration::from_secs(5);
/// Pause between fake attempts, standing in for waiting on camera frames.
const FAKE_ATTEMPT_TIME: Duration = Duration::from_millis(5);

fn my_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

/// What the fake source answers per attempt.
#[derive(Clone, Copy)]
enum Mode {
    /// Never recognized.
    Never,
    /// Recognized from the n-th attempt on.
    FromAttempt(usize),
    /// An attempt never finishes (camera never delivers).
    Hang,
}

struct FakeSource {
    mode: Mode,
    attempts: Arc<AtomicUsize>,
    /// Set when an in-flight attempt is dropped.
    dropped: Arc<AtomicUsize>,
}

struct DropGuard(Arc<AtomicUsize>);
impl Drop for DropGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait]
impl DecisionSource for FakeSource {
    async fn attempt(&self, _username: &str) -> AttemptOutcome {
        let n = self.attempts.fetch_add(1, Ordering::SeqCst) + 1;
        let _guard = DropGuard(self.dropped.clone());
        tokio::time::sleep(FAKE_ATTEMPT_TIME).await;
        match self.mode {
            Mode::Never => AttemptOutcome::NotMatched,
            Mode::FromAttempt(k) if n >= k => AttemptOutcome::Matched { similarity: 0.8 },
            Mode::FromAttempt(_) => AttemptOutcome::NotMatched,
            Mode::Hang => std::future::pending().await,
        }
    }
}

struct Harness {
    sock: PathBuf,
    attempts: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
    _dir: tempfile::TempDir,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.server.abort();
    }
}

/// `alice` resolves to `alice_uid`; the peer is this test process.
fn start(cfg: IpcConfig, alice_uid: u32, mode: Mode) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let gate = Arc::new(Gatekeeper::with_parts(
        cfg.clone(),
        Box::new(move |name| (name == "alice").then_some(alice_uid)),
        Box::new(std::time::Instant::now),
    ));
    let attempts = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let source = Arc::new(FakeSource {
        mode,
        attempts: attempts.clone(),
        dropped: dropped.clone(),
    });
    let max_ms = cfg.wait_for_match_max_ms;
    let handler = Arc::new(move |_peer: PeerCred, req: Request| {
        let source = source.clone();
        async move {
            match req {
                Request::WaitForMatch {
                    username,
                    timeout_ms,
                } => {
                    unlock_wait::wait_for_match(
                        &*source,
                        None,
                        &username,
                        unlock_wait::effective_timeout(timeout_ms, max_ms),
                        None,
                        &unlock_wait::WaitTiming::new(),
                    )
                    .await
                }
                _ => Response::Failure {
                    reason: "unexpected".into(),
                },
            }
        }
    });
    let server = tokio::spawn(ipc_policy::serve(listener, gate, handler));
    Harness {
        sock,
        attempts,
        dropped,
        _dir: dir,
        server,
    }
}

fn wait_line(user: &str, timeout_ms: u64) -> String {
    format!(r#"{{"type":"wait_for_match","username":"{user}","timeout_ms":{timeout_ms}}}"#)
}

/// Send one request, return every response line until the server closes.
async fn exchange(sock: &Path, line: &str) -> Vec<Response> {
    let mut s = UnixStream::connect(sock).await.unwrap();
    s.write_all(format!("{line}\n").as_bytes()).await.unwrap();
    let mut out = String::new();
    tokio::time::timeout(PROMPT, s.read_to_string(&mut out))
        .await
        .expect("server must finish the exchange")
        .unwrap();
    out.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn self_trusting_cfg() -> IpcConfig {
    IpcConfig::default()
}

/// Gate a request as if it came from `uid` (the test process may be root, so
/// the kernel-reported credentials cannot be used to impersonate a user).
async fn gated_as(uid: u32, line: &str, attempts: &AtomicUsize) -> Response {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let gate = Gatekeeper::with_parts(
        IpcConfig::default(),
        Box::new(|name| (name == "alice").then_some(ALICE_UID)),
        Box::new(std::time::Instant::now),
    );
    let injected = PeerCred {
        uid,
        gid: uid,
        pid: Some(4242),
    };
    let server = async {
        let (stream, _) = listener.accept().await.unwrap();
        let mut reader = BufReader::new(stream);
        let response = match ipc_policy::read_gated_request(&mut reader, &injected, &gate)
            .await
            .unwrap()
        {
            // A request that passes the gate would start recognition.
            Gated::Allowed(_) => {
                attempts.fetch_add(1, Ordering::SeqCst);
                Response::Failure {
                    reason: "reached the handler".into(),
                }
            }
            Gated::Rejected(r) => r,
        };
        let mut stream = reader.into_inner();
        stream
            .write_all(format!("{}\n", serde_json::to_string(&response).unwrap()).as_bytes())
            .await
            .unwrap();
    };
    let client = async {
        let mut s = UnixStream::connect(&sock).await.unwrap();
        s.write_all(format!("{line}\n").as_bytes()).await.unwrap();
        let mut out = String::new();
        BufReader::new(s).read_line(&mut out).await.unwrap();
        serde_json::from_str::<Response>(out.trim()).unwrap()
    };
    tokio::join!(server, client).1
}

const ALICE_UID: u32 = 1000;
const BOB_UID: u32 = 1001;

#[tokio::test]
async fn other_uid_is_denied_and_never_reaches_recognition() {
    let attempts = AtomicUsize::new(0);
    let r = gated_as(BOB_UID, &wait_line("alice", 1000), &attempts).await;
    assert!(matches!(r, Response::Denied { .. }), "{r:?}");
    // Unknown user: indistinguishable from another user's account.
    let ghost = gated_as(BOB_UID, &wait_line("ghost", 1000), &attempts).await;
    assert_eq!(format!("{r:?}"), format!("{ghost:?}"));
    assert_eq!(attempts.load(Ordering::SeqCst), 0);

    // The user themself passes the gate.
    let r = gated_as(ALICE_UID, &wait_line("alice", 1000), &attempts).await;
    assert!(matches!(r, Response::Failure { .. }), "{r:?}");
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn trusted_uid_may_wait_for_another_user() {
    let cfg = IpcConfig {
        trusted_uids: vec![my_uid()],
        ..IpcConfig::default()
    };
    let h = start(cfg, my_uid() + 1, Mode::FromAttempt(1));
    let r = exchange(&h.sock, &wait_line("alice", 1000)).await;
    assert!(matches!(r.as_slice(), [Response::Matched { .. }]), "{r:?}");
}

#[tokio::test]
async fn own_user_gets_exactly_one_matched_and_no_further_attempts() {
    let h = start(self_trusting_cfg(), my_uid(), Mode::FromAttempt(3));
    let r = exchange(&h.sock, &wait_line("alice", 5000)).await;
    match r.as_slice() {
        [Response::Matched {
            username,
            similarity,
            ts_unix_ms,
        }] => {
            assert_eq!(username, "alice");
            assert!((similarity - 0.8).abs() < 1e-6);
            assert!(*ts_unix_ms > 0);
        }
        other => panic!("expected exactly one Matched, got {other:?}"),
    }
    // Let a stray background attempt show up if one existed.
    tokio::time::sleep(FAKE_ATTEMPT_TIME * 10).await;
    assert_eq!(h.attempts.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn unmatched_wait_ends_with_timeout() {
    let h = start(self_trusting_cfg(), my_uid(), Mode::Never);
    // Longer than two retry pauses, so several attempts fit.
    let wait_ms = (unlock_wait::RETRY_PAUSE * 3).as_millis() as u64;
    let r = exchange(&h.sock, &wait_line("alice", wait_ms)).await;
    assert!(
        matches!(
            r.as_slice(),
            [Response::NoMatch {
                reason: NoMatchReason::Timeout
            }]
        ),
        "{r:?}"
    );
    assert!(
        h.attempts.load(Ordering::SeqCst) > 1,
        "kept trying until the deadline"
    );
}

#[tokio::test]
async fn requested_timeout_is_capped_by_config() {
    let cfg = IpcConfig {
        wait_for_match_max_ms: 100,
        ..IpcConfig::default()
    };
    let h = start(cfg, my_uid(), Mode::Never);
    let started = std::time::Instant::now();
    let r = exchange(&h.sock, &wait_line("alice", u64::MAX)).await;
    assert!(matches!(
        r.as_slice(),
        [Response::NoMatch {
            reason: NoMatchReason::Timeout
        }]
    ));
    assert!(started.elapsed() < PROMPT);
}

#[tokio::test]
async fn closing_the_connection_cancels_the_wait() {
    let h = start(self_trusting_cfg(), my_uid(), Mode::Hang);
    let mut s = UnixStream::connect(&h.sock).await.unwrap();
    s.write_all(format!("{}\n", wait_line("alice", 60_000)).as_bytes())
        .await
        .unwrap();
    while h.attempts.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(FAKE_ATTEMPT_TIME).await;
    }
    drop(s);
    tokio::time::timeout(PROMPT, async {
        while h.dropped.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(FAKE_ATTEMPT_TIME).await;
        }
    })
    .await
    .expect("in-flight attempt must be dropped when the client disconnects");
}

#[tokio::test]
async fn each_call_counts_once_against_the_rate_limit() {
    let cfg = IpcConfig {
        max_auth_attempts_per_minute: 2,
        ..IpcConfig::default()
    };
    let h = start(cfg, my_uid(), Mode::Never);
    // Many internal attempts per call, but only one rate-limit hit each.
    let wait_ms = (unlock_wait::RETRY_PAUSE * 3).as_millis() as u64;
    for _ in 0..2 {
        let r = exchange(&h.sock, &wait_line("alice", wait_ms)).await;
        assert!(matches!(r.as_slice(), [Response::NoMatch { .. }]), "{r:?}");
    }
    assert!(h.attempts.load(Ordering::SeqCst) > 2);
    let r = exchange(&h.sock, &wait_line("alice", 60)).await;
    assert!(
        matches!(r.as_slice(), [Response::RateLimited { .. }]),
        "{r:?}"
    );
}

#[tokio::test]
async fn status_is_not_blocked_by_a_waiting_client() {
    let h = start(self_trusting_cfg(), my_uid(), Mode::Hang);
    let mut waiting = UnixStream::connect(&h.sock).await.unwrap();
    waiting
        .write_all(format!("{}\n", wait_line("alice", 60_000)).as_bytes())
        .await
        .unwrap();
    while h.attempts.load(Ordering::SeqCst) == 0 {
        tokio::time::sleep(FAKE_ATTEMPT_TIME).await;
    }
    let mut s = UnixStream::connect(&h.sock).await.unwrap();
    s.write_all(b"{\"type\":\"status\"}\n").await.unwrap();
    let mut line = String::new();
    tokio::time::timeout(PROMPT, BufReader::new(s).read_line(&mut line))
        .await
        .expect("Status must answer while a WaitForMatch is pending")
        .unwrap();
    assert!(line.contains("\"status\""), "{line}");
}
