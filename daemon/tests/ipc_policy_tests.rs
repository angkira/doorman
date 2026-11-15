//! Handler-level IPC security tests: a real unix socket and the real gate
//! (`read_gated_request`), with an injected `PeerCred` and a stub dispatcher so
//! no camera, models or root are required.

use doorman_shared::{IpcConfig, Response};
use doormand::ipc_policy::{self, Gated, Gatekeeper, PeerCred};
use std::os::unix::fs::PermissionsExt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

const ALICE: u32 = 1000;
const BOB: u32 = 1001;

fn resolve(name: &str) -> Option<u32> {
    match name {
        "alice" => Some(ALICE),
        "bob" => Some(BOB),
        _ => None,
    }
}

fn peer(uid: u32) -> PeerCred {
    PeerCred {
        uid,
        gid: uid,
        pid: Some(4242),
    }
}

/// Serve one connection: gate it, answer Success for allowed requests.
async fn serve_one(listener: &UnixListener, gate: &Gatekeeper, injected: PeerCred) {
    let (stream, _) = listener.accept().await.unwrap();
    let mut reader = BufReader::new(stream);
    let response = match ipc_policy::read_gated_request(&mut reader, &injected, gate)
        .await
        .unwrap()
    {
        Gated::Allowed(_) => Response::Success {
            message: Some("stub".into()),
            data: None,
        },
        Gated::Rejected(r) => r,
    };
    let mut stream = reader.into_inner();
    stream
        .write_all(format!("{}\n", serde_json::to_string(&response).unwrap()).as_bytes())
        .await
        .unwrap();
}

async fn roundtrip(
    gate: &Gatekeeper,
    injected: PeerCred,
    sock: &std::path::Path,
    listener: &UnixListener,
    line: &str,
) -> Response {
    let client = async {
        let mut s = UnixStream::connect(sock).await.unwrap();
        s.write_all(format!("{line}\n").as_bytes()).await.unwrap();
        let mut out = String::new();
        BufReader::new(s).read_line(&mut out).await.unwrap();
        serde_json::from_str::<Response>(out.trim()).unwrap()
    };
    let (_, resp) = tokio::join!(serve_one(listener, gate, injected), client);
    resp
}

fn gate_with(cfg: IpcConfig, clock: Arc<Mutex<Instant>>) -> Gatekeeper {
    Gatekeeper::with_parts(
        cfg,
        Box::new(resolve),
        Box::new(move || *clock.lock().unwrap()),
    )
}

const AUTH_ALICE: &str = r#"{"type":"authenticate","username":"alice"}"#;

#[tokio::test]
async fn same_uid_allowed_other_uid_denied() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let gate = gate_with(IpcConfig::default(), Arc::new(Mutex::new(Instant::now())));

    let r = roundtrip(&gate, peer(ALICE), &sock, &listener, AUTH_ALICE).await;
    assert!(matches!(r, Response::Success { .. }), "{r:?}");

    let r = roundtrip(&gate, peer(0), &sock, &listener, AUTH_ALICE).await;
    assert!(matches!(r, Response::Success { .. }), "root: {r:?}");

    let r = roundtrip(&gate, peer(BOB), &sock, &listener, AUTH_ALICE).await;
    assert!(matches!(r, Response::Denied { .. }), "{r:?}");

    // Unknown user is indistinguishable from another user's account.
    let ghost = r#"{"type":"authenticate","username":"ghost"}"#;
    let r2 = roundtrip(&gate, peer(BOB), &sock, &listener, ghost).await;
    assert_eq!(format!("{r:?}"), format!("{r2:?}"));
}

#[tokio::test]
async fn admin_ops_denied_for_non_root_allowed_for_root() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let gate = gate_with(IpcConfig::default(), Arc::new(Mutex::new(Instant::now())));

    for line in [
        r#"{"type":"enroll","username":"alice"}"#,
        r#"{"type":"remove_user","username":"alice"}"#,
        r#"{"type":"list_users"}"#,
        r#"{"type":"shutdown"}"#,
    ] {
        let r = roundtrip(&gate, peer(ALICE), &sock, &listener, line).await;
        assert!(matches!(r, Response::Denied { .. }), "{line}: {r:?}");
        let r = roundtrip(&gate, peer(0), &sock, &listener, line).await;
        assert!(matches!(r, Response::Success { .. }), "{line}: {r:?}");
    }
}

#[tokio::test]
async fn rate_limit_kicks_in_then_recovers() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let cfg = IpcConfig {
        max_auth_attempts_per_minute: 2,
        ..IpcConfig::default()
    };
    let t0 = Instant::now();
    let clock = Arc::new(Mutex::new(t0));
    let gate = gate_with(cfg, clock.clone());

    for _ in 0..2 {
        let r = roundtrip(&gate, peer(ALICE), &sock, &listener, AUTH_ALICE).await;
        assert!(matches!(r, Response::Success { .. }), "{r:?}");
    }
    let r = roundtrip(&gate, peer(ALICE), &sock, &listener, AUTH_ALICE).await;
    assert!(
        matches!(r, Response::RateLimited { retry_after_s } if retry_after_s > 0),
        "{r:?}"
    );
    // Root is limited too.
    for _ in 0..2 {
        roundtrip(&gate, peer(0), &sock, &listener, AUTH_ALICE).await;
    }
    let r = roundtrip(&gate, peer(0), &sock, &listener, AUTH_ALICE).await;
    assert!(matches!(r, Response::RateLimited { .. }), "root: {r:?}");

    *clock.lock().unwrap() = t0 + Duration::from_secs(61);
    let r = roundtrip(&gate, peer(ALICE), &sock, &listener, AUTH_ALICE).await;
    assert!(matches!(r, Response::Success { .. }), "{r:?}");
}

#[tokio::test]
async fn protocol_version_handling() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let gate = gate_with(IpcConfig::default(), Arc::new(Mutex::new(Instant::now())));

    // Absent (legacy) and 1 are accepted; 2 gets an explicit error.
    let ok1 = r#"{"type":"status"}"#;
    let ok2 = r#"{"type":"status","protocol_version":1}"#;
    let bad = r#"{"type":"status","protocol_version":2}"#;
    for l in [ok1, ok2] {
        let r = roundtrip(&gate, peer(ALICE), &sock, &listener, l).await;
        assert!(matches!(r, Response::Success { .. }), "{l}: {r:?}");
    }
    let r = roundtrip(&gate, peer(ALICE), &sock, &listener, bad).await;
    assert!(
        matches!(
            r,
            Response::UnsupportedProtocol {
                supported_version: 1,
                ..
            }
        ),
        "{r:?}"
    );
}

#[tokio::test]
async fn real_peer_cred_matches_current_process() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let (client, accepted) = tokio::join!(UnixStream::connect(&sock), listener.accept());
    let _client = client.unwrap();
    let got = ipc_policy::peer_from_stream(&accepted.unwrap().0).unwrap();
    assert_eq!(got.uid, nix::unistd::getuid().as_raw());
    assert_eq!(got.pid, Some(std::process::id() as i32));
}

fn mode_of(p: &std::path::Path) -> u32 {
    std::fs::metadata(p).unwrap().permissions().mode() & 0o7777
}

#[tokio::test]
async fn created_socket_modes() {
    let dir = tempfile::tempdir().unwrap();

    let cmd = dir.path().join("cmd.sock");
    let _l = ipc_policy::bind_socket(cmd.to_str().unwrap(), ipc_policy::COMMAND_SOCKET_MODE, None)
        .unwrap();
    assert_eq!(mode_of(&cmd), 0o666);

    let frames = dir.path().join("frames.sock");
    let _l =
        ipc_policy::bind_stream_socket(frames.to_str().unwrap(), &IpcConfig::default()).unwrap();
    assert_eq!(mode_of(&frames), 0o600);

    // With a group configured: 0660 and group-owned (use our own primary group
    // so chown works without root).
    let gid = nix::unistd::getgid();
    let group = nix::unistd::Group::from_gid(gid).unwrap().unwrap().name;
    let dbg = dir.path().join("debug.sock");
    let cfg = IpcConfig {
        stream_socket_group: Some(group),
        ..IpcConfig::default()
    };
    let _l = ipc_policy::bind_stream_socket(dbg.to_str().unwrap(), &cfg).unwrap();
    assert_eq!(mode_of(&dbg), 0o660);
    use std::os::unix::fs::MetadataExt;
    assert_eq!(std::fs::metadata(&dbg).unwrap().gid(), gid.as_raw());

    // Unknown group fails startup explicitly.
    let cfg = IpcConfig {
        stream_socket_group: Some("no-such-group-doorman".into()),
        ..IpcConfig::default()
    };
    let bad = dir.path().join("bad.sock");
    assert!(ipc_policy::bind_stream_socket(bad.to_str().unwrap(), &cfg).is_err());
}

#[test]
fn stream_accept_policy() {
    let def = IpcConfig::default();
    assert!(ipc_policy::stream_peer_allowed(&peer(0), &def));
    assert!(!ipc_policy::stream_peer_allowed(&peer(ALICE), &def));
    let admin = IpcConfig {
        admin_uids: vec![ALICE],
        ..IpcConfig::default()
    };
    assert!(ipc_policy::stream_peer_allowed(&peer(ALICE), &admin));
}

#[tokio::test]
async fn oversized_request_line_without_newline_is_rejected_with_bounded_read() {
    let gate = gate_with(IpcConfig::default(), Arc::new(Mutex::new(Instant::now())));
    // Four times the cap, no newline anywhere.
    let flood = vec![b'a'; 4 * ipc_policy::MAX_REQUEST_LINE_BYTES as usize];
    let mut reader = BufReader::new(&flood[..]);
    let err = match ipc_policy::read_gated_request(&mut reader, &peer(ALICE), &gate).await {
        Ok(_) => panic!("an oversized line must not be accepted"),
        Err(e) => e,
    };
    assert!(format!("{err:#}").contains("exceeds"), "{err:#}");
    // Never pulled much more than the cap off the wire (cap + one read buffer).
    let consumed = flood.len() - reader.get_ref().len();
    let slack = 64 * 1024;
    assert!(
        consumed as u64 <= ipc_policy::MAX_REQUEST_LINE_BYTES + 1 + slack,
        "consumed {consumed} bytes"
    );
}

#[tokio::test]
async fn request_line_at_the_cap_is_still_parsed() {
    let gate = gate_with(IpcConfig::default(), Arc::new(Mutex::new(Instant::now())));
    // A valid request padded with whitespace to exactly the cap (incl. newline).
    let body = r#"{"type":"status"}"#;
    let pad = ipc_policy::MAX_REQUEST_LINE_BYTES as usize - body.len() - 1;
    let line = format!("{body}{}\n", " ".repeat(pad));
    assert_eq!(line.len() as u64, ipc_policy::MAX_REQUEST_LINE_BYTES);
    let mut reader = BufReader::new(line.as_bytes());
    let got = ipc_policy::read_gated_request(&mut reader, &peer(ALICE), &gate)
        .await
        .unwrap();
    assert!(matches!(got, Gated::Allowed(_)));
}

/// Round 9b-2a: over a real socket and the real gate, calibration start and end
/// are refused to every non-root peer with a reason that names root, a
/// configured admin uid included; status is open; root is let through.
#[tokio::test]
async fn calibration_requests_are_root_only_on_the_wire() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let cfg = IpcConfig {
        admin_uids: vec![ALICE],
        trusted_uids: vec![BOB],
        ..IpcConfig::default()
    };
    let gate = gate_with(cfg, Arc::new(Mutex::new(Instant::now())));
    let start = r#"{"type":"calibrate","ttl_secs":600,"protocol_version":1}"#;
    let end = r#"{"type":"calibrate_end","protocol_version":1}"#;
    let status = r#"{"type":"calibrate_status","protocol_version":1}"#;
    for uid in [ALICE, BOB, 65534] {
        for line in [start, end] {
            match roundtrip(&gate, peer(uid), &sock, &listener, line).await {
                Response::Denied { reason } => {
                    assert!(reason.contains("requires root"), "uid {uid}: {reason}");
                    assert!(
                        reason.contains("sudo"),
                        "tells the user what to do: {reason}"
                    );
                }
                other => panic!("uid {uid} {line}: {other:?}"),
            }
        }
        assert!(matches!(
            roundtrip(&gate, peer(uid), &sock, &listener, status).await,
            Response::Success { .. }
        ));
    }
    for line in [start, end, status] {
        assert!(matches!(
            roundtrip(&gate, peer(0), &sock, &listener, line).await,
            Response::Success { .. }
        ));
    }
}
