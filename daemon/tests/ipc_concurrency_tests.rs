//! The command socket serves connections concurrently: a long request must not
//! block others. Real socket, real SO_PEERCRED, stub handler (no camera/models).

use doorman_shared::{IpcConfig, Request, Response};
use doormand::ipc_policy::{self, Gatekeeper, PeerCred};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::Notify;

/// Longest a request that must answer immediately may take.
const PROMPT: Duration = Duration::from_secs(2);

fn my_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

async fn ask(sock: &std::path::Path, line: &str) -> Response {
    let mut s = UnixStream::connect(sock).await.unwrap();
    s.write_all(format!("{line}\n").as_bytes()).await.unwrap();
    let mut out = String::new();
    BufReader::new(s).read_line(&mut out).await.unwrap();
    serde_json::from_str(out.trim()).unwrap()
}

#[tokio::test]
async fn status_answers_while_authenticate_is_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let cfg = IpcConfig {
        trusted_uids: vec![my_uid()],
        ..IpcConfig::default()
    };
    let gate = Arc::new(Gatekeeper::new(cfg));

    let release = Arc::new(Notify::new());
    let started = Arc::new(Notify::new());
    let (r, st) = (release.clone(), started.clone());
    let handler = Arc::new(move |_peer: PeerCred, req: Request| {
        let (r, st) = (r.clone(), st.clone());
        async move {
            if matches!(req, Request::Authenticate { .. }) {
                st.notify_one();
                r.notified().await;
                return Response::Success {
                    message: Some("auth done".into()),
                    data: None,
                };
            }
            Response::Success {
                message: Some("status".into()),
                data: None,
            }
        }
    });
    let server = tokio::spawn(ipc_policy::serve(listener, gate, handler));

    let auth_sock = sock.clone();
    let auth = tokio::spawn(async move {
        ask(&auth_sock, r#"{"type":"authenticate","username":"alice"}"#).await
    });
    started.notified().await;

    // A silent connection must not stall anyone either.
    let _idle = UnixStream::connect(&sock).await.unwrap();

    let status = tokio::time::timeout(PROMPT, ask(&sock, r#"{"type":"status"}"#))
        .await
        .expect("Status must answer while Authenticate is in flight");
    assert!(
        matches!(&status, Response::Success { message: Some(m), .. } if m == "status"),
        "{status:?}"
    );
    assert!(!auth.is_finished(), "authenticate is still in flight");

    release.notify_one();
    let done = tokio::time::timeout(PROMPT, auth).await.unwrap().unwrap();
    assert!(
        matches!(&done, Response::Success { message: Some(m), .. } if m == "auth done"),
        "{done:?}"
    );
    server.abort();
}

#[tokio::test]
async fn oversized_line_drops_the_connection_without_a_reply() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let gate = Arc::new(Gatekeeper::new(IpcConfig::default()));
    let handler = Arc::new(|_peer: PeerCred, _req: Request| async {
        Response::Success {
            message: None,
            data: None,
        }
    });
    let server = tokio::spawn(ipc_policy::serve(listener, gate, handler));

    let mut s = UnixStream::connect(&sock).await.unwrap();
    let flood = vec![b'a'; 4 * ipc_policy::MAX_REQUEST_LINE_BYTES as usize];
    // The server stops reading at the cap, so the tail of the write may fail.
    let _ = tokio::time::timeout(PROMPT, s.write_all(&flood)).await;
    let mut out = String::new();
    let n = tokio::time::timeout(PROMPT, BufReader::new(s).read_line(&mut out))
        .await
        .expect("connection must be closed promptly")
        .unwrap_or(0);
    assert_eq!(n, 0, "no reply to an oversized line: {out:?}");
    server.abort();
}

#[tokio::test]
async fn per_uid_connection_limit_drops_extra_connections() {
    const LIMIT: u32 = 2;
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("d.sock");
    let listener = ipc_policy::bind_socket(sock.to_str().unwrap(), 0o666, None).unwrap();
    let cfg = IpcConfig {
        trusted_uids: vec![my_uid()],
        max_connections_per_uid: LIMIT,
        ..IpcConfig::default()
    };
    let gate = Arc::new(Gatekeeper::new(cfg));

    let release = Arc::new(Notify::new());
    let started = Arc::new(Notify::new());
    let (r, st) = (release.clone(), started.clone());
    let handler = Arc::new(move |_peer: PeerCred, req: Request| {
        let (r, st) = (r.clone(), st.clone());
        async move {
            if matches!(req, Request::Authenticate { .. }) {
                st.notify_one();
                r.notified().await;
            }
            Response::Success {
                message: Some("done".into()),
                data: None,
            }
        }
    });
    let server = tokio::spawn(ipc_policy::serve(listener, gate, handler));

    // LIMIT in-flight authenticates occupy the uid's slots.
    let mut inflight = Vec::new();
    for _ in 0..LIMIT {
        let s = sock.clone();
        inflight.push(tokio::spawn(async move {
            ask(&s, r#"{"type":"authenticate","username":"alice"}"#).await
        }));
        tokio::time::timeout(PROMPT, started.notified())
            .await
            .expect("authenticate reaches the handler");
    }

    // One more connection from the same uid is dropped without a reply.
    let mut extra = UnixStream::connect(&sock).await.unwrap();
    extra.write_all(b"{\"type\":\"status\"}\n").await.unwrap();
    let mut out = String::new();
    let n = tokio::time::timeout(PROMPT, BufReader::new(extra).read_line(&mut out))
        .await
        .expect("over-limit connection must be closed promptly")
        .unwrap_or(0);
    assert_eq!(n, 0, "over-limit connection got a reply: {out:?}");

    // Slots are released when the in-flight requests finish.
    for _ in 0..LIMIT {
        release.notify_one();
    }
    for t in inflight {
        tokio::time::timeout(PROMPT, t).await.unwrap().unwrap();
    }
    let status = tokio::time::timeout(PROMPT, async {
        loop {
            let mut s = UnixStream::connect(&sock).await.unwrap();
            s.write_all(b"{\"type\":\"status\"}\n").await.unwrap();
            let mut out = String::new();
            if BufReader::new(s).read_line(&mut out).await.unwrap_or(0) > 0 {
                return out;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("a slot frees up after the requests finish");
    assert!(status.contains("done"), "{status}");
    server.abort();
}
