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
