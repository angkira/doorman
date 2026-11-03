//! Entry points shared by the cargo-fuzz targets (`fuzz/`) and the seeded
//! no-panic tests (`daemon/tests/fuzz_smoke.rs`), so both exercise exactly the
//! same code. Each function feeds untrusted bytes to the production path and
//! asserts the invariants that must hold; a panic is a bug.
//!
//! No behaviour of the daemon depends on this module.

use crate::ipc_policy::{read_gated_request, request_name, Gated, Gatekeeper, PeerCred};
use crate::storage::{Storage, UserEmbedding};
use doorman_shared::config_check::check_str;
use doorman_shared::{IpcConfig, Request, Response, PROTOCOL_VERSION};
use std::time::Instant;

/// uid of the fixed unprivileged peer.
pub const FUZZ_USER_UID: u32 = 1000;
/// uid of a second unprivileged user, never the peer.
const FUZZ_OTHER_UID: u32 = 1001;
/// uid listed in `admin_uids` of the fuzz config (not the peer).
pub const FUZZ_ADMIN_UID: u32 = 2000;
/// Usernames the fixed resolver knows.
pub const FUZZ_USER: &str = "alice";
const FUZZ_OTHER_USER: &str = "bob";

/// What the gate did with one input.
#[derive(Debug, PartialEq, Eq)]
pub enum IpcOutcome {
    /// No request: oversize, not UTF-8, or not a valid request.
    Unparsed,
    /// Authorized for the peer; the request is handed to the handler.
    Allowed(&'static str),
    /// Answered by the gate instead (`Denied`, `UnsupportedProtocol`, `RateLimited`).
    Rejected,
}

fn fuzz_resolver(name: &str) -> Option<u32> {
    match name {
        FUZZ_USER => Some(FUZZ_USER_UID),
        FUZZ_OTHER_USER => Some(FUZZ_OTHER_UID),
        _ => None,
    }
}

fn is_admin_request(req: &Request) -> bool {
    !matches!(
        req,
        Request::Authenticate { .. } | Request::WaitForMatch { .. } | Request::Status
    )
}

/// Parse one request line exactly as the daemon does
/// ([`read_gated_request`]: size cap, UTF-8, `RequestEnvelope`, protocol
/// version gate, `authorize()`) for an unprivileged peer, and check that the
/// gate never grants more than the documented policy.
pub fn ipc_request_line(data: &[u8]) -> IpcOutcome {
    let cfg = IpcConfig {
        admin_uids: vec![FUZZ_ADMIN_UID],
        ..IpcConfig::default()
    };
    let gate = Gatekeeper::with_parts(cfg, Box::new(fuzz_resolver), Box::new(Instant::now));
    let peer = PeerCred {
        uid: FUZZ_USER_UID,
        gid: FUZZ_USER_UID,
        pid: Some(1),
    };
    let mut reader = data;
    let gated = match futures::executor::block_on(read_gated_request(&mut reader, &peer, &gate)) {
        Ok(g) => g,
        Err(_) => return IpcOutcome::Unparsed,
    };
    match gated {
        Gated::Allowed(req) => {
            assert!(
                !is_admin_request(&req),
                "unprivileged peer was allowed an admin request: {req:?}"
            );
            if let Request::Authenticate { username, .. } | Request::WaitForMatch { username, .. } =
                &req
            {
                assert_eq!(
                    username, FUZZ_USER,
                    "peer authorized for a different user's account"
                );
            }
            IpcOutcome::Allowed(request_name(&req))
        }
        Gated::Rejected(resp) => {
            match resp {
                Response::Denied { .. } | Response::RateLimited { .. } => {}
                Response::UnsupportedProtocol {
                    supported_version, ..
                } => assert_eq!(supported_version, PROTOCOL_VERSION),
                other => panic!("gate answered with an unexpected response: {other:?}"),
            }
            IpcOutcome::Rejected
        }
    }
}

/// Parse, validate and unknown-key-diff a config file (`doorman config check`).
/// Returns whether the text was accepted. An accepted config must stay valid
/// when validated again.
pub fn config_toml(data: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(data) else {
        return false;
    };
    match check_str(text) {
        Ok(report) => {
            report
                .config
                .validate()
                .expect("an accepted config validates again");
            true
        }
        Err(_) => false,
    }
}

/// Load an `embeddings.bin` (version 1 headerless, or version 2). Returns the
/// number of users. A file that decodes must re-encode and decode to the same
/// users.
pub fn embeddings_file(data: &[u8]) -> Option<usize> {
    let users = Storage::decode_embeddings(data).ok()?;
    let bytes = Storage::encode_embeddings(&users).expect("decoded users re-encode");
    let again = Storage::decode_embeddings(&bytes).expect("re-encoded file decodes");
    assert_eq!(users.len(), again.len());
    for (a, b) in users.iter().zip(&again) {
        assert_eq!(a.username, b.username);
        assert_eq!(a.embeddings.len(), b.embeddings.len());
    }
    Some(users.len())
}

/// A small synthetic user (no biometric data) for seed corpora and tests.
pub fn synthetic_user(name: &str, vectors: usize, dim: usize) -> UserEmbedding {
    UserEmbedding {
        username: name.to_string(),
        embeddings: (0..vectors)
            .map(|v| (0..dim).map(|i| ((v + i) % 7) as f32 / 7.0).collect())
            .collect(),
        enrolled_at: "2026-01-01T00:00:00Z".to_string(),
    }
}
