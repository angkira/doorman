//! `doorman test` against a fake daemon: the stdout lines scripts parse stay
//! byte-identical, `--json` adds the structured record, exit codes are fixed.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::process::{Command, Output, Stdio};

const DOORMAN: &str = env!("CARGO_BIN_EXE_doorman");

fn run(reply: &str, json: bool) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("d.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let reply = reply.to_string();
    let server = std::thread::spawn(move || {
        let (conn, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(conn);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let mut conn = reader.into_inner();
        writeln!(conn, "{reply}").unwrap();
    });
    let mut cmd = Command::new(DOORMAN);
    cmd.arg("--socket").arg(&socket);
    if json {
        cmd.arg("--json");
    }
    let out = cmd
        .args(["test", "alice"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    server.join().unwrap();
    out
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

const SUCCESS: &str = r#"{"status":"success","message":"Authenticated (confidence: 81.23%)"}"#;
const FAILURE: &str = r#"{"status":"failure","reason":"Face not recognized (confidence: 31.00%)"}"#;
const SPOOF: &str = r#"{"status":"failure","reason":"Liveness check failed (not a live face)."}"#;
const SUSPENDED: &str = r#"{"status":"denied","reason":"face auth suspended [face_auth_suspended]: use the password (retry in 20 s)"}"#;
const DENIED: &str = r#"{"status":"denied","reason":"peer uid"}"#;

#[test]
fn plain_output_lines_are_byte_compatible() {
    let o = run(SUCCESS, false);
    assert_eq!(
        stdout(&o),
        "recognized: alice (Authenticated (confidence: 81.23%))\n"
    );
    assert_eq!(o.status.code(), Some(0));
    let o = run(FAILURE, false);
    assert_eq!(
        stdout(&o),
        "!recognized: alice (Face not recognized (confidence: 31.00%))\n"
    );
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn a_suspension_is_a_rejection_line_with_exit_2_and_a_denial_stays_an_error() {
    let o = run(SUSPENDED, false);
    assert!(
        stdout(&o).starts_with("!recognized: alice (face auth suspended"),
        "{}",
        stdout(&o)
    );
    assert_eq!(o.status.code(), Some(2));
    let o = run(DENIED, false);
    assert_eq!(stdout(&o), "");
    assert_eq!(o.status.code(), Some(1));
}

#[test]
fn json_output_is_structured_and_carries_no_exact_score() {
    let o = run(SUCCESS, true);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["decision"], "recognized");
    assert_eq!(v["reason_code"], "matched");
    assert_eq!(v["similarity_bucket"], 0.8);
    assert_eq!(v["liveness_ok"], true);
    assert!(v["duration_ms"].is_u64());
    assert!(!stdout(&o).contains("81.23") && !stdout(&o).contains("0.8123"));
    assert_eq!(o.status.code(), Some(0));

    let o = run(SPOOF, true);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        (v["decision"].as_str(), v["reason_code"].as_str()),
        (Some("not_recognized"), Some("liveness_rejected"))
    );
    assert_eq!(v["liveness_ok"], false);
    assert_eq!(o.status.code(), Some(2));

    let o = run(SUSPENDED, true);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(v["decision"], "suspended");
    assert!(v["suspended_until"].as_str().unwrap().ends_with('Z'));
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn json_reports_errors_as_json_with_exit_1() {
    let o = run(DENIED, true);
    let v: serde_json::Value = serde_json::from_str(&stdout(&o)).unwrap();
    assert_eq!(
        (v["decision"].as_str(), v["reason_code"].as_str()),
        (Some("error"), Some("denied"))
    );
    assert_eq!(o.status.code(), Some(1));
}

#[test]
fn json_with_an_unreachable_daemon_is_still_one_json_object() {
    let out = Command::new(DOORMAN)
        .args([
            "--socket",
            "/nonexistent/doorman.sock",
            "--json",
            "test",
            "alice",
        ])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&stdout(&out)).unwrap();
    assert_eq!(v["decision"], "error");
    assert_eq!(out.status.code(), Some(1));
}
