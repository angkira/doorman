//! A bad explicit config must stop the daemon instead of running on defaults.

use std::process::{Command, Output};

fn daemon() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_doormand"));
    cmd.env_remove("DOORMAN_CONFIG");
    cmd
}

fn run_with_config(contents: &str, via_env: bool) -> Output {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("doorman.toml");
    std::fs::write(&path, contents).unwrap();
    let mut cmd = daemon();
    if via_env {
        cmd.env("DOORMAN_CONFIG", &path);
    } else {
        cmd.arg("--config").arg(&path);
    }
    cmd.output().unwrap()
}

fn assert_refused(out: &Output) {
    assert!(!out.status.success(), "daemon must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("refusing to start: invalid configuration"),
        "stderr: {stderr}"
    );
}

#[test]
fn unparsable_config_flag_is_fatal() {
    assert_refused(&run_with_config("this is = = not toml", false));
}

#[test]
fn invalid_value_in_env_config_is_fatal() {
    assert_refused(&run_with_config(
        "[enrollment]\nprune_cosine = 0.0\n",
        true,
    ));
}

#[test]
fn missing_explicit_config_is_fatal() {
    let out = daemon()
        .args(["--config", "/nonexistent/doorman-test/doorman.toml"])
        .output()
        .unwrap();
    assert_refused(&out);
}
