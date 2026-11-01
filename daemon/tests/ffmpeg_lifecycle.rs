//! Lifecycle of the ffmpeg capture child: dropping the camera must kill and reap it.
//!
//! Uses a stub `ffmpeg` script on PATH (no real ffmpeg or camera needed). This
//! lives in its own integration-test binary because it overrides the process
//! PATH, which must not leak into other tests.

#![cfg(all(feature = "camera-ffmpeg", target_os = "linux"))]

use doorman_shared::Config;
use doormand::camera::{CameraBackend, FfmpegCamera};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::{Duration, Instant};

/// How long the stub child sleeps if nobody kills it.
const STUB_SLEEP_SECS: u32 = 300;
/// How long to wait for the stub to publish its pid.
const PID_WAIT: Duration = Duration::from_secs(5);

fn write_stub(dir: &Path, pid_file: &Path) {
    let stub = dir.join("ffmpeg");
    let script = format!(
        "#!/bin/sh\necho $$ > '{}'\nexec sleep {}\n",
        pid_file.display(),
        STUB_SLEEP_SECS
    );
    std::fs::write(&stub, script).unwrap();
    std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn wait_for_pid(pid_file: &Path) -> u32 {
    let start = Instant::now();
    while start.elapsed() < PID_WAIT {
        if let Ok(s) = std::fs::read_to_string(pid_file) {
            if let Ok(pid) = s.trim().parse() {
                return pid;
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("stub ffmpeg did not publish its pid within {PID_WAIT:?}");
}

#[test]
fn dropping_ffmpeg_camera_kills_and_reaps_child() {
    let dir = std::env::temp_dir().join(format!("doorman-ffmpeg-stub-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let pid_file = dir.join("pid");
    write_stub(&dir, &pid_file);

    let old_path = std::env::var("PATH").unwrap_or_default();
    std::env::set_var("PATH", format!("{}:{}", dir.display(), old_path));

    let cam = futures::executor::block_on(FfmpegCamera::new_with_config(&Config::default()))
        .expect("stub ffmpeg should spawn");
    let pid = wait_for_pid(&pid_file);
    let proc_dir = format!("/proc/{pid}");
    assert!(Path::new(&proc_dir).exists(), "child must be running");

    drop(cam);

    // kill() + wait() in Drop reaps the child: no /proc entry (a zombie would keep it).
    assert!(
        !Path::new(&proc_dir).exists(),
        "ffmpeg child {pid} still present (running or zombie) after drop"
    );

    std::env::set_var("PATH", old_path);
    let _ = std::fs::remove_dir_all(&dir);
}
