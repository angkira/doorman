//! Terminal side of `doorman calibrate`, `doorman pad run` and
//! `doorman config get|set|unset`. Included by the `doorman` binary (like
//! `install.rs`); the logic it drives lives in the `doormand` library
//! (`calibration`, `pad_protocol`, `config_edit`) where it is unit-tested.

use chrono::{DateTime, Utc};
use doorman_shared::{CalibrationInfo, Request, Response, ResponseData};
use doormand::config_edit::{self, EditError, Source};
use doormand::pad_protocol::{self as pad, Backend, Counts, Ending, Session, ShadowLog, Terminal};
use doormand::test_report::{self, TestReport};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

/// Sends one request to the daemon; the error text is shown to the user.
pub type Transport<'a> = &'a dyn Fn(&Request, Duration) -> Result<Response, String>;

/// Same ceiling as `doorman test`.
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(15);
const STATUS_TIMEOUT: Duration = Duration::from_secs(5);
const INPUT_POLL: Duration = Duration::from_millis(100);

// ---------------------------------------------------------------------------
// calibrate
// ---------------------------------------------------------------------------

/// The command the operator must run when this process is not root.
pub fn sudo_instruction(args: &str) -> String {
    format!("calibration mode needs root. Run: sudo doorman calibrate{args}")
}

fn calibration_of(response: Response) -> Result<CalibrationInfo, String> {
    match response {
        Response::Success {
            data: Some(ResponseData::Calibration { info }),
            ..
        } => Ok(info),
        Response::Failure { reason } | Response::Denied { reason } => Err(reason),
        other => Err(format!("unexpected response: {other:?}")),
    }
}

fn print_calibration(info: &CalibrationInfo) {
    if info.active {
        println!(
            "CALIBRATION MODE ACTIVE: {} left of {}.",
            fmt_duration(info.remaining_secs),
            fmt_duration(info.ttl_secs)
        );
        println!(
            "  Failure budget suspended, shadow PAD logging on; every decision rule is unchanged."
        );
        println!(
            "  Judged failures not counted toward the budget so far: {}",
            info.judged_failures_counted
        );
        println!("  It ends by itself (and with a daemon restart). End it now: sudo doorman calibrate --end");
    } else {
        println!("calibration mode: off (failure budget and logging are as configured)");
    }
}

pub fn fmt_duration(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else {
        format!("{m}m{s:02}s")
    }
}

/// `doorman calibrate [--ttl T] [--end] [--status]`. Returns the exit code.
pub fn cmd_calibrate(
    send: Transport<'_>,
    ttl: Option<&str>,
    end: bool,
    status: bool,
    json: bool,
    is_root: bool,
) -> Result<i32, String> {
    let request = if status {
        Request::CalibrateStatus
    } else if end {
        Request::CalibrateEnd
    } else {
        let ttl_secs = ttl.map(doormand::calibration::parse_ttl).transpose()?;
        Request::Calibrate { ttl_secs }
    };
    if !status && !is_root {
        let args = if end {
            " --end".to_string()
        } else {
            ttl.map(|t| format!(" --ttl {t}")).unwrap_or_default()
        };
        eprintln!("{}", sudo_instruction(&args));
        return Ok(1);
    }
    let response = send(&request, STATUS_TIMEOUT)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?
        );
    }
    let info = calibration_of(response)?;
    if !json {
        print_calibration(&info);
        if info.active && !status && !end {
            println!("Next: doorman pad run");
        }
    }
    Ok(0)
}

/// Current calibration state, `None` when the daemon cannot be asked.
pub fn calibration_state(send: Transport<'_>) -> Option<CalibrationInfo> {
    calibration_of(send(&Request::CalibrateStatus, STATUS_TIMEOUT).ok()?).ok()
}

// ---------------------------------------------------------------------------
// pad run
// ---------------------------------------------------------------------------

struct DaemonBackend<'a> {
    send: Transport<'a>,
    user: String,
}

impl Backend for DaemonBackend<'_> {
    fn attempt(&mut self) -> TestReport {
        let started = std::time::Instant::now();
        let answer = (self.send)(
            &Request::Authenticate {
                username: self.user.clone(),
                service: None,
                tty: None,
            },
            ATTEMPT_TIMEOUT,
        );
        let ms = started.elapsed().as_millis() as u64;
        match answer {
            Ok(response) => test_report::classify(&response, ms, Utc::now().timestamp()),
            Err(_) => test_report::error_report(ms),
        }
    }

    fn calibration(&mut self) -> Option<CalibrationInfo> {
        calibration_state(self.send)
    }
}

/// Real terminal: prompts on stdout, lines from a stdin thread, Ctrl-C as a
/// flag (the first one ends the session cleanly, a second one kills the process).
struct StdTerminal {
    interrupted: Arc<AtomicBool>,
    lines: Receiver<String>,
}

impl StdTerminal {
    fn new() -> std::io::Result<Self> {
        let interrupted = Arc::new(AtomicBool::new(false));
        signal_hook::flag::register_conditional_shutdown(
            signal_hook::consts::SIGINT,
            1,
            interrupted.clone(),
        )?;
        signal_hook::flag::register(signal_hook::consts::SIGINT, interrupted.clone())?;
        let (tx, lines) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Self { interrupted, lines })
    }
}

impl Terminal for StdTerminal {
    fn say(&mut self, line: &str) {
        println!("{line}");
    }

    fn ask(&mut self, prompt: &str) -> Option<String> {
        print!("{prompt}");
        let _ = std::io::stdout().flush();
        loop {
            if self.interrupted.load(Ordering::Relaxed) {
                println!();
                return None;
            }
            match self.lines.recv_timeout(INPUT_POLL) {
                Ok(line) => return Some(line),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
    }

    fn sleep(&mut self, d: Duration) -> bool {
        let deadline = std::time::Instant::now() + d;
        while std::time::Instant::now() < deadline {
            if self.interrupted.load(Ordering::Relaxed) {
                return false;
            }
            std::thread::sleep(INPUT_POLL.min(deadline - std::time::Instant::now()));
        }
        !self.interrupted.load(Ordering::Relaxed)
    }

    fn interrupted(&self) -> bool {
        self.interrupted.load(Ordering::Relaxed)
    }

    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// The owner to test: `--user`, else the invoking user (`SUDO_USER` when run
/// through sudo, which the daemon would otherwise read as uid 0).
pub fn resolve_user(explicit: Option<&str>) -> Result<String, String> {
    if let Some(u) = explicit {
        return Ok(u.to_string());
    }
    if let Some(u) = std::env::var("SUDO_USER").ok().filter(|u| !u.is_empty()) {
        return Ok(u);
    }
    nix::unistd::User::from_uid(nix::unistd::getuid())
        .ok()
        .flatten()
        .map(|u| u.name)
        .ok_or_else(|| "cannot tell which user you are; pass --user NAME".to_string())
}

/// The shadow log path from the config the daemon would load (readable
/// copies only; built-in default otherwise).
fn shadow_log_path() -> String {
    doorman_shared::Config::load_with(None)
        .map(|c| c.telemetry.shadow_pad_log_path)
        .unwrap_or_else(|_| {
            doorman_shared::Config::default()
                .telemetry
                .shadow_pad_log_path
        })
}

/// `doorman pad run [--user U] [--quick] [--out PREFIX]`. Returns the exit code.
pub fn cmd_pad_run(
    send: Transport<'_>,
    user: Option<&str>,
    quick: bool,
    out: Option<&Path>,
) -> Result<i32, String> {
    let user = resolve_user(user)?;
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or("HOME is not set: cannot place the result files")?;
    let prefix = out
        .map(Path::to_path_buf)
        .unwrap_or_else(|| pad::default_prefix(&home, Utc::now().date_naive()));
    let counts = if quick { Counts::QUICK } else { Counts::FULL };
    let log_path = shadow_log_path();
    let shadow = ShadowLog::open(Path::new(&log_path));
    let mut backend = DaemonBackend { send, user };
    let mut term = StdTerminal::new().map_err(|e| format!("cannot set up Ctrl-C handling: {e}"))?;
    let mut session = Session::new(&mut backend, &mut term, counts, shadow);
    session.run();

    let rows = pad::summarize(&session.attempts);
    let headline = pad::headline(&rows);
    println!("\n{}", pad::format_table(&rows));
    println!("{headline}\n");
    match session.ending {
        Ending::Completed => {}
        Ending::NotStarted => println!("Nothing was run."),
        other => println!("Session ended early ({other:?}): the partial results are saved."),
    }
    if !session.attempts.is_empty() {
        let paths = pad::write_results(&prefix, &session, &rows, &headline)
            .map_err(|e| format!("cannot write the result files: {e}"))?;
        println!(
            "written (mode 0600): {}, {}, {}",
            paths[0].display(),
            paths[1].display(),
            paths[2].display()
        );
        println!(
            "cross-check: doorman-bench pad-report --scores {} --threshold 0.5",
            paths[2].display()
        );
    }
    println!("{}", pad::shadow_hint(&session, &log_path));
    Ok(if session.ending == Ending::Completed {
        0
    } else {
        1
    })
}

// ---------------------------------------------------------------------------
// config get | set | unset
// ---------------------------------------------------------------------------

/// Where `config get` reads: `--config`, else the file the daemon would load.
pub fn config_read_path(explicit: Option<&Path>) -> Option<PathBuf> {
    explicit
        .map(Path::to_path_buf)
        .or_else(doorman_shared::Config::find_config_path)
}

pub fn cmd_config_get(explicit: Option<&Path>, key: &str) -> Result<i32, String> {
    let (content, origin) = match config_read_path(explicit) {
        Some(p) => (
            std::fs::read_to_string(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?,
            p.display().to_string(),
        ),
        None => (String::new(), "built-in defaults".to_string()),
    };
    for entry in config_edit::get(&content, key).map_err(|e| e.to_string())? {
        let note = match entry.source {
            Source::File => format!("from {origin}"),
            Source::Default => "default".to_string(),
        };
        println!("{} = {}   # {note}", entry.key, entry.value);
    }
    Ok(0)
}

/// `config set` / `config unset` (`value = None`). Root edits the system file;
/// a non-root user may edit only an explicit `--config` file of their own.
pub fn cmd_config_edit(
    explicit: Option<&Path>,
    key: &str,
    value: Option<&str>,
    is_root: bool,
) -> Result<i32, String> {
    let verb = if value.is_some() { "set" } else { "unset" };
    if !is_root && explicit.is_none() {
        let tail = value.map(|v| format!(" {v}")).unwrap_or_default();
        eprintln!(
            "editing the system config needs root. Run: sudo doorman config {verb} {key}{tail}"
        );
        return Ok(1);
    }
    let path = explicit
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(doorman_shared::config_trust::SYSTEM_CONFIG_PATH));
    if !path.exists() {
        return Err(format!(
            "{} does not exist; copy packaging/doorman.toml.example there first",
            path.display()
        ));
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let plan = match value {
        Some(v) => config_edit::plan_set(&content, key, v),
        None => config_edit::plan_unset(&content, key),
    }
    .map_err(|e: EditError| e.to_string())?;
    if !plan.changed {
        println!(
            "no change: {key} already has that value in {}",
            path.display()
        );
        return Ok(0);
    }
    let timestamp = Utc::now().format("%Y%m%dT%H%M%S%3fZ").to_string();
    let trust = is_root.then_some(
        &doorman_shared::config_trust::FsMetadata
            as &dyn doorman_shared::config_trust::MetadataProvider,
    );
    let applied = config_edit::apply(&path, &content, &plan.new_content, &timestamp, trust)
        .map_err(|e| e.to_string())?;
    print!("{}", plan.diff);
    println!(
        "{verb}: {key}: {} -> {}",
        plan.before.as_deref().unwrap_or("(not set)"),
        plan.after.as_deref().unwrap_or("(default)")
    );
    println!("backup: {}", applied.backup.display());
    println!("{}", config_edit::RESTART_HINT);
    Ok(0)
}
