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

// ---------------------------------------------------------------------------
// storage key (`doorman key ...`; see docs/STORAGE-ENCRYPTION.md)
// ---------------------------------------------------------------------------

/// The command the operator must run when this process is not root.
pub fn key_sudo_instruction(args: &str) -> String {
    format!("storage key management needs root. Run: sudo doorman key{args}")
}

fn storage_key_of(response: Response) -> Result<doorman_shared::StorageKeyInfo, String> {
    match response {
        Response::Success {
            data: Some(ResponseData::StorageKey { info }),
            ..
        } => Ok(info),
        Response::Failure { reason } | Response::Denied { reason } => Err(reason),
        other => Err(format!("unexpected response: {other:?}")),
    }
}

fn print_storage_key(info: &doorman_shared::StorageKeyInfo) {
    println!(
        "sealed file: {}   key loaded: {}   PIN enrolled: {}",
        on_off(info.sealed),
        on_off(info.loaded),
        on_off(info.pin_enrolled)
    );
    if info.sealed && !info.loaded {
        println!("The store is LOCKED: face auth and enrollment refuse with `storage_locked`.");
        println!("Load the key: restart (TPM credential) or: sudo doorman key unlock");
    }
    if !info.sealed && info.loaded {
        println!(
            "The key is loaded but the file is not sealed yet: the next store write seals it (v4)."
        );
    }
}

fn on_off(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// `doorman key status`. World-readable, no secrets.
pub fn cmd_key_status(send: Transport<'_>, json: bool) -> Result<i32, String> {
    let response = send(&Request::StorageKeyStatus, STATUS_TIMEOUT)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?
        );
    }
    print_storage_key(&storage_key_of(response)?);
    Ok(0)
}

/// Read one secret line without echo via systemd-ask-password; falls back to
/// an echoed stdin read (warned) when it is unavailable.
fn ask_secret(prompt: &str) -> Result<String, String> {
    if let Ok(out) = std::process::Command::new("systemd-ask-password")
        .arg(prompt)
        .output()
    {
        if out.status.success() {
            return Ok(String::from_utf8_lossy(&out.stdout)
                .trim_end_matches('\n')
                .to_string());
        }
    }
    eprintln!(
        "warning: systemd-ask-password unavailable; the PIN will echo. Prefer a local terminal."
    );
    eprint!("{prompt}: ");
    use std::io::Write as _;
    std::io::stderr()
        .flush()
        .map_err(|e| format!("cannot prompt: {e}"))?;
    let mut line = String::new();
    std::io::BufRead::read_line(&mut std::io::stdin().lock(), &mut line)
        .map_err(|e| format!("cannot read PIN: {e}"))?;
    Ok(line.trim_end_matches('\n').to_string())
}

/// Write secret bytes with mode 0600 at creation (`create_new`: never clobbers).
pub fn write_secret_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    use std::io::Write as _;
    f.write_all(bytes)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    f.sync_all()
        .map_err(|e| format!("cannot fsync {}: {e}", path.display()))?;
    Ok(())
}

/// Best-effort shred of a temp key file (overwrite + remove).
fn shred_file(path: &Path) {
    if let Ok(meta) = std::fs::metadata(path) {
        let zeros = vec![0u8; meta.len().min(4096) as usize];
        let _ = std::fs::write(path, &zeros);
    }
    let _ = std::fs::remove_file(path);
}

/// `doorman key unlock`. Root only. Reads the PIN sidecar (as root), unwraps
/// the data key locally, and hands it to the daemon over the root-only IPC.
/// The PIN and the key never touch the disk outside their 0600 homes.
pub fn cmd_key_unlock(send: Transport<'_>, json: bool, is_root: bool) -> Result<i32, String> {
    if !is_root {
        eprintln!("{}", key_sudo_instruction(" unlock"));
        return Ok(1);
    }
    let data_dir = PathBuf::from(doorman_shared::DATA_DIR);
    let blob_path = doormand::storage_key::pin_blob_path(&data_dir);
    let text = std::fs::read_to_string(&blob_path)
        .map_err(|e| format!("no PIN enrolled ({}: {e})", blob_path.display()))?;
    let blob: doormand::storage_key::PinBlob =
        serde_json::from_str(&text).map_err(|e| format!("PIN blob unreadable: {e}"))?;
    let pin = ask_secret("doorman storage PIN")?;
    let key = doormand::storage_key::unwrap_with_pin(&blob, &pin).map_err(|e| format!("{e:#}"))?;
    let response = send(
        &Request::UnlockStorage { key: key.to_vec() },
        STATUS_TIMEOUT,
    )?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?
        );
    }
    match response {
        Response::Success {
            data: Some(ResponseData::StorageKey { info }),
            ..
        } => {
            print_storage_key(&info);
            Ok(0)
        }
        Response::Failure { reason } | Response::Denied { reason } => Err(reason),
        other => Err(format!("unexpected response: {other:?}")),
    }
}

/// `doorman key enroll [--tpm-pcrs 7] [--pin]`. Root only. First-time setup:
/// refuses when the live store is already sealed or keyed (use `rotate` for a
/// live store — a fresh enroll key would orphan the live one). Generates the
/// data key, seals it to the TPM via `systemd-creds encrypt`, installs the
/// unit drop-in, and optionally writes the PIN sidecar. Takes effect on
/// restart; the store file itself is sealed by the daemon on its first write
/// after that (with the `.v3.bak` backup). Never touches embeddings.bin.
pub fn cmd_key_enroll(
    send: Transport<'_>,
    tpm_pcrs: &str,
    with_pin: bool,
    json: bool,
    is_root: bool,
) -> Result<i32, String> {
    if !is_root {
        let mut args = String::from(" enroll");
        if tpm_pcrs != "7" {
            args.push_str(&format!(" --tpm-pcrs {tpm_pcrs}"));
        }
        if with_pin {
            args.push_str(" --pin");
        }
        eprintln!("{}", key_sudo_instruction(&args));
        return Ok(1);
    }
    let status = storage_key_of(send(&Request::StorageKeyStatus, STATUS_TIMEOUT)?)?;
    if status.sealed || status.loaded {
        return Err(
            "the live store is already sealed or keyed; use `sudo doorman key rotate` instead (a fresh enroll key would orphan the live one)".into(),
        );
    }
    if std::process::Command::new("systemd-creds")
        .arg("--version")
        .output()
        .map(|o| !o.status.success())
        .unwrap_or(true)
    {
        return Err(
            "systemd-creds is missing or unusable (systemd >= 248 with +TPM2 required)".into(),
        );
    }
    // Temp key file outside the data dir, 0600 from creation.
    let tmp = std::env::temp_dir().join(format!("doorman-key-{}.bin", std::process::id()));
    let key = doormand::storage_key::DataKey::generate().map_err(|e| format!("{e:#}"))?;
    write_secret_file(&tmp, key.bytes())?;
    let sealed = run_tpm_seal(&tmp, tpm_pcrs);
    shred_file(&tmp);
    let sealed = sealed?;
    let dropin = doormand::storage_key::render_dropin(&sealed).map_err(|e| format!("{e:#}"))?;
    let dropin_path = doormand::storage_key::dropin_path();
    if let Some(parent) = dropin_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    std::fs::write(&dropin_path, dropin)
        .map_err(|e| format!("cannot write {}: {e}", dropin_path.display()))?;
    if with_pin {
        let pin = ask_secret("new doorman storage PIN")?;
        let again = ask_secret("repeat the PIN")?;
        if pin != again {
            return Err("PINs do not match; nothing enrolled".into());
        }
        let blob = doormand::storage_key::wrap_with_pin(key.bytes(), &pin)
            .map_err(|e| format!("{e:#}"))?;
        let text = serde_json::to_string_pretty(&blob).map_err(|e| format!("{e}"))?;
        let blob_path =
            doormand::storage_key::pin_blob_path(&PathBuf::from(doorman_shared::DATA_DIR));
        if blob_path.exists() {
            return Err(format!(
                "PIN sidecar {} already exists; refusing to overwrite",
                blob_path.display()
            ));
        }
        write_secret_file(&blob_path, text.as_bytes())?;
    }
    if json {
        println!(
            "{}",
            serde_json::json!({"enrolled": true, "tpm_pcrs": tpm_pcrs, "pin": with_pin})
        );
    }
    println!("sealed key installed: {}", dropin_path.display());
    if with_pin {
        println!("PIN recovery enrolled.");
    }
    println!("Takes effect on restart: sudo systemctl restart doormand");
    println!("The store seals on the first write after that (backup: embeddings.bin.v3.bak).");
    Ok(0)
}

/// Run `systemd-creds encrypt` over `key_file` and return its stdout
/// (`SetCredentialEncrypted=` line). Shared by enroll and rotate.
fn run_tpm_seal(key_file: &Path, pcrs: &str) -> Result<String, String> {
    let (prog, argv) = doormand::storage_key::tpm_seal_argv(key_file, pcrs);
    let out = std::process::Command::new(&prog)
        .args(&argv)
        .output()
        .map_err(|e| format!("cannot run {prog}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "systemd-creds encrypt failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| format!("sealed key is not UTF-8: {e}"))
}

/// `doorman key rotate [--pin]`. Root only. Refuses when the file is sealed
/// and no PIN recovery exists (a lost new key would strand the store). Order:
/// PIN blob first (recovery for the new key), daemon rekey, drop-in last, so
/// every prefix of a crash still leaves a recovery path.
pub fn cmd_key_rotate(
    send: Transport<'_>,
    with_pin: bool,
    tpm_pcrs: &str,
    json: bool,
    is_root: bool,
) -> Result<i32, String> {
    if !is_root {
        eprintln!(
            "{}",
            key_sudo_instruction(if with_pin { " rotate --pin" } else { " rotate" })
        );
        return Ok(1);
    }
    let status = storage_key_of(send(&Request::StorageKeyStatus, STATUS_TIMEOUT)?)?;
    if !status.loaded {
        return Err(
            "store key is not loaded (restart with the credential, or unlock first)".into(),
        );
    }
    if status.sealed && !status.pin_enrolled && !with_pin {
        return Err(
            "refusing: the file is sealed and no PIN recovery is enrolled; re-run with --pin"
                .into(),
        );
    }
    let new_key = doormand::storage_key::DataKey::generate().map_err(|e| format!("{e:#}"))?;
    if with_pin {
        let pin = ask_secret("new doorman storage PIN")?;
        let again = ask_secret("repeat the PIN")?;
        if pin != again {
            return Err("PINs do not match; nothing rotated".into());
        }
        let blob = doormand::storage_key::wrap_with_pin(new_key.bytes(), &pin)
            .map_err(|e| format!("{e:#}"))?;
        // Atomic replace of the sidecar: 0600 temp + rename in the same dir.
        let blob_path =
            doormand::storage_key::pin_blob_path(&PathBuf::from(doorman_shared::DATA_DIR));
        let tmp = blob_path.with_extension("pin.tmp");
        write_secret_file(
            &tmp,
            serde_json::to_string_pretty(&blob)
                .map_err(|e| format!("{e}"))?
                .as_bytes(),
        )?;
        std::fs::rename(&tmp, &blob_path)
            .map_err(|e| format!("cannot replace {}: {e}", blob_path.display()))?;
    }
    let response = send(
        &Request::RotateStorage {
            key: new_key.bytes().to_vec(),
        },
        STATUS_TIMEOUT,
    )?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).map_err(|e| e.to_string())?
        );
    }
    match response {
        Response::Success {
            data: Some(ResponseData::StorageKey { info }),
            ..
        } => {
            print_storage_key(&info);
            // Keep the boot credential in step with the live key, but only
            // when a TPM drop-in already exists (PIN-only mode has none and
            // must stay that way).
            let dropin_path = doormand::storage_key::dropin_path();
            if dropin_path.exists() {
                let tmp =
                    std::env::temp_dir().join(format!("doorman-key-{}.bin", std::process::id()));
                write_secret_file(&tmp, new_key.bytes())?;
                let sealed = run_tpm_seal(&tmp, tpm_pcrs);
                shred_file(&tmp);
                let sealed = sealed?;
                let dropin =
                    doormand::storage_key::render_dropin(&sealed).map_err(|e| format!("{e:#}"))?;
                std::fs::write(&dropin_path, dropin)
                    .map_err(|e| format!("cannot write {}: {e}", dropin_path.display()))?;
                println!("boot credential re-sealed: {}", dropin_path.display());
            } else {
                println!(
                    "no TPM drop-in present (PIN-only mode); next boot needs `key unlock` again"
                );
            }
            println!("Restart to confirm the new key chain: sudo systemctl restart doormand");
            Ok(0)
        }
        Response::Failure { reason } | Response::Denied { reason } => Err(reason),
        other => Err(format!("unexpected response: {other:?}")),
    }
}
