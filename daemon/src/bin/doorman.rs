//! `doorman` — friendly command-line tool to enroll and manage users by talking
//! to a running `doormand` daemon over its IPC socket.
//!
//! ## Wire protocol (matched exactly to `daemon/src/ipc.rs`)
//!
//! The command socket speaks **newline-delimited JSON**. For every request the
//! client opens a connection, writes exactly one JSON-serialized
//! [`Request`] followed by a single `\n`, then reads exactly one line back and
//! parses it as a [`Response`]. The daemon handles one request per connection
//! and then closes it.
//!
//! Note that `Enroll` does **not** stream `Progress` responses on the command
//! socket — the daemon blocks for the full recording duration and then replies
//! with a single `Success`/`Failure`. Live enrollment progress is published
//! separately on the **debug socket** (`doorman-debug.sock`) as
//! newline-delimited [`StreamMessage::Enrollment`] frames. So `enroll` opens the
//! debug socket in a background thread to render a progress bar while the
//! command connection waits for the final reply.
//!
//! ## Socket path resolution (matched to `daemon/src/main.rs` `--user` mode)
//!
//! Priority: `--socket` > `--runtime-dir` > `$XDG_RUNTIME_DIR` > `/tmp/doorman-run`.
//! The command socket is `<runtime>/doorman.sock` and the debug socket is
//! `<runtime>/doorman-debug.sock`, exactly as the daemon builds them.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use doorman_shared::{
    DaemonInfo, EnrollmentPhase, Request, Response, ResponseData, StreamMessage, UserInfo,
};

// Self-contained installer (doctor / install / uninstall). Kept in a sibling
// file so this bin stays a pure std+clap client that still builds with
// --no-default-features.
#[path = "../install.rs"]
mod install;
use install::{DesktopChoice, EpChoice, InstallOpts, PamScope, UninstallOpts};

/// Default runtime directory when neither `--socket`/`--runtime-dir` nor
/// `$XDG_RUNTIME_DIR` is set. Matches how the daemon is run on this Mac.
const FALLBACK_RUNTIME_DIR: &str = "/tmp/doorman-run";
const COMMAND_SOCKET_NAME: &str = "doorman.sock";
const DEBUG_SOCKET_NAME: &str = "doorman-debug.sock";

/// Read timeout for a normal request/response round-trip.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
/// Enrollment records ~10s of video and then processes every captured frame
/// (detect → align → embed) on the daemon side. On CPU that can be hundreds of
/// frames at a few hundred ms each, so the client must wait well beyond the
/// recording window. This is the wall-clock ceiling before we give up.
const ENROLL_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Parser)]
#[command(
    name = "doorman",
    about = "Manage the doorman face-unlock daemon (enroll, list, remove, test, status)",
    version
)]
struct Cli {
    /// Override the full path to the daemon command socket.
    #[arg(long, global = true, value_name = "PATH")]
    socket: Option<PathBuf>,

    /// Runtime directory holding the daemon sockets (default: $XDG_RUNTIME_DIR or /tmp/doorman-run).
    #[arg(long, global = true, value_name = "DIR")]
    runtime_dir: Option<PathBuf>,

    /// Emit raw JSON responses instead of human-friendly output.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Enroll a new user (records from the daemon's current camera source).
    Enroll {
        /// Username to enroll.
        username: String,
    },
    /// List all enrolled users.
    List,
    /// Remove a user's enrollment.
    Remove {
        /// Username to remove.
        username: String,
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Authenticate a user against the live camera (alias: auth).
    #[command(alias = "auth")]
    Test {
        /// Username to authenticate.
        username: String,
    },
    /// Show daemon status (version, uptime, camera, models, enrolled users).
    Status,
    /// Build an offline self-improvement candidate from the attempt-log dataset.
    Train {
        /// Data directory containing dataset/attempts-*.jsonl and labels-*.jsonl.
        #[arg(long, value_name = "DIR")]
        data_dir: Option<PathBuf>,
        /// Output models directory. Defaults to <data-dir>/models.
        #[arg(long, value_name = "DIR")]
        models_dir: Option<PathBuf>,
        /// Minimum confirmed-genuine label events required before writing a candidate.
        #[arg(long, default_value_t = 1)]
        min_confirmed: usize,
    },
    /// Inspect offline dataset files and print a summary.
    DatasetReport {
        /// Data directory containing dataset/attempts-*.jsonl and labels-*.jsonl.
        #[arg(long, value_name = "DIR")]
        data_dir: Option<PathBuf>,
    },
    /// Promote a projector candidate after offline validation passes.
    Promote {
        /// Candidate directory containing manifest.json.
        #[arg(long, value_name = "DIR")]
        candidate_dir: PathBuf,
        /// False-accept rate for the currently active projector.
        #[arg(long)]
        incumbent_far: f32,
        /// False-accept rate for the candidate projector.
        #[arg(long)]
        candidate_far: f32,
        /// False-reject rate for the currently active projector.
        #[arg(long)]
        incumbent_frr: f32,
        /// False-reject rate for the candidate projector.
        #[arg(long)]
        candidate_frr: f32,
        /// Maximum allowed absolute FRR regression.
        #[arg(long, default_value_t = 0.01)]
        max_frr_regression: f32,
    },
    /// Diagnose the system (OS, desktop/PAM, GPU/EP, camera, install state). Read-only.
    Doctor,
    /// Install doorman: daemon service, models, config, and (opt-in) PAM integration.
    Install {
        /// Print every planned action and mutate nothing.
        #[arg(long)]
        dry_run: bool,
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
        /// Auto-detect the execution provider (default).
        #[arg(long, conflicts_with_all = ["cpu", "rocm", "ncnn"])]
        auto: bool,
        /// Force the CPU backend.
        #[arg(long, conflicts_with_all = ["auto", "rocm", "ncnn"])]
        cpu: bool,
        /// Force the AMD ROCm backend (iGPU).
        #[arg(long, conflicts_with_all = ["auto", "cpu", "ncnn"])]
        rocm: bool,
        /// Force the ncnn-Vulkan hybrid: depth-liveness on the iGPU via Vulkan,
        /// detection + recognition on the CPU (ORT). Builds libncnn + converts the
        /// depth model at install time.
        #[arg(long, conflicts_with_all = ["auto", "cpu", "rocm"])]
        ncnn: bool,
        /// Integrate with screen unlock only (default, safest).
        #[arg(long, conflicts_with_all = ["login", "sudo", "no_pam"])]
        screen_unlock: bool,
        /// Also integrate with display-manager login (sddm). High-risk opt-in.
        #[arg(long, conflicts_with_all = ["screen_unlock", "sudo", "no_pam"])]
        login: bool,
        /// Integrate with sudo authentication. High-risk opt-in.
        #[arg(long, conflicts_with_all = ["screen_unlock", "login", "no_pam"])]
        sudo: bool,
        /// Do not touch PAM at all (service + enroll only).
        #[arg(long, conflicts_with_all = ["screen_unlock", "login", "sudo"])]
        no_pam: bool,
        /// Desktop adapter to install/configure (auto, sway, kde, gnome, none).
        #[arg(long, value_name = "DESKTOP", default_value = "auto")]
        desktop: String,
        /// HIGH-TRUST opt-in: grant the `doorman` user the logind unlock-sessions
        /// polkit action AND set proactive_unlock=true, so a recognized face (with
        /// liveness PASS) unlocks the session WITHOUT a keypress. Password remains a
        /// fallback. Default OFF (proactive unlock off; PAM is the unlock path).
        #[arg(long)]
        auto_unlock: bool,
    },
    /// Uninstall doorman: remove PAM lines, service, and unit (idempotent).
    Uninstall {
        /// Print every planned action and mutate nothing.
        #[arg(long)]
        dry_run: bool,
        /// Also remove /var/lib/doorman, /etc/doorman, and the system user.
        #[arg(long)]
        purge: bool,
    },
    /// Open the live preview window (camera + detection overlay). Thin wrapper that
    /// launches the `doorman-preview` GUI, defaulting --runtime-dir to the system
    /// daemon's socket dir. Extra args are forwarded to doorman-preview.
    Preview {
        /// Arguments forwarded to doorman-preview (e.g. --runtime-dir <dir>).
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

/// Resolved socket locations for this invocation.
struct Sockets {
    command: PathBuf,
    debug: PathBuf,
}

impl Sockets {
    fn resolve(socket: Option<PathBuf>, runtime_dir: Option<PathBuf>) -> Self {
        // --runtime-dir wins for both sockets; --socket only overrides the command
        // socket (debug is still derived from the runtime dir so `enroll` progress
        // works). This mirrors the daemon's own derivation in `--user` mode.
        let runtime = runtime_dir.clone().unwrap_or_else(default_runtime_dir);
        let command = match socket {
            Some(s) => s,
            None => {
                let user_sock = runtime.join(COMMAND_SOCKET_NAME);
                // Pure default (no --socket, no --runtime-dir): prefer the per-user
                // (XDG) socket, but if it's absent and the installed SYSTEM daemon's
                // socket exists, talk to that. Lets the CLI "just work" against a
                // system-installed daemon without needing --socket.
                if runtime_dir.is_none()
                    && !user_sock.exists()
                    && std::path::Path::new(doorman_shared::SOCKET_PATH).exists()
                {
                    PathBuf::from(doorman_shared::SOCKET_PATH)
                } else {
                    user_sock
                }
            }
        };
        let debug = runtime.join(DEBUG_SOCKET_NAME);
        Sockets { command, debug }
    }
}

fn default_runtime_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FALLBACK_RUNTIME_DIR))
}

/// A flat error with a friendly, actionable message.
struct CliError(String);

impl CliError {
    fn new(msg: impl Into<String>) -> Self {
        CliError(msg.into())
    }
}

fn main() {
    let cli = Cli::parse();
    let sockets = Sockets::resolve(cli.socket.clone(), cli.runtime_dir.clone());

    let result = match &cli.command {
        Command::Enroll { username } => cmd_enroll(&sockets, username, cli.json),
        Command::List => cmd_list(&sockets, cli.json),
        Command::Remove { username, yes } => cmd_remove(&sockets, username, *yes, cli.json),
        Command::Test { username } => cmd_test(&sockets, username, cli.json),
        Command::Status => cmd_status(&sockets, cli.json),
        Command::Train {
            data_dir,
            models_dir,
            min_confirmed,
        } => cmd_train(data_dir.as_deref(), models_dir.as_deref(), *min_confirmed, cli.json),
        Command::DatasetReport { data_dir } => cmd_dataset_report(data_dir.as_deref(), cli.json),
        Command::Promote {
            candidate_dir,
            incumbent_far,
            candidate_far,
            incumbent_frr,
            candidate_frr,
            max_frr_regression,
        } => cmd_promote(
            candidate_dir,
            *incumbent_far,
            *candidate_far,
            *incumbent_frr,
            *candidate_frr,
            *max_frr_regression,
            cli.json,
        ),
        Command::Doctor => cmd_doctor(),
        Command::Install {
            dry_run,
            yes,
            auto: _,
            cpu,
            rocm,
            ncnn,
            screen_unlock: _,
            login,
            sudo,
            no_pam,
            desktop,
            auto_unlock,
        } => cmd_install(*dry_run, *yes, *cpu, *rocm, *ncnn, *login, *sudo, *no_pam, desktop, *auto_unlock),
        Command::Uninstall { dry_run, purge } => cmd_uninstall(*dry_run, *purge),
        Command::Preview { args } => cmd_preview(args),
    };

    if let Err(CliError(msg)) = result {
        eprintln!("error: {msg}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// doctor / install / uninstall
// ---------------------------------------------------------------------------

fn cmd_doctor() -> Result<(), CliError> {
    let d = install::diagnose(EpChoice::Auto, PamScope::ScreenUnlock, DesktopChoice::Auto);
    install::print_diagnosis(&d);
    Ok(())
}

fn cmd_train(
    data_dir: Option<&Path>,
    models_dir: Option<&Path>,
    min_confirmed: usize,
    json: bool,
) -> Result<(), CliError> {
    let data_dir = data_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(doorman_shared::DATA_DIR));
    let models_dir = models_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| data_dir.join("models"));
    let report = doormand::training::run_offline_training(&data_dir, &models_dir, min_confirmed)
        .map_err(|e| CliError::new(format!("offline training refused: {e}")))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|e| CliError::new(format!("failed to render training report: {e}")))?
        );
    } else {
        println!("doorman offline training candidate");
        println!("  version: {}", report.version);
        println!("  attempts seen: {}", report.attempts_seen);
        println!("  confirmed genuine labels: {}", report.confirmed_genuine);
        println!("  rejected attempts: {}", report.rejected);
        println!("  suspected spoof attempts: {}", report.suspected_spoof);
        println!("  unknown attempts: {}", report.unknown);
        println!("  output: {}", report.output_dir.display());
        println!("  promoted: {}", if report.promoted { "yes" } else { "no" });
    }
    Ok(())
}

fn cmd_dataset_report(data_dir: Option<&Path>, json: bool) -> Result<(), CliError> {
    let data_dir = data_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(doorman_shared::DATA_DIR));
    let report = doormand::training::inspect_dataset(&data_dir)
        .map_err(|e| CliError::new(format!("dataset inspection failed: {e}")))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|e| CliError::new(format!("failed to render dataset report: {e}")))?
        );
    } else {
        println!("doorman dataset report");
        println!("  data dir: {}", data_dir.display());
        println!("  attempts seen: {}", report.attempts_seen);
        println!("  labels seen: {}", report.labels_seen);
        println!("  confirmed genuine labels: {}", report.confirmed_genuine);
        println!("  false negative attempts: {}", report.false_negative);
        println!("  rejected attempts: {}", report.rejected);
        println!("  suspected spoof attempts: {}", report.suspected_spoof);
        println!("  proactive unlocked attempts: {}", report.proactive_unlocked);
        println!("  unknown attempts: {}", report.unknown);
        println!("  matched attempts: {}", report.matched_attempts);
        println!("  unlabeled attempts: {}", report.unlabeled_attempts);
        println!("  capture packs: {}", report.capture_packs);
        println!("  capture frames: {}", report.capture_frames);
        if report.failure_reasons.is_empty() {
            println!("  failure reasons: none");
        } else {
            println!("  failure reasons:");
            for (reason, count) in report.failure_reasons {
                println!("    {reason}: {count}");
            }
        }
    }
    Ok(())
}

fn cmd_promote(
    candidate_dir: &Path,
    incumbent_far: f32,
    candidate_far: f32,
    incumbent_frr: f32,
    candidate_frr: f32,
    max_frr_regression: f32,
    json: bool,
) -> Result<(), CliError> {
    let validation = doormand::training::ValidationMetrics {
        incumbent_far,
        candidate_far,
        incumbent_frr,
        candidate_frr,
    };
    let report = doormand::training::promote_projector_candidate(
        candidate_dir,
        validation,
        max_frr_regression,
    )
    .map_err(|e| CliError::new(format!("projector promotion refused: {e}")))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|e| CliError::new(format!("failed to render promotion report: {e}")))?
        );
    } else {
        println!("doorman projector promotion");
        println!("  candidate: {}", report.candidate_dir.display());
        println!("  current: {}", report.current_path.display());
        println!("  incumbent FAR: {:.6}", report.validation.incumbent_far);
        println!("  candidate FAR: {:.6}", report.validation.candidate_far);
        println!("  incumbent FRR: {:.6}", report.validation.incumbent_frr);
        println!("  candidate FRR: {:.6}", report.validation.candidate_frr);
        println!("  promoted: {}", if report.promoted { "yes" } else { "no" });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_install(
    dry_run: bool,
    yes: bool,
    cpu: bool,
    rocm: bool,
    ncnn: bool,
    login: bool,
    sudo: bool,
    no_pam: bool,
    desktop: &str,
    auto_unlock: bool,
) -> Result<(), CliError> {
    let ep = if cpu {
        EpChoice::Cpu
    } else if rocm {
        EpChoice::Rocm
    } else if ncnn {
        EpChoice::Ncnn
    } else {
        EpChoice::Auto
    };
    let pam_scope = if no_pam {
        PamScope::None
    } else if login {
        PamScope::Login
    } else if sudo {
        PamScope::Sudo
    } else {
        PamScope::ScreenUnlock
    };
    let desktop = DesktopChoice::parse(desktop).map_err(CliError::new)?;
    install::run_install(&InstallOpts {
        dry_run,
        assume_yes: yes,
        ep,
        pam_scope,
        desktop,
        auto_unlock,
    })
    .map_err(CliError::new)
}

fn cmd_uninstall(dry_run: bool, purge: bool) -> Result<(), CliError> {
    install::run_uninstall(&UninstallOpts { dry_run, purge }).map_err(CliError::new)
}

/// Launch the `doorman-preview` GUI. Resolves the binary next to this executable
/// (installed layout) or on `$PATH`, and defaults `--runtime-dir` to the system
/// daemon's socket directory (so `doorman preview` "just works" against a system
/// install) unless the caller passed one. Extra args are forwarded verbatim.
fn cmd_preview(args: &[String]) -> Result<(), CliError> {
    let bin = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("doorman-preview")))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("doorman-preview"));

    let mut cmd = std::process::Command::new(&bin);
    // Default the runtime dir to the system daemon's socket dir if the caller didn't
    // set one and the system frame socket is present.
    let has_runtime = args
        .iter()
        .any(|a| a == "--runtime-dir" || a.starts_with("--runtime-dir="));
    if !has_runtime {
        if let Some(dir) = Path::new(doorman_shared::SOCKET_PATH).parent() {
            if dir.join("doorman-frames.sock").exists() {
                cmd.arg("--runtime-dir").arg(dir);
            }
        }
    }
    cmd.args(args);

    match cmd.status() {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(CliError::new(format!("doorman-preview exited with {s}"))),
        Err(e) => Err(CliError::new(format!(
            "failed to launch '{}': {e}. Build/install it: \
             cargo build --release --features preview --bin doorman-preview",
            bin.display()
        ))),
    }
}

/// Connect, send one request line, read one response line. This is the exact
/// framing the daemon expects: newline-delimited JSON, one round-trip per
/// connection.
fn send_request(
    socket_path: &Path,
    request: &Request,
    timeout: Duration,
) -> Result<Response, CliError> {
    let stream = UnixStream::connect(socket_path).map_err(|e| {
        CliError::new(format!(
            "could not connect to daemon socket {} ({e}). Is doormand running? \
             Start it (e.g. `doormand --user`) or point --socket/--runtime-dir at it.",
            socket_path.display()
        ))
    })?;

    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| CliError::new(format!("failed to set read timeout: {e}")))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(|e| CliError::new(format!("failed to set write timeout: {e}")))?;

    let mut writer = stream;
    let request_json = serde_json::to_string(request)
        .map_err(|e| CliError::new(format!("failed to serialize request: {e}")))?;
    writeln!(writer, "{request_json}")
        .map_err(|e| CliError::new(format!("failed to send request: {e}")))?;
    writer
        .flush()
        .map_err(|e| CliError::new(format!("failed to flush request: {e}")))?;

    let mut reader = BufReader::new(writer);
    let mut line = String::new();
    let n = reader.read_line(&mut line).map_err(|e| {
        if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut {
            CliError::new("timed out waiting for daemon response")
        } else {
            CliError::new(format!("failed to read daemon response: {e}"))
        }
    })?;
    if n == 0 {
        return Err(CliError::new(
            "daemon closed the connection without responding",
        ));
    }

    serde_json::from_str(line.trim())
        .map_err(|e| CliError::new(format!("failed to parse daemon response: {e} (raw: {})", line.trim())))
}

fn print_raw(response: &Response) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(response)
        .map_err(|e| CliError::new(format!("failed to serialize response: {e}")))?;
    println!("{json}");
    Ok(())
}

// ---------------------------------------------------------------------------
// status
// ---------------------------------------------------------------------------

fn cmd_status(sockets: &Sockets, json: bool) -> Result<(), CliError> {
    let response = send_request(&sockets.command, &Request::Status, DEFAULT_TIMEOUT)?;
    if json {
        return print_raw(&response);
    }
    match response {
        Response::Success {
            data: Some(ResponseData::DaemonStatus { info }),
            ..
        } => {
            print_status(&info);
            Ok(())
        }
        Response::Failure { reason } => Err(CliError::new(reason)),
        other => Err(CliError::new(format!("unexpected response: {other:?}"))),
    }
}

fn print_status(info: &DaemonInfo) {
    let uptime = format_uptime(info.uptime_secs);
    println!("doorman daemon status");
    println!("  version        : {}", info.version);
    println!("  uptime         : {uptime}");
    println!(
        "  camera         : {}",
        if info.camera_available { "available" } else { "unavailable" }
    );
    println!(
        "  models loaded  : {}",
        if info.models_loaded { "yes" } else { "no" }
    );
    println!("  enrolled users : {}", info.enrolled_users);
}

fn format_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        format!("{h}h {m}m {s}s")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

// ---------------------------------------------------------------------------
// list
// ---------------------------------------------------------------------------

fn cmd_list(sockets: &Sockets, json: bool) -> Result<(), CliError> {
    let response = send_request(&sockets.command, &Request::ListUsers, DEFAULT_TIMEOUT)?;
    if json {
        return print_raw(&response);
    }
    match response {
        Response::Success {
            data: Some(ResponseData::UserList { users }),
            ..
        } => {
            print_user_table(&users);
            Ok(())
        }
        Response::Failure { reason } => Err(CliError::new(reason)),
        other => Err(CliError::new(format!("unexpected response: {other:?}"))),
    }
}

fn print_user_table(users: &[UserInfo]) {
    if users.is_empty() {
        println!("No users enrolled.");
        return;
    }

    let user_h = "USERNAME";
    let enrolled_h = "ENROLLED AT";
    let emb_h = "EMBEDDINGS";

    let user_w = users
        .iter()
        .map(|u| u.username.len())
        .chain(std::iter::once(user_h.len()))
        .max()
        .unwrap_or(user_h.len());
    let enrolled_w = users
        .iter()
        .map(|u| u.enrolled_at.len())
        .chain(std::iter::once(enrolled_h.len()))
        .max()
        .unwrap_or(enrolled_h.len());

    println!("{user_h:<user_w$}  {enrolled_h:<enrolled_w$}  {emb_h}");
    println!(
        "{}  {}  {}",
        "-".repeat(user_w),
        "-".repeat(enrolled_w),
        "-".repeat(emb_h.len())
    );
    for u in users {
        println!(
            "{:<user_w$}  {:<enrolled_w$}  {}",
            u.username, u.enrolled_at, u.num_embeddings
        );
    }
    println!("\n{} user(s) enrolled.", users.len());
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------

fn cmd_remove(
    sockets: &Sockets,
    username: &str,
    yes: bool,
    json: bool,
) -> Result<(), CliError> {
    if !yes && !json {
        print!("Remove enrollment for '{username}'? [y/N] ");
        std::io::stdout()
            .flush()
            .map_err(|e| CliError::new(format!("failed to flush prompt: {e}")))?;
        let mut answer = String::new();
        std::io::stdin()
            .read_line(&mut answer)
            .map_err(|e| CliError::new(format!("failed to read confirmation: {e}")))?;
        let answer = answer.trim().to_lowercase();
        if answer != "y" && answer != "yes" {
            println!("Aborted.");
            return Ok(());
        }
    }

    let response = send_request(
        &sockets.command,
        &Request::RemoveUser {
            username: username.to_string(),
        },
        DEFAULT_TIMEOUT,
    )?;
    if json {
        return print_raw(&response);
    }
    match response {
        Response::Success { message, .. } => {
            println!("{}", message.unwrap_or_else(|| format!("Removed user: {username}")));
            Ok(())
        }
        Response::Failure { reason } => Err(CliError::new(reason)),
        other => Err(CliError::new(format!("unexpected response: {other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// test / auth
// ---------------------------------------------------------------------------

fn cmd_test(sockets: &Sockets, username: &str, json: bool) -> Result<(), CliError> {
    let response = send_request(
        &sockets.command,
        &Request::Authenticate {
            username: username.to_string(),
        },
        // Daemon captures AUTH_FRAMES frames; allow a little headroom over DEFAULT.
        Duration::from_secs(15),
    )?;
    if json {
        return print_raw(&response);
    }
    match response {
        Response::Success { message, .. } => {
            let detail = message.unwrap_or_default();
            if detail.is_empty() {
                println!("recognized: {username}");
            } else {
                println!("recognized: {username} ({detail})");
            }
            Ok(())
        }
        // Authentication failure is a normal, expected outcome — report it but do
        // not treat it as a CLI error (exit 0) so scripts can distinguish "not
        // recognized" from "daemon unreachable".
        Response::Failure { reason } => {
            println!("!recognized: {username} ({reason})");
            Ok(())
        }
        other => Err(CliError::new(format!("unexpected response: {other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// enroll
// ---------------------------------------------------------------------------

fn cmd_enroll(sockets: &Sockets, username: &str, json: bool) -> Result<(), CliError> {
    // Start a background progress watcher on the debug socket. Enrollment
    // progress is published there as StreamMessage::Enrollment frames while the
    // command connection below blocks waiting for the final reply.
    let stop = Arc::new(AtomicBool::new(false));
    let watcher = if json {
        None
    } else {
        Some(spawn_progress_watcher(
            sockets.debug.clone(),
            username.to_string(),
            stop.clone(),
        ))
    };

    if !json {
        println!("Enrolling '{username}' — look at the camera...");
    }

    let response = send_request(
        &sockets.command,
        &Request::Enroll {
            username: username.to_string(),
        },
        ENROLL_TIMEOUT,
    );

    // Stop the watcher and wait for it to finish drawing.
    stop.store(true, Ordering::SeqCst);
    if let Some(handle) = watcher {
        let _ = handle.join();
    }

    let response = response?;
    if json {
        return print_raw(&response);
    }
    match response {
        Response::Success { message, .. } => {
            println!(
                "\n✓ Enrolled '{username}'. {}",
                message.unwrap_or_default()
            );
            Ok(())
        }
        Response::Failure { reason } => Err(CliError::new(format!(
            "enrollment failed: {reason}"
        ))),
        // The daemon never streams Progress on the command socket, but handle it
        // defensively so an unexpected frame doesn't crash the CLI.
        Response::Progress { message, current, total } => Err(CliError::new(format!(
            "unexpected progress on command socket: {message} ({current}/{total})"
        ))),
    }
}

/// Connect to the debug socket and render enrollment progress until `stop` is
/// set or the socket closes. Best-effort: if the debug socket is unavailable
/// (e.g. daemon not in --preview/debug mode) we silently skip the live bar.
fn spawn_progress_watcher(
    debug_socket: PathBuf,
    username: String,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let stream = match UnixStream::connect(&debug_socket) {
            Ok(s) => s,
            Err(_) => return, // No live progress available; final result still prints.
        };
        // Short read timeout so we can poll the stop flag between frames.
        let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
        let mut reader = BufReader::new(stream);
        let mut line = String::new();

        while !stop.load(Ordering::SeqCst) {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // socket closed
                Ok(_) => {
                    if let Ok(StreamMessage::Enrollment {
                        phase,
                        current,
                        total,
                        username: msg_user,
                        ..
                    }) = serde_json::from_str::<StreamMessage>(line.trim())
                    {
                        if msg_user == username {
                            render_progress(&phase, current, total);
                        }
                    }
                }
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    continue; // poll stop flag again
                }
                Err(_) => break,
            }
        }
    })
}

fn render_progress(phase: &EnrollmentPhase, current: usize, total: usize) {
    let (label, total) = match phase {
        EnrollmentPhase::Recording => ("recording ", total.max(1)),
        EnrollmentPhase::Processing => ("processing", total.max(1)),
        EnrollmentPhase::Selecting => ("selecting ", total.max(1)),
        EnrollmentPhase::Complete => ("done      ", total.max(1)),
    };
    let frac = (current as f64 / total as f64).clamp(0.0, 1.0);
    let width = 30usize;
    let filled = (frac * width as f64).round() as usize;
    let bar: String = "#".repeat(filled) + &"-".repeat(width - filled);
    // \r keeps the bar on one line; stdout flush so it updates live.
    print!("\r  {label} [{bar}] {current}/{total}");
    let _ = std::io::stdout().flush();
}
