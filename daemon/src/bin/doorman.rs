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
    DaemonInfo, EnrollmentPhase, Request, RequestEnvelope, Response, ResponseData, StreamMessage,
    UserInfo,
};

// Self-contained installer (doctor / install / uninstall). Kept in a sibling
// file so this bin stays a pure std+clap client that still builds with
// --no-default-features.
#[path = "../install.rs"]
mod install;
use install::{DesktopChoice, EpChoice, InstallOpts, PamScope, UninstallOpts};

// Terminal side of `calibrate`, `pad run` and `config get|set|unset`.
#[path = "../cli_pad.rs"]
mod cli_pad;

// Event-driven unlock helper for screen lockers (`doorman unlock-watch`).
#[path = "../unlock_watch.rs"]
mod unlock_watch;

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
enum ConfigAction {
    /// Parse and validate the config and report unknown or deprecated keys.
    /// Exit codes: 0 ok, 1 invalid, 2 valid but unknown/deprecated keys.
    Check {
        /// Config file to check (default: the file the daemon would load).
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
    /// Print every config key with its default as Markdown (docs/CONFIG.md).
    DumpDefaults,
    /// Show the effective value of a key (or of every key of a section).
    Get {
        /// Dotted key, for example `ipc.face_cooldown_secs`, or a section.
        key: String,
        /// Config file (default: the file the daemon would load).
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
    /// Set a key in /etc/doorman/doorman.toml (root). The key and value are
    /// validated like the daemon does, comments are kept, the file is replaced
    /// atomically after a backup. Never restarts anything.
    Set {
        /// Dotted key, for example `ipc.face_max_judged_failures`.
        key: String,
        /// TOML value: `7`, `true`, `"text"` (a bare word works for strings), `[10, 30, 120]`.
        value: String,
        /// Edit this file instead of the system config.
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
    /// Remove a key from /etc/doorman/doorman.toml (root) so its default applies.
    Unset {
        key: String,
        /// Edit this file instead of the system config.
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum PadAction {
    /// Guided presentation-attack screening against the live camera: genuine
    /// baseline, phone photo / video, print, mixed, lighting, glasses, yaw,
    /// distance. Needs no root. Run `sudo doorman calibrate` first for a
    /// session without pauses. Screening, not certification.
    Run {
        /// Enrolled user to test (default: you).
        #[arg(long)]
        user: Option<String>,
        /// Short smoke run: 3 genuine, 2 per attack class, 2 per condition.
        #[arg(long)]
        quick: bool,
        /// Result file prefix (default: ~/doorman-pad-<date>).
        #[arg(long, value_name = "PREFIX")]
        out: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ModelsAction {
    /// Hash the model files the daemon loads and compare them with the
    /// integrity manifest (built-in, plus `[models] manifest_path`). Read-only,
    /// needs no daemon. Exit codes: 0 all verified, 1 mismatch or error,
    /// 2 no mismatch but some files unlisted or missing.
    Verify {
        /// Models directory (default: `[ml] models_dir`).
        #[arg(long, value_name = "DIR")]
        models_dir: Option<PathBuf>,
        /// Extra manifest to merge over the built-in one (default:
        /// `[models] manifest_path`).
        #[arg(long, value_name = "FILE")]
        manifest: Option<PathBuf>,
        /// Print the loaded files in `sha256sum` format (for a user manifest)
        /// instead of verifying. Only run it on files you trust.
        #[arg(long)]
        print_manifest: bool,
    },
}

#[derive(Subcommand)]
enum DatasetAction {
    /// Delete dataset files (attempt/label shards, capture packs) older than N days.
    /// Root only; never run automatically.
    Purge {
        /// Delete entries whose UTC day is more than this many days old
        /// (default: `[dataset] retention_days`).
        #[arg(long, value_name = "DAYS")]
        older_than: Option<u32>,
        /// List what would be deleted and delete nothing.
        #[arg(long)]
        dry_run: bool,
        /// Data directory containing dataset/ (default: `[daemon] data_dir`).
        #[arg(long, value_name = "DIR")]
        data_dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum Command {
    /// Enroll a new user (records from the daemon's current camera source).
    Enroll {
        /// Username to enroll.
        username: String,
        /// Add templates (for example another lighting condition) to the
        /// existing ones without evicting any.
        #[arg(long)]
        add: bool,
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
    /// Authenticate a user against the live camera (alias: auth). Exit code: 0
    /// recognized, 2 not recognized (output starts with `!recognized:`), 1 error.
    #[command(alias = "auth")]
    Test {
        /// Username to authenticate.
        username: String,
    },
    /// Show daemon status (version, uptime, camera, models, enrolled users,
    /// calibration mode).
    Status,
    /// Calibration mode for measuring presentation attacks: root only, time
    /// limited, never persisted. `sudo doorman calibrate` starts it (default 45
    /// minutes), `--end` ends it, `--status` shows it (any user).
    Calibrate {
        /// Session length: 45m, 2700s, 1h30m (bare number = minutes). Max 120m.
        #[arg(long, value_name = "TIME", conflicts_with_all = ["end", "status"])]
        ttl: Option<String>,
        /// End calibration mode now.
        #[arg(long, conflicts_with = "status")]
        end: bool,
        /// Show whether calibration mode is active and for how long.
        #[arg(long)]
        status: bool,
    },
    /// Presentation-attack screening (see `doorman pad run --help`).
    Pad {
        #[command(subcommand)]
        action: PadAction,
    },
    /// Unlock a screen locker on face match: waits on the daemon (WaitForMatch)
    /// and SIGTERMs the given locker PID. Exit codes: see docs/unlock-path.md.
    UnlockWatch {
        /// User whose face unlocks the session.
        #[arg(long)]
        user: String,
        /// PID of the locker process the lock script started.
        #[arg(long, value_name = "PID")]
        locker_pid: i32,
        /// Expected process name of the locker (/proc/<pid>/comm); nothing is
        /// signalled if the PID now belongs to another process.
        #[arg(long, value_name = "NAME")]
        locker_name: Option<String>,
        /// Length of one WaitForMatch call; calls are re-issued while the locker lives.
        #[arg(long, value_name = "MS", default_value_t = unlock_watch::DEFAULT_WAIT_TIMEOUT_MS)]
        timeout_ms: u64,
        /// Time the locker gets to exit after SIGTERM before SIGKILL.
        #[arg(long, value_name = "MS", default_value_t = unlock_watch::DEFAULT_TERM_GRACE_MS)]
        term_grace_ms: u64,
    },
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
    /// Manage stored dataset files (see docs/datasets.md).
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },
    /// Verify the model files against the integrity manifest (docs/MODELS.md).
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    /// Validate or document the daemon config (see docs/CONFIG.md).
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Diagnose the system (OS, desktop/PAM, GPU/EP, camera, install state). Read-only.
    Doctor,
    /// From-source developer installer (rebuilds from a checkout; daemon service,
    /// models, config, opt-in PAM). Not for upgrading a packaged (.deb) install:
    /// use the package for that.
    Install {
        /// Print every planned action and mutate nothing.
        #[arg(long)]
        dry_run: bool,
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
        /// Default: CPU on Linux (CoreML on macOS). A GPU is never picked
        /// automatically; use --rocm, --cuda or --ncnn to opt in.
        #[arg(long, conflicts_with_all = ["cpu", "rocm", "cuda", "ncnn"])]
        auto: bool,
        /// Force the CPU backend.
        #[arg(long, conflicts_with_all = ["auto", "rocm", "cuda", "ncnn"])]
        cpu: bool,
        /// Opt in to the AMD ROCm backend (iGPU).
        #[arg(long, conflicts_with_all = ["auto", "cpu", "cuda", "ncnn"])]
        rocm: bool,
        /// Opt in to the NVIDIA CUDA backend.
        #[arg(long, conflicts_with_all = ["auto", "cpu", "rocm", "ncnn"])]
        cuda: bool,
        /// Force the ncnn-Vulkan hybrid: depth-liveness on the iGPU via Vulkan,
        /// detection + recognition on the CPU (ORT). Builds libncnn + converts the
        /// depth model at install time.
        #[arg(long, conflicts_with_all = ["auto", "cpu", "rocm", "cuda"])]
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

fn is_root() -> bool {
    doorman_shared::config_trust::effective_uid() == 0
}

/// A request sender for the `cli_pad` commands: refusals come back as errors
/// with the daemon's own reason.
fn transport(sockets: &Sockets) -> impl Fn(&Request, Duration) -> Result<Response, String> {
    let socket = sockets.command.clone();
    move |request, timeout| match send_request_raw(&socket, request, timeout) {
        Ok(Response::Denied { reason }) => Err(format!("denied by daemon: {reason}")),
        Ok(other) => Ok(other),
        Err(CliError(msg)) => Err(msg),
    }
}

/// Turn a `cli_pad` result into the CLI's error handling and exit code.
fn exit_with(result: Result<i32, String>) -> Result<(), CliError> {
    match result {
        Ok(0) => Ok(()),
        Ok(code) => std::process::exit(code),
        Err(e) => Err(CliError::new(e)),
    }
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
        Command::Enroll { username, add } => cmd_enroll(&sockets, username, *add, cli.json),
        Command::List => cmd_list(&sockets, cli.json),
        Command::Remove { username, yes } => cmd_remove(&sockets, username, *yes, cli.json),
        Command::Test { username } => cmd_test(&sockets, username, cli.json).map(|code| {
            if code != TEST_EXIT_RECOGNIZED {
                std::process::exit(code);
            }
        }),
        Command::Status => cmd_status(&sockets, cli.json),
        Command::Calibrate { ttl, end, status } => {
            let send = transport(&sockets);
            match cli_pad::cmd_calibrate(&send, ttl.as_deref(), *end, *status, cli.json, is_root())
            {
                Ok(0) => Ok(()),
                Ok(code) => std::process::exit(code),
                Err(e) => Err(CliError::new(e)),
            }
        }
        Command::Pad {
            action: PadAction::Run { user, quick, out },
        } => {
            let send = transport(&sockets);
            match cli_pad::cmd_pad_run(&send, user.as_deref(), *quick, out.as_deref()) {
                Ok(0) => Ok(()),
                Ok(code) => std::process::exit(code),
                Err(e) => Err(CliError::new(e)),
            }
        }
        Command::UnlockWatch {
            user,
            locker_pid,
            locker_name,
            timeout_ms,
            term_grace_ms,
        } => std::process::exit(
            unlock_watch::run(&unlock_watch::WatchArgs {
                socket: sockets.command.clone(),
                user: user.clone(),
                locker_pid: *locker_pid,
                locker_name: locker_name.clone(),
                timeout_ms: *timeout_ms,
                term_grace: Duration::from_millis(*term_grace_ms),
            })
            .code(),
        ),
        Command::Train {
            data_dir,
            models_dir,
            min_confirmed,
        } => cmd_train(
            data_dir.as_deref(),
            models_dir.as_deref(),
            *min_confirmed,
            cli.json,
        ),
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
        Command::Dataset { action } => match action {
            DatasetAction::Purge {
                older_than,
                dry_run,
                data_dir,
            } => cmd_dataset_purge(*older_than, *dry_run, data_dir.as_deref()),
        },
        Command::Models { action } => match action {
            ModelsAction::Verify {
                models_dir,
                manifest,
                print_manifest,
            } => std::process::exit(cmd_models_verify(
                models_dir.as_deref(),
                manifest.as_deref(),
                *print_manifest,
            )),
        },
        Command::Config { action } => match action {
            ConfigAction::Check { config } => {
                std::process::exit(cmd_config_check(config.as_deref()))
            }
            ConfigAction::DumpDefaults => {
                print!(
                    "{}",
                    doorman_shared::config_check::render_reference_markdown()
                );
                Ok(())
            }
            ConfigAction::Get { key, config } => {
                exit_with(cli_pad::cmd_config_get(config.as_deref(), key))
            }
            ConfigAction::Set { key, value, config } => exit_with(cli_pad::cmd_config_edit(
                config.as_deref(),
                key,
                Some(value),
                is_root(),
            )),
            ConfigAction::Unset { key, config } => exit_with(cli_pad::cmd_config_edit(
                config.as_deref(),
                key,
                None,
                is_root(),
            )),
        },
        Command::Doctor => cmd_doctor(),
        Command::Install {
            dry_run,
            yes,
            auto: _,
            cpu,
            rocm,
            cuda,
            ncnn,
            screen_unlock: _,
            login,
            sudo,
            no_pam,
            desktop,
            auto_unlock,
        } => cmd_install(
            *dry_run,
            *yes,
            *cpu,
            *rocm,
            *cuda,
            *ncnn,
            *login,
            *sudo,
            *no_pam,
            desktop,
            *auto_unlock,
        ),
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

/// Exit code: every loaded model file matches the manifest.
const MODELS_EXIT_OK: i32 = 0;
/// Exit code: a hash mismatch, an unreadable file, or an error.
const MODELS_EXIT_MISMATCH: i32 = 1;
/// Exit code: no mismatch, but a file is unlisted (unverifiable) or missing.
const MODELS_EXIT_UNVERIFIED: i32 = 2;

fn cmd_models_verify(
    models_dir: Option<&Path>,
    manifest_file: Option<&Path>,
    print_manifest: bool,
) -> i32 {
    use doormand::model_integrity::{self as mi, FileStatus};
    let config = match doorman_shared::Config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: cannot load config: {e}");
            return MODELS_EXIT_MISMATCH;
        }
    };
    let dir = models_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&config.ml.models_dir));
    let files = mi::loaded_model_files(config.authentication.liveness_enabled);
    if print_manifest {
        for name in &files {
            match mi::sha256_file(&dir.join(name)) {
                Ok(hash) => println!("{hash}  {name}"),
                Err(e) => eprintln!("# {name}: {e}"),
            }
        }
        return MODELS_EXIT_OK;
    }
    let user_manifest = manifest_file
        .map(PathBuf::from)
        .or_else(|| config.models.manifest_path.as_ref().map(PathBuf::from));
    let manifest = match mi::manifest_for(user_manifest.as_deref()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: {e}");
            return MODELS_EXIT_MISMATCH;
        }
    };
    let report = mi::verify(&dir, &files, &manifest);
    println!("models dir: {}", dir.display());
    for (name, status) in &report.files {
        match status {
            FileStatus::Verified => println!("  ok         {name}"),
            FileStatus::Mismatch { expected, actual } => {
                println!("  MISMATCH   {name}\n             expected {expected}\n             actual   {actual}")
            }
            FileStatus::Unlisted { actual } => {
                println!("  unlisted   {name} (sha256 {actual}; not in the manifest)")
            }
            FileStatus::Missing => println!("  missing    {name}"),
            FileStatus::Unreadable(e) => println!("  UNREADABLE {name}: {e}"),
        }
    }
    if !report.mismatches().is_empty() {
        MODELS_EXIT_MISMATCH
    } else if report.all_verified() {
        MODELS_EXIT_OK
    } else {
        MODELS_EXIT_UNVERIFIED
    }
}

fn cmd_dataset_purge(
    older_than: Option<u32>,
    dry_run: bool,
    data_dir: Option<&Path>,
) -> Result<(), CliError> {
    // Dataset files are root-owned 0700; refuse early with a clear message
    // instead of failing half way through a deletion.
    if !nix::unistd::geteuid().is_root() {
        return Err(CliError::new(
            "dataset purge must run as root (the dataset directory is private to the daemon); try: sudo doorman dataset purge ...",
        ));
    }
    let config = doorman_shared::Config::load()
        .map_err(|e| CliError::new(format!("cannot load config: {e}")))?;
    let days = older_than.unwrap_or(config.dataset.retention_days);
    let data_dir = data_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&config.daemon.data_dir));
    let dataset_dir = data_dir.join("dataset");
    let today = chrono::Utc::now().date_naive();
    let entries = doormand::dataset_purge::purge_candidates(&dataset_dir, today, days)
        .map_err(|e| CliError::new(format!("cannot scan {}: {e}", dataset_dir.display())))?;
    for path in &entries {
        println!(
            "{}{}",
            if dry_run {
                "would delete "
            } else {
                "deleting "
            },
            path.display()
        );
    }
    if dry_run {
        println!(
            "{} entries older than {days} days (dry run, nothing deleted)",
            entries.len()
        );
        return Ok(());
    }
    let removed = doormand::dataset_purge::delete_entries(&dataset_dir, &entries)
        .map_err(|e| CliError::new(format!("delete failed: {e}")))?;
    println!(
        "deleted {removed} entries older than {days} days from {}",
        dataset_dir.display()
    );
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
        println!(
            "  proactive unlocked attempts: {}",
            report.proactive_unlocked
        );
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
    cuda: bool,
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
    } else if cuda {
        EpChoice::Cuda
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
    match send_request_raw(socket_path, request, timeout)? {
        Response::Denied { reason } => Err(CliError::new(format!("denied by daemon: {reason}"))),
        Response::RateLimited { retry_after_s } => Err(CliError::new(format!(
            "rate limited by daemon: retry in {retry_after_s}s"
        ))),
        Response::UnsupportedProtocol {
            reason,
            supported_version,
        } => Err(CliError::new(format!(
            "daemon rejected the request: {reason} (daemon speaks protocol {supported_version})"
        ))),
        other => Ok(other),
    }
}

/// [`send_request`] without turning refusals into errors: the caller sees
/// `Denied`, `RateLimited` and `UnsupportedProtocol` as responses.
fn send_request_raw(
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
    let request_json = serde_json::to_string(&RequestEnvelope::new(request.clone()))
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

    let response: Response = serde_json::from_str(line.trim()).map_err(|e| {
        CliError::new(format!(
            "failed to parse daemon response: {e} (raw: {})",
            line.trim()
        ))
    })?;
    Ok(response)
}

fn print_raw(response: &Response) -> Result<(), CliError> {
    let json = serde_json::to_string_pretty(response)
        .map_err(|e| CliError::new(format!("failed to serialize response: {e}")))?;
    println!("{json}");
    Ok(())
}

// ---------------------------------------------------------------------------
// config check
// ---------------------------------------------------------------------------

/// Exit code: config valid, no unknown or deprecated keys.
const CONFIG_EXIT_OK: i32 = 0;
/// Exit code: config unreadable, unparsable or failing validation.
const CONFIG_EXIT_INVALID: i32 = 1;
/// Exit code: config valid but has unknown or deprecated keys.
const CONFIG_EXIT_UNKNOWN_KEYS: i32 = 2;

fn cmd_config_check(explicit: Option<&Path>) -> i32 {
    let path = match explicit {
        Some(p) => p.to_path_buf(),
        None => match doorman_shared::Config::find_config_path() {
            Some(p) => p,
            None => {
                println!("no config file found in the standard locations; built-in defaults apply");
                return CONFIG_EXIT_OK;
            }
        },
    };
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("invalid: cannot read {}: {e}", path.display());
            return CONFIG_EXIT_INVALID;
        }
    };
    let report = match doorman_shared::config_check::check_str(&content) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("invalid: {}: {e}", path.display());
            return CONFIG_EXIT_INVALID;
        }
    };
    println!(
        "{}: valid (schema_version {})",
        path.display(),
        report.config.schema_version
    );
    for key in &report.unknown_keys {
        println!("unknown key: {key}");
    }
    for (key, hint) in &report.deprecated_keys {
        println!("deprecated key: {key} ({hint})");
    }
    if report.unknown_keys.is_empty() && report.deprecated_keys.is_empty() {
        CONFIG_EXIT_OK
    } else {
        CONFIG_EXIT_UNKNOWN_KEYS
    }
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
        if info.camera_available {
            "available"
        } else {
            "unavailable"
        }
    );
    println!(
        "  models loaded  : {}",
        if info.models_loaded { "yes" } else { "no" }
    );
    println!("  enrolled users : {}", info.enrolled_users);
    if let Some(ort) = &info.ort_runtime {
        println!("  onnx runtime   : {ort}");
    }
    if let Some(left) = info.calibration_remaining_secs {
        println!(
            "  CALIBRATION    : ACTIVE, {} left (failure budget suspended; end: sudo doorman calibrate --end)",
            cli_pad::fmt_duration(left)
        );
    }
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

fn cmd_remove(sockets: &Sockets, username: &str, yes: bool, json: bool) -> Result<(), CliError> {
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
            println!(
                "{}",
                message.unwrap_or_else(|| format!("Removed user: {username}"))
            );
            Ok(())
        }
        Response::Failure { reason } => Err(CliError::new(reason)),
        other => Err(CliError::new(format!("unexpected response: {other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// test / auth
// ---------------------------------------------------------------------------

/// Exit codes of `doorman test` (see `doormand::test_report`): 0 recognized,
/// 1 error (daemon unreachable, denied, unexpected answer), 2 not recognized,
/// rejected, unavailable or suspended. The `recognized:` / `!recognized:` lines
/// on stdout are unchanged.
use doormand::test_report::{
    classify, error_report, EXIT_NOT_RECOGNIZED as TEST_EXIT_NOT_RECOGNIZED,
    EXIT_RECOGNIZED as TEST_EXIT_RECOGNIZED,
};

fn cmd_test(sockets: &Sockets, username: &str, json: bool) -> Result<i32, CliError> {
    let started = std::time::Instant::now();
    let answer = send_request_raw(
        &sockets.command,
        &Request::Authenticate {
            username: username.to_string(),
            service: None,
            tty: None,
        },
        // Daemon captures AUTH_FRAMES frames; allow a little headroom over DEFAULT.
        Duration::from_secs(15),
    );
    let duration_ms = started.elapsed().as_millis() as u64;
    if json {
        // One JSON object on stdout in every case, errors included.
        let mut report = match &answer {
            Ok(response) => classify(response, duration_ms, chrono::Utc::now().timestamp()),
            Err(_) => error_report(duration_ms),
        };
        // Loud in the data too: results taken in calibration mode say so.
        let send = transport(sockets);
        report.calibration = cli_pad::calibration_state(&send).is_some_and(|c| c.active);
        let line = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::new(format!("failed to serialize report: {e}")))?;
        println!("{line}");
        if let Err(CliError(msg)) = &answer {
            eprintln!("error: {msg}");
        }
        return Ok(report.exit_code);
    }
    let response = match answer? {
        // A suspension is an expected outcome, shown like a rejection.
        Response::Denied { reason }
            if reason.contains(doormand::face_budget::FACE_SUSPENDED_CODE) =>
        {
            println!("!recognized: {username} ({reason})");
            return Ok(TEST_EXIT_NOT_RECOGNIZED);
        }
        Response::Denied { reason } => {
            return Err(CliError::new(format!("denied by daemon: {reason}")))
        }
        Response::RateLimited { retry_after_s } => {
            return Err(CliError::new(format!(
                "rate limited by daemon: retry in {retry_after_s}s"
            )))
        }
        Response::UnsupportedProtocol {
            reason,
            supported_version,
        } => {
            return Err(CliError::new(format!(
                "daemon rejected the request: {reason} (daemon speaks protocol {supported_version})"
            )))
        }
        other => other,
    };
    match response {
        Response::Success { message, .. } => {
            let detail = message.unwrap_or_default();
            if detail.is_empty() {
                println!("recognized: {username}");
            } else {
                println!("recognized: {username} ({detail})");
            }
            Ok(TEST_EXIT_RECOGNIZED)
        }
        // Authentication failure is a normal, expected outcome: the line is not an
        // `error:`, but the exit code is non-zero so exit-code-only consumers are
        // not fooled by a rejected or camera-less attempt.
        Response::Failure { reason } => {
            println!("!recognized: {username} ({reason})");
            Ok(TEST_EXIT_NOT_RECOGNIZED)
        }
        other => Err(CliError::new(format!("unexpected response: {other:?}"))),
    }
}

// ---------------------------------------------------------------------------
// enroll
// ---------------------------------------------------------------------------

fn cmd_enroll(sockets: &Sockets, username: &str, add: bool, json: bool) -> Result<(), CliError> {
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
        println!("Hint: slowly turn your head a little and vary the lighting; use --add for another lighting condition.");
    }

    let response = send_request(
        &sockets.command,
        &Request::Enroll {
            username: username.to_string(),
            add,
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
            println!("\n✓ Enrolled '{username}'. {}", message.unwrap_or_default());
            Ok(())
        }
        Response::Failure { reason } => Err(CliError::new(format!("enrollment failed: {reason}"))),
        // The daemon never streams Progress on the command socket, but handle it
        // defensively so an unexpected frame doesn't crash the CLI.
        Response::Progress {
            message,
            current,
            total,
        } => Err(CliError::new(format!(
            "unexpected progress on command socket: {message} ({current}/{total})"
        ))),
        // send_request already turns these into errors; arm kept for exhaustiveness.
        Response::Denied { .. }
        | Response::RateLimited { .. }
        | Response::UnsupportedProtocol { .. }
        | Response::Matched { .. }
        | Response::NoMatch { .. } => {
            Err(CliError::new(format!("unexpected response: {response:?}")))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use doorman_shared::NoMatchReason;
    use doormand::test_report::EXIT_ERROR as TEST_EXIT_ERROR;

    fn test_exit_code(response: &Response) -> i32 {
        classify(response, 0, 0).exit_code
    }

    #[test]
    fn test_exit_code_is_zero_only_for_a_recognized_face() {
        let table = [
            (
                Response::Success {
                    message: None,
                    data: None,
                },
                TEST_EXIT_RECOGNIZED,
            ),
            (
                Response::Failure {
                    reason: "Camera not available (no frames)".into(),
                },
                TEST_EXIT_NOT_RECOGNIZED,
            ),
            (
                Response::Denied {
                    reason: "peer uid".into(),
                },
                TEST_EXIT_ERROR,
            ),
            (Response::RateLimited { retry_after_s: 1 }, TEST_EXIT_ERROR),
            (
                Response::Denied {
                    reason: doormand::decision::suspended_reason(30),
                },
                TEST_EXIT_NOT_RECOGNIZED,
            ),
            (
                Response::NoMatch {
                    reason: NoMatchReason::Timeout,
                },
                TEST_EXIT_ERROR,
            ),
        ];
        for (response, want) in table {
            assert_eq!(test_exit_code(&response), want, "{response:?}");
        }
    }

    // --- Round 9b-2a: calibrate, pad run, config get|set|unset ---

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut argv = vec!["doorman"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv)
    }

    #[test]
    fn calibrate_parses_ttl_end_and_status_and_rejects_mixtures() {
        match parse(&["calibrate"]).unwrap().command {
            Command::Calibrate {
                ttl: None,
                end: false,
                status: false,
            } => {}
            _ => panic!("plain calibrate"),
        }
        match parse(&["calibrate", "--ttl", "45m"]).unwrap().command {
            Command::Calibrate { ttl: Some(t), .. } => assert_eq!(t, "45m"),
            _ => panic!("ttl"),
        }
        assert!(matches!(
            parse(&["calibrate", "--end"]).unwrap().command,
            Command::Calibrate { end: true, .. }
        ));
        assert!(matches!(
            parse(&["calibrate", "--status"]).unwrap().command,
            Command::Calibrate { status: true, .. }
        ));
        assert!(parse(&["calibrate", "--end", "--status"]).is_err());
        assert!(parse(&["calibrate", "--end", "--ttl", "5m"]).is_err());
    }

    #[test]
    fn pad_run_has_sensible_defaults_and_one_optional_quick() {
        match parse(&["pad", "run"]).unwrap().command {
            Command::Pad {
                action:
                    PadAction::Run {
                        user: None,
                        quick: false,
                        out: None,
                    },
            } => {}
            _ => panic!("defaults"),
        }
        assert!(matches!(
            parse(&["pad", "run", "--quick"]).unwrap().command,
            Command::Pad {
                action: PadAction::Run { quick: true, .. }
            }
        ));
        assert!(parse(&["pad"]).is_err());
        assert!(
            parse(&["pad", "run", "--genuine", "5"]).is_err(),
            "no flag soup"
        );
    }

    #[test]
    fn config_get_set_unset_parse() {
        match parse(&["config", "set", "ipc.face_cooldown_secs", "[10, 30, 120]"])
            .unwrap()
            .command
        {
            Command::Config {
                action:
                    ConfigAction::Set {
                        key,
                        value,
                        config: None,
                    },
            } => {
                assert_eq!(
                    (key.as_str(), value.as_str()),
                    ("ipc.face_cooldown_secs", "[10, 30, 120]")
                );
            }
            _ => panic!("set"),
        }
        assert!(matches!(
            parse(&["config", "get", "ipc"]).unwrap().command,
            Command::Config {
                action: ConfigAction::Get { .. }
            }
        ));
        assert!(matches!(
            parse(&["config", "unset", "ipc.face_max_judged_failures"])
                .unwrap()
                .command,
            Command::Config {
                action: ConfigAction::Unset { .. }
            }
        ));
        assert!(parse(&["config", "set", "only.key"]).is_err());
        assert!(
            parse(&["config", "set", "k", "v", "--restart"]).is_err(),
            "no --restart flag"
        );
    }

    #[test]
    fn calibrate_from_a_non_root_user_prints_the_sudo_line_and_sends_nothing() {
        let sent = std::cell::Cell::new(0);
        let send = |_: &Request, _: Duration| -> Result<Response, String> {
            sent.set(sent.get() + 1);
            Err("must not be called".into())
        };
        assert_eq!(
            cli_pad::cmd_calibrate(&send, Some("30m"), false, false, false, false),
            Ok(1)
        );
        assert_eq!(
            cli_pad::cmd_calibrate(&send, None, true, false, false, false),
            Ok(1)
        );
        assert_eq!(sent.get(), 0);
        assert!(
            cli_pad::sudo_instruction(" --ttl 30m").contains("sudo doorman calibrate --ttl 30m")
        );
    }

    #[test]
    fn calibrate_status_works_for_anyone_and_root_sends_the_request() {
        use doorman_shared::CalibrationInfo;
        let seen = std::cell::RefCell::new(Vec::new());
        let send = |r: &Request, _: Duration| -> Result<Response, String> {
            seen.borrow_mut().push(serde_json::to_string(r).unwrap());
            Ok(Response::Success {
                message: None,
                data: Some(ResponseData::Calibration {
                    info: CalibrationInfo {
                        active: true,
                        remaining_secs: 60,
                        ttl_secs: 2700,
                        judged_failures_counted: 0,
                    },
                }),
            })
        };
        assert_eq!(
            cli_pad::cmd_calibrate(&send, None, false, true, false, false),
            Ok(0)
        );
        assert_eq!(
            cli_pad::cmd_calibrate(&send, Some("30m"), false, false, false, true),
            Ok(0)
        );
        assert_eq!(
            cli_pad::cmd_calibrate(&send, None, true, false, false, true),
            Ok(0)
        );
        assert_eq!(
            *seen.borrow(),
            vec![
                r#"{"type":"calibrate_status"}"#.to_string(),
                r#"{"type":"calibrate","ttl_secs":1800}"#.to_string(),
                r#"{"type":"calibrate_end"}"#.to_string(),
            ]
        );
        assert!(cli_pad::cmd_calibrate(&send, Some("soon"), false, false, false, true).is_err());
    }

    #[test]
    fn calibrate_surfaces_the_daemons_refusal() {
        let send = |_: &Request, _: Duration| -> Result<Response, String> {
            Ok(Response::Failure {
                reason: "ttl 99999s exceeds the maximum of 7200s (120 min)".into(),
            })
        };
        let err =
            cli_pad::cmd_calibrate(&send, Some("99999s"), false, false, false, true).unwrap_err();
        assert!(err.contains("exceeds the maximum"), "{err}");
    }
}
