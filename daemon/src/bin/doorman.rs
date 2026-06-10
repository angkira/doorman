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
#[path = "doorman_install.rs"]
mod install;
use install::{EpChoice, InstallOpts, PamScope, UninstallOpts};

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
        #[arg(long, conflicts_with_all = ["cpu", "rocm"])]
        auto: bool,
        /// Force the CPU backend.
        #[arg(long, conflicts_with_all = ["auto", "rocm"])]
        cpu: bool,
        /// Force the AMD ROCm backend (iGPU).
        #[arg(long, conflicts_with_all = ["auto", "cpu"])]
        rocm: bool,
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
        Command::Doctor => cmd_doctor(),
        Command::Install {
            dry_run,
            yes,
            auto: _,
            cpu,
            rocm,
            screen_unlock: _,
            login,
            sudo,
            no_pam,
        } => cmd_install(*dry_run, *yes, *cpu, *rocm, *login, *sudo, *no_pam),
        Command::Uninstall { dry_run, purge } => cmd_uninstall(*dry_run, *purge),
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
    let d = install::diagnose(EpChoice::Auto, PamScope::ScreenUnlock);
    install::print_diagnosis(&d);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_install(
    dry_run: bool,
    yes: bool,
    cpu: bool,
    rocm: bool,
    login: bool,
    sudo: bool,
    no_pam: bool,
) -> Result<(), CliError> {
    let ep = if cpu {
        EpChoice::Cpu
    } else if rocm {
        EpChoice::Rocm
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
    install::run_install(&InstallOpts {
        dry_run,
        assume_yes: yes,
        ep,
        pam_scope,
    })
    .map_err(CliError::new)
}

fn cmd_uninstall(dry_run: bool, purge: bool) -> Result<(), CliError> {
    install::run_uninstall(&UninstallOpts { dry_run, purge }).map_err(CliError::new)
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
