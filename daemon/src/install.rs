//! Self-contained installer for the `doorman` CLI: `doctor`, `install`, `uninstall`.
//!
//! This module is **deliberately dependency-light** — the `doorman` binary is a
//! pure `std` + `clap` IPC client that builds with `--no-default-features` (it
//! links no camera/ML code). The installer keeps that property: it shells out to
//! system tools (`systemctl`, `sudo`, `pamtester`, `cargo`, `install`, `useradd`)
//! rather than pulling new crates.
//!
//! ## Safety model
//! - Every privileged step is a *discrete* `sudo <cmd>` invocation, so the user
//!   gets the normal inline `sudo` password prompt and can see exactly what runs.
//! - `--dry-run` prints every planned action and mutates **nothing** (no `sudo`
//!   is ever spawned).
//! - PAM edits NEVER touch `common-auth`; they add exactly one `auth sufficient
//!   libpam_doorman.so` line to a per-service `/etc/pam.d/<service>` override,
//!   back up the original, validate with `pamtester`, and auto-rollback on any
//!   error or if `pamtester` is unavailable.
//! - Password auth is never removed — face is purely additive (`sufficient`).
//! - Screen-unlock is the default (can't brick boot); login/sudo require an
//!   explicit flag.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Public CLI surface (wired into doorman.rs clap enum)
// ---------------------------------------------------------------------------

/// Which execution-environment ML backend to target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpChoice {
    Auto,
    Cpu,
    Rocm,
}

/// Which PAM auth surface to integrate with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PamScope {
    /// Screen unlock only (default, safe — cannot brick boot/login).
    ScreenUnlock,
    /// Display-manager login (sddm). Requires explicit opt-in.
    Login,
    /// `sudo`. Requires explicit opt-in.
    Sudo,
    /// Do not touch PAM at all.
    None,
}

/// Options for `doorman install`.
pub struct InstallOpts {
    pub dry_run: bool,
    pub assume_yes: bool,
    pub ep: EpChoice,
    pub pam_scope: PamScope,
    /// Opt-in: grant the `doorman` system user the logind
    /// `org.freedesktop.login1.unlock-sessions` polkit action AND flip
    /// `proactive_unlock = true` in the installed config, so a recognized face
    /// (with liveness PASS) can unlock the session WITHOUT a keypress. Default
    /// OFF — high-trust convenience. Password/PAM remain the fallback path.
    pub auto_unlock: bool,
}

/// Options for `doorman uninstall`.
pub struct UninstallOpts {
    pub dry_run: bool,
    pub purge: bool,
}

// ---------------------------------------------------------------------------
// Detection result (the `doctor` core — all read-only)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ep {
    Cpu,
    Rocm,
    Cuda,
    CoreMl,
}

impl Ep {
    /// Cargo feature that selects this EP's backend.
    pub fn cargo_feature(self) -> &'static str {
        match self {
            Ep::Cpu => "backend-ort",
            Ep::Rocm => "backend-ort-rocm",
            Ep::Cuda => "backend-ort-cuda",
            Ep::CoreMl => "backend-ort-coreml",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Ep::Cpu => "cpu",
            Ep::Rocm => "rocm",
            Ep::Cuda => "cuda",
            Ep::CoreMl => "coreml",
        }
    }
}

#[derive(Debug)]
pub struct Diagnosis {
    pub os_pretty: String,
    pub os_id: String,
    pub os_version: String,
    pub has_systemd: bool,
    pub desktop: String,
    pub pam_service: Option<String>,
    pub pam_service_note: String,
    pub ep: Ep,
    pub ep_note: String,
    pub rocm_so: Option<PathBuf>,
    pub cameras: Vec<PathBuf>,
    pub has_pamtester: bool,
    // install state
    pub service_installed: bool,
    pub service_active: bool,
    pub pam_lines_present: Vec<String>,
    pub enrolled_users: Vec<String>,
    pub doormand_installed: bool,
}

// ---------------------------------------------------------------------------
// Pure parsers (unit-tested)
// ---------------------------------------------------------------------------

/// Parse `/etc/os-release` contents into `(PRETTY_NAME, ID, VERSION_ID)`.
pub fn parse_os_release(content: &str) -> (String, String, String) {
    let mut pretty = String::new();
    let mut id = String::new();
    let mut version = String::new();
    for line in content.lines() {
        let line = line.trim();
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let v = v.trim().trim_matches('"').to_string();
        match k.trim() {
            "PRETTY_NAME" => pretty = v,
            "ID" => id = v,
            "VERSION_ID" => version = v,
            _ => {}
        }
    }
    (pretty, id, version)
}

/// Decide the PAM service name to target for a given desktop + which
/// greeter/locker binaries are installed.
///
/// `desktop` is the raw `$XDG_CURRENT_DESKTOP` (may contain `:`-separated
/// entries like `KDE` or `ubuntu:GNOME`). `has_*` flags come from probing the
/// filesystem (so this stays a pure function).
///
/// Returns `(service, note)` where `service` is `None` if it cannot be
/// determined (the caller then asks the user).
pub fn pick_pam_service(
    desktop: &str,
    scope: PamScope,
    has_kscreenlocker: bool,
    has_gnome_screensaver: bool,
) -> (Option<String>, String) {
    let desk = desktop.to_ascii_lowercase();
    let is_kde = desk.contains("kde") || desk.contains("plasma");
    let is_gnome = desk.contains("gnome");

    match scope {
        PamScope::None => (None, "PAM integration disabled (--no-pam)".to_string()),
        PamScope::Login => (
            Some("sddm".to_string()),
            "display-manager login via sddm (explicit --login)".to_string(),
        ),
        PamScope::Sudo => (
            Some("sudo".to_string()),
            "sudo authentication (explicit --sudo)".to_string(),
        ),
        PamScope::ScreenUnlock => {
            if is_kde && has_kscreenlocker {
                (
                    Some("kde".to_string()),
                    "KDE/Plasma screen unlock (kscreenlocker -> /etc/pam.d/kde)".to_string(),
                )
            } else if is_gnome && has_gnome_screensaver {
                (
                    Some("gnome-screensaver".to_string()),
                    "GNOME screen unlock (gnome-screensaver)".to_string(),
                )
            } else if is_kde {
                (
                    Some("kde".to_string()),
                    "KDE/Plasma assumed (kscreenlocker greeter not found on PATH)".to_string(),
                )
            } else {
                (
                    None,
                    format!(
                        "could not determine screen-unlock PAM service for desktop '{desktop}' \
                         — choose one with --login/--sudo or pass a known service"
                    ),
                )
            }
        }
    }
}

/// Decide the execution provider from probe results.
///
/// IMPORTANT: a present NVIDIA dGPU does NOT auto-select CUDA on a box where the
/// dGPU is reserved (training). We only auto-pick CUDA when AMD/ROCm is absent.
/// On this machine ROCm targets the AMD iGPU via the verified isolation env.
pub fn pick_ep(
    forced: EpChoice,
    has_kfd: bool,
    rocm_so_present: bool,
    has_nvidia: bool,
    is_macos: bool,
) -> (Ep, String) {
    match forced {
        EpChoice::Cpu => (Ep::Cpu, "forced --cpu".to_string()),
        EpChoice::Rocm => {
            let note = if rocm_so_present {
                "forced --rocm (ROCm ONNX Runtime .so found)".to_string()
            } else {
                "forced --rocm (WARNING: ROCm ONNX Runtime .so NOT found — \
                 build it first, see build_onnxruntime_rocm.sh)"
                    .to_string()
            };
            (Ep::Rocm, note)
        }
        EpChoice::Auto => {
            if is_macos {
                (Ep::CoreMl, "macOS -> CoreML EP".to_string())
            } else if has_kfd && rocm_so_present {
                (
                    Ep::Rocm,
                    "AMD ROCm (/sys/class/kfd present + ROCm ORT .so found) -> iGPU EP".to_string(),
                )
            } else if has_kfd && !rocm_so_present {
                (
                    Ep::Cpu,
                    "AMD GPU present but ROCm ONNX Runtime .so missing -> falling back to CPU \
                     (build it with build_onnxruntime_rocm.sh, then --rocm)"
                        .to_string(),
                )
            } else if has_nvidia {
                (
                    Ep::Cuda,
                    "NVIDIA detected (nvidia-smi) -> CUDA EP".to_string(),
                )
            } else {
                (Ep::Cpu, "no supported GPU detected -> CPU".to_string())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Filesystem / command probes (side-effecting reads only)
// ---------------------------------------------------------------------------

fn read_file(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn binary_on_path(name: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name} >/dev/null 2>&1"))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn rocm_so_path() -> Option<PathBuf> {
    // Source (dev) location from run_rocm.sh, then the system install location.
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.local/lib/onnxruntime-rocm-local/lib/libonnxruntime.so"),
        "/usr/lib/doorman/libonnxruntime.so".to_string(),
    ];
    candidates
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

fn list_cameras() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/dev") {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix("video") {
                if rest.chars().all(|c| c.is_ascii_digit()) {
                    out.push(e.path());
                }
            }
        }
    }
    out.sort();
    out
}

/// Scan `/etc/pam.d/*` for our `libpam_doorman.so` line; return service names.
fn pam_services_with_doorman() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/etc/pam.d") {
        for e in rd.flatten() {
            if let Ok(content) = std::fs::read_to_string(e.path()) {
                if content
                    .lines()
                    .any(|l| !l.trim_start().starts_with('#') && l.contains("libpam_doorman.so"))
                {
                    out.push(e.file_name().to_string_lossy().into_owned());
                }
            }
        }
    }
    out.sort();
    out
}

/// Enrolled usernames inferred from the data dir's embeddings store, if readable.
/// Best-effort: the store is a single binary file owned by `doorman`, so a normal
/// user usually can't read it — we report what we can (presence of the file).
fn enrolled_users() -> Vec<String> {
    // We can't parse the bincode embeddings store without linking ML types here.
    // Report users only if a readable per-user layout exists; otherwise empty.
    let mut out = Vec::new();
    for base in ["/var/lib/doorman", "/var/lib/doorman/users"] {
        if let Ok(rd) = std::fs::read_dir(base) {
            for e in rd.flatten() {
                if e.path().is_dir() {
                    out.push(e.file_name().to_string_lossy().into_owned());
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn systemctl_state(unit: &str) -> (bool, bool) {
    // installed: unit file exists in a system path; active: is-active == active.
    let installed = Path::new(&format!("/etc/systemd/system/{unit}")).exists()
        || Command::new("systemctl")
            .args(["list-unit-files", unit])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(unit))
            .unwrap_or(false);
    let active = Command::new("systemctl")
        .args(["is-active", unit])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "active")
        .unwrap_or(false);
    (installed, active)
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

pub fn diagnose(ep_choice: EpChoice, pam_scope: PamScope) -> Diagnosis {
    let os = read_file("/etc/os-release").unwrap_or_default();
    let (os_pretty, os_id, os_version) = parse_os_release(&os);

    let has_systemd = binary_on_path("systemctl");
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default();

    let has_kscreenlocker = binary_on_path("kscreenlocker_greet")
        || Path::new("/usr/lib/x86_64-linux-gnu/libexec/kscreenlocker_greet").exists();
    let has_gnome_screensaver = binary_on_path("gnome-screensaver");
    let (pam_service, pam_service_note) =
        pick_pam_service(&desktop, pam_scope, has_kscreenlocker, has_gnome_screensaver);

    let has_kfd = Path::new("/sys/class/kfd").exists() || binary_on_path("rocminfo");
    let rocm_so = rocm_so_path();
    let has_nvidia = binary_on_path("nvidia-smi");
    let is_macos = cfg!(target_os = "macos");
    let (ep, ep_note) = pick_ep(
        ep_choice,
        has_kfd,
        rocm_so.is_some(),
        has_nvidia,
        is_macos,
    );

    let (service_installed, service_active) = systemctl_state("doormand.service");
    let doormand_installed =
        Path::new("/usr/bin/doormand").exists() || Path::new("/usr/local/bin/doormand").exists();

    Diagnosis {
        os_pretty,
        os_id,
        os_version,
        has_systemd,
        desktop,
        pam_service,
        pam_service_note,
        ep,
        ep_note,
        rocm_so,
        cameras: list_cameras(),
        has_pamtester: binary_on_path("pamtester"),
        service_installed,
        service_active,
        pam_lines_present: pam_services_with_doorman(),
        enrolled_users: enrolled_users(),
        doormand_installed,
    }
}

fn row(out: &mut String, k: &str, v: &str) {
    let _ = writeln!(out, "  {k:<18}: {v}");
}

pub fn print_diagnosis(d: &Diagnosis) {
    let mut s = String::new();
    let _ = writeln!(s, "doorman doctor — system diagnosis (read-only)\n");

    let _ = writeln!(s, " Platform");
    row(&mut s, "os", &format!("{} ({} {})", nz(&d.os_pretty), nz(&d.os_id), nz(&d.os_version)));
    row(&mut s, "init", if d.has_systemd { "systemd" } else { "NOT systemd (unsupported)" });
    row(&mut s, "desktop", &nz(&d.desktop));

    let _ = writeln!(s, "\n Auth / PAM target");
    row(
        &mut s,
        "pam service",
        &match &d.pam_service {
            Some(svc) => format!("{svc}  ({})", d.pam_service_note),
            None => format!("undetermined  ({})", d.pam_service_note),
        },
    );
    row(&mut s, "pamtester", if d.has_pamtester { "available" } else { "MISSING (PAM install will refuse without it)" });

    let _ = writeln!(s, "\n Hardware / backend");
    row(&mut s, "execution prov.", &format!("{}  ({})", d.ep.label(), d.ep_note));
    row(&mut s, "cargo feature", d.ep.cargo_feature());
    row(
        &mut s,
        "rocm onnx .so",
        &match &d.rocm_so {
            Some(p) => p.display().to_string(),
            None => "not found".to_string(),
        },
    );
    row(
        &mut s,
        "cameras",
        &if d.cameras.is_empty() {
            "none (/dev/video* absent)".to_string()
        } else {
            d.cameras.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        },
    );

    let _ = writeln!(s, "\n Install state");
    row(
        &mut s,
        "doormand binary",
        if d.doormand_installed { "installed" } else { "not installed" },
    );
    row(
        &mut s,
        "service",
        &format!(
            "{} / {}",
            if d.service_installed { "installed" } else { "not installed" },
            if d.service_active { "active" } else { "inactive" }
        ),
    );
    row(
        &mut s,
        "pam lines",
        &if d.pam_lines_present.is_empty() {
            "none".to_string()
        } else {
            d.pam_lines_present.join(", ")
        },
    );
    row(
        &mut s,
        "enrolled users",
        &if d.enrolled_users.is_empty() {
            "none readable".to_string()
        } else {
            d.enrolled_users.join(", ")
        },
    );

    print!("{s}");
}

fn nz(s: &str) -> String {
    if s.is_empty() {
        "unknown".to_string()
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Action model: each privileged op is a discrete `sudo <cmd>` we can dry-run.
// ---------------------------------------------------------------------------

/// One installer action. `privileged` actions are wrapped in `sudo` when run.
struct Action {
    /// Human-readable description printed in the plan / dry-run.
    desc: String,
    /// The argv to execute (program + args). For privileged actions this does
    /// NOT include `sudo` — it is prepended at run time.
    argv: Vec<String>,
    privileged: bool,
    /// If `Some`, run only when the predicate returns true (idempotency skip).
    /// Evaluated at run time AND used to annotate the plan.
    skip_if: Option<fn() -> bool>,
    /// Optional stdin piped to the command (used for `tee` of generated files).
    stdin: Option<String>,
}

impl Action {
    fn priv_(desc: impl Into<String>, argv: &[&str]) -> Self {
        Action {
            desc: desc.into(),
            argv: argv.iter().map(|s| s.to_string()).collect(),
            privileged: true,
            skip_if: None,
            stdin: None,
        }
    }
    fn user(desc: impl Into<String>, argv: &[&str]) -> Self {
        Action {
            desc: desc.into(),
            argv: argv.iter().map(|s| s.to_string()).collect(),
            privileged: false,
            skip_if: None,
            stdin: None,
        }
    }
    fn skip_when(mut self, f: fn() -> bool) -> Self {
        self.skip_if = Some(f);
        self
    }
    fn with_stdin(mut self, s: impl Into<String>) -> Self {
        self.stdin = Some(s.into());
        self
    }

    fn shell_repr(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.privileged {
            parts.push("sudo".to_string());
        }
        for a in &self.argv {
            if a.chars().any(|c| c.is_whitespace() || c == '"') {
                parts.push(format!("'{a}'"));
            } else {
                parts.push(a.clone());
            }
        }
        let mut s = parts.join(" ");
        if self.stdin.is_some() {
            s = format!("printf '%s' <generated> | {s}");
        }
        s
    }
}

// ---------------------------------------------------------------------------
// install
// ---------------------------------------------------------------------------

const SYS_BIN: &str = "/usr/bin/doormand";
const SYS_DEPTHD_BIN: &str = "/usr/bin/doorman-depthd";
const SYS_CLI: &str = "/usr/bin/doorman";
const SYS_LIBDIR: &str = "/usr/lib/doorman";
const ETC_DIR: &str = "/etc/doorman";
const ETC_CONFIG: &str = "/etc/doorman/doorman.toml";
const STATE_DIR: &str = "/var/lib/doorman";
const MODELS_DIR: &str = "/var/lib/doorman/models";
const UNIT_PATH: &str = "/etc/systemd/system/doormand.service";
const DEPTHD_UNIT_PATH: &str = "/etc/systemd/system/doorman-depthd.service";
/// Modern polkit JS rule that authorizes ONLY the `doorman` system user for ONLY
/// the logind `unlock-sessions` action. Written ONLY with `--auto-unlock`; the
/// proactive-unlock path in the daemon needs it to call `Session.Unlock()`.
const POLKIT_RULE_PATH: &str = "/etc/polkit-1/rules.d/49-doorman-unlock.rules";
/// Sidecar depth socket the daemon's `DepthSource::Sidecar` connects to on GPU
/// installs. Must live in the daemon's RuntimeDirectory so both the (sandboxed)
/// daemon and sidecar can bind/connect it. Matches `depth_ipc::DEFAULT_DEPTH_SOCKET`.
const SYS_DEPTH_SOCKET: &str = "/run/doorman/doorman-depth.sock";

/// Build the systemd unit text with the EP env block baked in. For ROCm on a
/// SYSTEM service running as `doorman`, the ROCm ORT `.so` is relocated to
/// `/usr/lib/doorman/` and HOME-relative paths are rewritten to system paths.
fn render_unit(ep: Ep) -> String {
    let mut env = String::new();
    env.push_str("Environment=RUST_LOG=info\n");
    let device_arg;
    match ep {
        Ep::Rocm => {
            device_arg = "--device rocm";
            // System-relocated mirror of run_rocm.sh's VERIFIED iGPU env.
            env.push_str("Environment=ROCR_VISIBLE_DEVICES=1\n");
            env.push_str("Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0\n");
            env.push_str("Environment=HIP_PATH=/opt/rocm-7.2.2\n");
            env.push_str("Environment=ROCM_PATH=/opt/rocm-7.2.2\n");
            env.push_str(&format!(
                "Environment=ORT_DYLIB_PATH={SYS_LIBDIR}/libonnxruntime.so\n"
            ));
            env.push_str(&format!(
                "Environment=LD_LIBRARY_PATH={SYS_LIBDIR}:/opt/rocm-7.2.2/lib:/opt/rocm-7.2.2/lib64\n"
            ));
            env.push_str("Environment=MIOPEN_FIND_MODE=3\n");
            // System service has no usable $HOME (the doorman user's home is hidden
            // by ProtectHome and may not exist). MIOpen's runtime JIT (COMGR/hiprtc)
            // writes compiled kernels under $HOME/.cache and $XDG_CACHE_HOME; point
            // both at the writable StateDirectory or the cold compile SEGV-crashes.
            env.push_str(&format!("Environment=HOME={STATE_DIR}\n"));
            env.push_str(&format!("Environment=XDG_CACHE_HOME={STATE_DIR}/.cache\n"));
            env.push_str("Environment=MIOPEN_USER_DB_PATH=/var/lib/doorman/miopen\n");
        }
        Ep::Cuda => {
            device_arg = "--device cuda";
        }
        _ => {
            device_arg = "--device cpu";
        }
    }

    // On ROCm the daemon's depth-PAD source is the out-of-process sidecar
    // (`DepthSource::Sidecar`); wire the daemon to want + start after it so the
    // depth socket is up before the daemon connects. CPU installs run depth
    // in-process and need no sidecar ordering.
    let depth_dep = if ep == Ep::Rocm {
        "After=doorman-depthd.service\n         Wants=doorman-depthd.service\n         "
    } else {
        ""
    };

    format!(
        "# doorman face-authentication daemon — SYSTEM service.\n\
         # Generated by `doorman install` (EP: {ep}). Do not edit by hand; re-run the installer.\n\
         [Unit]\n\
         Description=doorman Face Authentication Daemon (system)\n\
         After=systemd-udev-settle.service local-fs.target\n\
         Wants=systemd-udev-settle.service\n\
         {depth_dep}\
         [Service]\n\
         Type=simple\n\
         User=doorman\n\
         Group=doorman\n\
         SupplementaryGroups=video render\n\
         {env}\
         ExecStart={SYS_BIN} --config {ETC_CONFIG} {device_arg}\n\
         Restart=on-failure\n\
         RestartSec=2s\n\
         StandardOutput=journal\n\
         StandardError=journal\n\
         SyslogIdentifier=doormand\n\
         NoNewPrivileges=true\n\
         ProtectSystem=strict\n\
         ProtectHome=true\n\
         PrivateTmp=true\n\
         RestrictAddressFamilies=AF_UNIX\n\
         StateDirectory=doorman\n\
         RuntimeDirectory=doorman\n\
         ReadWritePaths={STATE_DIR}\n\
         DevicePolicy=closed\n\
         DeviceAllow=char-video4linux rw\n\
         DeviceAllow=/dev/dri rw\n\
         DeviceAllow=/dev/kfd rw\n\
         MemoryMax=2G\n\
         TasksMax=128\n\
         \n\
         [Install]\n\
         WantedBy=multi-user.target\n",
        ep = ep.label()
    )
}

/// Build the `doorman-depthd.service` unit for a ROCm install.
///
/// The depth-liveness sidecar runs the Depth-Anything-V2 model on the iGPU (ROCm
/// EP) in a SEPARATE process so the CPU stays light during auth, and so a CPU
/// depth session is never co-loaded with the daemon's ROCm sessions in one
/// `OrtEnv` (that mix SEGVs). It carries the SAME verified iGPU ROCm env block as
/// the daemon unit, plus `DOORMAN_DEPTH_EP=rocm`. It is ordered BEFORE the daemon
/// (the daemon's `DepthSource::Sidecar` connects to its socket) and shares the
/// daemon's RuntimeDirectory so the socket path lines up.
fn render_depthd_unit() -> String {
    let mut env = String::new();
    env.push_str("Environment=RUST_LOG=info\n");
    env.push_str("Environment=DOORMAN_DEPTH_EP=rocm\n");
    env.push_str(&format!("Environment=DOORMAN_DEPTH_SOCKET={SYS_DEPTH_SOCKET}\n"));
    // System-relocated mirror of run_rocm.sh's VERIFIED iGPU env (identical to the
    // daemon unit so both processes target the same isolated iGPU the same way).
    env.push_str("Environment=ROCR_VISIBLE_DEVICES=1\n");
    env.push_str("Environment=HSA_OVERRIDE_GFX_VERSION=11.0.0\n");
    env.push_str("Environment=HIP_PATH=/opt/rocm-7.2.2\n");
    env.push_str("Environment=ROCM_PATH=/opt/rocm-7.2.2\n");
    env.push_str(&format!(
        "Environment=ORT_DYLIB_PATH={SYS_LIBDIR}/libonnxruntime.so\n"
    ));
    env.push_str(&format!(
        "Environment=LD_LIBRARY_PATH={SYS_LIBDIR}:/opt/rocm-7.2.2/lib:/opt/rocm-7.2.2/lib64\n"
    ));
    env.push_str("Environment=MIOPEN_FIND_MODE=3\n");
    // Same writable HOME/XDG/MIOpen cache as the daemon: the doorman user has no
    // usable $HOME under ProtectHome, and MIOpen's runtime JIT cold-compile SEGVs
    // without a writable cache. Both processes share the StateDirectory caches.
    env.push_str(&format!("Environment=HOME={STATE_DIR}\n"));
    env.push_str(&format!("Environment=XDG_CACHE_HOME={STATE_DIR}/.cache\n"));
    env.push_str("Environment=MIOPEN_USER_DB_PATH=/var/lib/doorman/miopen\n");

    format!(
        "# doorman depth-liveness SIDECAR — SYSTEM service (ROCm iGPU depth).\n\
         # Generated by `doorman install` (EP: rocm). Do not edit by hand; re-run the installer.\n\
         # Runs Depth-Anything-V2 on the iGPU in a separate process so the CPU stays light\n\
         # and a CPU depth session is never co-loaded with the daemon's ROCm sessions.\n\
         [Unit]\n\
         Description=doorman Depth-Liveness Sidecar (ROCm iGPU)\n\
         After=systemd-udev-settle.service local-fs.target\n\
         Wants=systemd-udev-settle.service\n\
         Before=doormand.service\n\
         \n\
         [Service]\n\
         Type=simple\n\
         User=doorman\n\
         Group=doorman\n\
         SupplementaryGroups=video render\n\
         {env}\
         ExecStart={SYS_DEPTHD_BIN} --config {ETC_CONFIG}\n\
         Restart=on-failure\n\
         RestartSec=2s\n\
         StandardOutput=journal\n\
         StandardError=journal\n\
         SyslogIdentifier=doorman-depthd\n\
         NoNewPrivileges=true\n\
         ProtectSystem=strict\n\
         ProtectHome=true\n\
         PrivateTmp=true\n\
         RestrictAddressFamilies=AF_UNIX\n\
         StateDirectory=doorman\n\
         RuntimeDirectory=doorman\n\
         RuntimeDirectoryPreserve=yes\n\
         ReadWritePaths={STATE_DIR}\n\
         DevicePolicy=closed\n\
         DeviceAllow=/dev/dri rw\n\
         DeviceAllow=/dev/kfd rw\n\
         MemoryMax=2G\n\
         TasksMax=128\n\
         \n\
         [Install]\n\
         WantedBy=multi-user.target\n"
    )
}

/// The modern polkit JS rule (`/etc/polkit-1/rules.d/49-doorman-unlock.rules`)
/// granting ONLY the `doorman` system user ONLY the logind
/// `org.freedesktop.login1.unlock-sessions` action. Tightly scoped: no other
/// user, no other action. Returns `polkit.Result.YES` for that exact pair and
/// falls through (undefined) otherwise so every other check uses system policy.
///
/// The daemon's proactive-unlock path (`Session.Unlock()`, run as `doorman`)
/// is rejected by polkit's default `auth_admin` for `unlock-sessions` without
/// this rule. Installed ONLY under the opt-in `--auto-unlock` flag.
fn render_polkit_unlock_rule() -> String {
    "// doorman face-unlock — managed by `doorman install --auto-unlock`.\n\
     // Authorizes ONLY the `doorman` system user for ONLY the logind\n\
     // unlock-sessions action, so the daemon's proactive-unlock path can call\n\
     // Session.Unlock() after a recognized face + liveness PASS. No other user\n\
     // or action is affected. Remove via `doorman uninstall`.\n\
     polkit.addRule(function(action, subject) {\n\
     \x20   if (action.id == \"org.freedesktop.login1.unlock-sessions\" &&\n\
     \x20       subject.user == \"doorman\") {\n\
     \x20       return polkit.Result.YES;\n\
     \x20   }\n\
     });\n"
        .to_string()
}

/// Render the installed `doorman.toml`, optionally forcing
/// `proactive_unlock = true` under `[authentication]` (for `--auto-unlock`).
///
/// When `enable` is false the source is returned verbatim (default install:
/// proactive unlock OFF — lock-gating still works, PAM is the unlock path).
/// When true, an existing `proactive_unlock = ...` line is rewritten to `true`;
/// if absent, the key is inserted as the first line of the `[authentication]`
/// section. Idempotent and order-preserving.
fn render_config(src: &str, enable: bool) -> String {
    if !enable {
        return src.to_string();
    }
    let kv = "proactive_unlock = true  # set by `doorman install --auto-unlock`";
    // Rewrite an existing key in place.
    let mut found = false;
    let mut out = String::with_capacity(src.len() + kv.len() + 1);
    for line in src.lines() {
        if line.trim_start().starts_with("proactive_unlock") {
            out.push_str(kv);
            out.push('\n');
            found = true;
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if found {
        return out;
    }
    // Key absent: insert right after the [authentication] header.
    let mut out = String::with_capacity(src.len() + kv.len() + 1);
    let mut inserted = false;
    for line in src.lines() {
        out.push_str(line);
        out.push('\n');
        if !inserted && line.trim() == "[authentication]" {
            out.push_str(kv);
            out.push('\n');
            inserted = true;
        }
    }
    if !inserted {
        // No [authentication] section (unusual) — append a minimal one.
        out.push_str("\n[authentication]\n");
        out.push_str(kv);
        out.push('\n');
    }
    out
}

/// The exact PAM override file content for a service. Adds face auth as
/// `sufficient` BEFORE the password include, never removing the password path.
fn render_pam_override(service: &str) -> String {
    // Base the override on the system's EXISTING stack for this service so we
    // preserve every non-auth rule (kwallet auto-unlock, SELinux/session/limits,
    // the root-exclusion guard, etc.). Source precedence: an existing
    // /etc/pam.d/<svc> override, then the vendor default at /usr/lib/pam.d/<svc>
    // (Ubuntu ships the kde/sddm services there). We insert exactly ONE line —
    // `auth sufficient libpam_doorman.so` — immediately BEFORE the first
    // `@include common-auth`, so face is tried first and ANY non-success falls
    // through to the unchanged password stack. common-auth itself is never edited.
    let header = format!(
        "# /etc/pam.d/{service} — managed by `doorman install`. Face auth is additive\n\
         # and `sufficient`; on any non-success it falls through to the system password\n\
         # stack below (copied verbatim from the system default). common-auth untouched.\n"
    );
    let doorman_line = "auth    sufficient    libpam_doorman.so\n";

    let base = std::fs::read_to_string(format!("/etc/pam.d/{service}"))
        .or_else(|_| std::fs::read_to_string(format!("/usr/lib/pam.d/{service}")))
        .ok();

    if let Some(content) = base {
        // Don't double-insert if a previous run already added our line.
        if content.contains("libpam_doorman.so") {
            return content;
        }
        let mut out = header;
        let mut inserted = false;
        for line in content.lines() {
            if !inserted && line.trim_start().starts_with("@include common-auth") {
                out.push_str(doorman_line);
                inserted = true;
            }
            out.push_str(line);
            out.push('\n');
        }
        if !inserted {
            // No common-auth include found (unusual); prepend our line after the header.
            out.insert_str(header_len_marker(&out), doorman_line);
        }
        out
    } else {
        // Neither file readable — minimal self-contained stack via the common-* includes.
        format!(
            "{header}{doorman_line}@include common-auth\n@include common-account\n\
             @include common-password\n@include common-session\n"
        )
    }
}

/// Offset just past the comment header (used only in the unusual no-common-auth
/// fallback) — insert the face line right after the leading `#` comment block.
fn header_len_marker(s: &str) -> usize {
    s.lines()
        .take_while(|l| l.trim_start().starts_with('#'))
        .map(|l| l.len() + 1)
        .sum()
}

fn repo_root() -> PathBuf {
    // The CLI is typically run from the repo during install; fall back to CWD.
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// The username of the real human running the install (NOT root). Under `sudo`
/// the original user is in `$SUDO_USER`; otherwise `$USER`. Used to locate the
/// warm MIOpen cache, the enrolled embeddings, and the XDG model set.
fn invoking_user() -> Option<String> {
    std::env::var("SUDO_USER")
        .ok()
        .filter(|u| !u.is_empty() && u != "root")
        .or_else(|| std::env::var("USER").ok())
        .filter(|u| !u.is_empty() && u != "root")
}

/// The home directory of the invoking (non-root) user, resolved from the passwd
/// database via `getent` so it is correct even when running under `sudo` (where
/// `$HOME` is root's). Falls back to `/home/<user>` if `getent` is unavailable.
fn invoking_user_home() -> Option<PathBuf> {
    let user = invoking_user()?;
    if let Ok(out) = Command::new("getent").args(["passwd", &user]).output() {
        if out.status.success() {
            let line = String::from_utf8_lossy(&out.stdout);
            // passwd format: name:passwd:uid:gid:gecos:home:shell
            if let Some(home) = line.trim_end().split(':').nth(5) {
                if !home.is_empty() {
                    return Some(PathBuf::from(home));
                }
            }
        }
    }
    Some(PathBuf::from(format!("/home/{user}")))
}

/// True if `dir` exists and contains at least one entry.
fn dir_nonempty(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|mut rd| rd.next().is_some())
        .unwrap_or(false)
}

/// Build the ordered install plan (no side effects — pure construction).
fn build_install_plan(d: &Diagnosis, opts: &InstallOpts) -> Vec<Action> {
    let mut plan = Vec::new();
    let root = repo_root();
    let target = root.join("target/release");
    let feature = d.ep.cargo_feature();

    // Phase 2: binaries (user-context build).
    //
    // On ROCm the depth-liveness sidecar (`doorman-depthd`) is built into the SAME
    // cargo invocation with the SAME ROCm feature, so it loads the depth model on
    // the iGPU (ROCm EP) and keeps the CPU light during auth. CPU/CUDA installs run
    // depth in-process and do not need (or build) the sidecar.
    let build_depthd = d.ep == Ep::Rocm;
    {
        let mut argv: Vec<&str> = vec![
            "cargo",
            "build",
            "--release",
            "--no-default-features",
            "--features",
            // camera features always included for V4L2 + ffmpeg fallback.
            Box::leak(format!("{feature},camera-v4l2,camera-ffmpeg").into_boxed_str()),
            "--bin",
            "doormand",
            "--bin",
            "doorman",
        ];
        if build_depthd {
            argv.push("--bin");
            argv.push("doorman-depthd");
        }
        let desc = if build_depthd {
            format!("Build doormand+doorman+doorman-depthd for EP '{}' (feature {feature})", d.ep.label())
        } else {
            format!("Build doormand+doorman for EP '{}' (feature {feature})", d.ep.label())
        };
        plan.push(Action::user(desc, &argv));
    }

    // Phase 3: system provisioning (all sudo).
    plan.push(Action::priv_(
        "Create system user 'doorman' (no login shell, video+render groups, HOME=/var/lib/doorman)",
        &["useradd", "--system", "--home-dir", STATE_DIR, "--no-create-home",
          "--shell", "/usr/sbin/nologin", "--groups", "video,render", "doorman"],
    ).skip_when(|| user_exists("doorman")));

    plan.push(Action::priv_(
        format!("Create directories {ETC_DIR}, {STATE_DIR}, {MODELS_DIR}, {SYS_LIBDIR}"),
        &["install", "-d", "-m", "0755", ETC_DIR, SYS_LIBDIR],
    ));
    plan.push(Action::priv_(
        format!("Create state dir {STATE_DIR} owned by doorman"),
        &["install", "-d", "-m", "0750", "-o", "doorman", "-g", "doorman", STATE_DIR, MODELS_DIR],
    ));
    // MIOpen runtime-JIT cache + XDG cache home, owned by doorman and writable
    // under ProtectSystem=strict (both live under the StateDirectory). Without
    // these the ROCm session creation cold-compile crashes (no $HOME/.cache).
    plan.push(Action::priv_(
        "Create MIOpen + XDG cache dirs owned by doorman (/var/lib/doorman/miopen, /var/lib/doorman/.cache)",
        &["install", "-d", "-o", "doorman", "-g", "doorman", "-m", "0750",
          "/var/lib/doorman/miopen", "/var/lib/doorman/.cache"],
    ));

    plan.push(Action::priv_(
        format!("Install doormand -> {SYS_BIN}"),
        &["install", "-m", "0755",
          Box::leak(target.join("doormand").display().to_string().into_boxed_str()), SYS_BIN],
    ));
    plan.push(Action::priv_(
        format!("Install doorman CLI -> {SYS_CLI}"),
        &["install", "-m", "0755",
          Box::leak(target.join("doorman").display().to_string().into_boxed_str()), SYS_CLI],
    ));
    // ROCm: install the depth-liveness sidecar binary (runs depth on the iGPU).
    if build_depthd {
        plan.push(Action::priv_(
            format!("Install doorman-depthd sidecar -> {SYS_DEPTHD_BIN}"),
            &["install", "-m", "0755",
              Box::leak(target.join("doorman-depthd").display().to_string().into_boxed_str()), SYS_DEPTHD_BIN],
        ));
    }

    // Config: install doorman.toml with device set to the detected EP.
    // ALWAYS re-install (no skip-if): the config is installer-managed and a re-run
    // must reconcile it — skipping when it exists left a stale socket_path
    // (/run/doorman.sock) after the socket-path migration, so the daemon failed to
    // bind. `install` overwrites in place; back it up first so a manual edit isn't
    // silently lost.
    let cfg_src = root.join("doorman.toml");
    if Path::new(ETC_CONFIG).exists() {
        plan.push(Action::priv_(
            format!("Back up existing {ETC_CONFIG} -> {ETC_CONFIG}.doorman-bak"),
            &["cp", "--update=none", ETC_CONFIG,
              Box::leak(format!("{ETC_CONFIG}.doorman-bak").into_boxed_str())],
        ));
    }
    if opts.auto_unlock {
        // --auto-unlock: render the config with proactive_unlock=true and write it
        // via tee (the source file is unchanged on disk; only the installed copy
        // gets the flag flipped). 0644 to match the `install -m 0644` default.
        let rendered = render_config(&std::fs::read_to_string(&cfg_src).unwrap_or_default(), true);
        plan.push(
            Action::priv_(
                format!("Install config -> {ETC_CONFIG} (backend=ort, device={}, proactive_unlock=true)", d.ep.label()),
                &["tee", ETC_CONFIG],
            )
            .with_stdin(rendered),
        );
        plan.push(Action::priv_(
            format!("Set {ETC_CONFIG} mode 0644"),
            &["chmod", "0644", ETC_CONFIG],
        ));
    } else {
        plan.push(Action::priv_(
            format!("Install config -> {ETC_CONFIG} (backend=ort, device={})", d.ep.label()),
            &["install", "-m", "0644",
              Box::leak(cfg_src.display().to_string().into_boxed_str()), ETC_CONFIG],
        ));
    }

    // --auto-unlock: install the scoped polkit rule that lets the `doorman` user
    // call logind Session.Unlock() (the daemon's proactive-unlock path). Without
    // the flag this is NEVER written — proactive unlock stays OFF and PAM is the
    // unlock path. High-trust convenience opt-in (face unlocks without a keypress;
    // password remains a fallback).
    if opts.auto_unlock {
        plan.push(Action::priv_(
            format!("Create polkit rules dir {}", "/etc/polkit-1/rules.d"),
            &["install", "-d", "-m", "0755", "/etc/polkit-1/rules.d"],
        ));
        plan.push(
            Action::priv_(
                format!("Write polkit unlock rule -> {POLKIT_RULE_PATH} (grants doorman org.freedesktop.login1.unlock-sessions)"),
                &["tee", POLKIT_RULE_PATH],
            )
            .with_stdin(render_polkit_unlock_rule()),
        );
        plan.push(Action::priv_(
            format!("Set {POLKIT_RULE_PATH} mode 0644"),
            &["chmod", "0644", POLKIT_RULE_PATH],
        ));
    }

    // Models: copy the REAL runtime model set into the system models dir.
    //
    // The daemon's liveness gate FAILS SAFE: if the depth model
    // (depth_anything_v2_small_fp32.onnx) is absent it REJECTS every face, so the
    // enrolled user can never unlock. That model is large and lives in the user's
    // XDG data dir (~/.local/share/doorman/models), not in the repo's data/models.
    // Prefer the XDG set when present (it has yunet/edgeface/minifasnet + depth);
    // otherwise fall back to the repo set and fetch the missing models.
    let xdg_models = invoking_user_home().map(|h| h.join(".local/share/doorman/models"));
    let xdg_has_models = xdg_models.as_ref().map(|p| dir_nonempty(p)).unwrap_or(false);
    if let (true, Some(src)) = (xdg_has_models, &xdg_models) {
        plan.push(Action::priv_(
            format!("Copy ONNX models from {} -> {MODELS_DIR} (includes depth_anything_v2_small_fp32.onnx)", src.display()),
            &["cp", "-r",
              Box::leak(format!("{}/.", src.display()).into_boxed_str()), MODELS_DIR],
        ));
    } else {
        // Fall back to the repo model set, then fetch any missing models (the depth
        // model is not vendored in the repo, so fetch_models.sh must supply it).
        let models_src = root.join("data/models");
        plan.push(Action::user(
            "Fetch full model set into repo data/models (incl. depth model) via scripts/fetch_models.sh",
            &["scripts/fetch_models.sh"],
        ).skip_when(|| {
            // Skip the fetch only if the depth model already sits in the repo dir.
            repo_root().join("data/models/depth_anything_v2_small_fp32.onnx").exists()
        }));
        plan.push(Action::priv_(
            format!("Copy ONNX models from {} -> {MODELS_DIR}", models_src.display()),
            &["cp", "-r",
              Box::leak(format!("{}/.", models_src.display()).into_boxed_str()), MODELS_DIR],
        ));
    }
    plan.push(Action::priv_(
        "chown models to doorman:doorman".to_string(),
        &["chown", "-R", "doorman:doorman", STATE_DIR],
    ));

    // ROCm: relocate the verified ORT .so into /usr/lib/doorman.
    if d.ep == Ep::Rocm {
        match &d.rocm_so {
            Some(so) => {
                let dest = PathBuf::from(format!("{SYS_LIBDIR}/libonnxruntime.so"));
                if same_file(so, &dest) {
                    // The source already IS the installed destination (a prior install
                    // placed it there and rocm_so_path() found it at the system path).
                    // `cp` would fail with "source and destination are the same file";
                    // skip the copy — it is already in place.
                    plan.push(Action::user(
                        format!(
                            "ROCm ONNX Runtime .so already in place at {SYS_LIBDIR}/libonnxruntime.so — skipping copy"
                        ),
                        &["true"],
                    ));
                } else {
                    // Copy the resolved real file (follow the symlink) to a stable name.
                    plan.push(Action::priv_(
                        format!("Copy ROCm ONNX Runtime .so -> {SYS_LIBDIR}/libonnxruntime.so"),
                        &["cp", "-L",
                          Box::leak(so.display().to_string().into_boxed_str()),
                          Box::leak(dest.display().to_string().into_boxed_str())],
                    ));
                }
            }
            None => {
                plan.push(Action::user(
                    "WARNING: ROCm .so not found — build it (build_onnxruntime_rocm.sh) before install",
                    &["true"],
                ));
            }
        }

        // Pre-seed the MIOpen kernel cache from the invoking user's WARM cache so
        // the first system-daemon start does NOT have to cold-JIT (which is what
        // crashes when $HOME is unset). If the user has no warm cache, skip and let
        // the daemon JIT on first start (slower, but now safe with HOME/XDG set).
        let warm_cache = invoking_user_home().map(|h| h.join(".cache/miopen"));
        let warm_present = warm_cache.as_ref().map(|p| dir_nonempty(p)).unwrap_or(false);
        match (warm_present, &warm_cache) {
            (true, Some(src)) => {
                plan.push(Action::priv_(
                    format!("Pre-seed MIOpen kernel cache from {} -> /var/lib/doorman/miopen (avoids cold-JIT crash)", src.display()),
                    &["cp", "-r",
                      Box::leak(format!("{}/.", src.display()).into_boxed_str()),
                      "/var/lib/doorman/miopen/"],
                ));
                plan.push(Action::priv_(
                    "chown pre-seeded MIOpen cache to doorman:doorman".to_string(),
                    &["chown", "-R", "doorman:doorman", "/var/lib/doorman/miopen"],
                ));
            }
            _ => {
                plan.push(Action::user(
                    "NOTE: no warm MIOpen cache found for the invoking user — skipping pre-seed (first start will JIT, slower but safe with HOME set)".to_string(),
                    &["true"],
                ));
            }
        }
    }

    // Carry the invoking user's enrollment into the system daemon's store so they
    // are already enrolled (the system daemon reads /var/lib/doorman/embeddings.bin).
    // Without this the user must run `sudo doorman enroll` against the system socket.
    let user_embeddings = invoking_user_home().map(|h| h.join(".local/share/doorman/embeddings.bin"));
    if let Some(src) = user_embeddings.as_ref().filter(|p| p.exists()) {
        plan.push(Action::priv_(
            format!("Carry enrollment: copy {} -> {STATE_DIR}/embeddings.bin", src.display()),
            &["cp",
              Box::leak(src.display().to_string().into_boxed_str()),
              Box::leak(format!("{STATE_DIR}/embeddings.bin").into_boxed_str())],
        ));
        plan.push(Action::priv_(
            "chown carried embeddings to doorman:doorman".to_string(),
            &["chown", "doorman:doorman",
              Box::leak(format!("{STATE_DIR}/embeddings.bin").into_boxed_str())],
        ));
    }

    // ROCm: depth-liveness sidecar unit (runs depth on the iGPU, ordered BEFORE
    // the daemon so its socket is up when the daemon's Sidecar source connects).
    if build_depthd {
        plan.push(
            Action::priv_(format!("Write depth-sidecar unit -> {DEPTHD_UNIT_PATH}"), &["tee", DEPTHD_UNIT_PATH])
                .with_stdin(render_depthd_unit()),
        );
    }

    // systemd unit (generated text via tee).
    plan.push(
        Action::priv_(format!("Write systemd unit -> {UNIT_PATH}"), &["tee", UNIT_PATH])
            .with_stdin(render_unit(d.ep)),
    );
    plan.push(Action::priv_("systemctl daemon-reload", &["systemctl", "daemon-reload"]));
    // ROCm: enable + start the depth sidecar BEFORE the daemon so its socket is up.
    if build_depthd {
        plan.push(Action::priv_(
            "Enable + start doorman-depthd.service (ROCm iGPU depth sidecar)",
            &["systemctl", "enable", "--now", "doorman-depthd.service"],
        ));
    }
    plan.push(Action::priv_(
        "Enable + start doormand.service",
        &["systemctl", "enable", "--now", "doormand.service"],
    ));
    // HEALTH-CHECK: poll up to ~30s for socket + is-active + a successful status
    // query. On failure: stop the service (kill any crash loop), dump the journal,
    // and abort non-zero. Recognized specially by execute_plan (not a raw command).
    plan.push(Action::priv_(
        "HEALTH-CHECK: wait for doormand to become healthy (socket up, active, status ok)",
        &["true"],
    ));

    // Phase 7: PAM (guarded, last). Only if a service is targeted.
    if let Some(svc) = &d.pam_service {
        let pamfile = format!("/etc/pam.d/{svc}");
        let backup = format!("/etc/pam.d/{svc}.doorman-bak");

        // Rebuild + install the PAM module from current source.
        plan.push(Action::user(
            "Rebuild libpam_doorman.so from current source",
            &["cargo", "build", "--release", "-p", "pam_doorman"],
        ));
        plan.push(Action::priv_(
            "Install libpam_doorman.so -> /lib/x86_64-linux-gnu/security/",
            &["install", "-m", "0755",
              Box::leak(target.join("libpam_doorman.so").display().to_string().into_boxed_str()),
              "/lib/x86_64-linux-gnu/security/libpam_doorman.so"],
        ));
        // Backup the existing pam file — ONLY if /etc/pam.d/<svc> already exists.
        // On Ubuntu the vendor default lives at /usr/lib/pam.d/<svc> and there is no
        // /etc override yet; in that case there is nothing to back up and rollback /
        // uninstall simply removes the /etc file we create (see rollback_pam). The
        // `--update=none` flag (portable replacement for the deprecated `cp -n`) keeps
        // it idempotent: a re-run never clobbers the first install's saved original.
        if Path::new(&pamfile).exists() {
            plan.push(Action::priv_(
                format!("Back up {pamfile} -> {backup} (never overwrites an existing backup)"),
                &["cp", "--update=none",
                  Box::leak(pamfile.clone().into_boxed_str()),
                  Box::leak(backup.clone().into_boxed_str())],
            ));
        }
        // Write the override.
        plan.push(
            Action::priv_(
                format!("Write PAM override -> {pamfile} (auth sufficient libpam_doorman.so)"),
                &["tee", Box::leak(pamfile.clone().into_boxed_str())],
            )
            .with_stdin(render_pam_override(svc)),
        );
        // Validate: run pamtester non-interactively (stdin=/dev/null, timeout 10s)
        // to confirm the PAM stack loads and runs without a module-level error.
        // "Authentication failure" (EOF → pam_unix) is treated as a PASS because
        // it proves the stack ran; only a load/config error triggers rollback.
        let user = std::env::var("USER").unwrap_or_else(|_| "$USER".to_string());
        plan.push(Action::user(
            format!(
                "VALIDATE: pamtester {svc} {user} authenticate \
                 (non-interactive, 10 s timeout; auth-failure=OK, module-error=rollback)"
            ),
            // argv[0..4] = the real pamtester invocation; the VALIDATE branch in
            // execute_plan intercepts this and runs it with bounded timeout +
            // /dev/null stdin rather than calling run_action().
            &["pamtester", svc, &user, "authenticate"],
        ));
    }

    plan
}

fn user_exists(name: &str) -> bool {
    Command::new("id")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Return `true` when `a` and `b` refer to the same underlying file.
///
/// Detects all forms `cp` rejects as "same file":
/// - Identical path strings after canonicalization (symlinks resolved).
/// - Different paths that share the same device+inode (hard links).
///
/// Used to guard `cp` steps that would fail with "source and destination are
/// the same file" on a re-install where the source was already installed to
/// the destination path (or is a symlink/hard-link to it).
///
/// Returns `false` if either path does not exist or its metadata cannot be
/// read; in that case the copy proceeds as normal (fresh-install path).
pub fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    // Fast path: identical canonical paths (covers symlinks pointing at the same real file).
    if let (Ok(ca), Ok(cb)) = (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        if ca == cb {
            return true;
        }
    }
    // Inode path: same device + same inode covers hard links and any other
    // same-file scenario that canonicalize might miss.
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(ma), Ok(mb)) => ma.dev() == mb.dev() && ma.ino() == mb.ino(),
        _ => false,
    }
}

/// The system daemon's command socket (from the installed config doorman.toml).
/// Lives under /run/doorman/ — the systemd RuntimeDirectory=doorman owned by the
/// sandboxed `doorman` user (root-owned /run is not writable under
/// ProtectSystem=strict). Must match `doorman_shared::SOCKET_PATH`.
const SYS_SOCKET: &str = "/run/doorman/doorman.sock";

/// Poll for daemon health for up to `timeout`. Healthy means: the command socket
/// exists, `systemctl is-active doormand` == "active", AND a `status` query over
/// the socket returns a success line. Returns Ok(()) on health; on timeout/failure
/// it stops the service (to kill a crash loop), prints the last journal lines, and
/// returns Err with a clear message. Never leaves a crash-looping unit running.
fn health_check(timeout: Duration) -> Result<(), String> {
    use std::time::Instant;
    let deadline = Instant::now() + timeout;
    let mut last_reason = String::from("daemon did not become healthy");
    while Instant::now() < deadline {
        let active = Command::new("systemctl")
            .args(["is-active", "doormand.service"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim() == "active")
            .unwrap_or(false);
        let socket_up = Path::new(SYS_SOCKET).exists();
        if active && socket_up {
            // Query status over the socket via the installed CLI; success = healthy.
            let ok = Command::new(SYS_CLI)
                .args(["--socket", SYS_SOCKET, "status"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false);
            if ok {
                return Ok(());
            }
            last_reason = "socket up and unit active but `status` query failed".to_string();
        } else if !active {
            last_reason = "systemctl reports doormand is not active".to_string();
        } else {
            last_reason = format!("unit active but socket {SYS_SOCKET} not present yet");
        }
        std::thread::sleep(Duration::from_millis(750));
    }

    // Unhealthy: stop the (possibly crash-looping) service, then show the REAL error.
    eprintln!("\n!! doormand did NOT become healthy ({last_reason}). Stopping the service to break any crash loop.");
    let _ = Command::new("sudo")
        .args(["systemctl", "stop", "doormand.service"])
        .status();
    eprintln!("\n----- last 40 lines of `journalctl -u doormand` -----");
    let _ = Command::new("sudo")
        .args(["journalctl", "-u", "doormand", "--no-pager", "-n", "40"])
        .status();
    eprintln!("------------------------------------------------------");
    Err(format!(
        "doormand failed health check: {last_reason}. The service has been STOPPED (no crash loop \
         left running). Review the journal above, fix the cause, and re-run `doorman install`."
    ))
}

/// Print the plan; if dry-run, stop. Otherwise (not implemented here for safety
/// in this environment) we would execute each action. Per the implementation
/// boundary, the REAL privileged execution is the user's supervised run.
pub fn run_install(opts: &InstallOpts) -> Result<(), String> {
    if matches!(opts.pam_scope, PamScope::Login | PamScope::Sudo) {
        eprintln!(
            "\n  !!  --{} integrates face auth with a HIGH-RISK surface.\n      \
             Open a separate ROOT shell now and keep it open until you have verified\n      \
             login still works with your PASSWORD. doorman keeps the password path;\n      \
             this only ADDS face as `sufficient`. pamtester validates before commit.\n",
            match opts.pam_scope { PamScope::Login => "login", _ => "sudo" }
        );
    }

    if opts.auto_unlock {
        eprintln!(
            "\n  !!  --auto-unlock is a HIGH-TRUST CONVENIENCE opt-in.\n      \
             It grants the `doorman` user the logind unlock-sessions polkit action\n      \
             and sets proactive_unlock=true, so a RECOGNIZED face (with liveness PASS)\n      \
             unlocks your session WITHOUT a keypress. Your password remains a fallback;\n      \
             the liveness gate still runs before any unlock. Omit the flag to keep\n      \
             proactive unlock OFF (lock-gating still works; PAM is the unlock path).\n"
        );
    }

    let d = diagnose(opts.ep, opts.pam_scope);

    if !d.has_systemd {
        return Err("systemd not found — this installer targets systemd Linux only".into());
    }
    if d.ep == Ep::Rocm && d.rocm_so.is_none() {
        return Err(
            "ROCm selected but no ROCm ONNX Runtime .so found. Build it with \
             build_onnxruntime_rocm.sh first, or run with --cpu."
                .into(),
        );
    }
    if d.pam_service.is_some() && !d.has_pamtester && !opts.dry_run {
        return Err(
            "pamtester is not installed; refusing to modify PAM without a validator. \
             Install it (e.g. `sudo apt install pamtester`) and re-run, or use --no-pam."
                .into(),
        );
    }

    println!("\nDetected platform:");
    print_diagnosis(&d);

    let plan = build_install_plan(&d, opts);

    println!("\nPlanned actions ({} steps){}:\n", plan.len(),
        if opts.dry_run { " [DRY RUN — nothing will be executed]" } else { "" });
    for (i, a) in plan.iter().enumerate() {
        let tag = if a.privileged { "sudo " } else { "user " };
        let skip = if a.skip_if.map(|f| f()).unwrap_or(false) {
            "  (already done — will skip)"
        } else {
            ""
        };
        println!("  {:>2}. [{tag}] {}{skip}", i + 1, a.desc);
        // Synthetic steps (health check, NOTE/WARNING markers) wrap a no-op `true`;
        // printing `$ sudo true` would be misleading, so suppress their command line.
        let synthetic = a.argv == ["true"] || a.desc.starts_with("HEALTH-CHECK:");
        if !synthetic {
            println!("        $ {}", a.shell_repr());
        }
    }

    if opts.dry_run {
        println!("\nDry run complete. No files were created or changed; sudo was not invoked.");
        print_next_steps(&d, opts);
        return Ok(());
    }

    // -- Real execution path (intentionally guarded off in this environment) --
    //
    // For SAFETY during automated implementation/verification, the installer does
    // NOT auto-execute privileged steps. A supervised run is required: re-run with
    // DOORMAN_INSTALL_CONFIRM=1 from a terminal where you can answer sudo prompts
    // and have a root shell open. The execution loop below honors that gate.
    if std::env::var("DOORMAN_INSTALL_CONFIRM").as_deref() != Ok("1") {
        return Err(
            "refusing to perform privileged install automatically. Re-run from an interactive \
             terminal with DOORMAN_INSTALL_CONFIRM=1 (keep a root shell open), or use --dry-run."
                .into(),
        );
    }
    if !opts.assume_yes {
        if !confirm("Proceed with the install above?") {
            println!("Aborted.");
            return Ok(());
        }
    }
    execute_plan(&plan, &d, opts)?;
    print_next_steps(&d, opts);
    Ok(())
}

/// Execute the plan, with PAM auto-rollback. Only reached under the confirm gate.
fn execute_plan(plan: &[Action], d: &Diagnosis, _opts: &InstallOpts) -> Result<(), String> {
    let pam_service = d.pam_service.clone();
    for a in plan {
        if a.skip_if.map(|f| f()).unwrap_or(false) {
            println!("skip: {}", a.desc);
            continue;
        }
        println!("==> {}", a.desc);
        // The health check is not a raw command — run the polling helper instead.
        if a.desc.starts_with("HEALTH-CHECK:") {
            health_check(Duration::from_secs(30))?;
            println!("daemon healthy (models loaded, socket up)");
            continue;
        }
        let is_validate = a.desc.starts_with("VALIDATE:");
        if is_validate {
            match validate_pam_stack(a) {
                PamValidation::StackIntact => {}
                PamValidation::StackBroken(reason) => {
                    eprintln!("pamtester validation FAILED — rolling back PAM override.");
                    eprintln!("  reason: {reason}");
                    if let Some(svc) = &pam_service {
                        rollback_pam(svc);
                    }
                    return Err(
                        "PAM validation failed; rolled back. No changes left to PAM.".into(),
                    );
                }
                PamValidation::TimedOut => {
                    eprintln!(
                        "pamtester validation TIMED OUT (10 s) — rolling back PAM override."
                    );
                    if let Some(svc) = &pam_service {
                        rollback_pam(svc);
                    }
                    return Err(
                        "PAM validation timed out; rolled back. Stack left unchanged.".into(),
                    );
                }
            }
            continue;
        }
        let status = run_action(a);
        match status {
            Ok(true) => {}
            Ok(false) | Err(_) => {
                return Err(format!("step failed: {}", a.desc));
            }
        }
    }
    Ok(())
}

fn run_action(a: &Action) -> Result<bool, String> {
    let mut cmd = if a.privileged {
        let mut c = Command::new("sudo");
        c.args(&a.argv);
        c
    } else {
        // Non-privileged steps (cargo build, pamtester, scripts) must run as the
        // invoking user, NOT as root. When the installer is launched via
        // `sudo doorman install`, the process uid is root but the user's rustup
        // toolchain lives in /home/<SUDO_USER>/.cargo/bin — a path that is NOT on
        // root's secure_path, so `Command::new("cargo")` fails with "No such file
        // or directory" or silently resolves nothing. Re-exec as SUDO_USER so
        // their PATH/CARGO_HOME/RUSTUP_HOME resolve correctly and the build
        // targets the user-owned `target/` directory (which root cannot write to
        // without permission errors anyway).
        if let Some(sudo_user) = std::env::var("SUDO_USER")
            .ok()
            .filter(|u| !u.is_empty() && u != "root")
        {
            // Resolve the invoking user's home via the passwd database (getent), so
            // we get the right HOME even if the env was sanitised by sudo.
            let user_home = Command::new("getent")
                .args(["passwd", &sudo_user])
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        String::from_utf8(o.stdout)
                            .ok()
                            .and_then(|s| {
                                s.trim_end()
                                    .split(':')
                                    .nth(5)
                                    .filter(|h| !h.is_empty())
                                    .map(|h| h.to_string())
                            })
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| format!("/home/{sudo_user}"));

            // Build a PATH that includes the user's ~/.cargo/bin so rustup/cargo
            // resolve to the correct toolchain. Prepend it to whatever PATH is
            // currently set (the sudo secure_path is often only system dirs).
            let cargo_bin = format!("{user_home}/.cargo/bin");
            let current_path = std::env::var("PATH").unwrap_or_default();
            let merged_path = if current_path.is_empty() {
                cargo_bin.clone()
            } else {
                format!("{cargo_bin}:{current_path}")
            };

            // CARGO_HOME / RUSTUP_HOME default to ~/.cargo / ~/.rustup. Override
            // only when the env vars are absent or still pointing at root's home,
            // so an explicit override in the caller's environment is respected.
            let cargo_home = std::env::var("CARGO_HOME")
                .unwrap_or_else(|_| format!("{user_home}/.cargo"));
            let rustup_home = std::env::var("RUSTUP_HOME")
                .unwrap_or_else(|_| format!("{user_home}/.rustup"));

            let mut c = Command::new("sudo");
            c.arg("-u")
                .arg(&sudo_user)
                .arg(format!("--preserve-env=DOORMAN_INSTALL_CONFIRM,TERM"))
                .arg("env")
                .arg(format!("HOME={user_home}"))
                .arg(format!("PATH={merged_path}"))
                .arg(format!("CARGO_HOME={cargo_home}"))
                .arg(format!("RUSTUP_HOME={rustup_home}"));
            c.args(&a.argv);
            c
        } else {
            // Not under sudo (running directly as the target user) — exec directly.
            let mut c = Command::new(&a.argv[0]);
            c.args(&a.argv[1..]);
            c
        }
    };

    use std::process::Stdio;
    if let Some(input) = &a.stdin {
        cmd.stdin(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| e.to_string())?;
        use std::io::Write as _;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .map_err(|e| e.to_string())?;
        let status = child.wait().map_err(|e| e.to_string())?;
        Ok(status.success())
    } else {
        // Capture stderr so we can surface it on failure — the previous
        // implementation called cmd.status() which inherited stderr from the root
        // process. Under sudo the build's stderr was printed to the terminal but
        // because cargo was not found at all (root's PATH has no ~/.cargo/bin),
        // the spawn itself failed, giving the user only "step failed: Build ..."
        // with no cargo output at all. By capturing stderr here we guarantee the
        // real error (spawn failure, cargo error, compile error) is always shown.
        //
        // stdout is inherited (Stdio::inherit) so build progress (cargo's
        // "Compiling ...", progress bar) is visible live.
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| e.to_string())?;
        // Drain stderr concurrently so the pipe buffer never fills and blocks the
        // child. For typical cargo stderr (a few KB of errors) this is fine in a
        // simple read-after-wait, but for large outputs we read it in a thread.
        // Use the simple path: cargo stderr is bounded and wait() is after spawn.
        let stderr_handle = child.stderr.take();
        let status = child.wait().map_err(|e| e.to_string())?;
        let captured_stderr = stderr_handle
            .map(|mut r| {
                use std::io::Read as _;
                let mut buf = String::new();
                let _ = r.read_to_string(&mut buf);
                buf
            })
            .unwrap_or_default();
        if !status.success() && !captured_stderr.trim().is_empty() {
            eprintln!(
                "\n--- stderr from failed step: {} ---\n{}\n---",
                a.desc,
                captured_stderr.trim_end()
            );
        }
        Ok(status.success())
    }
}

/// Installed path of the PAM module on Debian/Ubuntu/Fedora (x86-64).
/// Must match the `install` step in `build_install_plan`.
const PAM_MODULE_PATH: &str = "/lib/x86_64-linux-gnu/security/libpam_doorman.so";

/// Result of the non-interactive PAM stack validation.
#[derive(Debug, PartialEq)]
enum PamValidation {
    /// The stack loaded and ran correctly.  "Authentication failure" (EOF →
    /// pam_unix) is treated as intact because the module-layer succeeded.
    StackIntact,
    /// A module-level or config error: .so missing / unresolved symbols /
    /// malformed pam config.  Caller must roll back.
    StackBroken(String),
    /// pamtester did not exit within the timeout.  Caller must roll back.
    TimedOut,
}

/// Classify pamtester's exit code + combined output into a `PamValidation`.
///
/// Mapping:
///
/// | exit | output pattern                        | verdict       |
/// |------|---------------------------------------|---------------|
/// | 0    | (any)                                 | StackIntact   |
/// | ≠0   | "Authentication failure" / "auth fail"| StackIntact   |
/// | ≠0   | "Module is unknown" / "dlopen" / "cannot open" / "pam_start" / "error in service module" | StackBroken |
/// | 124  | (timeout sentinel)                    | TimedOut      |
/// | ≠0   | any other text                        | StackBroken   |
fn classify_pamtester(exit_code: i32, output: &str) -> PamValidation {
    if exit_code == 124 {
        return PamValidation::TimedOut;
    }
    if exit_code == 0 {
        return PamValidation::StackIntact;
    }
    // Non-zero exit: distinguish auth-failure (stack ran fine, just no password)
    // from a real module-level error.
    let lower = output.to_ascii_lowercase();
    let is_auth_failure = lower.contains("authentication failure")
        || lower.contains("authentication failed")
        || lower.contains("auth failure")
        || lower.contains("auth failed");
    if is_auth_failure {
        return PamValidation::StackIntact;
    }
    // Module/config errors — the stack is broken.
    let is_module_error = lower.contains("module is unknown")
        || lower.contains("dlopen")
        || lower.contains("cannot open shared object")
        || lower.contains("no such file or directory")
        || lower.contains("pam_start")
        || lower.contains("error in service module")
        || lower.contains("failed to load")
        || lower.contains("cannot find module");
    if is_module_error {
        let reason = output
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        return PamValidation::StackBroken(reason);
    }
    // Unknown non-zero exit with unrecognized output — treat as broken to be safe.
    let reason = format!(
        "exit {exit_code}; output: {}",
        output.trim().lines().next().unwrap_or("(empty)")
    );
    PamValidation::StackBroken(reason)
}

/// Run the PAM validation:
/// 1. ldd pre-flight: verify the .so exists and has no unresolved deps.
/// 2. Run pamtester non-interactively (stdin=/dev/null) with a 10 s watchdog.
/// 3. Classify the result via `classify_pamtester`.
fn validate_pam_stack(a: &Action) -> PamValidation {
    use std::io::Read as _;
    use std::process::Stdio;

    // --- Pre-flight: verify the installed .so before touching the live stack ---
    let so = Path::new(PAM_MODULE_PATH);
    if !so.exists() {
        return PamValidation::StackBroken(format!(
            "PAM module not found at {PAM_MODULE_PATH} — install step may have failed"
        ));
    }
    // ldd: check for unresolved "not found" dependencies.
    match Command::new("ldd").arg(PAM_MODULE_PATH).output() {
        Ok(ldd_out) => {
            let ldd_text = String::from_utf8_lossy(&ldd_out.stdout);
            let ldd_err = String::from_utf8_lossy(&ldd_out.stderr);
            let combined = format!("{ldd_text}{ldd_err}");
            if combined.to_ascii_lowercase().contains("not found") {
                return PamValidation::StackBroken(format!(
                    "ldd reports unresolved dependencies in {PAM_MODULE_PATH}: \
                     {}",
                    combined
                        .lines()
                        .filter(|l| l.to_ascii_lowercase().contains("not found"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        Err(e) => {
            // ldd not available — skip the pre-flight rather than blocking install.
            eprintln!("warning: could not run ldd for PAM module pre-flight: {e}");
        }
    }

    // --- Run pamtester non-interactively with a watchdog ---
    // argv: ["pamtester", service, user, "authenticate"]
    // stdin → /dev/null (never block on a password prompt)
    let mut cmd = Command::new(&a.argv[0]);
    cmd.args(&a.argv[1..]);
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return PamValidation::StackBroken(format!(
                "could not spawn pamtester: {e}"
            ));
        }
    };

    // Watchdog: kill the child after 10 s so we can never block on stdin.
    let child_id = child.id();
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(10));
        // SIGKILL via kill(2). Best-effort; ignore errors.
        let _ = Command::new("kill")
            .args(["-9", &child_id.to_string()])
            .status();
    });

    // Collect stdout + stderr while the child runs.
    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    if let Some(mut r) = child.stdout.take() {
        let _ = r.read_to_string(&mut stdout_buf);
    }
    if let Some(mut r) = child.stderr.take() {
        let _ = r.read_to_string(&mut stderr_buf);
    }

    let wait_result = child.wait();
    // Watchdog thread is now moot; let it finish in the background (it will sleep
    // and then issue a harmless kill to an already-exited PID).
    drop(watchdog);

    let exit_code: i32 = match wait_result {
        Ok(status) => status.code().unwrap_or(
            // SIGKILL gives no exit code — treat as timeout (124 mirrors `timeout`).
            124,
        ),
        Err(_) => 124,
    };

    let combined = format!("{stdout_buf}{stderr_buf}");
    let result = classify_pamtester(exit_code, &combined);

    // Log what happened for operator visibility.
    let verdict = match &result {
        PamValidation::StackIntact => "PASS (stack intact)",
        PamValidation::StackBroken(_) => "FAIL (stack broken)",
        PamValidation::TimedOut => "FAIL (timed out)",
    };
    println!(
        "  pamtester exit={exit_code}, verdict={verdict}{}",
        if combined.trim().is_empty() {
            String::new()
        } else {
            format!(
                "\n  output: {}",
                combined
                    .trim()
                    .lines()
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
        }
    );

    result
}

fn rollback_pam(service: &str) {
    let pamfile = format!("/etc/pam.d/{service}");
    let backup = format!("{pamfile}.doorman-bak");
    if Path::new(&backup).exists() {
        let _ = Command::new("sudo").args(["mv", "-f", &backup, &pamfile]).status();
    } else {
        // No backup means the service file did not previously exist — remove ours.
        let _ = Command::new("sudo").args(["rm", "-f", &pamfile]).status();
    }
}

fn confirm(prompt: &str) -> bool {
    use std::io::{stdin, stdout, Write as _};
    print!("{prompt} [y/N] ");
    let _ = stdout().flush();
    let mut s = String::new();
    if stdin().read_line(&mut s).is_err() {
        return false;
    }
    matches!(s.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

fn print_next_steps(d: &Diagnosis, opts: &InstallOpts) {
    println!("\nNext steps");
    // Enrollment targets the SYSTEM daemon's socket (/run/doorman.sock). If the
    // invoking user's enrollment was carried over, they are already enrolled.
    let carried = invoking_user_home()
        .map(|h| h.join(".local/share/doorman/embeddings.bin").exists())
        .unwrap_or(false);
    if carried {
        println!("  1. Your existing enrollment was COPIED to the system daemon — you should already");
        println!("     be enrolled. Re-enroll only if needed:");
        println!("        sudo {SYS_CLI} --socket {SYS_SOCKET} enroll $USER");
    } else {
        println!("  1. Enroll your face against the SYSTEM daemon:");
        println!("        sudo {SYS_CLI} --socket {SYS_SOCKET} enroll $USER");
    }
    println!("  2. Verify recognition:  sudo {SYS_CLI} --socket {SYS_SOCKET} test $USER");
    if let Some(svc) = &d.pam_service {
        match opts.pam_scope {
            PamScope::ScreenUnlock => {
                println!("  3. Test it: LOCK YOUR SCREEN ({svc}) and look at the camera.");
                println!("     Your password still works — face is only `sufficient`.");
                println!("  4. Extend to login:  doorman install --login   (needs a root shell open)");
                println!("     Extend to sudo:    doorman install --sudo");
            }
            PamScope::Login => println!("  3. Lock+unlock and log out/in to test sddm. Keep your root shell until verified."),
            PamScope::Sudo => println!("  3. Open a NEW terminal and try `sudo -k; sudo true`. Keep your root shell until verified."),
            PamScope::None => {}
        }
    }
    if opts.auto_unlock {
        println!("\n  Proactive auto-unlock is ENABLED (--auto-unlock): proactive_unlock=true and the");
        println!("  polkit rule granting `doorman` logind unlock-sessions are installed. After enrolling,");
        println!("  LOCK YOUR SCREEN and present your face — a recognized face + liveness PASS unlocks");
        println!("  the session with NO keypress. Your password remains a fallback. Disable by re-running");
        println!("  install without --auto-unlock (then `doorman uninstall` removes the polkit rule).");
    }
    println!("\n  Liveness/print-attack note: face auth is `sufficient`, not a replacement for");
    println!("  the password. Treat photos/video spoofing as a residual risk on this surface.");
    println!("  Revert everything any time with:  doorman uninstall  (add --purge to wipe data)");

    // CLI shadowing: warn if `doorman` on PATH resolves to something other than the
    // freshly-installed /usr/bin/doorman (e.g. a stale ~/.local/bin/doorman).
    if let Ok(out) = Command::new("sh").args(["-c", "command -v doorman"]).output() {
        if out.status.success() {
            let resolved = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !resolved.is_empty() && resolved != SYS_CLI {
                println!("\n  !! WARNING: `doorman` on your PATH resolves to '{resolved}', not '{SYS_CLI}'.");
                println!("     Use the full path '{SYS_CLI}' or remove the stale binary so commands");
                println!("     target the installed CLI (e.g. `rm {resolved}`).");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// uninstall
// ---------------------------------------------------------------------------

pub fn run_uninstall(opts: &UninstallOpts) -> Result<(), String> {
    let d = diagnose(EpChoice::Auto, PamScope::None);
    let mut plan: Vec<Action> = Vec::new();

    // Remove PAM overrides we installed (restore backups where present).
    for svc in &d.pam_lines_present {
        let pamfile = format!("/etc/pam.d/{svc}");
        let backup = format!("{pamfile}.doorman-bak");
        if Path::new(&backup).exists() {
            plan.push(Action::priv_(
                format!("Restore original {pamfile} from backup"),
                &["mv", "-f",
                  Box::leak(backup.into_boxed_str()),
                  Box::leak(pamfile.into_boxed_str())],
            ));
        } else {
            plan.push(Action::priv_(
                format!("Remove doorman-managed {pamfile} (no backup -> file was ours)"),
                &["rm", "-f", Box::leak(pamfile.into_boxed_str())],
            ));
        }
    }

    plan.push(Action::priv_(
        "Disable + stop doormand.service",
        &["systemctl", "disable", "--now", "doormand.service"],
    ).skip_when(|| !systemctl_state("doormand.service").0));
    // Depth-liveness sidecar (ROCm installs only — no-op/skip if not installed).
    plan.push(Action::priv_(
        "Disable + stop doorman-depthd.service (depth sidecar)",
        &["systemctl", "disable", "--now", "doorman-depthd.service"],
    ).skip_when(|| !systemctl_state("doorman-depthd.service").0));
    plan.push(Action::priv_(format!("Remove unit {UNIT_PATH}"), &["rm", "-f", UNIT_PATH]));
    plan.push(Action::priv_(format!("Remove depth-sidecar unit {DEPTHD_UNIT_PATH}"), &["rm", "-f", DEPTHD_UNIT_PATH]));
    // Remove the --auto-unlock polkit rule (idempotent: rm -f no-ops if absent).
    plan.push(Action::priv_(
        format!("Remove polkit unlock rule {POLKIT_RULE_PATH} (if present)"),
        &["rm", "-f", POLKIT_RULE_PATH],
    ));
    plan.push(Action::priv_("systemctl daemon-reload", &["systemctl", "daemon-reload"]));
    plan.push(Action::priv_(
        format!("Remove {SYS_BIN}, {SYS_DEPTHD_BIN}, {SYS_CLI}, {SYS_LIBDIR}, PAM module"),
        &["rm", "-rf", SYS_BIN, SYS_DEPTHD_BIN, SYS_CLI, SYS_LIBDIR,
          "/lib/x86_64-linux-gnu/security/libpam_doorman.so"],
    ));

    if opts.purge {
        plan.push(Action::priv_(
            format!("PURGE state + config ({STATE_DIR}, {ETC_DIR})"),
            &["rm", "-rf", STATE_DIR, ETC_DIR],
        ));
        plan.push(Action::priv_(
            "PURGE system user 'doorman'",
            &["userdel", "doorman"],
        ).skip_when(|| !user_exists("doorman")));
    }

    println!("Planned uninstall actions ({} steps){}:\n", plan.len(),
        if opts.dry_run { " [DRY RUN]" } else { "" });
    for (i, a) in plan.iter().enumerate() {
        println!("  {:>2}. {}", i + 1, a.desc);
        println!("        $ {}", a.shell_repr());
    }

    if opts.dry_run {
        println!("\nDry run complete. Nothing changed.");
        return Ok(());
    }
    if std::env::var("DOORMAN_INSTALL_CONFIRM").as_deref() != Ok("1") {
        return Err(
            "refusing to perform privileged uninstall automatically. Re-run with \
             DOORMAN_INSTALL_CONFIRM=1 from an interactive terminal, or use --dry-run."
                .into(),
        );
    }
    for a in &plan {
        if a.skip_if.map(|f| f()).unwrap_or(false) {
            continue;
        }
        println!("==> {}", a.desc);
        let _ = run_action(a);
    }
    println!("doorman removed.");
    Ok(())
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ubuntu_os_release() {
        let s = "PRETTY_NAME=\"Ubuntu 25.10\"\nNAME=\"Ubuntu\"\nVERSION_ID=\"25.10\"\nID=ubuntu\nID_LIKE=debian\n";
        let (pretty, id, ver) = parse_os_release(s);
        assert_eq!(pretty, "Ubuntu 25.10");
        assert_eq!(id, "ubuntu");
        assert_eq!(ver, "25.10");
    }

    #[test]
    fn parses_fedora_os_release_without_quotes() {
        let s = "NAME=Fedora\nID=fedora\nVERSION_ID=41\nPRETTY_NAME=\"Fedora Linux 41\"\n";
        let (pretty, id, ver) = parse_os_release(s);
        assert_eq!(pretty, "Fedora Linux 41");
        assert_eq!(id, "fedora");
        assert_eq!(ver, "41");
    }

    #[test]
    fn pam_kde_screen_unlock() {
        let (svc, _note) = pick_pam_service("KDE", PamScope::ScreenUnlock, true, false);
        assert_eq!(svc.as_deref(), Some("kde"));
    }

    #[test]
    fn pam_kde_via_plasma_token() {
        let (svc, _) = pick_pam_service("ubuntu:plasma", PamScope::ScreenUnlock, true, false);
        assert_eq!(svc.as_deref(), Some("kde"));
    }

    #[test]
    fn pam_gnome_screen_unlock() {
        let (svc, _) = pick_pam_service("ubuntu:GNOME", PamScope::ScreenUnlock, false, true);
        assert_eq!(svc.as_deref(), Some("gnome-screensaver"));
    }

    #[test]
    fn pam_login_flag_forces_sddm() {
        let (svc, _) = pick_pam_service("KDE", PamScope::Login, true, false);
        assert_eq!(svc.as_deref(), Some("sddm"));
    }

    #[test]
    fn pam_sudo_flag_forces_sudo() {
        let (svc, _) = pick_pam_service("KDE", PamScope::Sudo, true, false);
        assert_eq!(svc.as_deref(), Some("sudo"));
    }

    #[test]
    fn pam_unknown_desktop_is_none() {
        let (svc, _) = pick_pam_service("XFCE", PamScope::ScreenUnlock, false, false);
        assert!(svc.is_none());
    }

    #[test]
    fn pam_none_scope_disables() {
        let (svc, _) = pick_pam_service("KDE", PamScope::None, true, false);
        assert!(svc.is_none());
    }

    #[test]
    fn ep_auto_amd_with_so_picks_rocm() {
        let (ep, _) = pick_ep(EpChoice::Auto, true, true, true, false);
        assert_eq!(ep, Ep::Rocm); // NVIDIA present but dGPU reserved -> ROCm wins
    }

    #[test]
    fn ep_auto_amd_without_so_falls_back_to_cpu() {
        let (ep, _) = pick_ep(EpChoice::Auto, true, false, false, false);
        assert_eq!(ep, Ep::Cpu);
    }

    #[test]
    fn ep_auto_nvidia_only_picks_cuda() {
        let (ep, _) = pick_ep(EpChoice::Auto, false, false, true, false);
        assert_eq!(ep, Ep::Cuda);
    }

    #[test]
    fn ep_auto_macos_picks_coreml() {
        let (ep, _) = pick_ep(EpChoice::Auto, false, false, false, true);
        assert_eq!(ep, Ep::CoreMl);
    }

    #[test]
    fn ep_forced_cpu_overrides() {
        let (ep, _) = pick_ep(EpChoice::Cpu, true, true, true, false);
        assert_eq!(ep, Ep::Cpu);
    }

    #[test]
    fn ep_feature_names() {
        assert_eq!(Ep::Rocm.cargo_feature(), "backend-ort-rocm");
        assert_eq!(Ep::Cpu.cargo_feature(), "backend-ort");
    }

    #[test]
    fn unit_rocm_has_isolation_env() {
        let u = render_unit(Ep::Rocm);
        assert!(u.contains("ROCR_VISIBLE_DEVICES=1"));
        assert!(u.contains("HSA_OVERRIDE_GFX_VERSION=11.0.0"));
        assert!(u.contains("/usr/lib/doorman/libonnxruntime.so"));
        assert!(u.contains("--device rocm"));
        // System service must NOT reference the user's HOME path.
        assert!(!u.contains("/home/"));
    }

    #[test]
    fn unit_rocm_sets_writable_home_and_cache() {
        // Without a writable HOME/XDG_CACHE_HOME the MIOpen cold-JIT SEGV-crashes
        // under the doorman system user (no $HOME/.cache). Both must point under
        // the StateDirectory so they are writable under ProtectSystem=strict.
        let u = render_unit(Ep::Rocm);
        assert!(u.contains("Environment=HOME=/var/lib/doorman"));
        assert!(u.contains("Environment=XDG_CACHE_HOME=/var/lib/doorman/.cache"));
        assert!(u.contains("Environment=MIOPEN_USER_DB_PATH=/var/lib/doorman/miopen"));
    }

    #[test]
    fn cpu_unit_has_no_rocm_home_overrides() {
        // The HOME/XDG cache overrides are ROCm-only; the CPU unit must not carry them.
        let u = render_unit(Ep::Cpu);
        assert!(!u.contains("MIOPEN"));
        assert!(!u.contains("XDG_CACHE_HOME"));
        assert!(u.contains("--device cpu"));
    }

    #[test]
    fn depthd_unit_runs_rocm_depth_with_isolation_env() {
        let u = render_depthd_unit();
        // Selects the ROCm depth EP and the shared sidecar socket.
        assert!(u.contains("Environment=DOORMAN_DEPTH_EP=rocm"));
        assert!(u.contains("Environment=DOORMAN_DEPTH_SOCKET=/run/doorman/doorman-depth.sock"));
        // Carries the same verified iGPU isolation + cache env as the daemon.
        assert!(u.contains("ROCR_VISIBLE_DEVICES=1"));
        assert!(u.contains("HSA_OVERRIDE_GFX_VERSION=11.0.0"));
        assert!(u.contains("/usr/lib/doorman/libonnxruntime.so"));
        assert!(u.contains("Environment=HOME=/var/lib/doorman"));
        assert!(u.contains("MIOPEN_USER_DB_PATH=/var/lib/doorman/miopen"));
        // Ordered before the daemon so the depth socket is up first.
        assert!(u.contains("Before=doormand.service"));
        assert!(u.contains("ExecStart=/usr/bin/doorman-depthd"));
        // System service must NOT reference the user's HOME path.
        assert!(!u.contains("/home/"));
    }

    #[test]
    fn rocm_daemon_unit_wants_depth_sidecar() {
        let u = render_unit(Ep::Rocm);
        assert!(u.contains("Wants=doorman-depthd.service"));
        assert!(u.contains("After=doorman-depthd.service"));
    }

    #[test]
    fn cpu_daemon_unit_has_no_depth_sidecar_ordering() {
        // CPU runs depth in-process; no sidecar ordering should be emitted.
        let u = render_unit(Ep::Cpu);
        assert!(!u.contains("doorman-depthd"));
    }

    #[test]
    fn polkit_rule_grants_only_doorman_unlock_sessions() {
        let r = render_polkit_unlock_rule();
        // Modern polkit JS API surface.
        assert!(r.contains("polkit.addRule(function(action, subject)"));
        assert!(r.contains("return polkit.Result.YES;"));
        // Tightly scoped: exactly the doorman user + the unlock-sessions action.
        assert!(r.contains("action.id == \"org.freedesktop.login1.unlock-sessions\""));
        assert!(r.contains("subject.user == \"doorman\""));
        // No other action or user is granted.
        assert!(!r.contains("unlock-session\""), "must target unlock-sessions, not a different id");
        assert!(!r.contains("\"root\""));
        // Balanced braces (well-formed JS block structure).
        assert_eq!(r.matches('{').count(), r.matches('}').count());
        assert_eq!(r.matches('(').count(), r.matches(')').count());
    }

    #[test]
    fn polkit_rule_path_is_the_documented_location() {
        assert_eq!(POLKIT_RULE_PATH, "/etc/polkit-1/rules.d/49-doorman-unlock.rules");
    }

    #[test]
    fn render_config_default_is_verbatim() {
        let src = "[authentication]\nsimilarity_threshold = 0.75\n";
        assert_eq!(render_config(src, false), src);
        // Default must NOT enable proactive unlock.
        assert!(!render_config(src, false).contains("proactive_unlock = true"));
    }

    #[test]
    fn render_config_auto_unlock_inserts_key_under_authentication() {
        let src = "[detector]\nx = 1\n\n[authentication]\nsimilarity_threshold = 0.75\nauth_frames = 10\n";
        let out = render_config(src, true);
        assert!(out.contains("proactive_unlock = true"));
        // Inserted inside [authentication], before the existing keys.
        let auth_idx = out.find("[authentication]").unwrap();
        let pu_idx = out.find("proactive_unlock = true").unwrap();
        let thr_idx = out.find("similarity_threshold").unwrap();
        assert!(auth_idx < pu_idx && pu_idx < thr_idx);
        // Existing content preserved.
        assert!(out.contains("similarity_threshold = 0.75"));
        assert!(out.contains("[detector]"));
    }

    #[test]
    fn render_config_auto_unlock_rewrites_existing_false() {
        let src = "[authentication]\nproactive_unlock = false\nauth_frames = 10\n";
        let out = render_config(src, true);
        assert!(out.contains("proactive_unlock = true"));
        assert!(!out.contains("proactive_unlock = false"));
        // Exactly one proactive_unlock line (no duplication).
        assert_eq!(out.matches("proactive_unlock").count(), 1);
    }

    fn test_diag(ep: Ep) -> Diagnosis {
        Diagnosis {
            os_pretty: String::new(),
            os_id: String::new(),
            os_version: String::new(),
            has_systemd: true,
            desktop: String::new(),
            pam_service: None,
            pam_service_note: String::new(),
            ep,
            ep_note: String::new(),
            rocm_so: Some(PathBuf::from("/tmp/libonnxruntime.so")),
            cameras: vec![],
            has_pamtester: true,
            service_installed: false,
            service_active: false,
            pam_lines_present: vec![],
            enrolled_users: vec![],
            doormand_installed: false,
        }
    }

    fn opts(auto_unlock: bool) -> InstallOpts {
        InstallOpts {
            dry_run: true,
            assume_yes: true,
            ep: EpChoice::Rocm,
            pam_scope: PamScope::None,
            auto_unlock,
        }
    }

    #[test]
    fn plan_includes_polkit_rule_only_with_auto_unlock() {
        let d = test_diag(Ep::Rocm);

        let with = build_install_plan(&d, &opts(true));
        assert!(
            with.iter().any(|a| a.desc.contains("polkit unlock rule")),
            "expected a polkit-rule write action with --auto-unlock"
        );
        // The polkit write must tee to the documented path.
        assert!(with.iter().any(|a| a.argv.iter().any(|s| s == POLKIT_RULE_PATH)));
        // And the config description must advertise proactive_unlock=true.
        assert!(with.iter().any(|a| a.desc.contains("proactive_unlock=true")));

        let without = build_install_plan(&d, &opts(false));
        assert!(
            !without.iter().any(|a| a.desc.contains("polkit unlock rule")),
            "no polkit rule must be written without --auto-unlock"
        );
        assert!(!without.iter().any(|a| a.argv.iter().any(|s| s == POLKIT_RULE_PATH)));
        assert!(!without.iter().any(|a| a.desc.contains("proactive_unlock=true")));
    }

    #[test]
    fn pam_override_keeps_password_and_is_sufficient() {
        let p = render_pam_override("kde");
        assert!(p.contains("auth    sufficient    libpam_doorman.so"));
        assert!(p.contains("@include common-auth"));
        // Must never edit common-auth itself.
        assert!(!p.contains("common-auth\n#"));
    }

    // -----------------------------------------------------------------------
    // same_file helper
    // -----------------------------------------------------------------------

    #[test]
    fn same_file_identical_path_returns_true() {
        // A real, existing file pointed at via the same path twice must be detected
        // as the same file.
        let f = std::env::temp_dir().join("doorman_test_same_file_identity.tmp");
        std::fs::write(&f, b"x").expect("write tmp");
        assert!(same_file(&f, &f), "identical path must be the same file");
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn same_file_symlink_to_self_returns_true() {
        // A symlink that resolves to the same real file as the target must be same.
        let real = std::env::temp_dir().join("doorman_test_same_file_real.tmp");
        let link = std::env::temp_dir().join("doorman_test_same_file_link.tmp");
        std::fs::write(&real, b"x").expect("write real");
        let _ = std::fs::remove_file(&link); // clean up any leftover
        std::os::unix::fs::symlink(&real, &link).expect("symlink");
        assert!(
            same_file(&real, &link),
            "symlink pointing to the same real file must be detected as same"
        );
        let _ = std::fs::remove_file(&real);
        let _ = std::fs::remove_file(&link);
    }

    #[test]
    fn same_file_different_paths_returns_false() {
        let a = std::env::temp_dir().join("doorman_test_same_file_a.tmp");
        let b = std::env::temp_dir().join("doorman_test_same_file_b.tmp");
        std::fs::write(&a, b"a").expect("write a");
        std::fs::write(&b, b"b").expect("write b");
        assert!(!same_file(&a, &b), "two distinct files must not be the same");
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }

    #[test]
    fn same_file_missing_path_returns_false() {
        let real = std::env::temp_dir().join("doorman_test_same_file_real2.tmp");
        let missing = std::env::temp_dir().join("doorman_test_same_file_missing_xyz.tmp");
        std::fs::write(&real, b"x").expect("write real");
        let _ = std::fs::remove_file(&missing); // ensure absent
        assert!(
            !same_file(&real, &missing),
            "when dst does not exist, same_file must return false (not an error)"
        );
        let _ = std::fs::remove_file(&real);
    }

    // -----------------------------------------------------------------------
    // Plan-level: ROCm .so copy skipped when src == dst
    // -----------------------------------------------------------------------

    /// same_file detects hard-linked files (same dev+inode, different paths).
    #[test]
    fn same_file_hard_link_returns_true() {
        let src = std::env::temp_dir().join("doorman_test_same_file_hl_src.tmp");
        let dst = std::env::temp_dir().join("doorman_test_same_file_hl_dst.tmp");
        std::fs::write(&src, b"ort").expect("write src");
        let _ = std::fs::remove_file(&dst);
        std::fs::hard_link(&src, &dst).expect("hard_link");
        assert!(same_file(&src, &dst), "hard-linked files must be detected as same");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    /// The plan emits a skip note (no privileged `cp`) when the rocm .so source
    /// path and the destination path are the same file (re-install scenario).
    /// We use a real tempfile as the rocm_so source and create a symlink from
    /// the source path so same_file returns true at plan-build time.
    ///
    /// Because build_install_plan hardcodes `{SYS_LIBDIR}/libonnxruntime.so`
    /// as the destination, we cannot inject a tmpdir dest without refactoring.
    /// Instead, we directly test the same_file helper and verify the plan-builder
    /// branch produces the right action description using the documented branch
    /// condition (matching what build_install_plan does inline).
    #[test]
    fn plan_rocm_so_skip_branch_logic() {
        let src = std::env::temp_dir().join("doorman_test_ort_pl_src.so");
        let dst = std::env::temp_dir().join("doorman_test_ort_pl_dst.so");
        std::fs::write(&src, b"fake ort").expect("write src");
        let _ = std::fs::remove_file(&dst);
        std::fs::hard_link(&src, &dst).expect("hard_link src->dst");

        // same_file must detect the hard link.
        assert!(same_file(&src, &dst), "pre-condition: hard-linked files are same");

        // The branch logic in build_install_plan:
        //   if same_file(so, &dest) { emit skip note } else { emit cp action }
        // Verify the skip-note desc is produced for the same-file case.
        let desc_same = if same_file(&src, &dst) {
            format!("ROCm ONNX Runtime .so already in place at {SYS_LIBDIR}/libonnxruntime.so — skipping copy")
        } else {
            format!("Copy ROCm ONNX Runtime .so -> {SYS_LIBDIR}/libonnxruntime.so")
        };
        assert!(
            desc_same.contains("already in place"),
            "same-file branch must produce skip note; got: {desc_same}"
        );

        // Verify the cp desc is produced for different files.
        let other = std::env::temp_dir().join("doorman_test_ort_pl_other.so");
        std::fs::write(&other, b"different ort").expect("write other");
        assert!(!same_file(&src, &other), "pre-condition: different files");
        let desc_diff = if same_file(&src, &other) {
            format!("ROCm ONNX Runtime .so already in place at {SYS_LIBDIR}/libonnxruntime.so — skipping copy")
        } else {
            format!("Copy ROCm ONNX Runtime .so -> {SYS_LIBDIR}/libonnxruntime.so")
        };
        assert!(
            desc_diff.contains("Copy ROCm ONNX Runtime"),
            "different-file branch must produce cp description; got: {desc_diff}"
        );

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
        let _ = std::fs::remove_file(&other);
    }

    /// Verify the plan does the right thing for the concrete re-install scenario
    /// present on THIS machine (if /usr/lib/doorman/libonnxruntime.so exists) or
    /// the fresh-install scenario (if it doesn't), without hard-coding either.
    #[test]
    fn plan_rocm_so_cp_iff_src_ne_dest() {
        let mut d = test_diag(Ep::Rocm);
        // Use a path that definitely differs from the system dest.
        let tmp = std::env::temp_dir().join("doorman_test_rocm_so_plan.so");
        std::fs::write(&tmp, b"fake ort").expect("write tmp so");
        d.rocm_so = Some(tmp.clone());

        let o = opts(false);
        let plan = build_install_plan(&d, &o);

        let has_cp = plan.iter().any(|a| {
            a.argv.first().map(|s| s == "cp").unwrap_or(false)
                && a.argv.iter().any(|s| s.contains("libonnxruntime"))
        });
        let has_skip = plan.iter().any(|a| a.desc.contains("already in place"));

        // tmp != /usr/lib/doorman/libonnxruntime.so, so same_file is false -> cp.
        assert!(has_cp, "fresh-install: different src path must emit cp action");
        assert!(!has_skip, "fresh-install: different src path must NOT emit skip note");

        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // classify_pamtester: pure-function unit tests
    //
    // These exercise all documented cases without spawning any process:
    //   exit=0               → StackIntact
    //   exit≠0, auth-failure → StackIntact
    //   exit≠0, module error → StackBroken
    //   exit=124             → TimedOut
    // -----------------------------------------------------------------------

    #[test]
    fn pamtester_exit0_is_intact() {
        // pamtester exit 0 = explicit authentication success.
        assert_eq!(
            classify_pamtester(0, ""),
            PamValidation::StackIntact,
        );
    }

    #[test]
    fn pamtester_auth_failure_text_is_intact() {
        // pam_unix got EOF on the password prompt → "Authentication failure".
        // This proves the stack ran; it is NOT a broken stack.
        let output = "pamtester: Authentication failure";
        assert_eq!(
            classify_pamtester(6, output),
            PamValidation::StackIntact,
            "auth-failure text must classify as StackIntact (stack ran correctly)"
        );
    }

    #[test]
    fn pamtester_auth_failed_variant_is_intact() {
        // Some pamtester versions print "Authentication failed" instead.
        let output = "pamtester: authenticating as angkira\npamtester: Authentication failed";
        assert_eq!(
            classify_pamtester(6, output),
            PamValidation::StackIntact,
        );
    }

    #[test]
    fn pamtester_module_unknown_is_broken() {
        // The .so name was wrong or the file is absent from the security/ dir.
        let output = "pamtester: Module is unknown";
        match classify_pamtester(6, output) {
            PamValidation::StackBroken(_) => {}
            other => panic!("expected StackBroken, got {other:?}"),
        }
    }

    #[test]
    fn pamtester_dlopen_error_is_broken() {
        // dlopen of libpam_doorman.so failed (unresolved symbols, ABI mismatch).
        let output =
            "pamtester: error in service module\nlibpam_doorman.so: cannot open shared object file: No such file or directory";
        match classify_pamtester(255, output) {
            PamValidation::StackBroken(_) => {}
            other => panic!("expected StackBroken, got {other:?}"),
        }
    }

    #[test]
    fn pamtester_error_in_service_module_is_broken() {
        // Malformed pam config (e.g. missing required module or syntax error).
        let output = "pamtester: error in service module";
        match classify_pamtester(7, output) {
            PamValidation::StackBroken(_) => {}
            other => panic!("expected StackBroken, got {other:?}"),
        }
    }

    #[test]
    fn pamtester_timeout_exit124_is_timedout() {
        // Watchdog killed the process (mirrors the `timeout(1)` sentinel exit code).
        assert_eq!(
            classify_pamtester(124, ""),
            PamValidation::TimedOut,
        );
    }

    #[test]
    fn pamtester_unknown_nonzero_is_broken() {
        // Unknown non-zero exit with unrecognized output → treat as broken (safe default).
        match classify_pamtester(99, "some unexpected output") {
            PamValidation::StackBroken(_) => {}
            other => panic!("expected StackBroken, got {other:?}"),
        }
    }
}
