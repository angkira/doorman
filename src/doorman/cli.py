#!/usr/bin/env python3
"""
doorman CLI - Management interface for the doorman face authentication system
"""

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.syntax import Syntax

from .models import ModelManager
from .model_specs import get_model_spec, print_model_spec, MODEL_SPECS

app = typer.Typer(
    name="doorman",
    help="Secure face unlock system for Linux",
    add_completion=False,
)
console = Console()

# Constants - detect if using user service or system service
def get_socket_path() -> str:
    """Get socket path - prefer user socket if it exists"""
    user_runtime_dir = os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    user_socket = f"{user_runtime_dir}/doorman.sock"
    system_socket = "/run/doorman.sock"

    # Prefer user socket if it exists
    if os.path.exists(user_socket):
        return user_socket
    elif os.path.exists(system_socket):
        return system_socket
    else:
        # Default to user socket for better error messages
        return user_socket

SOCKET_PATH = get_socket_path()
DATA_DIR = "/var/lib/doorman"
MODELS_DIR = f"{DATA_DIR}/models"
PAM_LIB_PATH = "/usr/lib/x86_64-linux-gnu/security/pam_doorman.so"
DAEMON_BIN_PATH = "/usr/local/bin/doormand"
SERVICE_FILE_PATH = "/etc/systemd/system/doormand.service"


def is_root() -> bool:
    """Check if running as root"""
    return os.geteuid() == 0


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if required dependencies are installed"""
    missing = []

    # Check for rustc and cargo
    if not shutil.which("rustc"):
        missing.append(
            "rustc (install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)"
        )
    if not shutil.which("cargo"):
        missing.append("cargo (included with rustc)")

    # Check for gcc/build tools
    if not shutil.which("gcc"):
        missing.append("gcc (install via: sudo apt install build-essential)")

    # Check for PAM development headers
    pam_header = Path("/usr/include/security/pam_appl.h")
    if not pam_header.exists():
        missing.append("libpam0g-dev (install via: sudo apt install libpam0g-dev)")

    return len(missing) == 0, missing


def send_daemon_request(request: dict, timeout: int = 5) -> Optional[dict]:
    """Send a request to the daemon via UNIX socket"""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)

        # Send request
        request_json = json.dumps(request) + "\n"
        sock.sendall(request_json.encode())

        # Receive response
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break

        sock.close()

        if data:
            return json.loads(data.decode().strip())
        return None

    except socket.timeout:
        console.print("[red]Error:[/red] Request timed out")
        return None
    except socket.error as e:
        console.print(f"[red]Error:[/red] Cannot connect to daemon: {e}")
        console.print(
            "[yellow]Hint:[/yellow] Is the daemon running? Try: sudo systemctl status doormand"
        )
        return None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return None


@app.command()
def setup(
    skip_checks: bool = typer.Option(
        False, "--skip-checks", help="Skip dependency checks"
    ),
):
    """
    Initial setup: build, install, and configure doorman
    """
    if not is_root():
        console.print("[red]Error:[/red] Setup must be run as root (use sudo)")
        raise typer.Exit(1)

    console.print(Panel.fit("🚪 doorman Setup", style="bold blue"))

    # Check dependencies
    if not skip_checks:
        console.print("\n[bold]Checking dependencies...[/bold]")
        deps_ok, missing = check_dependencies()
        if not deps_ok:
            console.print("[red]Missing dependencies:[/red]")
            for dep in missing:
                console.print(f"  • {dep}")
            raise typer.Exit(1)
        console.print("[green]✓[/green] All dependencies found")

    # Build Rust components
    console.print("\n[bold]Building Rust components...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Compiling (this may take a few minutes)...", total=None
        )

        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            console.print(f"[red]Build failed:[/red]\n{result.stderr}")
            raise typer.Exit(1)

        progress.update(task, completed=True)

    console.print("[green]✓[/green] Build successful")

    # Install binaries
    console.print("\n[bold]Installing binaries...[/bold]")
    project_root = Path(__file__).parent.parent.parent

    # Install PAM module
    pam_src = project_root / "target/release/libpam_doorman.so"
    if not pam_src.exists():
        console.print(f"[red]Error:[/red] PAM module not found at {pam_src}")
        raise typer.Exit(1)

    shutil.copy2(pam_src, PAM_LIB_PATH)
    os.chmod(PAM_LIB_PATH, 0o644)
    console.print(f"[green]✓[/green] Installed PAM module to {PAM_LIB_PATH}")

    # Install daemon
    daemon_src = project_root / "target/release/doormand"
    if not daemon_src.exists():
        console.print(f"[red]Error:[/red] Daemon binary not found at {daemon_src}")
        raise typer.Exit(1)

    shutil.copy2(daemon_src, DAEMON_BIN_PATH)
    os.chmod(DAEMON_BIN_PATH, 0o755)
    console.print(f"[green]✓[/green] Installed daemon to {DAEMON_BIN_PATH}")

    # Create data directories
    console.print("\n[bold]Setting up data directories...[/bold]")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    os.chmod(DATA_DIR, 0o700)
    os.chmod(MODELS_DIR, 0o755)
    console.print(f"[green]✓[/green] Created {DATA_DIR}")
    console.print(f"[green]✓[/green] Created {MODELS_DIR}")

    # Install systemd service
    console.print("\n[bold]Installing systemd service...[/bold]")
    service_content = f"""[Unit]
Description=doorman Face Authentication Daemon
After=network.target

[Service]
Type=simple
ExecStart={DAEMON_BIN_PATH}
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

# Security hardening
NoNewPrivileges=false
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={DATA_DIR} /run

[Install]
WantedBy=multi-user.target
"""

    Path(SERVICE_FILE_PATH).write_text(service_content)
    console.print(f"[green]✓[/green] Installed service file to {SERVICE_FILE_PATH}")

    # Enable and start service
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "doormand.service"], check=True)
    subprocess.run(["systemctl", "start", "doormand.service"], check=True)
    console.print("[green]✓[/green] Service enabled and started")

    # Configure PAM
    console.print("\n[bold]Configuring PAM...[/bold]")
    pam_files = ["/etc/pam.d/kde", "/etc/pam.d/sudo"]
    pam_line = "auth sufficient pam_doorman.so"

    for pam_file in pam_files:
        pam_path = Path(pam_file)
        if not pam_path.exists():
            console.print(f"[yellow]⚠[/yellow] {pam_file} not found, skipping")
            continue

        # Backup original
        backup_path = Path(f"{pam_file}.doorman.bak")
        if not backup_path.exists():
            shutil.copy2(pam_path, backup_path)
            console.print(f"[green]✓[/green] Backed up {pam_file}")

        # Read current content
        content = pam_path.read_text()

        # Check if already configured
        if "pam_doorman.so" in content:
            console.print(f"[yellow]⚠[/yellow] {pam_file} already configured")
            continue

        # Add doorman line after #%PAM-1.0
        lines = content.split("\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.strip() == "#%PAM-1.0":
                new_lines.append(f"{pam_line}")

        pam_path.write_text("\n".join(new_lines))
        console.print(f"[green]✓[/green] Configured {pam_file}")

    # Success message
    console.print("\n" + "=" * 60)
    console.print(
        Panel.fit(
            "[bold green]Setup Complete! 🎉[/bold green]\n\n"
            "Next steps:\n"
            "1. Download models: [cyan]doorman models download[/cyan]\n\n"
            "2. Enroll your face: [cyan]doorman enroll[/cyan]\n\n"
            "3. [bold yellow]Test it works:[/bold yellow] [cyan]doorman test[/cyan]\n\n"
            "4. Lock screen (Meta+L) to test face unlock!\n\n"
            "[yellow]⚠️  Always run[/yellow] [cyan]doorman test[/cyan] [yellow]before relying on PAM![/yellow]",
            title="✅ Success",
            border_style="green",
        )
    )


@app.command()
def enroll(
    username: Optional[str] = typer.Argument(
        None, help="Username to enroll (default: current user)"
    ),
):
    """
    Enroll a user's face for authentication
    """
    # Get username
    if username is None:
        username = os.getenv("USER") or "unknown"

    # Security: users can only enroll themselves (unless root)
    current_user = os.getenv("USER")
    if not is_root() and username != current_user:
        console.print(
            f"[red]Error:[/red] You can only enroll yourself ({current_user})"
        )
        console.print(
            "To enroll other users, use: [cyan]sudo doorman enroll <username>[/cyan]"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold]Enrolling user:[/bold] [cyan]{username}[/cyan]")
    console.print("[yellow]Look at the camera and remain still...[/yellow]\n")

    # Send enroll request
    request = {"type": "enroll", "username": username}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Capturing frames...", total=None)

        response = send_daemon_request(request, timeout=30)

        progress.update(task, completed=True)

    if response is None:
        raise typer.Exit(1)

    if response.get("status") == "success":
        msg = response.get("message", "Enrollment successful")
        console.print(f"\n[green]✓ {msg}[/green]")
    else:
        reason = response.get("reason", "Unknown error")
        console.print(f"\n[red]✗ Enrollment failed:[/red] {reason}")
        raise typer.Exit(1)


@app.command()
def list():
    """
    List all enrolled users
    """
    request = {"type": "list_users"}
    response = send_daemon_request(request)

    if response is None:
        raise typer.Exit(1)

    if response.get("status") != "success":
        console.print("[red]Failed to list users[/red]")
        raise typer.Exit(1)

    data = response.get("data", {})
    users = data.get("users", [])

    if not users:
        console.print("\n[yellow]No users enrolled yet[/yellow]")
        console.print("Run: [cyan]doorman enroll[/cyan]\n")
        return

    table = Table(title="Enrolled Users", show_header=True, header_style="bold magenta")
    table.add_column("Username", style="cyan")
    table.add_column("Enrolled At", style="green")
    table.add_column("Embeddings", justify="right")

    for user in users:
        table.add_row(
            user["username"], user["enrolled_at"], str(user["num_embeddings"])
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def test(
    username: Optional[str] = typer.Argument(
        None, help="Username to test (default: current user)"
    ),
):
    """
    Test face authentication before enabling in PAM
    """
    # Get username
    if username is None:
        username = os.getenv("USER") or "unknown"

    console.print(f"\n[bold]Testing authentication for:[/bold] [cyan]{username}[/cyan]")
    console.print("[yellow]Look at the camera...[/yellow]\n")

    # Send authenticate request
    request = {"type": "authenticate", "username": username}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Capturing and analyzing frames...", total=None)

        response = send_daemon_request(request, timeout=30)

        progress.update(task, completed=True)

    if response is None:
        raise typer.Exit(1)

    console.print()

    if response.get("status") == "success":
        msg = response.get("message", "Authentication successful")
        console.print(
            Panel.fit(
                f"[bold green]✓ {msg}[/bold green]\n\n"
                "Face recognition is working correctly!\n"
                "Your face authentication is ready to use.",
                title="✅ Test Passed",
                border_style="green",
            )
        )
        console.print()
    else:
        reason = response.get("reason", "Unknown error")
        console.print(
            Panel.fit(
                f"[bold red]✗ Authentication failed[/bold red]\n\n"
                f"Reason: {reason}\n\n"
                "[yellow]Troubleshooting:[/yellow]\n"
                "• Ensure you're enrolled: [cyan]doorman list[/cyan]\n"
                "• Try re-enrolling: [cyan]doorman enroll[/cyan]\n"
                "• Check lighting conditions\n"
                "• Verify camera works: [cyan]doorman status[/cyan]",
                title="❌ Test Failed",
                border_style="red",
            )
        )
        console.print()
        raise typer.Exit(1)


@app.command()
def preview(
    camera: int = typer.Option(0, "--camera", "-c", help="Camera device index"),
    console_mode: bool = typer.Option(False, "--console", help="Console mode: text output only, no GUI window"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
    models_dir: Optional[str] = typer.Option(
        None, "--models-dir", "-m", help="Path to ONNX models directory"
    ),
):
    """
    Live camera preview with face detection visualization

    Uses Python + OpenCV for maximum compatibility.
    Works around Rust opencv-rust binding issues.
    """
    if console_mode:
        console.print("\n[bold cyan]Starting console preview...[/bold cyan]")
        console.print("[yellow]Shows:[/yellow]")
        console.print("  • Real-time face detection status")
        console.print("  • Recognition results")
        console.print("  • Bounding box coordinates")
        console.print("  • Processing FPS")
        console.print("\n[dim]Press Ctrl+C to quit[/dim]\n")
    else:
        console.print("\n[bold cyan]Starting camera preview...[/bold cyan]")
        console.print("[yellow]Shows:[/yellow]")
        console.print("  • Live camera feed from daemon")
        console.print("  • Face detection (red box = unknown, green box = recognized)")
        console.print("  • Recognition results with username")
        console.print("  • Frame processing time and FPS")
        console.print("\n[dim]Press 'q' or ESC to quit[/dim]\n")

    try:
        # Setup logging if debug mode enabled
        if debug:
            import logging
            logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
        
        # Import IPC-based preview module
        from .preview_ipc import FacePreview

        # Create and run preview (no models needed - uses daemon)
        preview_obj = FacePreview(camera_index=camera)
        if console_mode:
            preview_obj.run_console(debug=debug)
        else:
            preview_obj.run()

    except ImportError as e:
        console.print(f"[red]Error:[/red] Missing dependencies: {e}")
        console.print(
            "[yellow]Install with:[/yellow] uv pip install opencv-python numpy"
        )
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Preview interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Preview failed:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def remove(
    username: str = typer.Argument(..., help="Username to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """
    Remove a user's enrollment
    """
    if not is_root():
        console.print("[red]Error:[/red] Remove must be run as root (use sudo)")
        raise typer.Exit(1)

    if not yes:
        confirm = typer.confirm(f"Remove enrollment for '{username}'?")
        if not confirm:
            console.print("Cancelled")
            return

    request = {"type": "remove_user", "username": username}
    response = send_daemon_request(request)

    if response is None:
        raise typer.Exit(1)

    if response.get("status") == "success":
        console.print(f"[green]✓ Removed user:[/green] {username}")
    else:
        reason = response.get("reason", "Unknown error")
        console.print(f"[red]✗ Failed:[/red] {reason}")
        raise typer.Exit(1)


@app.command()
def start():
    """
    Start the doorman daemon
    """
    if not is_root():
        console.print("[red]Error:[/red] Starting daemon requires root (use sudo)")
        raise typer.Exit(1)

    result = subprocess.run(
        ["systemctl", "start", "doormand.service"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]✓ Daemon started[/green]")
    else:
        console.print(f"[red]✗ Failed to start daemon[/red]\n{result.stderr}")
        raise typer.Exit(1)


@app.command()
def stop():
    """
    Stop the doorman daemon
    """
    if not is_root():
        console.print("[red]Error:[/red] Stopping daemon requires root (use sudo)")
        raise typer.Exit(1)

    result = subprocess.run(
        ["systemctl", "stop", "doormand.service"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]✓ Daemon stopped[/green]")
    else:
        console.print(f"[red]✗ Failed to stop daemon[/red]\n{result.stderr}")
        raise typer.Exit(1)


@app.command()
def restart():
    """
    Restart the doorman daemon
    """
    if not is_root():
        console.print("[red]Error:[/red] Restarting daemon requires root (use sudo)")
        raise typer.Exit(1)

    result = subprocess.run(
        ["systemctl", "restart", "doormand.service"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]✓ Daemon restarted[/green]")
    else:
        console.print(f"[red]✗ Failed to restart daemon[/red]\n{result.stderr}")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Show daemon status
    """
    # Check systemd status (try user service first, then system service)
    result_user = subprocess.run(
        ["systemctl", "--user", "is-active", "doormand-user.service"],
        capture_output=True,
        text=True,
    )
    result_system = subprocess.run(
        ["systemctl", "is-active", "doormand.service"],
        capture_output=True,
        text=True,
    )

    systemd_active = (result_user.stdout.strip() == "active" or
                     result_system.stdout.strip() == "active")

    console.print("\n[bold]System Status[/bold]")
    console.print(
        f"Daemon: {'[green]running[/green]' if systemd_active else '[red]stopped[/red]'}"
    )

    if not systemd_active:
        console.print("\n[yellow]Daemon is not running.[/yellow]")
        console.print("Start it with: [cyan]sudo doorman start[/cyan]\n")
        return

    # Get daemon info
    request = {"type": "status"}
    response = send_daemon_request(request)

    if response and response.get("status") == "success":
        data = response.get("data", {})
        info = data.get("info", {})

        uptime = info.get("uptime_secs", 0)
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60

        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Version", info.get("version", "unknown"))
        table.add_row("Uptime", f"{hours}h {minutes}m")
        table.add_row("Camera", "✓" if info.get("camera_available") else "✗")
        table.add_row("Models", "✓" if info.get("models_loaded") else "✗ (see README)")
        table.add_row("Users", str(info.get("enrolled_users", 0)))

        console.print()
        console.print(table)

    console.print()


@app.command()
def uninstall(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """
    Uninstall doorman completely
    """
    if not is_root():
        console.print("[red]Error:[/red] Uninstall must be run as root (use sudo)")
        raise typer.Exit(1)

    if not yes:
        console.print("[yellow]This will:[/yellow]")
        console.print("  • Stop and remove the daemon service")
        console.print("  • Remove PAM configuration")
        console.print("  • Remove binaries")
        console.print(
            "  • Keep user data at /var/lib/doorman (delete manually if needed)\n"
        )

        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Cancelled")
            return

    console.print("\n[bold]Uninstalling doorman...[/bold]\n")

    # Stop and disable service
    subprocess.run(["systemctl", "stop", "doormand.service"], check=False)
    subprocess.run(["systemctl", "disable", "doormand.service"], check=False)
    console.print("[green]✓[/green] Service stopped and disabled")

    # Remove service file
    if Path(SERVICE_FILE_PATH).exists():
        Path(SERVICE_FILE_PATH).unlink()
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        console.print("[green]✓[/green] Service file removed")

    # Restore PAM configs
    pam_files = ["/etc/pam.d/kde", "/etc/pam.d/sudo"]
    for pam_file in pam_files:
        backup = Path(f"{pam_file}.doorman.bak")
        if backup.exists():
            shutil.copy2(backup, pam_file)
            backup.unlink()
            console.print(f"[green]✓[/green] Restored {pam_file}")

    # Remove binaries
    for path in [PAM_LIB_PATH, DAEMON_BIN_PATH]:
        if Path(path).exists():
            Path(path).unlink()
            console.print(f"[green]✓[/green] Removed {path}")

    # Remove socket
    if Path(SOCKET_PATH).exists():
        Path(SOCKET_PATH).unlink()

    console.print(f"\n[green]✓ Uninstall complete[/green]")
    console.print(f"\n[yellow]Note:[/yellow] User data preserved at {DATA_DIR}")
    console.print("To remove completely: [cyan]sudo rm -rf /var/lib/doorman[/cyan]\n")


# ============================================================================
# Model Management Commands
# ============================================================================

models_app = typer.Typer(
    name="models",
    help="Manage ONNX models for face recognition",
)
app.add_typer(models_app)


@models_app.command("list")
def models_list(
    models_dir: Optional[str] = typer.Option(
        None, "--models-dir", help="Custom models directory"
    ),
):
    """List all available models and their status"""
    manager = ModelManager(models_dir)

    console.print("\n[bold cyan]Doorman ONNX Models[/bold cyan]")
    console.print(f"Location: [yellow]{manager.models_dir}[/yellow]\n")

    models = manager.list_models()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("File", style="white")
    table.add_column("Size", justify="right")
    table.add_column("Status", style="green")
    table.add_column("Description", style="dim")

    for model in models:
        status_style = "green" if model["installed"] else "red"
        status_icon = "✅" if model["installed"] else "❌"
        table.add_row(
            model["name"],
            model["filename"],
            f"{model['size_mb']:.1f} MB",
            f"[{status_style}]{status_icon}[/{status_style}]",
            model["description"],
        )

    console.print(table)

    installed = manager.get_installed_models()
    missing = manager.get_missing_models()

    console.print(
        f"\n[bold]Summary:[/bold] {len(installed)}/{len(manager.MODELS)} installed"
    )

    if missing:
        console.print(f"\n[yellow]Missing models:[/yellow] {', '.join(missing)}")
        console.print(
            "[dim]Run[/dim] [cyan]doorman models download[/cyan] [dim]to install them[/dim]\n"
        )
    else:
        console.print("[green]✓ All models installed![/green]\n")


@models_app.command("download")
def models_download(
    model: Optional[str] = typer.Argument(
        None, help="Specific model to download (or 'all')"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-download even if already installed"
    ),
    models_dir: Optional[str] = typer.Option(
        None, "--models-dir", help="Custom models directory"
    ),
):
    """Download ONNX models for face recognition"""

    manager = ModelManager(models_dir)

    # Check if we need root for creating models dir
    try:
        manager.ensure_models_dir()
    except PermissionError:
        console.print(f"[red]✗ Permission denied:[/red] {manager.models_dir}")
        console.print(
            "Either run with sudo or use [cyan]--models-dir[/cyan] to specify a writable directory\n"
        )
        raise typer.Exit(1)

    # Determine what to download
    if model is None or model == "all":
        to_download = (
            manager.get_missing_models() if not force else list(manager.MODELS.keys())
        )
        if not to_download:
            console.print("[green]✓ All models already installed[/green]")
            console.print("Use [cyan]--force[/cyan] to re-download\n")
            return
    else:
        if model not in manager.MODELS:
            console.print(f"[red]✗ Unknown model:[/red] {model}")
            console.print(f"Available: {', '.join(manager.MODELS.keys())}\n")
            raise typer.Exit(1)
        to_download = [model]
        if manager.is_model_installed(model) and not force:
            console.print(f"[yellow]Model '{model}' already installed[/yellow]")
            console.print("Use [cyan]--force[/cyan] to re-download\n")
            return

    console.print(
        f"\n[bold cyan]Downloading {len(to_download)} model(s)...[/bold cyan]\n"
    )

    success_count = 0
    for model_key in to_download:
        model_info = manager.MODELS[model_key]

        console.print(f"[bold]{model_info.name}[/bold]")
        console.print(f"  File: {model_info.filename}")
        console.print(f"  Size: ~{model_info.size_mb:.1f} MB")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)

            def progress_callback(msg):
                progress.update(task, description=msg)

            try:
                manager.download_model(model_key, progress_callback)
                console.print(f"[green]✓ {model_info.name} installed[/green]\n")
                success_count += 1
            except Exception as e:
                console.print(f"[red]✗ Failed:[/red] {e}\n")

    console.print(
        f"[bold]Downloaded {success_count}/{len(to_download)} models[/bold]\n"
    )


@models_app.command("verify")
def models_verify(
    model: Optional[str] = typer.Argument(
        None, help="Specific model to verify (or 'all')"
    ),
    models_dir: Optional[str] = typer.Option(
        None, "--models-dir", help="Custom models directory"
    ),
):
    """Verify installed models are valid"""

    manager = ModelManager(models_dir)

    if model is None or model == "all":
        to_verify = manager.get_installed_models()
        if not to_verify:
            console.print("[yellow]No models installed[/yellow]\n")
            return
    else:
        if model not in manager.MODELS:
            console.print(f"[red]✗ Unknown model:[/red] {model}\n")
            raise typer.Exit(1)
        if not manager.is_model_installed(model):
            console.print(f"[red]✗ Model not installed:[/red] {model}\n")
            raise typer.Exit(1)
        to_verify = [model]

    console.print(f"\n[bold cyan]Verifying {len(to_verify)} model(s)...[/bold cyan]\n")

    all_valid = True
    for model_key in to_verify:
        model_info = manager.MODELS[model_key]
        is_valid, message = manager.verify_model(model_key)

        status = "[green]✓[/green]" if is_valid else "[red]✗[/red]"
        console.print(f"{status} {model_info.name}: {message}")

        if not is_valid:
            all_valid = False

    console.print()
    if all_valid:
        console.print("[green]✓ All models verified successfully[/green]\n")
    else:
        console.print("[red]✗ Some models failed verification[/red]")
        console.print(
            "Run [cyan]doorman models download --force[/cyan] to re-download\n"
        )
        raise typer.Exit(1)


@models_app.command("remove")
def models_remove(
    model: str = typer.Argument(..., help="Model to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    models_dir: Optional[str] = typer.Option(
        None, "--models-dir", help="Custom models directory"
    ),
):
    """Remove an installed model"""

    manager = ModelManager(models_dir)

    if model not in manager.MODELS:
        console.print(f"[red]✗ Unknown model:[/red] {model}")
        console.print(f"Available: {', '.join(manager.MODELS.keys())}\n")
        raise typer.Exit(1)

    if not manager.is_model_installed(model):
        console.print(f"[yellow]Model '{model}' not installed[/yellow]\n")
        return

    model_info = manager.MODELS[model]

    if not yes:
        confirm = typer.confirm(f"Remove {model_info.name} ({model_info.filename})?")
        if not confirm:
            console.print("Cancelled\n")
            return

    try:
        manager.remove_model(model)
        console.print(f"[green]✓ Removed {model_info.name}[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to remove:[/red] {e}\n")
        raise typer.Exit(1)


@models_app.command("info")
def models_info():
    """Show detailed information about model requirements"""

    info_text = """
[bold cyan]Doorman Model Requirements[/bold cyan]

Doorman requires 3 ONNX models for face authentication:

[bold]1. Face Detection (blazeface.onnx)[/bold]
   • Detects faces in camera frames
   • Based on BlazeFace/UltraFace architecture
   • Input: RGB image (any size)
   • Output: Face bounding boxes and confidence scores

[bold]2. Liveness Detection (liveness.onnx)[/bold]
   • Anti-spoofing: detects if face is real or fake (photo/video)
   • Prevents authentication with printed photos or screens
   • Input: Cropped face region (80x80 or 224x224)
   • Output: Real vs Fake classification score

[bold]3. Face Recognition (mobilefacenet.onnx)[/bold]
   • Extracts unique face embeddings (512-dimensional vectors)
   • Used to compare and identify faces
   • Based on MobileFaceNet or ArcFace architecture
   • Input: Normalized face region (112x112 or 224x224)
   • Output: Feature embedding vector

[bold yellow]Model Sources:[/bold yellow]
• ONNX Model Zoo: https://github.com/onnx/models
• Silent Face Anti-Spoofing: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
• ArcFace/InsightFace: https://github.com/deepinsight/insightface

[bold green]Quick Start:[/bold green]
[cyan]doorman models download[/cyan]         # Download all models
[cyan]doorman models list[/cyan]             # Check installation status
[cyan]doorman models verify[/cyan]           # Verify model integrity
[cyan]doorman models spec <model>[/cyan]     # Show detailed specifications
"""

    console.print(Panel(info_text, expand=False))


@models_app.command("spec")
def models_spec(
    model: str = typer.Argument(
        ..., help="Model to show specification for (blazeface, liveness, mobilefacenet)"
    ),
):
    """Show detailed technical specifications for a model"""

    if model not in MODEL_SPECS:
        console.print(f"[red]✗ Unknown model:[/red] {model}")
        console.print(f"Available: {', '.join(MODEL_SPECS.keys())}\n")
        raise typer.Exit(1)

    spec_text = print_model_spec(model)
    console.print(f"\n[bold cyan]Model Specification: {model}[/bold cyan]\n")
    console.print(spec_text)
    console.print()


if __name__ == "__main__":
    app()
