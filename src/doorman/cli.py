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

app = typer.Typer(
    name="doorman",
    help="Secure face unlock system for Linux",
    add_completion=False,
)
console = Console()

# Constants
SOCKET_PATH = "/run/doorman.sock"
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
        missing.append("rustc (install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh)")
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
        console.print("[yellow]Hint:[/yellow] Is the daemon running? Try: sudo systemctl status doormand")
        return None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return None


@app.command()
def setup(
    skip_checks: bool = typer.Option(False, "--skip-checks", help="Skip dependency checks"),
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
        task = progress.add_task("Compiling (this may take a few minutes)...", total=None)
        
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
    console.print(Panel.fit(
        "[bold green]Setup Complete! 🎉[/bold green]\n\n"
        "Next steps:\n"
        "1. Download ONNX models to: [cyan]/var/lib/doorman/models/[/cyan]\n"
        "   • blazeface.onnx\n"
        "   • liveness.onnx\n"
        "   • mobilefacenet.onnx\n\n"
        "2. Enroll your face: [cyan]sudo doorman enroll[/cyan]\n\n"
        "3. Test: Lock your screen and use your face to unlock!\n\n"
        "[yellow]Note:[/yellow] See README.md for model download links.",
        title="✅ Success",
        border_style="green"
    ))


@app.command()
def enroll(
    username: Optional[str] = typer.Argument(None, help="Username to enroll (default: current user)"),
):
    """
    Enroll a user's face for authentication
    """
    if not is_root():
        console.print("[red]Error:[/red] Enrollment must be run as root (use sudo)")
        raise typer.Exit(1)
    
    # Get username
    if username is None:
        username = os.getenv("SUDO_USER") or os.getenv("USER") or "unknown"
    
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
        console.print("Run: [cyan]sudo doorman enroll[/cyan]\n")
        return
    
    table = Table(title="Enrolled Users", show_header=True, header_style="bold magenta")
    table.add_column("Username", style="cyan")
    table.add_column("Enrolled At", style="green")
    table.add_column("Embeddings", justify="right")
    
    for user in users:
        table.add_row(
            user["username"],
            user["enrolled_at"],
            str(user["num_embeddings"])
        )
    
    console.print()
    console.print(table)
    console.print()


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
def status():
    """
    Show daemon status
    """
    # Check systemd status
    result = subprocess.run(
        ["systemctl", "is-active", "doormand.service"],
        capture_output=True,
        text=True,
    )
    
    systemd_active = result.stdout.strip() == "active"
    
    console.print("\n[bold]System Status[/bold]")
    console.print(f"Daemon: {'[green]running[/green]' if systemd_active else '[red]stopped[/red]'}")
    
    if not systemd_active:
        console.print("\n[yellow]Daemon is not running.[/yellow]")
        console.print("Start it with: [cyan]sudo systemctl start doormand[/cyan]\n")
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
        console.print("  • Keep user data at /var/lib/doorman (delete manually if needed)\n")
        
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


if __name__ == "__main__":
    app()

