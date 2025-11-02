#!/bin/bash
# Doorman Installation Script
# Run with: sudo ./install.sh

set -e

echo "🔧 Installing Doorman Face Unlock System..."
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run as root"
    echo "Usage: sudo ./install.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAM_LIB_DIR="/usr/lib/security"
SYSTEMD_DIR="/etc/systemd/system"

# Find cargo - check user's home first
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    export PATH="$USER_HOME/.cargo/bin:$PATH"
fi

# Verify cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Build the PAM module
echo "📦 Building PAM module..."
cd "$SCRIPT_DIR"
cargo build --release -p pam_doorman

# Copy PAM module
echo "📋 Installing PAM module..."
cp target/release/libpam_doorman.so "$PAM_LIB_DIR/"
chmod 644 "$PAM_LIB_DIR/libpam_doorman.so"
echo "✅ PAM module installed to $PAM_LIB_DIR/libpam_doorman.so"

# Build daemon
echo "📦 Building daemon..."
cargo build --release --features backend-tract

# Copy daemon
echo "📋 Installing daemon..."
cp target/release/doormand /usr/local/bin/
chmod 755 /usr/local/bin/doormand
echo "✅ Daemon installed to /usr/local/bin/doormand"

# Install systemd service
echo "📋 Installing systemd service..."
cp doormand.service "$SYSTEMD_DIR/"
systemctl daemon-reload
echo "✅ Systemd service installed"

# Create data directory
echo "📁 Creating data directory..."
mkdir -p /var/lib/doorman
chmod 755 /var/lib/doorman
echo "✅ Data directory created at /var/lib/doorman"

# Copy models if they exist
if [ -d "$SCRIPT_DIR/data/models" ]; then
    echo "📋 Copying models..."
    mkdir -p /var/lib/doorman/models
    cp -r "$SCRIPT_DIR/data/models/"* /var/lib/doorman/models/ 2>/dev/null || true
    echo "✅ Models copied to /var/lib/doorman/models"
fi

# Copy config
if [ -f "$SCRIPT_DIR/doorman.toml" ]; then
    echo "📋 Installing configuration..."
    cp "$SCRIPT_DIR/doorman.toml" /etc/doorman.toml
    chmod 644 /etc/doorman.toml
    echo "✅ Configuration installed to /etc/doorman.toml"
fi

echo
echo "✅ Installation complete!"
echo
echo "Next steps:"
echo "  1. Start the daemon: sudo systemctl start doormand"
echo "  2. Enable at boot: sudo systemctl enable doormand"
echo "  3. Enroll your face: /home/$SUDO_USER/.local/bin/doorman enroll \$(whoami)"
echo "  4. Configure PAM to use doorman (edit /etc/pam.d/common-auth or /etc/pam.d/system-auth)"
echo
echo "For PAM configuration, add this line:"
echo "  auth sufficient pam_doorman.so"
echo

