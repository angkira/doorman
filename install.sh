#!/bin/bash
# Doorman Installation Script
# Run WITHOUT sudo: ./install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

# Detect PAM module directory
PAM_LIB_DIR=$(find /lib /usr/lib -name "pam_unix.so" 2>/dev/null | head -1 | xargs dirname)
if [ -z "$PAM_LIB_DIR" ]; then
    echo "❌ Could not find PAM modules directory"
    exit 1
fi
echo "Detected PAM directory: $PAM_LIB_DIR"

echo "🔧 Installing Doorman Face Unlock System..."
echo

# Step 1: Build as regular user (no sudo needed)
if [ "$EUID" -eq 0 ]; then
    echo "❌ Do NOT run this script with sudo!"
    echo "Usage: ./install.sh"
    echo
    echo "The script will:"
    echo "  1. Build binaries as your user (no sudo)"
    echo "  2. Ask for sudo password to install system files"
    exit 1
fi

# Verify cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ cargo not found. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Build the PAM module
echo "📦 Building PAM module (as user)..."
cd "$SCRIPT_DIR"
cargo build --release -p pam_doorman

# Build daemon
echo "📦 Building daemon (as user)..."
cargo build --release --features backend-tract

echo
echo "✅ Build complete! Now installing system files (requires sudo)..."
echo

# Step 2: Install as root (asks for sudo password once)
sudo bash <<EOF
set -e

# Copy PAM module
echo "📋 Installing PAM module..."
cp "$SCRIPT_DIR/target/release/libpam_doorman.so" "$PAM_LIB_DIR/"
chmod 644 "$PAM_LIB_DIR/libpam_doorman.so"
echo "✅ PAM module installed to $PAM_LIB_DIR/libpam_doorman.so"

# Copy daemon
echo "📋 Installing daemon..."
cp "$SCRIPT_DIR/target/release/doormand" /usr/local/bin/
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
EOF

echo
echo "✅ Installation complete!"
echo
echo "Next steps:"
echo "  1. Start the daemon: sudo systemctl start doormand"
echo "  2. Enable at boot: sudo systemctl enable doormand"
echo "  3. Enroll your face: doorman enroll \$USER"
echo "  4. Configure PAM to use doorman (edit /etc/pam.d/common-auth)"
echo
echo "For PAM configuration, add this line:"
echo "  auth sufficient pam_doorman.so"
echo

