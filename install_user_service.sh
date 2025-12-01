#!/bin/bash
# Install doorman as a systemd user service
# This enables proper PipeWire/GStreamer camera access

set -e

echo "🔧 Installing doorman user service..."
echo ""

# Check if binary exists
if [ ! -f "target/release/doormand" ]; then
    echo "❌ Binary not found. Please run 'cargo build --release' first."
    exit 1
fi

# 1. Install binary
echo "📦 Installing binary to /usr/local/bin/doormand..."
sudo cp target/release/doormand /usr/local/bin/doormand
sudo chmod +x /usr/local/bin/doormand
echo "✅ Binary installed"
echo ""

# 2. Install user service file
echo "📄 Installing systemd user service..."
mkdir -p ~/.config/systemd/user
cp doormand-user.service ~/.config/systemd/user/
echo "✅ Service file installed to ~/.config/systemd/user/doormand-user.service"
echo ""

# 3. Create data directory
echo "📁 Creating data directory..."
mkdir -p ~/.local/share/doorman
echo "✅ Data directory created at ~/.local/share/doorman"
echo ""

# 4. Reload systemd user daemon
echo "🔄 Reloading systemd user daemon..."
systemctl --user daemon-reload
echo "✅ Systemd reloaded"
echo ""

# 5. Stop old system service if running
echo "⏹️  Stopping old system service (if running)..."
sudo systemctl stop doormand 2>/dev/null || true
sudo systemctl disable doormand 2>/dev/null || true
echo "✅ Old system service stopped"
echo ""

# 6. Start user service
echo "▶️  Starting user service..."
systemctl --user start doormand-user.service
echo "✅ User service started"
echo ""

# 7. Enable autostart
echo "🚀 Enabling autostart..."
systemctl --user enable doormand-user.service
echo "✅ Autostart enabled"
echo ""

# 8. Show status
echo "📊 Service Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
systemctl --user status doormand-user.service --no-pager || true
echo ""

# 9. Show socket location
USER_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
echo "🔌 Socket locations:"
echo "   Main: $USER_RUNTIME_DIR/doorman.sock"
echo "   Debug: $USER_RUNTIME_DIR/doorman-debug.sock"
echo ""

# 10. Check if sockets exist
if [ -S "$USER_RUNTIME_DIR/doorman.sock" ]; then
    echo "✅ Main socket exists and ready"
else
    echo "⚠️  Main socket not found - check logs below"
fi
echo ""

# 11. Show recent logs
echo "📋 Recent logs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
journalctl --user -u doormand-user -n 20 --no-pager
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Check logs: journalctl --user -u doormand-user -f"
echo "   2. Enroll yourself: doorman enroll \$USER"
echo "   3. Test preview: doorman preview"
echo ""
echo "💡 Tips:"
echo "   - User service runs with YOUR permissions (can access camera!)"
echo "   - Data stored in: ~/.local/share/doorman"
echo "   - To restart: systemctl --user restart doormand-user"
echo "   - To stop: systemctl --user stop doormand-user"
echo "   - To disable autostart: systemctl --user disable doormand-user"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
