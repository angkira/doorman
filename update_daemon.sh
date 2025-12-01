#!/bin/bash
# Quick update script - rebuild and restart user daemon

set -e

echo "🔄 Updating doorman daemon..."

# 1. Build
echo "📦 Building..."
cargo build --release --quiet

# 2. Stop service
echo "⏹️  Stopping service..."
systemctl --user stop doormand-user 2>/dev/null || true

# 3. Install binary
echo "📋 Installing binary..."
sudo cp target/release/doormand /usr/local/bin/doormand

# 4. Copy models if they exist in system location
if [ -d "/var/lib/doorman/models" ] && [ "$(ls -A /var/lib/doorman/models 2>/dev/null)" ]; then
    echo "📚 Copying models to user directory..."
    mkdir -p ~/.local/share/doorman/models
    cp -r /var/lib/doorman/models/* ~/.local/share/doorman/models/ 2>/dev/null || true
    echo "✅ Models copied"
fi

# 5. Start service
echo "▶️  Starting service..."
systemctl --user start doormand-user

# 6. Show status
echo ""
systemctl --user status doormand-user --no-pager -l | head -20

echo ""
echo "✅ Update complete!"
echo "📋 View logs: journalctl --user -u doormand-user -f"
