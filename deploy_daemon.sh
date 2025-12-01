#!/bin/bash
# Deploy updated doorman daemon

set -e

echo "📦 Deploying doorman daemon..."

# Stop daemon
echo "⏹️  Stopping daemon..."
sudo systemctl stop doormand || true

# Copy binary
echo "📋 Installing binary..."
sudo cp target/release/doormand /usr/local/bin/doormand

# Start daemon
echo "▶️  Starting daemon..."
sudo systemctl start doormand

# Show status
echo ""
echo "✅ Deployment complete!"
echo ""
echo "Status:"
sudo systemctl status doormand --no-pager -l | head -20

echo ""
echo "To view logs: journalctl -u doormand -f"
echo "To test preview: doorman preview"
