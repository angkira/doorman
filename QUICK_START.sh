#!/bin/bash
# Quick start script for morning testing

set -e

echo "================================"
echo "Doorman Quick Start"
echo "================================"
echo ""

# Check if daemon is already running
if pgrep -x "doormand" > /dev/null; then
    echo "⚠️  Daemon already running. Stopping it first..."
    pkill -x "doormand" || true
    sleep 1
fi

echo "1. Starting daemon with preview mode..."
cd /home/angkira/Home/doorman

# Start daemon in background
./target/release/doormand --user --preview > /tmp/doorman-daemon.log 2>&1 &
DAEMON_PID=$!

echo "   Daemon PID: $DAEMON_PID"
echo "   Logs: tail -f /tmp/doorman-daemon.log"
echo ""

# Wait for daemon to initialize
echo "2. Waiting for daemon to initialize (3 seconds)..."
sleep 3

echo ""
echo "3. Starting preview..."
echo "   (Press 'q' or ESC to quit preview)"
echo ""

# Activate venv and run preview
source .venv/bin/activate
doorman preview --debug

# When preview exits, stop daemon
echo ""
echo "4. Stopping daemon..."
kill $DAEMON_PID 2>/dev/null || true

echo ""
echo "================================"
echo "Session ended"
echo "================================"
echo ""
echo "Daemon logs available at: /tmp/doorman-daemon.log"
echo ""
