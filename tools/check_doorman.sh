#!/bin/bash
echo "=== DOORMAN SYSTEM STATE ==="
echo ""
echo "Running processes:"
pgrep -a doormand
pgrep -a ffmpeg
echo ""
echo "Camera lock:"
lsof /dev/video0 2>/dev/null || echo "No process has camera lock"
echo ""
echo "Sockets:"
ls -la /run/user/1000/doorman-*.sock 2>/dev/null || echo "No sockets found"
echo ""
echo "Camera device:"
ls -la /dev/video0
echo ""
echo "Python OpenCV:"
python3 -c "import cv2; print(cv2.__version__)" 2>&1
echo ""
echo "Last daemon log (if running via systemd):"
journalctl --user -u doormand -n 20 --no-pager 2>/dev/null || echo "Not running via systemd"
