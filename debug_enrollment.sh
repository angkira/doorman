#!/bin/bash
# Debug face detection issues

echo "🔍 Debugging Face Detection Issues"
echo "=================================="
echo ""

echo "1️⃣ Checking daemon status..."
systemctl is-active doormand.service >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Daemon is running"
else
    echo "❌ Daemon is NOT running"
    echo "   Start it: sudo systemctl start doormand"
    exit 1
fi

echo ""
echo "2️⃣ Checking camera access..."
if [ -e /dev/video0 ]; then
    echo "✅ Camera device exists: /dev/video0"
    ls -l /dev/video0
else
    echo "❌ No camera device found at /dev/video0"
    exit 1
fi

echo ""
echo "3️⃣ Checking ML models..."
for model in blazeface.onnx liveness.onnx mobilefacenet.onnx; do
    if [ -f "/var/lib/doorman/models/$model" ]; then
        size=$(du -h "/var/lib/doorman/models/$model" | cut -f1)
        echo "✅ $model ($size)"
    else
        echo "❌ $model NOT FOUND"
    fi
done

echo ""
echo "4️⃣ Testing camera with simple capture..."
if command -v ffmpeg >/dev/null 2>&1; then
    echo "📸 Capturing test frame with ffmpeg..."
    ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 /tmp/doorman_test_frame.jpg -y 2>&1 | grep -E "(Stream|Output)" || true
    if [ -f /tmp/doorman_test_frame.jpg ]; then
        echo "✅ Camera capture works! Frame saved to /tmp/doorman_test_frame.jpg"
    else
        echo "⚠️  Camera capture failed"
    fi
else
    echo "⚠️  ffmpeg not installed, skipping camera test"
    echo "   Install: sudo apt install ffmpeg"
fi

echo ""
echo "5️⃣ Now try enrollment with live logs:"
echo "   Terminal 1: sudo journalctl -u doormand -f"
echo "   Terminal 2: doorman enroll"
echo ""
echo "Look for these log messages:"
echo "  - 'Camera not available' → camera issue"
echo "  - 'No face detected' → positioning/lighting issue"  
echo "  - 'Failed liveness' → too close/photo/screen"
echo "  - 'ML pipeline' errors → model issue"

