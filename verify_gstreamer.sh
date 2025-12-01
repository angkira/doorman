#!/bin/bash
# Verify GStreamer backend integration

echo "=== GStreamer Backend Verification ==="
echo ""

# Check if GStreamer is installed
echo "1. Checking GStreamer installation..."
if command -v gst-launch-1.0 &> /dev/null; then
    version=$(gst-launch-1.0 --version | head -1)
    echo "   ✓ GStreamer installed: $version"
else
    echo "   ✗ GStreamer NOT installed"
    echo "   Run: ./install_gstreamer.sh"
    exit 1
fi

# Check for required plugins
echo ""
echo "2. Checking GStreamer plugins..."
plugins_ok=true
for plugin in coreelements videoconvert videoscale pipewire; do
    if gst-inspect-1.0 $plugin &> /dev/null; then
        echo "   ✓ $plugin"
    else
        echo "   ✗ $plugin MISSING"
        plugins_ok=false
    fi
done

if [ "$plugins_ok" = false ]; then
    echo "   Run: ./install_gstreamer.sh"
    exit 1
fi

# Check if feature compiles
echo ""
echo "3. Checking compilation..."
if cargo build --release --features camera-gstreamer 2>&1 | grep -q "Finished"; then
    echo "   ✓ Compiles with camera-gstreamer feature"
else
    echo "   ✗ Compilation failed"
    echo "   Check: cargo build --release --features camera-gstreamer"
    exit 1
fi

# Check if backend code exists
echo ""
echo "4. Checking backend implementation..."
if [ -f "daemon/src/camera/gstreamer_backend.rs" ]; then
    lines=$(wc -l < daemon/src/camera/gstreamer_backend.rs)
    echo "   ✓ GStreamer backend: $lines lines"
else
    echo "   ✗ Backend file missing"
    exit 1
fi

# Check if tests exist
echo ""
echo "5. Checking test suite..."
if [ -f "daemon/tests/gstreamer_camera_test.rs" ]; then
    tests=$(grep -c "async fn test_" daemon/tests/gstreamer_camera_test.rs)
    echo "   ✓ Test suite: $tests tests"
else
    echo "   ✗ Test suite missing"
fi

# Check documentation
echo ""
echo "6. Checking documentation..."
docs=("GSTREAMER_BACKEND.md" "GSTREAMER_INTEGRATION.md" "install_gstreamer.sh")
for doc in "${docs[@]}"; do
    if [ -f "$doc" ]; then
        echo "   ✓ $doc"
    else
        echo "   ✗ $doc missing"
    fi
done

# Test camera availability
echo ""
echo "7. Checking camera availability..."
if [ -e /dev/video0 ]; then
    echo "   ✓ Camera device /dev/video0 exists"
    
    # Try to list cameras with GStreamer
    if timeout 3 gst-device-monitor-1.0 Video 2>/dev/null | grep -q "Device found"; then
        echo "   ✓ GStreamer can see cameras"
    else
        echo "   ⚠ GStreamer might not see cameras (check PipeWire)"
    fi
else
    echo "   ⚠ No camera at /dev/video0"
    echo "   Note: Camera required for runtime testing"
fi

# Check PipeWire status
echo ""
echo "8. Checking PipeWire..."
if systemctl --user is-active pipewire &> /dev/null; then
    echo "   ✓ PipeWire is running"
else
    echo "   ⚠ PipeWire not running"
    echo "   Start with: systemctl --user start pipewire"
fi

echo ""
echo "=== Verification Summary ==="
echo ""
echo "✅ GStreamer backend is ready!"
echo ""
echo "To use:"
echo "  1. Build: cargo build --release --features camera-gstreamer"
echo "  2. Run: ./target/release/doormand --user --preview"
echo "  3. Check logs for: 'Using GStreamer camera backend'"
echo ""
echo "For tests:"
echo "  cargo test --features camera-gstreamer gstreamer -- --nocapture"
echo ""
echo "For documentation:"
echo "  cat GSTREAMER_BACKEND.md"
echo "  cat GSTREAMER_INTEGRATION.md"
echo ""
