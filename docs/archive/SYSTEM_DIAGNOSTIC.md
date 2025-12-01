# Doorman System Diagnostic - Complete Pipeline Analysis

**Purpose**: Systematically verify every component of the preview system from camera to display.

**Problem**: Preview shows corrupted/tiled frames with noise lines. Need to verify if the system is actually ready to run via `cargo run` and receive debug socket signals.

---

## Phase 1: Environment Cleanup & Process State

### 1.1 Kill All Running Instances
```bash
# Kill all daemon processes
pkill -9 doormand
pkill -9 ffmpeg

# Wait 2 seconds
sleep 2

# Verify nothing is running
pgrep doormand
pgrep ffmpeg

# Check what has camera lock
lsof /dev/video0

# If anything still has the camera, kill it
sudo fuser -k /dev/video0
```

**Expected**: No processes should be listed after cleanup.

### 1.2 Verify Camera Hardware
```bash
# List video devices
ls -la /dev/video*

# Get camera info
v4l2-ctl --device=/dev/video0 --all | grep -E "(Driver|Card|Format|Width|Height)"

# Test camera formats
v4l2-ctl --device=/dev/video0 --list-formats-ext | grep -E "(MJPEG|YUYV|H264)" -A 5
```

**Expected**:
- `/dev/video0` exists and is readable
- Camera supports at least one format (MJPEG, YUYV, or RGB)
- Camera reports actual resolution (likely 1024x720)

---

## Phase 2: Camera Capture Test (No Daemon)

### 2.1 Test FFmpeg Direct Capture
```bash
cd /home/angkira/Home/doorman

# Capture single frame to verify camera works
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 -y test_single_frame.jpg

# Check file was created and is valid JPEG
file test_single_frame.jpg
ls -lh test_single_frame.jpg
```

**Expected**:
- Command succeeds
- File is ~50-200KB JPEG
- File command shows: "JPEG image data"

### 2.2 Test FFmpeg Continuous Stream
```bash
# Stream 30 frames of raw RGB
timeout 3 ffmpeg \
  -f v4l2 \
  -video_size 1280x720 \
  -framerate 30 \
  -i /dev/video0 \
  -frames:v 30 \
  -f rawvideo \
  -pix_fmt rgb24 \
  - > /tmp/raw_frames.rgb 2>/tmp/ffmpeg_stream.log

# Check what resolution FFmpeg actually used
cat /tmp/ffmpeg_stream.log | grep -E "(changed the video|Stream|Input)"

# Calculate frame count from file size
actual_size=$(stat -f%z /tmp/raw_frames.rgb 2>/dev/null || stat -c%s /tmp/raw_frames.rgb)
echo "File size: $actual_size bytes"
echo "Expected for 30 frames at 1280x720: $((1280 * 720 * 3 * 30)) bytes"
echo "Expected for 30 frames at 1024x720: $((1024 * 720 * 3 * 30)) bytes"
```

**Expected**:
- File size matches one of the expected values
- FFmpeg log may show "changed the video from 1280x720 to 1024x720"
- If 1024x720, this confirms camera resolution mismatch

---

## Phase 3: Daemon Build & Run Test

### 3.1 Clean Build
```bash
cd /home/angkira/Home/doorman

# Clean build
cargo clean
cargo build --release 2>&1 | tee /tmp/build.log

# Check for errors
echo "Build exit code: $?"
```

**Expected**: Build succeeds with exit code 0

### 3.2 Verify Configuration
```bash
# Check config file exists
cat ~/.config/doorman/config.toml

# Verify camera settings
grep -A 5 "\[camera\]" ~/.config/doorman/config.toml
```

**Expected**:
- Config file exists
- Camera device_index = 0
- Width and height are set (note: daemon now auto-detects actual resolution)

### 3.3 Run Daemon in Foreground
```bash
# Run daemon with full logging
RUST_LOG=debug ./target/release/doormand 2>&1 | tee /tmp/daemon_run.log
```

**Watch for**:
1. "Initializing FFmpeg camera backend (continuous streaming)"
2. "Camera changed resolution from X to Y" (if resolution mismatch)
3. "FFmpeg camera test successful"
4. "FFmpeg continuous streaming started"
5. "Frame stream server listening on /run/user/1000/doorman-frames.sock"
6. "Debug stream server listening on /run/user/1000/doorman-debug.sock"

**Expected**: Daemon starts without errors and creates both sockets.

**STOP HERE** - Keep daemon running for Phase 4

---

## Phase 4: Socket Communication Test

### 4.1 Verify Sockets Exist
```bash
# In a NEW terminal
ls -la /run/user/1000/doorman-*.sock

# Check socket types
file /run/user/1000/doorman-frames.sock
file /run/user/1000/doorman-debug.sock
```

**Expected**: Both sockets exist and are type "socket"

### 4.2 Test Frame Socket Reception
```bash
cd /home/angkira/Home/doorman

# Run frame decode test
python3 test_frame_decode.py
```

**Watch for**:
1. "Connected" ✓
2. "Frame size: X bytes (Y KB)" ✓
3. "Valid JPEG magic bytes" ✓
4. "Saved to test_output.jpg" ✓
5. "OpenCV decoded: WxH, 3 channels" ✓

**Expected**:
- Script completes successfully
- `test_output.jpg` is created
- Image dimensions match camera resolution (1024x720 or 1280x720)
- Can open image and see camera view

### 4.3 Test Debug Socket Reception
```bash
# Connect to debug socket and print messages
nc -U /run/user/1000/doorman-debug.sock
```

**Expected**: Should see JSON messages with face detection data (if face visible) or empty arrays

---

## Phase 5: Python Preview Test

### 5.1 Verify Python Dependencies
```bash
cd /home/angkira/Home/doorman

# Check installed packages
uv pip list | grep -E "(opencv|numpy|insightface|onnx)"

# Verify OpenCV version
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Test OpenCV decode capability
python3 -c "import cv2, numpy as np; data=open('test_output.jpg','rb').read(); arr=np.frombuffer(data, np.uint8); img=cv2.imdecode(arr, cv2.IMREAD_COLOR); print(f'Decoded: {img.shape}' if img is not None else 'DECODE FAILED')"
```

**Expected**:
- opencv-python == 4.10.0.84
- OpenCV decode test prints: "Decoded: (720, 1024, 3)" or similar
- insightface is installed

### 5.2 Run Preview Client
```bash
# With daemon still running from Phase 3
python3 -m doorman.preview_ipc
```

**Watch for**:
1. Window opens showing camera feed
2. No tiling/noise artifacts
3. Face detection boxes appear if face is visible
4. FPS counter shows ~15-30 FPS
5. No corruption or frozen frames

**Expected**: Clean video preview with face detection overlays

---

## Phase 6: Full Pipeline Verification

### 6.1 Test Complete Flow
```bash
# Kill daemon from Phase 3
pkill doormand

# Run via cargo run to test normal workflow
cd /home/angkira/Home/doorman
RUST_LOG=info cargo run --release --bin doormand &

# Wait for startup
sleep 3

# Run preview
python3 -m doorman.preview_ipc
```

**Expected**: Same as Phase 5.2

### 6.2 Verify Frame Encoding Quality
```bash
# While preview is running, capture 10 frames
cd /home/angkira/Home/doorman

# Create test script to save frames
cat > test_save_frames.py << 'EOF'
#!/usr/bin/env python3
import socket, struct, time

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect("/run/user/1000/doorman-frames.sock")

for i in range(10):
    # Read size
    size_bytes = sock.recv(4)
    if len(size_bytes) != 4:
        break
    size = struct.unpack('>I', size_bytes)[0]

    # Read JPEG
    jpeg_data = b""
    while len(jpeg_data) < size:
        jpeg_data += sock.recv(size - len(jpeg_data))

    # Save
    with open(f"frame_{i:03d}.jpg", "wb") as f:
        f.write(jpeg_data)

    print(f"Saved frame_{i:03d}.jpg ({len(jpeg_data)} bytes)")
    time.sleep(0.1)

sock.close()
EOF

python3 test_save_frames.py
```

**Expected**:
- 10 JPEG files created
- Each is valid JPEG (check with `file frame_*.jpg`)
- Can open each image and see camera view
- No corruption/tiling in saved images

**If saved images are corrupted**: Problem is in daemon encoding
**If saved images are clean but preview is corrupted**: Problem is in Python preview client

---

## Phase 7: Diagnostic Summary

### Success Criteria Checklist

- [ ] Phase 1: All processes killed, camera accessible
- [ ] Phase 2.1: Single frame capture works
- [ ] Phase 2.2: Continuous stream works, resolution detected
- [ ] Phase 3.1: Daemon builds successfully
- [ ] Phase 3.3: Daemon runs without errors
- [ ] Phase 4.1: Both sockets created
- [ ] Phase 4.2: Frame socket delivers valid JPEG
- [ ] Phase 4.3: Debug socket delivers JSON
- [ ] Phase 5.1: Python dependencies correct
- [ ] Phase 5.2: Preview shows clean video
- [ ] Phase 6.2: Saved frames are not corrupted

### Failure Investigation

**If Phase 2 fails**: Camera hardware or driver issue
**If Phase 3.3 fails**: Daemon initialization problem
**If Phase 4.2 fails**: Frame encoding problem in daemon
**If Phase 5.2 fails but 6.2 passes**: Python client decoding problem
**If both 5.2 and 6.2 fail**: Frame encoding problem in daemon

---

## Expected Issues & Solutions

### Issue 1: Camera Resolution Mismatch
**Symptom**: FFmpeg reports "changed the video from 1280x720 to 1024x720"
**Solution**: Already fixed in `daemon/src/camera/ffmpeg_backend.rs` - daemon now auto-detects
**Verification**: Check daemon logs for detected resolution

### Issue 2: Multiple Daemon Instances
**Symptom**: "Device or resource busy" error
**Solution**: Phase 1 cleanup
**Verification**: `pgrep doormand` should return nothing when no daemon should be running

### Issue 3: OpenCV Version Conflict
**Symptom**: `AttributeError: module 'cv2' has no attribute 'imdecode'`
**Solution**: Already fixed in `pyproject.toml` - using opencv-python==4.10.0.84
**Verification**: Phase 5.1 checks

### Issue 4: JPEG Encoding Quality
**Symptom**: Frames are tiled/corrupted
**Possible Causes**:
1. Frame size calculation wrong (width * height * 3 mismatch)
2. JPEG encoder settings incorrect
3. Buffer overflow/underflow in encoding

**Investigation**: Check `daemon/src/frame_stream.rs` lines 100-110 for JPEG quality settings

---

## Quick Diagnostic Command

Run this to get current system state:
```bash
#!/bin/bash
echo "=== DOORMAN SYSTEM STATE ==="
echo ""
echo "Running processes:"
pgrep -a doormand
pgrep -a ffmpeg
echo ""
echo "Camera lock:"
lsof /dev/video0
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
```

Save as `check_doorman.sh` and run with `bash check_doorman.sh`
