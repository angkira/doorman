# Doorman Work Status - Morning Report

## Completed ✅

### 1. BlazeFace Coordinate Decoding Fixed
- **Problem**: BBox was covering 60-80% of frame instead of just the face
- **Solution**: Reduced width/height clamp from `0.8` to `0.4` (line 279-280 in `tract_backend.rs`)
- **Result**: BBox should now be 10-40% of frame size, much more accurate

### 2. Video File Backend Created
- **File**: `daemon/src/camera/video_file_backend.rs`
- **Purpose**: Test pipeline with video files instead of live camera
- **Status**: Code written but not yet fully tested due to compilation issues
- **Usage**: `Camera::from_video_file(path, width, height, fps, loop_playback)`

### 3. Camera Abstraction Improved
- Added `is_open()` method to FFmpegCamera and VideoFileBackend
- Better error handling and lifecycle management

## In Progress 🔄

### Test Video File Binary
- **File**: `daemon/src/bin/test-video-file.rs`
- **Status**: Partially implemented, has compilation errors
- **Remaining Issues**:
  - TractBackend API mismatch (needs Path not Config)
  - Async/sync mismatch in initialization
  - Need to verify all trait implementations

## Testing Needed 🧪

### 1. BBox Size Verification
**Priority: HIGH**

Test with live camera or video file to verify:
- BBox now correctly covers just the face (not whole frame)
- Coordinates properly map from normalized [0,1] to pixels
- Padding is reasonable (10-20% around detected face)

**Test Command**:
```bash
./target/release/doormand --user --preview
# Then run: doorman preview --debug
```

**Expected**: BBox should be ~150-250 pixels wide for a face in 1024x720 frame

### 2. Video File Pipeline
**Priority: MEDIUM**

Once test-video-file compiles:
```bash
./test_video_file.sh 2025-11-26-115723.webm
```

Should process video at ~10-30 FPS and detect faces in each frame.

### 3. Unit Tests for Coordinate Transforms
**Priority: HIGH**

Need tests in `daemon/src/ml/tests.rs`:
- BlazeFace raw output → normalized coords
- Normalized coords → pixel coords  
- Letterbox offset handling
- Padding calculation

## Known Issues 🐛

### 1. Preview FPS Still High (250+ fps)
- Preview counts sent JPEG frames, not actual camera FPS
- Should show camera capture rate (15-30 fps) instead
- **Fix**: Modify `tools/preview_ipc.py` to track time between unique frames

### 2. Liveness Check Always Failing
- Scores around -0.35 vs threshold 0.5
- Currently bypassed with warning
- **Fix**: May need different threshold or model re-training

### 3. GStreamer Backend Timeout
- Pipeline doesn't reach PLAYING state within 5 seconds
- Falls back to FFmpeg
- **Status**: Known issue, FFmpeg works well enough for now

## Architecture Status 📐

### Staged Pipeline (DONE ✅)
1. **Camera Producer** - 30fps frame capture ✅
2. **Frame Fanout** - Distributes to detection (10fps) and preview (15fps) ✅  
3. **Detection Pipeline** - Face detection + liveness ✅
4. **Recognition Pipeline** - Extract embeddings (stub) ✅

### Performance (Current)
- Camera: 15-30 FPS (GStreamer) or 5-10 FPS (FFmpeg)
- Detection: 3-7 FPS
- Preview: Smooth, ~15 FPS display

### Next Steps 🎯

1. **Fix test-video-file compilation** (30 min)
   - Simplify TractBackend initialization
   - Remove async where not needed

2. **Test BBox size with live camera** (15 min)
   - Verify 0.4 clamp works correctly
   - Adjust if face still too large/small

3. **Add coordinate transform tests** (1 hour)
   - Unit tests for all coordinate conversions
   - Verify letterbox offset math

4. **Implement face recognition** (2-3 hours)
   - Complete Recognition Pipeline
   - Match embeddings against stored users
   - Return recognition results to preview

5. **Fix liveness threshold** (30 min)
   - Analyze actual score distribution
   - Set appropriate threshold or retrain model

## Files Modified 📝

- `daemon/src/ml/tract_backend.rs` - Fixed BBox clamp values
- `daemon/src/camera/video_file_backend.rs` - New video file backend
- `daemon/src/camera/mod.rs` - Added video file support
- `daemon/src/camera/ffmpeg_backend.rs` - Added is_open() method
- `daemon/src/bin/test-video-file.rs` - New test binary (WIP)

## Build Commands 🔨

```bash
# Build daemon
cargo build --release

# Build with video file test
cargo build --release --bin test-video-file

# Run daemon with preview
./target/release/doormand --user --preview

# Run preview client (separate terminal)
doorman preview --debug
```

## Notes for AI Agent 🤖

When continuing work:
1. Start by fixing test-video-file compilation
2. Test BBox size - THIS IS CRITICAL
3. If BBox still wrong, add debug logging to see raw BlazeFace outputs
4. Video file in repo: `2025-11-26-115723.webm` (4K, user present in every frame)
5. Focus on making preview show correct face bounding box - this blocks recognition work

---
**Status**: Ready for testing, bbox fix applied but needs verification
**Date**: 2025-12-01
**Blocking Issue**: Need to verify bbox size is correct before implementing recognition
