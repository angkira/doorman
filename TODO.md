# TODO - Doorman Development

**Last Updated**: 2025-12-01 Morning  
**Status**: BBox fix applied, awaiting testing

---

## 🔴 Critical (Do First)

### 1. ✅ Test BBox Fix (15 min) **← START HERE**
- [ ] Run `./test_bbox.sh` in terminal 1
- [ ] Run `doorman preview --debug` in terminal 2
- [ ] Verify box size ~150-250px (not 600+px)
- [ ] Check logs for bbox values
- **Blocking**: All recognition work

### 2. Implement Face Recognition (2-3 hours)
**After** bbox is confirmed working:
- [ ] Complete `recognition_pipeline.rs` matching logic
- [ ] Calculate cosine similarity with stored embeddings
- [ ] Set threshold (e.g., 0.6 for match)
- [ ] Send recognition results to preview
- [ ] Color-code preview: green=known, red=unknown

### 3. User Enrollment Command (1 hour)
```bash
doorman enroll <username>
```
- [ ] Capture 5-10 face samples
- [ ] Extract and average embeddings
- [ ] Save to `~/.local/share/doorman/users/<name>.json`
- [ ] Add list/delete commands

---

## 🟡 High Priority

### 4. Unit Tests for Coordinates (1-2 hours)
Create `daemon/src/ml/tests.rs`:
- [ ] Test BlazeFace raw → normalized coords
- [ ] Test normalized → pixel conversion
- [ ] Test letterbox offset math
- [ ] Test padding calculation
- [ ] Test edge cases (face at border)

### 5. Fix Liveness Detection (30 min)
Currently bypassed with warning:
- [ ] Analyze actual score distribution
- [ ] Understand why scores are negative (-0.35 vs threshold 0.5)
- [ ] Either adjust threshold or document model issue
- [ ] Test with known-live faces

### 6. Optimize Camera Performance (2 hours)
**Problem**: FFmpeg only achieves 5-10 fps  
**Solution**: Fix GStreamer/PipeWire
- [ ] Debug GStreamer timeout (5 second limit)
- [ ] Test PipeWire node configuration
- [ ] Document working camera backend setup
- **Target**: 30 fps camera capture

---

## 🚀 Quick Commands

```bash
# Test bbox fix
./test_bbox.sh

# Full debug
RUST_LOG=debug ./target/release/doormand --user --preview

# Watch for bbox values
./target/release/doormand --user --preview 2>&1 | grep "bbox="

# Rebuild after changes
cargo build --release

# Preview client
doorman preview --debug
```

---

**Next Action**: Run `./test_bbox.sh` and verify bounding box size! 🎯
