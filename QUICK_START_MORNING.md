# Quick Start Guide - Morning Edition

## 🎯 Current Status
**Almost There!** Pipeline working, detection stable, just bbox sizing issue remaining.

## ⚡ Quick Commands

### Test with Live Camera
```bash
# Terminal 1: Start daemon
./target/release/doormand --user --preview

# Terminal 2: Start preview
doorman preview --debug
```

### Test with Video File (Reproducible)
```bash
# Terminal 1: Start daemon with video
./target/release/doormand --user --preview --video-file 2025-11-26-115723.webm

# Terminal 2: Start preview
doorman preview --debug
```

### Check Detection Output
```bash
RUST_LOG=info ./target/release/doormand --user --preview 2>&1 | grep -E "(BlazeFace raw|Final bbox)"
```

## 📊 Expected Output
```
BlazeFace raw: top_y=0.403, top_x=0.677, bot_y=0.416, bot_x=0.699, conf=0.196
Final bbox in original image: x=693.4, y=285.8, w=22.2, h=10.0
Broadcasting detection: bbox=(693, 285, 22, 10) = top_left + size, confidence=0.196
```

## 🐛 Known Issue
**Bounding box too small**: 22x10 pixels instead of ~150-200 pixels.

**Root cause**: BlazeFace model returns normalized size of w=0.022 (2.2%), h=0.013 (1.3%).

**Possible fixes**:
1. Apply 10x scaling factor (quick hack)
2. Research anchor-based decoding (proper fix)
3. Switch to different detector (YuNet, MTCNN)

## 📁 Important Files
- `GOOD_MORNING.txt` - Morning briefing
- `NIGHT_WORK_SUMMARY.md` - Technical details of night work
- `MORNING_STATUS.md` - Current system status
- `daemon/src/ml/tract_backend.rs` - Where bbox decoding happens (lines 265-330)

## 🔧 Next Steps (Priority)
1. ✅ Verify coordinate transform is correct (DONE!)
2. ⏳ Debug why BlazeFace outputs tiny boxes (IN PROGRESS)
3. ⏹️ Add padding to recognition crop
4. ⏹️ Complete face recognition pipeline
5. ⏹️ Add enrollment command

## 💡 Debugging Tips

### Save preprocessed frame
Add to `tract_backend.rs` after line 203:
```rust
image.save("/tmp/preprocessed_frame.jpg")?;
```

### Try 10x scaling hack
Change lines 278-279:
```rust
let w = (x2 - x).abs().clamp(0.01, 1.0) * 10.0;  // Temporary 10x scale
let h = (y2 - y).abs().clamp(0.01, 1.0) * 10.0;  // Temporary 10x scale
```

### Check model metadata
```bash
python3 << EOF
import onnx
model = onnx.load("/home/angkira/.local/share/doorman/models/blazeface.onnx")
print("Inputs:", model.graph.input)
print("Outputs:", model.graph.output)
EOF
```

## 🚀 Performance
- Camera: 15-30 fps ✅
- Detection: 7-8 fps ✅
- Preview: Smooth ✅
- Recognition: Not yet implemented ⏹️

Good luck! 🎯
