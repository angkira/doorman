# 🚀 Quick Test Instructions

## What Was Fixed

**BBox Size**: Changed from covering 60-80% of frame to 10-40% (just the face).

**File Changed**: `daemon/src/ml/tract_backend.rs`, lines 279-280

## Test Now

### Terminal 1:
```bash
cd ~/Home/doorman
./test_bbox.sh
```

### Terminal 2:
```bash
doorman preview --debug
```

## What to Look For

✅ **GOOD**: Green/red box tightly around your face  
✅ **GOOD**: Logs show bbox ~150-300 pixels wide  
✅ **GOOD**: Box tracks face smoothly  

❌ **BAD**: Box still covers most of the frame  
❌ **BAD**: Box is tiny or missing  
❌ **BAD**: Lots of "No faces detected" messages  

## If Box Still Wrong

### Too Large (still 400+ pixels):
```bash
# Edit daemon/src/ml/tract_backend.rs, line 279-280
# Change 0.4 to 0.3 or even 0.25
cargo build --release
./test_bbox.sh
```

### Too Small or Missing:
```bash
# Check confidence threshold in logs
# May need to lower detection threshold from 0.4 to 0.3
```

## Next Steps After Successful Test

1. ✅ Verify BBox size is correct
2. 📝 Implement face recognition matching
3. 👤 Add user enrollment  
4. 🎯 Test end-to-end recognition

## Debug Commands

```bash
# Check exact bbox values
./target/release/doormand --user --preview 2>&1 | grep "bbox="

# Full debug output
RUST_LOG=debug ./target/release/doormand --user --preview

# Just detection processing
./target/release/doormand --user --preview 2>&1 | grep -E "(Detection|bbox|confidence)"
```

---

**Status**: ✅ Built and ready to test  
**Critical**: Verify bbox before implementing recognition
