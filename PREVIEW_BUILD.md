# Building the Preview Tool

The `doorman-preview` GUI tool is **optional** - the main doorman system works perfectly without it.

## Known Issue: OpenCV Build Fails

The preview tool requires OpenCV Rust bindings, which have a known issue with `clang-sys` on some systems:

```
thread 'main' panicked at clang-sys-1.8.1/src/lib.rs:1859:1:
a `libclang` shared library is not loaded on this thread
```

## Workaround Options

### Option 1: Skip the Preview Tool
The main doorman functionality works without it:
```bash
# Build without preview
cargo build --release --features backend-tract

# Use doorman normally
doorman enroll
doorman test
```

### Option 2: Manual libclang Setup

If you really want the preview tool, try these fixes:

```bash
# 1. Find your libclang location
find /usr/lib -name "libclang.so*" 2>/dev/null

# 2. Set environment variables
export LIBCLANG_PATH=/usr/lib/llvm-20/lib
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 3. Clean and rebuild
cargo clean -p opencv
cargo build --release --bin doorman-preview --features backend-tract,video
```

### Option 3: Use System LD_PRELOAD

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libclang-20.so.20 \
  cargo build --release --bin doorman-preview --features backend-tract,video
```

### Option 4: Install Alternative OpenCV Bindings

The issue is with `opencv-rust`. You might have better luck with other computer vision libraries for the preview tool.

## Alternative: Use Direct Camera Testing

Instead of the GUI preview, you can:

1. **Test enrollment directly:**
   ```bash
   doorman enroll    # See if face is detected
   ```

2. **Check daemon logs:**
   ```bash
   sudo journalctl -u doormand -f
   ```
   
3. **Use the test command:**
   ```bash
   doorman test     # Tests full authentication
   ```

## Why This Happens

The `opencv` Rust crate uses `bindgen` which depends on `clang-sys`, which dynamically loads `libclang.so` at build time. The loading mechanism is sensitive to:
- Library path configurations
- Symlink structures  
- LLVM version mismatches
- Thread-local initialization

This is a known pain point in the Rust+OpenCV ecosystem, not a doorman bug.

## Bottom Line

**Don't let this block you!** The preview tool is a nice-to-have debugging feature. The core face recognition system works great without it.

