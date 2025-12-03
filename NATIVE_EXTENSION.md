# Native Extension Build Issues

## Problem

Native backend requires linking against libpython, but:

1. **System Python 3.13**: Has onnxruntime-rocm executable stack bug
   ```
   ImportError: cannot enable executable stack as shared object requires: Invalid argument
   ```

2. **Venv Python 3.12**: No shared library (libpython3.12.so)
   ```
   rust-lld: error: unable to find library -lpython3.12
   ```

3. **System Python 3.12**: Not installed (only 3.13 available)

## Working Solutions

### Option 1: IPC Backend (Currently Working)
```bash
cargo build --release --features backend-torch,camera-gstreamer
./target/release/doormand --config doorman-torch.toml
```

**Performance**: 8 FPS (Base64 overhead)
**Status**: ✅ Works reliably

### Option 2: Install Python 3.12 System-Wide

```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-dev python3.12-venv

# Then rebuild
export PYO3_PYTHON=/usr/bin/python3.12
cargo build --release --features backend-torch-native,camera-gstreamer
```

**Performance**: 36 FPS (native, no IPC)
**Status**: ⚠️ Requires Python 3.12 installation

### Option 3: Use PyO3 ABI3 (Stable ABI)

Build with stable ABI to work with any Python 3.x:

```toml
# Cargo.toml
[dependencies]
pyo3 = { version = "0.22", features = ["abi3-py38"] }
```

**Performance**: 36 FPS
**Status**: �� Requires code changes

## Testing Native Extension

Native extension works in isolation:

```bash
# Works with Python 3.12 venv
source .venv/bin/activate
python3 -c "import doorman_ml_native; ml = doorman_ml_native.DoormanML(...)"
# ✓ SUCCESS - 36 FPS
```

But daemon can't link against venv Python!

## Recommended Path Forward

**For Development**: Use IPC backend (8 FPS but reliable)

**For Production**: 
1. Install Python 3.12 system-wide
2. Or use containerized deployment (Docker)
3. Or implement shared memory IPC (50-55 FPS)

## Current Status

- ✅ Native extension builds and works
- ✅ iGPU performance excellent (36 FPS)
- ✅ IPC backend functional (8 FPS)
- ❌ Daemon + native backend linkage broken
- ⏳ Need Python 3.12 system installation

## Performance Summary

| Backend          | FPS  | Status | Notes                        |
|------------------|------|--------|------------------------------|
| Tract (CPU)      | 7.9  | ✅     | Pure Rust, reliable          |
| Torch IPC        | 8    | ✅     | Works, Base64 overhead       |
| Torch Native     | 36   | ❌     | Build issue (libpython3.12)  |
| Torch Shared Mem | 50+  | 📝     | TODO: eliminate Base64       |
