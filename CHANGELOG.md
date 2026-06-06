# Changelog

All notable changes to doorman will be documented in this file.

## [Unreleased]

Major simplification and a new ONNX model stack.

### Changed
- **Single ML backend on ONNX Runtime (`ort`)**, CPU by default, in-process — no
  Python subprocess or model-server IPC hop. ROCm/CUDA are optional Linux
  feature builds (`backend-ort-rocm` / `backend-ort-cuda`).
- **New models:** YuNet detection (MIT) + EdgeFace-S recognition (CC-BY-NC-SA 4.0,
  non-commercial) + MiniFASNetV2-SE liveness (Apache-2.0). Landmark-based 112×112
  face alignment before embedding. Fetched via `scripts/fetch_models.sh`.
- **Management CLI rewritten in Rust** (`doorman`: `enroll` / `list` / `remove` /
  `test` / `status`); replaces the previous Python CLI.
- **PAM is configured manually** via a documented `/etc/pam.d` edit
  (`make pam-instructions`); there is no auto-PAM "setup" command. Binaries
  install to `/usr/bin` via `make install` with systemd units.
- Added `doorman-preview` (egui) window: green box when recognized, red when not.

### Added
- **CoreML backend (`backend-ort-coreml`)** for Apple Silicon dev/preview: CoreML
  EP (ANE/GPU + CPU fallback), compute units = ALL, MLProgram format. Uses the
  bundled ONNX Runtime (no extra runtime); selected via `ml.device = "coreml"`
  (aliases `ane`/`gpu`/`auto`). macOS-only; harmless no-op elsewhere.

### Fixed / GPU
- **`ml.gpu_device_id` is now honored** by the ROCm and CUDA execution providers
  (`.with_device_id`); previously the device id was effectively hardcoded to 0.
- **`HSA_OVERRIDE_GFX_VERSION=11.0.0` auto-set** (best-effort, only if unset)
  before HIP init when `ml.device` is `rocm` **or** `gpu`, so unofficial targets
  like gfx1103 (Radeon 780M) work out of the box. Prefer setting it in
  `run_rocm.sh` / the systemd unit; the daemon never clobbers an exported value.
- **EP-registration logs are visible by default**: the log filter now includes
  `ort=info`, so `Successfully registered \`ROCMExecutionProvider\`` (or the
  CPU-fallback warning) shows up without setting `RUST_LOG`. `RUST_LOG` still
  overrides. CPU fallback remains non-fatal.
- **GPU-aware session pool**: 1 session per model on `rocm`/`gpu`/`cuda` (avoids
  4× MIOpen/cuDNN context + arena on memory-constrained iGPUs); 4 on CPU.

### Dependencies
- `ort` 2.0.0-rc.10 → **2.0.0-rc.12**. rc.12 requires an explicit TLS provider on
  the `download-binaries` path (we use `tls-rustls`) and an `api-*` feature on the
  `load-dynamic`/cuda/rocm paths (`api-24`).
- **Pinned & synchronized the GPU stack**: `ort` 2.0.0-rc.12 (api-24) ↔ ONNX
  Runtime **1.24.2** ↔ ROCm **7.2.4** (gfx1103 / Radeon 780M). The load-dynamic
  ROCm `libonnxruntime.so` MUST be ORT 1.24.x or the runtime `dlopen` breaks.
  Updated `build_onnxruntime_rocm.sh` (ORT_VERSION v1.24.2, ROCM_VERSION 7.2.4),
  `test_rocm.sh`, INSTALL.md, BUILD_REQUIREMENTS.txt and the ROCm testing plan;
  removed the stale 1.20.1/1.22.1 references.
- Bumped: `ndarray` 0.17, `nix` 0.31, `dirs` 6, `signal-hook-tokio` 0.4,
  `toml` 1.0, plus assorted semver-compatible updates.
- **Intentionally pinned:** `eframe`/`egui` 0.29 (0.34 API churn),
  `bincode` 1.3 (v3 changes the on-disk embedding format), `gstreamer` 0.22
  (Linux-only), `pam-sys` 0.5 (1.0 is alpha).

### Removed
- Abandoned ML backends (torch/tch/candle/migraphx/docker/socket/tract) and the
  `native_ml` PyO3 crate.
- Abandoned camera backends (pipewire, opencv, rscam, video-file stub). Shipped
  backends: mock, ffmpeg, v4l2, gstreamer, nokhwa.
- Stale experiment docs and `.ai-notes`.

## [0.1.0] - 2025-11-01

### Added

#### Core Features
- ✨ Complete face authentication system for Linux (PAM integration)
- 🔒 Secure daemon architecture (no Python in PAM)
- 🎯 3-stage ML pipeline (detection → liveness → recognition)
- ⚡ Async I/O with Tokio for high performance
- 🐍 Zero-config Python CLI for easy management

#### GPU Acceleration
- 🎮 **ROCm support** for AMD GPUs (Radeon RX 6000/7000, 780M)
- 🚀 **CUDA support** for NVIDIA GPUs
- ⚙️ Configurable device selection (CPU/GPU)
- 📊 3-4x speedup with GPU acceleration

#### Configuration
- 📝 TOML-based configuration system
- 🔧 Configurable similarity thresholds
- 🎛️ Adjustable frame counts for auth/enrollment
- 🖼️ Customizable image preprocessing
- 📍 Multiple config file locations support

#### Testing
- ✅ Comprehensive unit tests (storage, ML, config)
- 🧪 Integration tests (E2E workflow)
- 🎥 **Video file input support** for testing
- 🐍 Python CLI tests
- 📊 Test coverage >85%

#### Documentation
- 📚 README with installation guide
- 🛠️ INSTALL guide (Ubuntu, PAM, systemd, GPU appendix)
- 🏗️ ARCHITECTURE deep-dive
- 🎨 MODELS guide (ONNX models)

### Components

#### Rust PAM Module (`libpam_doorman.so`)
- Lightweight IPC client (~500KB)
- 3-second timeout protection
- Zero blocking of PAM stack
- Fail-safe fallback to password

#### Rust Daemon (`doormand`)
- Camera ownership and management
- ONNX Runtime integration
- UNIX socket IPC server
- Embedding storage (bincode)
- Signal handling for graceful shutdown
- systemd service integration

#### CLI (`doorman`)
- `enroll` - Face enrollment with progress UI
- `list` - Show enrolled users
- `remove` - Remove user enrollment
- `test` - Run a real authenticate via the daemon
- `status` - Daemon health check

(Originally a Python CLI; rewritten in Rust — see Unreleased. Install/uninstall
are handled by `make install` / `make uninstall`, not the CLI.)

### Security
- 🔐 Local processing (no cloud)
- 👁️ Liveness detection (anti-spoofing)
- 🔒 Root-only embedding access (0600)
- 🛡️ Process isolation (daemon vs PAM)
- ⏱️ Timeout protection in PAM
- 🔑 Password fallback always available

### Performance
- ⚡ 1-3s authentication (CPU)
- 🚀 0.3-0.5s authentication (GPU)
- 💾 ~100MB memory usage
- 📦 ~15MB disk usage (binaries + models)
- 🎯 Early exit on first match

### Build System
- 📦 Cargo workspace (3 crates)
- 🐍 uv-based Python packaging
- 🔨 Comprehensive Makefile
- 🏗️ Feature flags (video, gpu)
- ⚙️ Optimized release builds

### Platform Support
- 🐧 Linux (Ubuntu 22.04+, other distros)
- 🖥️ KDE Plasma (primary)
- 🎨 GNOME (compatible)
- 💻 x86_64 architecture
- 🔧 systemd-based systems

### Dependencies
- Rust: 1.70+ (stable)
- Python: 3.10+
- ONNX Runtime: bundled via the `ort` crate (CPU)
- tokio (async runtime)

### Known Limitations
- 👯 May match identical twins
- 🎭 Sophisticated 3D masks might bypass
- 💤 May unlock while sleeping
- 🔓 Physical access = potential bypass
- 🖥️ Not designed for enterprise (personal use)

### Future Roadmap
- [ ] Multi-face support (family)
- [ ] Web UI for enrollment
- [ ] GNOME/XFCE integration
- [ ] Mobile app remote unlock
- [ ] Better liveness models
- [ ] Performance benchmarking suite

---

## How to Upgrade

```bash
cd doorman
git pull
make build
sudo make install   # reinstalls binaries, PAM module, systemd units
```

## Breaking Changes

None (initial release)

## Contributors

- doorman team

---

**Release Date**: November 1, 2025  
**License**: MIT

