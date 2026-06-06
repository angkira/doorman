# doorman

Fast, lightweight face unlock for Linux — a cleaner-architected alternative to
howdy.

**3 components**: PAM module (`pam_doorman.so`, Rust) → auth daemon (`doormand`,
Rust) → management CLI (`doorman`, Rust).

**Key idea**: the daemon owns the camera and the ONNX models. PAM just sends an
IPC request and falls through to your password on any non-match, so it can never
lock you out or freeze the greeter.

Inference runs **in-process** via ONNX Runtime (the `ort` crate) — no Python
subprocess, no IPC hop to a model server. The default build is **CPU-only** and
ships the ONNX Runtime CPU library bundled; ROCm/CUDA are optional Linux feature
builds.

> macOS is dev/preview only (there is no PAM unlock target there). Production
> unlock targets Ubuntu (24.04 LTS reference).

## Install

Full, careful instructions (build deps, models, systemd, the manual `/etc/pam.d`
edit) are in [INSTALL.md](INSTALL.md). Short version on Ubuntu:

```bash
sudo apt install -y build-essential clang libclang-dev libpam0g-dev libssl-dev pkg-config v4l-utils
# optional camera backends:
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev ffmpeg

git clone https://github.com/doorman/doorman.git
cd doorman
make build            # builds doormand + doorman + pam_doorman.so (release)
./scripts/fetch_models.sh   # or: make models
sudo make install     # installs binaries, PAM module, systemd units; prints the PAM edit
```

`doorman` does **not** auto-edit PAM. `sudo make install` (or
`make pam-instructions`) prints the exact `auth sufficient pam_doorman.so` line
to add yourself — see [INSTALL.md §6](INSTALL.md) and keep a root shell open
while you do it.

## Models

Three ONNX files, fetched by `scripts/fetch_models.sh` into `data/models/` and
your runtime models dir (`/var/lib/doorman/models` for the system service):

1. `face_detection_yunet_2023mar.onnx` — detection (YuNet, **MIT**)
2. `edgeface_s.onnx` — recognition, 512-d embeddings (EdgeFace-S,
   **CC-BY-NC-SA 4.0 — non-commercial**)
3. `minifasnet_v2se.onnx` — anti-spoofing / liveness (MiniFASNetV2-SE,
   **Apache-2.0**)

> The default EdgeFace-S recognizer weights are **CC-BY-NC-SA 4.0
> (non-commercial only)**. For commercial use, swap in **AuraFace-v1** (fal,
> native ONNX, commercial-OK). See [MODELS.md](MODELS.md) for details.

Sources:
- YuNet: https://github.com/opencv/opencv_zoo
- EdgeFace: https://github.com/otroshi/edgeface
- MiniFASNetV2-SE: https://github.com/facenox/face-antispoof-onnx

## Usage

The `doorman` CLI talks to the daemon over its IPC socket:

```bash
doorman enroll "$USER"      # look at the camera; captures diverse embeddings
doorman list                # show enrolled users
doorman test "$USER"        # run a real authenticate via the daemon
doorman remove "$USER"      # remove an enrollment
doorman status              # daemon version, uptime, camera, models, users
```

**⚠️ Run `doorman test` and confirm it passes reliably BEFORE adding the PAM
line.** Then lock the screen (Meta+L) or run `sudo -k; sudo true` to exercise
face unlock for real.

The daemon itself is managed via systemd:

```bash
sudo systemctl enable --now doormand.service
sudo systemctl restart doormand.service
journalctl -u doormand.service -f
```

## Preview window

`doorman-preview` (built by the default `cargo build`) shows the live camera
with a bounding box that is **green when your face is recognized, red
otherwise**, plus the matched name and similarity score. Run the daemon in
preview mode first:

```bash
doormand --user --preview        # dev: webcam, stays unlocked with --start-unlocked
doorman-preview                  # in another terminal
```

## Performance

There is no Python subprocess or IPC model hop, so the old detection-FPS
collapse (~0.5 fps in the experimental PyTorch-IPC era) is gone. Detection input
is rate-capped by `[camera] processing_fps` (default 10).

Measured on Apple Silicon CPU (dev), 640×480, default config:

- **Sustained detection: ~8.6 fps** (capped at the 10 fps target; every 3rd
  frame runs the full detect → liveness → embed pipeline, the rest detect-only).
- **Enrollment: ~11 s end-to-end** (10 s recording + ~1.4 s processing). The
  daemon evenly samples up to 30 recorded frames before inference so a long
  recording does not serialize hundreds of CPU inferences or starve the live
  detection loop.

A ROCm/CUDA GPU build raises throughput further; CPU is sufficient for
single-camera unlock.

## GPU acceleration (optional, Linux-only)

```toml
# /etc/doorman/doorman.toml
[ml]
device = "rocm"   # or "cuda" for NVIDIA
gpu_device_id = 0
```

Build with `make build-rocm` (AMD) or `make build-cuda` (NVIDIA) and set
`ORT_DYLIB_PATH` to a GPU-enabled `libonnxruntime.so`. The ROCm/CUDA execution
provider falls back to CPU automatically if the GPU runtime is unavailable. Full
setup: [INSTALL.md, Appendix A — GPU builds](INSTALL.md#appendix-a--gpu-builds-optional).

## Config

Edit `/etc/doorman/doorman.toml` (system) or `~/.config/doorman/doorman.toml`
(user). See `packaging/doorman.toml.example` for all keys.

```toml
[authentication]
similarity_threshold = 0.4   # EdgeFace-S cosine threshold; higher = stricter
auth_frames = 10
liveness_enabled = true      # MiniFASNetV2-SE anti-spoof; non-fatal deterrent

[ml]
backend = "ort"
device = "cpu"               # or "rocm", "cuda"

[daemon]
start_locked = true          # boot locked; background loop unlocks on a match
```

## Camera backends

Backends are selected at compile time (Cargo features). The daemon auto-selects
a working one at runtime.

| Backend     | Feature             | Platform       | Notes                          |
|-------------|---------------------|----------------|--------------------------------|
| mock        | `camera-mock`       | any            | synthetic / video-file dev     |
| ffmpeg      | `camera-ffmpeg`     | macOS / Linux  | avfoundation / v4l2 via ffmpeg |
| v4l2        | `camera-v4l2`       | Linux          | direct V4L2                    |
| GStreamer   | `camera-gstreamer`  | Linux          | recommended for production     |
| nokhwa      | `camera-nokhwa`     | macOS          | AVFoundation webcam (dev)      |

Override the build set with e.g. `make build CAMERA_FEATURES=camera-v4l2`.

## Architecture

The daemon runs a 4-stage non-blocking tokio pipeline:

```
Camera Producer → Frame Fanout → Detection → Recognition
                       ↓
                    Preview / frame + debug sockets
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Testing

```bash
make test                       # Rust unit + integration tests
make test-docker                # Ubuntu container build/run (Dockerfile.test)
```

Model-dependent tests skip automatically when the ONNX files are absent, so a
fresh clone still builds and tests without fetching models.

## Security

**Good for**: personal-workstation convenience.
**Not for**: high-security or shared machines.

Password fallback is always available (PAM `sufficient`). Enrolled embeddings
are stored root-only (0600). Liveness is a convenience deterrent, not a
high-assurance anti-spoof.

## License

MIT (project code). Model weights carry their own licenses — see
[MODELS.md](MODELS.md): YuNet **MIT**, MiniFASNetV2-SE **Apache-2.0**, EdgeFace-S
**CC-BY-NC-SA 4.0 (non-commercial)**.
