# doorman — Installation (Ubuntu, lean CPU build)

doorman is a face-unlock daemon for Linux. It runs as an always-on system
service; a PAM module (`pam_doorman.so`) asks the daemon to authenticate you at
the login / lock screen and at `sudo`. The default build is **CPU-only** (ONNX
Runtime via the `ort` crate, prebuilt lib bundled — no Python, no CUDA needed).

> macOS is dev/preview only — there is no PAM unlock target there. Everything
> below targets Ubuntu (24.04 LTS reference).

---

## 1. Install build dependencies

```bash
sudo apt update
sudo apt install -y build-essential clang libclang-dev libpam0g-dev libssl-dev pkg-config v4l-utils

# Optional camera backends:
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev   # GStreamer camera backend
sudo apt install -y ffmpeg                                                  # ffmpeg CLI webcam path
```

Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Sanity check:

```bash
make check-deps
```

---

## 2. Build

```bash
git clone https://github.com/doorman/doorman.git
cd doorman
make build
```

This builds, in `--release`:

- `doormand` — the daemon (CPU ONNX Runtime + V4L2 + GStreamer + ffmpeg + mock camera)
- `doorman` — the management CLI (enroll / list / remove / test / status)
- `libpam_doorman.so` — the PAM module

Camera backends are chosen at compile time. Override with
`make build CAMERA_FEATURES=camera-v4l2` if you do not want GStreamer.

---

## 3. Fetch the models

```bash
./scripts/fetch_models.sh        # or: make models
```

Downloads/verifies the three ONNX models (YuNet detector [MIT], EdgeFace-S
recognizer [CC-BY-NC-SA 4.0, **non-commercial**], MiniFASNetV2-SE liveness
[Apache-2.0]) into `data/models/` and your runtime models dir. For the
system service they must end up in `/var/lib/doorman/models` (the installer’s
"next steps" remind you of this — run the fetch as the `doorman` user, or copy
the files in).

> The default EdgeFace-S recognizer is **non-commercial** (CC-BY-NC-SA 4.0). For
> commercial use, swap in **AuraFace-v1** (fal, native ONNX, commercial-OK).

---

## 4. Install (system service)

```bash
sudo make install
```

This:

- installs `doormand` and `doorman` to `/usr/bin`
- installs `pam_doorman.so` to the detected PAM dir
  (e.g. `/usr/lib/x86_64-linux-gnu/security/`)
- creates the `doorman` system user (in the `video` group) and
  `/var/lib/doorman/{,/models}`
- writes a default `/etc/doorman/doorman.toml` (if none exists)
- installs the systemd units and runs `daemon-reload`
- **prints the exact `/etc/pam.d` lines to add** (it does NOT edit PAM for you)

Enable the daemon:

```bash
sudo systemctl enable --now doormand.service
journalctl -u doormand.service -f      # watch it come up
```

---

## 5. Enroll and TEST before touching PAM

```bash
doorman enroll "$USER"     # look at the camera; captures diverse embeddings
doorman list               # confirm you are enrolled
doorman test "$USER"       # runs a real authenticate via the daemon — MUST pass
```

Do not proceed to PAM until `doorman test` succeeds reliably.

---

## 6. Configure PAM — ⚠️ keep a root shell open

> **Before editing any `/etc/pam.d` file, open a second terminal and run `sudo -i`
> (or `su -`) and KEEP IT OPEN.** A mistake here can lock you out of `sudo` and
> the desktop. With a live root shell you can always revert. Test the change
> while that root shell is still open; only log out once it works.

Print the exact instructions any time:

```bash
make pam-instructions
```

Add this as the **first** `auth` line in the file(s) you want:

```
auth    sufficient    pam_doorman.so
```

`sufficient` means a face match logs you in, and **anything else — no match,
daemon down, timeout — falls through to your normal password prompt**. The
module enforces a hard timeout (`AUTH_TIMEOUT_SECS + 1`s) so it can never freeze
the greeter, and returns `PAM_AUTHINFO_UNAVAIL` when the daemon is unreachable.

Where to add it (pick what you use):

| Use case                | File                       | Placement                       |
|-------------------------|----------------------------|---------------------------------|
| `sudo`                  | `/etc/pam.d/sudo`          | above `@include common-auth`    |
| GNOME login + lock      | `/etc/pam.d/gdm-password`  | above `@include common-auth`    |
| All PAM auth (broad)    | `/etc/pam.d/common-auth`   | top of file (riskier)           |

Example `/etc/pam.d/sudo`:

```
#%PAM-1.0
auth       sufficient   pam_doorman.so
@include common-auth
...
```

To undo: delete the `auth sufficient pam_doorman.so` line you added.

Start with `/etc/pam.d/sudo` only — it’s the safest place to validate the whole
chain (`sudo -k; sudo true` triggers a fresh auth) before enabling it for the
display manager.

---

## 7. Preview (optional, dev/desktop)

The preview window shows the live camera with a bounding box that is green when
your face is recognized and red otherwise. It needs the daemon running in
preview mode:

```bash
# Dev: run the daemon in preview mode against your webcam (stays unlocked)
doormand --user --preview

# In another terminal:
doorman-preview        # built by the default `cargo build`
```

For dev without auto-locking the desktop, run the daemon with
`--start-unlocked` (or set `start_locked = false` in the config).

---

## Uninstall

```bash
sudo make uninstall
```

Removes binaries, the PAM module, and the systemd units. It does **not** edit
`/etc/pam.d` — remove the `auth sufficient pam_doorman.so` lines you added
yourself. Enrolled data under `/var/lib/doorman` is preserved.

---

## Appendix A — GPU builds (optional)

CPU is the default and is sufficient for single-camera face unlock. GPU
execution providers are optional and Linux-only.

### AMD ROCm (e.g. Radeon 780M / gfx1103 iGPU)

The ROCm build links **no bundled ONNX Runtime** — the `backend-ort-rocm` feature
enables `ort/load-dynamic`, so the daemon loads a ROCm-enabled `libonnxruntime.so`
at runtime. You supply that library; the stock CPU build does not contain the
ROCm execution provider.

**1. Install ROCm runtime** (Ubuntu 24.04, ROCm 6.x):

```bash
# Follow https://rocm.docs.amd.com for the apt repo, then:
sudo apt install rocm-hip-runtime rocm-libs miopen-hip rocblas
sudo usermod -aG render,video "$USER"   # log out/in afterwards
```

**2. Obtain a ROCm-enabled ONNX Runtime shared library.** Either build ONNX
Runtime with `--use_rocm` (see onnxruntime build docs) or install the
`onnxruntime-rocm` package, then note the path to `libonnxruntime.so`
(e.g. `/opt/onnxruntime-rocm/lib/libonnxruntime.so`).

> Note: some pre-built `onnxruntime-rocm` wheels ship a `libonnxruntime.so` with
> an **executable stack**, which the loader may reject. The repo includes
> `build_onnxruntime_rocm.sh` (builds a clean ROCm ONNX Runtime locally) and
> `run_rocm.sh` (a ready-made launcher that exports `ORT_DYLIB_PATH`,
> `HSA_OVERRIDE_GFX_VERSION=11.0.0` and MIOpen tuning) as a working reference.

**3. Build doorman with the ROCm feature:**

```bash
make build-rocm     # = cargo build --release -p doormand \
                    #     --no-default-features --features backend-ort-rocm,camera-mock,camera-ffmpeg,<cameras>
```

**4. Point the loader + config at ROCm and run:**

```bash
export ORT_DYLIB_PATH=/opt/onnxruntime-rocm/lib/libonnxruntime.so
# gfx1103 (Radeon 780M) is not an official ROCm target; the daemon already sets
# HSA_OVERRIDE_GFX_VERSION=11.0.0 when device="rocm", but you can override it:
# export HSA_OVERRIDE_GFX_VERSION=11.0.0
```

In `/etc/doorman/doorman.toml` (or the user config):

```toml
[ml]
device = "rocm"        # selects the ROCm execution provider (see below)
gpu_device_id = 0
```

**Execution-provider selection & CPU fallback.** With `device = "rocm"` (or
`"gpu"`), `ort_backend.rs` registers `ROCMExecutionProvider` on every model
session. EP registration is **non-fatal** (we use `.build()`, not
`.error_on_failure()`): if the ROCm EP can't be registered — missing
`libonnxruntime.so` symbols, unsupported gfx target, no GPU — ONNX Runtime logs a
warning and **silently falls back to the CPU provider**, so the daemon keeps
working (just slower). Set `RUST_LOG=ort=info,doormand=info` and look for
`Successfully registered ROCMExecutionProvider` to confirm the GPU is actually in
use rather than the CPU fallback.

### NVIDIA CUDA

```bash
# Requires CUDA + cuDNN.
make build-cuda
# then in /etc/doorman/doorman.toml:  [ml] device = "cuda"
```

These map to the `backend-ort-rocm` / `backend-ort-cuda` Cargo features.

### Apple Silicon — CoreML (macOS dev, ANE/GPU)

macOS is a **dev/preview** target (no PAM-unlock), but on Apple Silicon you can
still accelerate inference with the CoreML execution provider (Neural Engine +
GPU + CPU fallback). The bundled ONNX Runtime already ships CoreML support, so
no extra runtime is needed — just the feature flag:

```bash
cargo build --release --features backend-ort-coreml
# then in ~/.config/doorman/doorman.toml:  [ml] device = "coreml"
# or per run:  doormand --device coreml   (use --device cpu to compare)
```

Maps to the `backend-ort-coreml` Cargo feature (macOS-only; enables `ort/coreml`
and registers `CoreMLExecutionProvider` with compute units = ALL, MLProgram
format, and non-fatal `.build()` CPU fallback). On macOS with this feature
built, an unset `device` auto-selects CoreML. Detection (YuNet) runs fully on
ANE/GPU (~5–6× faster); recognition/liveness stay on CPU. See the README
"Apple Silicon acceleration" section and the `coreml_bench` example for the
measured node-placement and latency table.

---

## Appendix B — Configuration

The daemon reads `/etc/doorman/doorman.toml` (system) or
`~/.config/doorman/doorman.toml` (user). See `packaging/doorman.toml.example`
for all keys. Notable ones:

- `daemon.start_locked` (default `true`) — boot locked; the background loop
  unlocks on a recognized face. PAM authentication is independent of this flag.
- `authentication.similarity_threshold` (default `0.4`) — EdgeFace-S match
  threshold; higher = stricter.
- `authentication.timeout_secs` (default `3`) — keep ≤ the PAM hard timeout.
- `authentication.liveness_enabled` (default `true`) — MiniFASNetV2-SE
  anti-spoof; a convenience deterrent, non-fatal on failure.
