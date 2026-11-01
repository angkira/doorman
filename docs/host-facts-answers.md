# Answers to the host-fact questions

Source of the facts: `.plans/2026-11-01_doorman-hardening-latency/HOST-FACTS.md`
(owner's host, 2026-11-01). "Installed build" means the binary the host runs,
believed to be about commit `5f78657` (2026-06-24, "Capture auth attempts as
positive and false-negative packs"). Paths below are relative to the repository;
`@5f78657` means `git show 5f78657:<path>`. Statements are **fact** (read from
code or from the collected output), **inference** (follows from facts, not
proven) or **hypothesis** (needs a host command to settle).

## 1. Why the nokhwa/V4L2 open failed in ~0.1 ms

**Fact: the request was a mode the camera does not offer.**

- `@5f78657:daemon/src/camera/v4l2_backend.rs:51-55` builds
  `CameraFormat::new(resolution, FrameFormat::YUYV, config.camera.fps)` and asks
  nokhwa for `RequestedFormatType::Closest(..)`, with the config width/height.
  The host config is `1280x720 @ 30` (HOST-FACTS section 5), so the request was
  **YUYV 1280x720@30**. `shared/src/config.rs` `default_camera_width/height`
  are also 1280/720.
- HOST-FACTS section 1: YUYV is offered only at 640x480, 640x360, 480x320,
  320x240, 800x600 (15 fps), 1024x720 (15), 1280x960 (10), 1920x1080 (5), ...;
  **1280x720 is not offered as YUYV**. MJPG offers 1280x720 at 60 and 30 fps.
- nokhwa's `Closest` fails with "Failed to Fufill" when the exact resolution is
  missing in the requested format (comment in `daemon/src/camera/format_select.rs:1-8`,
  `docs/camera.md`).
- The error text was hidden: `@5f78657:daemon/src/camera/mod.rs:128` logged
  `{}` of an `anyhow` error with only the outer context
  (`.context("Failed to open camera")`, `v4l2_backend.rs:55`), which is exactly
  the journal line `V4L2/nokhwa camera failed: Failed to open camera`.
  The failure is immediate (0.1-0.4 ms) because no streaming is attempted, only
  format negotiation.

**Current code (HEAD).** `daemon/src/camera/v4l2_backend.rs:94-132` opens with
`RequestedFormatType::None`, lists `compatible_camera_formats()`, picks a mode
with `select_mode` (`format_select.rs:65`) and requests it with
`RequestedFormatType::Exact`. With the default request (1280x720@30, no
`fourcc`), exact size beats format preference (`format_select.rs:32-37,65-95`),
so the only 720p@30 mode, **MJPG 1280x720@30**, is chosen. Errors carry the full
nokhwa error and the device list (`v4l2_backend.rs` `known_devices`;
`camera/mod.rs:153` logs `{:#}`).

**Test.** `host_camera_default_720p30_opens_mjpeg_not_yuyv` and
`host_camera_small_sizes_keep_yuyv_when_unset` (`format_select.rs:185-260`) use
the exact offered list from HOST-FACTS and assert: default 1280x720@30 resolves
to MJPG 1280x720@30 (never YUYV), an explicit `fourcc = "YUYV"` falls back to
MJPG at that size, and 640x480 without `fourcc` stays YUYV.

Still host-only: that nokhwa then streams MJPG 1280x720 without blocking
(the old comment at `@5f78657` v4l2_backend.rs:32-45 records a first-DQBUF hang
in an earlier variant). Run `doorman test` and read the journal line
`V4L2 camera opened: ... Mjpeg 1280x720`.

## 2. Which model produced the "depth gate" at 5f78657

**Fact: the only code that logs "Liveness REJECTED (depth gate)" is the
Depth-Anything-V2 relief gate; MiniFASNet is not involved.**

- `@5f78657:daemon/src/ml/mod.rs:144,195` (`process_frame`, `liveness_gate`)
  print the line; both call `backend.depth_relief()`. The comment at
  `mod.rs:135` says the old MiniFASNet `check_liveness` is "intentionally NOT used".
  The auth path calls `liveness_gate` once, on the first quality frame
  (`@5f78657:daemon/src/ipc.rs:286-309`).
- `depth_relief` (`@5f78657:daemon/src/ml/ort_backend.rs:1009-1022`) dispatches on
  `DepthSource`:
  - `InProcess(pool)` for a CPU device (`ort_backend.rs:268-287`): loads
    `models_dir/depth_anything_v2_small_fp32.onnx`
    (`@5f78657:daemon/src/ml/model_config.rs:160`, 518x518, ImageNet
    normalisation). An empty pool (file missing) returns `Ok(None)`, and the
    gate then logs `depth-PAD model not loaded`, **not** `relief=...`.
  - `Sidecar(sock)` for rocm/cuda/gpu (`ort_backend.rs:262-267`): the
    `doorman-depthd` process, which loads the same ONNX file.
  - The `ncnn` hybrid (`@5f78657:daemon/src/ml/mod.rs:57-60`,
    `ncnn_hybrid.rs:33-34`) is used only when `ml.backend == "ncnn"`; it needs
    `models_dir/depth.ncnn.param|bin` and computes the same relief on the iGPU.
- There is **no fallback model and no other search path**: every model is
  `models_dir.join(file)` (`ort_backend.rs:193,213,269,291`), with `models_dir`
  taken from `[ml] models_dir` (default `/var/lib/doorman/models`,
  `shared/src/config.rs` `default_models_dir`) and overridden only in user mode
  (`@5f78657:daemon/src/main.rs:311`). `/usr/share/doorman/models` is never read
  by the daemon; the deb `postinst` copies from it
  (`packaging/deb/postinst`, loop over four model files).
- The host log line `relief=0.0491 < threshold=0.0500` is the depth-relief
  formatting (`mod.rs:195`); the threshold 0.05 is
  `default_liveness_depth_threshold` (`shared/src/config.rs`). The host config
  has `backend = "ort"`, `device = "cpu"`, so the **in-process CPU
  Depth-Anything-V2-Small fp32 session** produced it (inference from config +
  code; the ncnn and sidecar paths are excluded by `backend`/`device`).

**Why `/usr/share/doorman/models` has no depth model, and `models_dir`
"does not exist".** Both observations are explained without any hidden
fallback:

- The deb ships only four files: yunet, edgeface_s, minifasnet_v2se(+int8)
  (`scripts/build_deb.sh:98-103`, `check_deb_packaging.sh`); the depth model is
  "not vendored" (`MODELS.md`, `@5f78657:daemon/src/install.rs:1360-1380`).
- `doorman install` copies the full set, "includes
  depth_anything_v2_small_fp32.onnx", from `~/.local/share/doorman/models` (or
  `data/models`) **into `/var/lib/doorman/models`**
  (`@5f78657:daemon/src/install.rs:1363-1383`; for `--ncnn` also
  `depth.ncnn.param|bin`, `install.rs:1396-1417`). The unit header "EP: ncnn"
  says that this installer ran with `--ncnn` (`install.rs` `render_unit`).
- HOST-FACTS section 3 ran `ls -l /var/lib/doorman 2>&1 | head -3`. The output
  has exactly three lines (`total 84`, `dataset`, `embeddings.bin`): `head -3`
  **cut the listing after `embeddings.bin`**, and `models` sorts later. So
  "models_dir does not exist" is **not established** by that output.
  Inference supporting that it exists: `ls -ld /var/lib/doorman` shows size 78;
  on btrfs-style directory sizes (2 x sum of entry-name lengths) the names
  `dataset`(7) + `embeddings.bin`(14) + `models`(6) + `miopen`(6) + `.cache`(6)
  = 39 -> 78, which are exactly the entries the installer and `postinst` create
  (`install.rs` state-dir steps, `packaging/deb/postinst`).

**Still hypothesis / host check:**
`ls -l /var/lib/doorman/models` should list `depth_anything_v2_small_fp32.onnx`
(and `depth.ncnn.*` if converted). If it is absent, the journal at startup would
show `Depth-PAD model ... unavailable` and every attempt would fail with
`depth-PAD model not loaded`, which the host log does not show.

## 3. Which ONNX Runtime is loaded

**Fact (packaging).**

- `daemon/Cargo.toml:127`: `backend-ort = ["_ort","ort","ort/download-binaries","ort/tls-rustls"]`
  -> the `ort` crate **statically links a prebuilt ONNX Runtime** downloaded at
  build time (pyke, the version that `ort 2.0.0-rc.12` targets, ORT 1.2x).
  `default = ["backend-ort", ...]` (`Cargo.toml:114`). The deb builds with
  `DOORMAN_DEB_FEATURES` default `backend-ort,camera-mock,camera-ffmpeg,camera-v4l2`
  (`scripts/build_deb.sh:45`, identical at 5f78657). The installer's `Ep::Ncnn`
  feature is `backend-ort,backend-ncnn` (`install.rs` `Ep::cargo_feature`),
  also `download-binaries`.
- Only `backend-ort-rocm` uses `ort/load-dynamic` (`daemon/Cargo.toml:137`),
  i.e. `dlopen` of `$ORT_DYLIB_PATH` (or the bare name `libonnxruntime.so`
  through the linker search path). `render_unit` sets
  `ORT_DYLIB_PATH=/usr/lib/doorman/libonnxruntime.so` **only for `Ep::Rocm`**
  (`daemon/src/install.rs:757` at HEAD; `:713` at 5f78657) and so does the
  depth sidecar unit (`:865`). The host unit (EP ncnn) has no `ORT_DYLIB_PATH`
  and no `LD_LIBRARY_PATH`.

**Fact (host).** `readelf -d /usr/bin/doormand`: NEEDED only libstdc++, libgcc_s,
libm, libc, ld-linux; `ldd` shows no onnx/ncnn/vulkan/avcodec/gst/v4l. That is
what a static ORT looks like. A load-dynamic build would `dlopen` at runtime and
also show no NEEDED entry, so this alone does not exclude it, but a bare
`dlopen("libonnxruntime.so")` would fail on this host: `ldconfig -p` lists only
`libonnxruntime.so.1.21` (no unversioned symlink) and `/usr/lib/doorman` is not
on a search path.

**Conclusion (inference, one host command from certain):** the installed build
runs its **bundled, statically linked** ONNX Runtime. Neither the distro
`libonnxruntime.so.1.21` (a package that happens to be installed) nor
`/usr/lib/doorman/libonnxruntime.so` + the 259 MB ROCm provider (left over from
an earlier ROCm install, dated Jun 12) is loaded by this unit.
`/proc/<pid>/maps` was unreadable for the unprivileged collector; run as root:
`grep -i onnx /proc/$(systemctl show -p MainPID --value doormand)/maps`
(no match = static).

**New in this change.** `ml::ort_backend::runtime_info()` (logged at startup as
`ONNX Runtime: ...` and reported by `Status`/bench) now ends with
`ORT_DYLIB_PATH=<value|unset>; mapped <paths from /proc/self/maps>` or
`no libonnxruntime mapping (runtime is statically linked)`
(`daemon/src/ml/ort_probe.rs`, `onnxruntime_mappings` is a pure function tested
on a synthetic maps string). `ort::info()` is evaluated first so a lazily
`dlopen`ed library is already mapped when `/proc/self/maps` is read.
The process reads its own maps, so no extra privilege is needed.

## 4. GPU access and the "EP: ncnn" header

- The generated unit used to grant `DeviceAllow=char-drm`, `/dev/dri`,
  `/dev/kfd` and `SupplementaryGroups=video render` for every EP
  (`render_unit`, comment: needed for "ncnn/Vulkan (radv)"). They are removed
  (see `docs/systemd-hardening.md`); the host's `10-drm.conf` drop-in is not
  generated by any code in this repository (only a comment in the header
  explains `char-drm`); remove it on the host by hand.
- Which backend does depth/liveness use in the installed build? Despite the
  header, **ORT on the CPU EP**: the config written to the host has
  `backend = "ort"`, `device = "cpu"`, and ExecStart passes `--device cpu`.
  The ncnn/Vulkan path only runs when `[ml] backend = "ncnn"`
  (`@5f78657:daemon/src/ml/mod.rs:57`). `render_config` writes that value for
  `--ncnn` installs (`install.rs:1294-1299` at 5f78657), yet the host file says
  `ort` and its header is the `make install` template, so the config on disk was
  not (or no longer) the rendered one (hypothesis: the deb `postinst` or a manual
  copy of the example config replaced it; the unit header and config disagree).
  The "radv ... Operation not permitted" comment describes an earlier ncnn run,
  not the current configuration. Host check: `journalctl -u doormand | grep -i
  'ML backend'` shows `Using ML backend: ...` (the ncnn hybrid reports a
  different name from the plain ORT backend).
