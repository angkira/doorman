# Camera: backends, open failure analysis, latency options

Status: written without access to the camera. Everything marked **proved** is
established by reading the code (file:line at the commit that introduced this
document); everything marked **hypothesis** needs the host output requested in
"Paste here" below.

## Backends and what the packaged builds enable

`Camera::new_with_config` (`daemon/src/camera/mod.rs`) tries, in order:
GStreamer (feature `camera-gstreamer`), V4L2 via nokhwa (`camera-v4l2` or
`camera-nokhwa`), ffmpeg CLI (`camera-ffmpeg`), then the mock camera.

| Build entry point | Camera features | Native V4L2 compiled in? |
|---|---|---|
| `daemon/Cargo.toml` default | `camera-mock, camera-ffmpeg, preview` | **no** |
| `Makefile` (`CAMERA_FEATURES ?= camera-v4l2,camera-gstreamer`) | mock, ffmpeg, v4l2, gstreamer | yes |
| `Dockerfile.test` | mock, ffmpeg, v4l2, gstreamer | yes |
| `build.sh` | `camera-gstreamer` only (plus `backend-tch`) | no |
| `install.sh` option 2 ("v4l") | `camera-v4l` | **no such feature**: cargo rejects the build |

Consequence for the "V4L2/nokhwa camera failed" log line: that line is only
emitted by a binary that was built with `camera-v4l2` or `camera-nokhwa`
(`mod.rs`, the `#[cfg(any(feature = "camera-v4l2", ...))]` block). If the host
log shows it, the backend **is** compiled in and the failure is a real open
failure, not a stub. If a host build never prints either "Using V4L2/nokhwa" or
"V4L2/nokhwa camera failed", the backend is not compiled in; check the build
command against the table above.

## Why the native open failed (ranked)

The reported failure is `Failed to open camera` after about 0.1 ms. Two things
are **proved**:

1. The underlying error was never logged. The backend wrapped the nokhwa error
   with `.context("Failed to open camera")` and `mod.rs` printed it with `{}`,
   which shows only the outermost context. The real cause was discarded. The
   warning now prints `{:#}` (full chain) and the open errors include the nokhwa
   error and the list of devices nokhwa can see.
2. nokhwa 0.10.11 (`nokhwa-core-0.1.9/src/types.rs`, `RequestedFormat::fulfill`,
   `Closest` arm) returns `None` ("Failed to Fufill", surfaced from
   `nokhwa-bindings-linux-0.1.4/src/lib.rs` `V4LCaptureDevice::new`) in these
   cases, all without touching the device again, hence the sub-millisecond time:
   - the device offers no mode in the requested pixel format (the old request
     was `Closest(YUYV, width x height, fps)`), or
   - the exact requested resolution is not offered in that pixel format: the
     frame-rate lookup filters on the *requested* resolution, not the nearest
     one it picked, so it comes back empty.

Hypotheses, most likely first:

1. **Requested size not offered as YUYV** (default config is 1280x720). UVC
   webcams commonly offer raw YUYV only at small sizes or low frame rates and
   MJPEG at 720p/1080p. The code's own comment records that this webcam exposes
   1024x720 as YUYV only. Fits the 0.1 ms failure and the "typical culprit" in
   the plan. Fixed in code by negotiating from the device's real mode list.
2. **`device_index = 0` is not the capture node.** UVC cameras expose a second
   `/dev/videoN` for metadata (and IR cameras another capture node). nokhwa
   `enum_formats` on a metadata node yields no usable formats, which ends in the
   same "Failed to Fufill". Check `v4l2-ctl --list-devices`; set
   `[camera] device_index` to the first node of the camera (usually 0, with the
   metadata node being `/dev/video1`).
3. **Device busy**, e.g. another process (a previous ffmpeg child, a browser)
   holds `/dev/video0`. The full error now says so.
4. **Permissions**: the daemon user is not in the `video` group. Also surfaces in
   the full error now.

## What changed in the code

- `format_select.rs`: pure mode selection from the device's offered modes.
  Exact size first (YUYV preferred, then MJPEG; MJPEG first when
  `fourcc = "MJPG"`), else nearest size; frame rate closest to the request.
  Errors list every offered mode.
- `v4l2_backend.rs`: opens the device without a mode, lists formats, selects,
  requests exactly that mode. Native MJPEG frames are decoded by nokhwa
  (mozjpeg, already in the dependency tree); no extra decoder was added.
  Synthetic 640x480 JPEG decode measured at 0.95 ms/frame (release build, single
  thread; a unit test prints the number, it does not assert it).
- `ffmpeg_backend.rs`: with `fourcc = "MJPG"` the V4L2 input gets
  `-input_format mjpeg` (`-video_size` and `-framerate` were already input-side).
  Output is still rawvideo rgb24, so nothing downstream changes.
- `camera/gate.rs`: pure `camera_wanted()` gate and `WarmupGate`.
- `pipeline/camera_producer.rs`: warm-up discard on cold opens, one summary line.

## Config (`[camera]`, all optional; old configs parse unchanged)

| Key | Default | Meaning |
|---|---|---|
| `backend` | `"auto"` | `"auto"`: native V4L2 first, accepted only after it decoded one real frame within `native_first_frame_timeout_ms`, otherwise the ffmpeg backend. `"native"`: native only, no ffmpeg fallback. `"ffmpeg"`: skip native, use ffmpeg (the pre-native behaviour; use it as an override if the native path misbehaves on a host). Anything else is rejected at config load. |
| `native_first_frame_timeout_ms` | `1500` | How long the native backend may take to deliver its first decoded frame. Valid 100..=10000. A timeout, a capture/decode error or a panic closes the native camera and moves on (see "Native first-frame validation"). |
| `native_warmup_frames` | `3` | Native backend only. Buffers read and dropped unexamined right after STREAMON before one may count as the first frame (UVC devices can deliver junk or partial buffers first). Every buffer after that is validated (see "Round 7") and, if rejected, skipped; the loop retries inside `native_first_frame_timeout_ms` instead of failing on frame 1. Valid 0..=10. Unrelated to `warmup_frames` (the producer's cold-open discard). |
| `native_failure_cooldown_secs` | `600` | `backend = "auto"` only. After the native backend failed in this daemon process, the next opens for the same device and requested mode skip it and go straight to ffmpeg for this many seconds (one INFO line: `native backend skipped: failed Ns ago: <reason>`), then native is probed again. The memory is per process: a restart re-probes. `0` disables it. Valid 0..=86400. `backend = "native"` ignores it (no fallback to protect); `backend = "ffmpeg"` never tries native. |
| `fourcc` | unset | `"MJPG"` or `"YUYV"`. Unset keeps today's behaviour. |
| `warmup_frames` | `0` | Frames dropped after a cold open. Valid 0..=10. Recommended: `3`. |
| `prearm_on_lock` | `false` | Open on the lock event even while `IdleHint` is set, but only for `prearm_window_ms` after it. Recommended: `true`. |
| `device_path` | unset | Pin the camera by a stable node such as `/dev/v4l/by-id/usb-..-video-index0` (resolved on every open; must end at a `/dev/videoN` character device). Overrides `device_index`. Recommended, because index numbering changes when devices enumerate in another order. |
| `allow_virtual_devices` | `false` | Virtual capture devices (v4l2loopback, vivid, akvcam, ...) are refused. Set `true` only for testing. See "Virtual cameras" below. |
| `prearm_window_ms` | `5000` | How long after the `LockedHint` false->true edge prearm may keep the camera open while idle. Valid 0..=30000 (0 disables prearm). After the window the normal `locked && !idle` rule applies, so a screen that stays off for hours does not keep the camera on. |

Defaults are unchanged on purpose: they change auth timing, so they are turned on
after the host measurement, not before. The default resolution is also
unchanged. The recommended starting point to measure is
`width = 640, height = 480, fourcc = "MJPG", warmup_frames = 3,
prearm_on_lock = true`, pending the WS6 recall comparison against 1280x720.

## Native first-frame validation and producer recovery

Host test of PR #13 (UGREEN Camera 4K, MJPG 1280x720@30): after the format
negotiation fix the native open succeeded, so the ffmpeg fallback was never
reached, and the producer thread then died about 570 ms after "Camera gated ON"
with no "Camera ready", no "Capture error" and no panic text.

**Proved by code:** the producer thread has exactly two ways to end: the
"all receivers gone" return (the receiver was alive, so not this one) and an
unwinding panic. There is no `?` in the thread. So it was a panic (or a
destructor panic) on the native capture path. **Not proved:** the exact panic
site; the candidates on that path are the nokhwa/v4l destructors
(`v4l-0.14.0` `Stream::drop` calls `panic!` on any STREAMOFF error except
ENODEV; `nokhwa::Camera::drop` calls `stop_stream().unwrap()`) and the mozjpeg
decode. Also note `v4l` mmap `dequeue` polls with no timeout, so a camera that
never delivers blocks `frame()` forever. This has not been reproduced on a host.

What the code now does regardless of the exact site:

- `v4l2_backend.rs` reads and decodes one real frame (on a helper thread, with
  `native_first_frame_timeout_ms`) before declaring the backend usable. On error,
  panic or timeout the camera is closed and `Camera::new_with_config` falls
  through to ffmpeg. Limit: on a timeout the blocked read cannot be interrupted,
  so the device stays held by the helper until that read returns and ffmpeg may
  find it busy; error and panic paths close the device before falling through.
- `camera_producer.rs` catches panics: the panic message is logged at ERROR, the
  camera is closed, consumers see "unavailable" (`latest_frame` cleared) and the
  next loop iteration re-opens it (and may fall back to ffmpeg). An unexpected
  thread exit is also logged at ERROR.

## Round 7: "Not a JPEG file: starts with 0x61 0xe8"

Owner's host journal (UGREEN Camera 4K, uvcvideo, MJPG 1280x720@30):
`native camera opened at DeviceMode { format: Mjpeg, 1280x720, 30 fps } but
delivered no usable first frame ... first frame read panicked: libjpeg fatal
error: Not a JPEG file: starts with 0x61 0xe8`, then ffmpeg works. So the
buffer handed to the decoder was not an MJPEG frame (no FF D8).

**Proved by code reading** (nokhwa 0.10.11, nokhwa-bindings-linux 0.1.4, v4l 0.14.0):
`v4l`'s mmap `next()` returns the whole mapped buffer, not `bytesused` bytes, and
`nokhwa::Camera::frame()`/`frame_raw()` pass it on without `bytesused`. A good
frame is therefore a JPEG plus slack, and truncation cannot be seen from the
length. The bindings already read the format back after `set_camera_format` and
return an error on a mismatch.

**Not proved (hypotheses until the new WARN lines are seen on the host):** why
the buffer began with `61 e8`: junk/partial first buffers after STREAMON, a
buffer for another format, or something inside nokhwa/the driver.

What the code does now:

- **Negotiated format is checked.** After the stream is open the driver's
  format is read back; a pixel format or resolution different from the request
  (or a format this backend cannot decode) fails the open with an explicit
  error. A different fps is tolerated.
- **Every candidate buffer is validated before libjpeg/mozjpeg sees it**
  (`camera/native_frames.rs`): MJPEG must start with FF D8 and a JPEG marker
  walk must reach FF D9; the decoder receives exactly the bytes up to that EOI
  (the slack is cut). YUYV must hold `width*height*2` bytes. The `catch_unwind`
  boundary stays as the last line of defence. Limit: without `bytesused`, a
  truncated frame whose missing tail was overwritten by an older frame's stale
  bytes can pass the walk; the decoder then rejects it and the frame is dropped.
- **Warm-up and retry.** The first `native_warmup_frames` buffers are dropped,
  then rejected candidates are skipped until one decodes or the budget
  (`native_first_frame_timeout_ms` minus up to 100 ms of slack) is spent. A read
  error fails at once.
- **Rejection log** (WARN, one line per rejection, after the first five per
  camera instance DEBUG): reason, the first 8 bytes of the buffer as hex (never
  more image data), buffer length, `bytesused` (`unavailable` through nokhwa),
  negotiated fourcc, resolution and frame index.
- **Negative cache** for `backend = "auto"`: see `native_failure_cooldown_secs`.
  Key = device (`device_path` or index) + requested width/height/fps/fourcc.

## Privacy invariant

The camera is closed whenever the session is unlocked, apart from the explicit
opt-ins that existed before (`always_track`, a connected preview client, an
enroll/test lease). `prearm_on_lock` and `IdleHint` can only matter while
`LockedHint` is true, and prearm only within `prearm_window_ms` of the lock edge: see `camera::gate::camera_wanted` and its tests.

## The open log line

One line per cold open, at info level:

    Camera ready: backend=<name> resolution=<w>x<h> fourcc=<configured|auto> open_ms=<..> first_frame_ms=<..> first_valid_ms=<..>

`open_ms` is the time to open the device, `first_frame_ms` the time to the first
frame (counted from the start of the open), `first_valid_ms` the time to the
first frame delivered after the warm-up discard. `fourcc` is the configured
value (`auto` when unset); the native backend logs the mode it actually chose
on its own line (`V4L2 camera opened: ...`).

## Paste here

Output of `v4l2-ctl --list-devices`:

```
(paste)
```

Output of `v4l2-ctl -d /dev/video0 --list-formats-ext`:

```
(paste)
```

Daemon log lines from one lock cycle (`journalctl -u doormand | grep -E "Camera|V4L2|ffmpeg"`):

```
(paste)
```

## Virtual cameras (v4l2loopback)

Nodes of `v4l2loopback` are writable by the `video` group, and the daemon is in
that group, so any member could feed arbitrary frames to it. `device_guard.rs`
refuses virtual devices before capture:

- native V4L2 backend: sysfs facts (`/sys/class/video4linux/videoN/name`,
  `device/driver`, presence of a parent device) before opening, then the
  `VIDIOC_QUERYCAP` driver, card and bus_info strings after opening (as reported
  by nokhwa). Markers: `v4l2loopback`, `akvcam`, `vivid`, `vimc`, `virtual`,
  `droidcam`, `dummy`.
- ffmpeg backend (Linux) and GStreamer backend: sysfs facts only, best effort. A
  node missing from sysfs cannot be verified and is refused.

Residual risk: root can load any module or rename a device; a malicious USB
device can report a real-camera identity; a loopback driver that uses a
different name and has a parent device would pass the sysfs checks. Pinning
`device_path` to the owner's physical camera narrows this to the last case. The
GStreamer backend is hard-wired to `/dev/video0` (checked, not configurable).

## WaitForMatch pre-arm and the camera

With `[ipc] prearm_camera = true` (default) the unlock helper's `WaitForMatch`
holds an on-demand capture lease (like `doorman test`) for at most
`[ipc] prearm_max_hold_secs` (default 5) while the session is not yet reported
locked. This is the one case where an unlock request, not a lock, opens the
camera; it is bounded, requested by the user's own helper, and gated by the same
peer-credential policy as `Authenticate`. See docs/unlock-path.md.
