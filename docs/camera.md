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
| `fourcc` | unset | `"MJPG"` or `"YUYV"`. Unset keeps today's behaviour. |
| `warmup_frames` | `0` | Frames dropped after a cold open. Valid 0..=10. Recommended: `3`. |
| `prearm_on_lock` | `false` | Open on `LockedHint` even while `IdleHint` is set. Recommended: `true`. |

Defaults are unchanged on purpose: they change auth timing, so they are turned on
after the host measurement, not before. The default resolution is also
unchanged. The recommended starting point to measure is
`width = 640, height = 480, fourcc = "MJPG", warmup_frames = 3,
prearm_on_lock = true`, pending the WS6 recall comparison against 1280x720.

## Privacy invariant

The camera is closed whenever the session is unlocked, apart from the explicit
opt-ins that existed before (`always_track`, a connected preview client, an
enroll/test lease). `prearm_on_lock` and `IdleHint` can only matter while
`LockedHint` is true: see `camera::gate::camera_wanted` and its tests.

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
