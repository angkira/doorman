//! Capture-mode negotiation for the native (V4L2/nokhwa) backend.
//!
//! The device is asked what it offers and one mode is chosen from that list, so
//! the open never requests a mode the device cannot deliver. This replaces
//! nokhwa's `RequestedFormatType::Closest`, which returns "Failed to Fufill"
//! whenever the exact requested resolution is missing in the requested pixel
//! format (see docs/camera.md). The logic is pure so it is tested with fake
//! device capabilities.

use doorman_shared::config::CameraFourcc;

/// One capture mode a device offers (e.g. one `v4l2-ctl --list-formats-ext` row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMode {
    pub format: CameraFourcc,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// What the configuration asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WantedMode {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub fourcc: Option<CameraFourcc>,
}

/// Pixel formats in order of preference. An explicit request goes first;
/// otherwise YUYV first (historical behaviour), then MJPEG.
fn format_preference(fourcc: Option<CameraFourcc>) -> [CameraFourcc; 2] {
    match fourcc {
        Some(CameraFourcc::Mjpeg) => [CameraFourcc::Mjpeg, CameraFourcc::Yuyv],
        Some(CameraFourcc::Yuyv) | None => [CameraFourcc::Yuyv, CameraFourcc::Mjpeg],
    }
}

fn resolution_distance(m: &DeviceMode, wanted: &WantedMode) -> u64 {
    let dx = i64::from(m.width) - i64::from(wanted.width);
    let dy = i64::from(m.height) - i64::from(wanted.height);
    (dx * dx + dy * dy) as u64
}

fn describe(modes: &[DeviceMode]) -> String {
    if modes.is_empty() {
        return "none".to_string();
    }
    modes
        .iter()
        .map(|m| format!("{:?} {}x{}@{}", m.format, m.width, m.height, m.fps))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Choose the mode to open.
///
/// 1. Exact requested size wins, in format-preference order (so a size the
///    device only offers as MJPEG is opened as MJPEG instead of failing).
/// 2. Otherwise the nearest offered size, ties broken by format preference.
/// 3. Within the chosen format and size, the frame rate closest to the request
///    (ties go to the higher rate).
///
/// Errors, listing everything the device offers, when it offers no usable mode.
pub fn select_mode(modes: &[DeviceMode], wanted: &WantedMode) -> Result<DeviceMode, String> {
    let prefs = format_preference(wanted.fourcc);
    let rank = |m: &DeviceMode| prefs.iter().position(|f| *f == m.format);

    let best = modes
        .iter()
        .filter(|m| rank(m).is_some())
        .min_by_key(|m| (resolution_distance(m, wanted), rank(m)))
        .ok_or_else(|| {
            format!(
                "device offers no MJPEG or YUYV capture mode (offered: {})",
                describe(modes)
            )
        })?;

    // Re-pick the frame rate by distance among the winning format + size.
    Ok(modes
        .iter()
        .filter(|m| m.format == best.format && m.width == best.width && m.height == best.height)
        .min_by_key(|m| {
            (
                (i64::from(m.fps) - i64::from(wanted.fps)).unsigned_abs(),
                std::cmp::Reverse(m.fps),
            )
        })
        .copied()
        .unwrap_or(*best))
}

/// Fail fast when the driver did not settle on the mode that was requested.
///
/// `actual` is the format the driver reports after the stream is open
/// (`None` when it is neither MJPEG nor YUYV, which this backend cannot
/// decode). The pixel format and the resolution must match exactly: decoding
/// the frames of another format with the requested format's decoder is how
/// "Not a JPEG file" errors arise. A different frame rate is harmless and is
/// not an error.
pub fn check_negotiated(
    requested: &DeviceMode,
    actual: Option<DeviceMode>,
    actual_description: &str,
) -> Result<(), String> {
    let Some(actual) = actual else {
        return Err(format!(
            "driver reports an unsupported pixel format ({actual_description}) instead of {:?}",
            requested.format
        ));
    };
    if actual.format != requested.format {
        return Err(format!(
            "driver negotiated {:?} instead of the requested {:?} ({actual_description})",
            actual.format, requested.format
        ));
    }
    if (actual.width, actual.height) != (requested.width, requested.height) {
        return Err(format!(
            "driver negotiated {}x{} instead of the requested {}x{} ({actual_description})",
            actual.width, actual.height, requested.width, requested.height
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use CameraFourcc::{Mjpeg, Yuyv};

    fn m(format: CameraFourcc, width: u32, height: u32, fps: u32) -> DeviceMode {
        DeviceMode {
            format,
            width,
            height,
            fps,
        }
    }

    fn want(width: u32, height: u32, fourcc: Option<CameraFourcc>) -> WantedMode {
        WantedMode {
            width,
            height,
            fps: 30,
            fourcc,
        }
    }

    /// Typical UVC webcam: raw YUYV only at small sizes, MJPEG at all sizes.
    fn uvc() -> Vec<DeviceMode> {
        vec![
            m(Yuyv, 640, 480, 30),
            m(Yuyv, 1280, 720, 10),
            m(Mjpeg, 640, 480, 30),
            m(Mjpeg, 1280, 720, 30),
            m(Mjpeg, 1920, 1080, 30),
        ]
    }

    #[test]
    fn exact_size_unset_fourcc_prefers_yuyv_as_before() {
        assert_eq!(
            select_mode(&uvc(), &want(640, 480, None)).unwrap(),
            m(Yuyv, 640, 480, 30)
        );
    }

    #[test]
    fn explicit_mjpeg_wins_at_same_size() {
        assert_eq!(
            select_mode(&uvc(), &want(640, 480, Some(Mjpeg))).unwrap(),
            m(Mjpeg, 640, 480, 30)
        );
    }

    #[test]
    fn size_only_offered_as_mjpeg_opens_mjpeg() {
        let modes = vec![m(Yuyv, 640, 480, 30), m(Mjpeg, 1280, 720, 30)];
        assert_eq!(
            select_mode(&modes, &want(1280, 720, None)).unwrap(),
            m(Mjpeg, 1280, 720, 30)
        );
    }

    #[test]
    fn mjpeg_only_device_works_without_fourcc() {
        let modes = vec![m(Mjpeg, 640, 480, 30), m(Mjpeg, 1280, 720, 30)];
        assert_eq!(
            select_mode(&modes, &want(1280, 720, None)).unwrap(),
            m(Mjpeg, 1280, 720, 30)
        );
    }

    #[test]
    fn missing_size_picks_nearest() {
        // 1024x720 is requested; 1280x720 (distance 256) beats 640x480.
        assert_eq!(
            select_mode(&uvc(), &want(1024, 720, Some(Mjpeg))).unwrap(),
            m(Mjpeg, 1280, 720, 30)
        );
    }

    #[test]
    fn frame_rate_closest_to_request() {
        let modes = vec![m(Mjpeg, 640, 480, 15), m(Mjpeg, 640, 480, 60)];
        assert_eq!(select_mode(&modes, &want(640, 480, None)).unwrap().fps, 15);
        let modes = vec![m(Mjpeg, 640, 480, 25), m(Mjpeg, 640, 480, 35)];
        assert_eq!(
            select_mode(&modes, &want(640, 480, None)).unwrap().fps,
            35,
            "ties go to the higher rate"
        );
    }

    /// Exactly the modes `v4l2-ctl --list-formats-ext` printed on the owner's host
    /// (UGREEN Camera 4K, uvcvideo); see HOST-FACTS.md section 1.
    fn host_ugreen() -> Vec<DeviceMode> {
        let mjpg: [(u32, u32, &[u32]); 13] = [
            (3840, 2160, &[30]),
            (1280, 720, &[60, 30]),
            (1920, 1080, &[60, 30]),
            (4000, 3000, &[30]),
            (640, 480, &[30]),
            (640, 360, &[60, 30]),
            (2560, 1440, &[30]),
            (2592, 1520, &[30]),
            (2048, 1536, &[30]),
            (1280, 960, &[30]),
            (1024, 768, &[30]),
            (480, 320, &[30]),
            (1280, 1024, &[30]),
        ];
        let yuyv: [(u32, u32, u32); 12] = [
            (640, 480, 30),
            (1920, 1080, 5),
            (640, 360, 30),
            (1280, 1024, 5),
            (2560, 1440, 3),
            (2592, 1944, 3),
            (1600, 1200, 5),
            (480, 320, 30),
            (320, 240, 30),
            (800, 600, 15),
            (1024, 720, 15),
            (1280, 960, 10),
        ];
        let mut modes = Vec::new();
        for (w, h, rates) in mjpg {
            modes.extend(rates.iter().map(|fps| m(Mjpeg, w, h, *fps)));
        }
        modes.extend(yuyv.iter().map(|(w, h, fps)| m(Yuyv, *w, *h, *fps)));
        modes
    }

    #[test]
    fn host_camera_default_720p30_opens_mjpeg_not_yuyv() {
        // The failing request at 5f78657 was YUYV 1280x720@30, which this camera
        // does not offer. The default 1280x720@30 must now resolve to MJPEG.
        let host = host_ugreen();
        assert!(!host
            .iter()
            .any(|d| d.format == Yuyv && d.width == 1280 && d.height == 720));
        assert_eq!(
            select_mode(&host, &want(1280, 720, None)).unwrap(),
            m(Mjpeg, 1280, 720, 30)
        );
        assert_eq!(
            select_mode(&host, &want(1280, 720, Some(Yuyv))).unwrap(),
            m(Mjpeg, 1280, 720, 30),
            "an explicit YUYV request at a size only MJPEG offers falls back to MJPEG"
        );
        assert_eq!(
            select_mode(&host, &want(1280, 720, Some(Mjpeg))).unwrap(),
            m(Mjpeg, 1280, 720, 30)
        );
    }

    #[test]
    fn host_camera_small_sizes_keep_yuyv_when_unset() {
        let host = host_ugreen();
        assert_eq!(
            select_mode(&host, &want(640, 480, None)).unwrap(),
            m(Yuyv, 640, 480, 30)
        );
        assert_eq!(
            select_mode(&host, &want(640, 480, Some(Mjpeg))).unwrap(),
            m(Mjpeg, 640, 480, 30)
        );
    }

    #[test]
    fn no_usable_mode_lists_what_is_offered() {
        let err = select_mode(&[], &want(640, 480, None)).unwrap_err();
        assert!(err.contains("offered: none"), "{err}");
    }

    #[test]
    fn negotiated_mode_must_match_format_and_resolution() {
        let want = DeviceMode {
            format: CameraFourcc::Mjpeg,
            width: 1280,
            height: 720,
            fps: 30,
        };
        assert!(check_negotiated(&want, Some(want), "x").is_ok());
        // fps drift is not an error
        let slower = DeviceMode { fps: 15, ..want };
        assert!(check_negotiated(&want, Some(slower), "x").is_ok());

        let yuyv = DeviceMode {
            format: CameraFourcc::Yuyv,
            ..want
        };
        let err = check_negotiated(&want, Some(yuyv), "YUYV 1280x720@30").unwrap_err();
        assert!(err.contains("Yuyv") && err.contains("Mjpeg"), "{err}");

        let small = DeviceMode { width: 640, ..want };
        let err = check_negotiated(&want, Some(small), "d").unwrap_err();
        assert!(err.contains("640x720") && err.contains("1280x720"), "{err}");

        let err = check_negotiated(&want, None, "NV12 1280x720@30").unwrap_err();
        assert!(
            err.contains("unsupported pixel format") && err.contains("NV12"),
            "{err}"
        );
    }
}
