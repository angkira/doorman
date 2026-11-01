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

    #[test]
    fn no_usable_mode_lists_what_is_offered() {
        let err = select_mode(&[], &want(640, 480, None)).unwrap_err();
        assert!(err.contains("offered: none"), "{err}");
    }
}
