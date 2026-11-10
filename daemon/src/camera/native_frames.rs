//! Frame validation and first-frame acquisition for the native camera backend.
//!
//! Host evidence (UGREEN Camera 4K, uvcvideo, MJPG 1280x720@30): the first
//! buffer handed to the JPEG decoder was not a JPEG (`libjpeg fatal error: Not
//! a JPEG file: starts with 0x61 0xe8`), which panicked inside the decoder.
//! Two facts about the stack matter here:
//!
//! * `nokhwa-bindings-linux` returns the whole mmap'd V4L2 buffer (its
//!   capacity), not the first `bytesused` bytes, and `nokhwa::Camera` does not
//!   expose `bytesused`. A valid frame is therefore a JPEG *followed by* stale
//!   or zero bytes, and the end of the frame has to be found in the data.
//! * UVC devices can deliver junk or partial buffers right after STREAMON.
//!
//! So every candidate buffer is checked before it reaches libjpeg/mozjpeg
//! ([`validate_frame`]): MJPEG must start with SOI (FF D8), and a marker walk
//! must reach EOI (FF D9) inside the buffer; the decoder gets exactly the bytes
//! up to that EOI. [`first_usable_frame`] discards the configured warm-up
//! buffers and then keeps reading until one candidate validates *and* decodes,
//! or the time budget is spent.
//!
//! Limit: without `bytesused`, a truncated frame whose missing tail is
//! overwritten by an older frame's stale bytes can still pass the marker walk;
//! the decoder then reports the damage (an error or a caught panic) and the
//! frame is dropped like any other rejected one.
//!
//! Logging: a rejection is logged at WARN with the first
//! [`HEX_PREVIEW_BYTES`] bytes as hex, the buffer length, `bytesused` if known,
//! the negotiated fourcc, the resolution and the frame index, and nothing else
//! of the image data.

// Production callers exist only in builds with a native backend; test builds
// without one compile this module for its tests alone.
#![cfg_attr(
    not(any(feature = "camera-v4l2", feature = "camera-nokhwa")),
    allow(dead_code)
)]

use anyhow::{anyhow, Result};
use doorman_shared::config::CameraFourcc;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

use super::probe::panic_message;

/// JPEG start-of-image marker.
const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
/// Smallest buffer that can be a JPEG (SOI + EOI).
const MIN_JPEG_LEN: usize = 4;
/// Number of leading buffer bytes shown in a rejection log (hex only).
pub const HEX_PREVIEW_BYTES: usize = 8;
/// YUYV (4:2:2) bytes per pixel.
const YUYV_BYTES_PER_PIXEL: usize = 2;
/// Largest share of the first-frame timeout given to the frame loop; the rest
/// is slack so the loop reports its reason before the outer timeout fires.
const LOOP_BUDGET_MARGIN_DIV: u32 = 4;
/// Upper bound of that slack.
const LOOP_BUDGET_MARGIN_MAX: Duration = Duration::from_millis(100);
/// Rejections logged with full detail at WARN per camera instance after the
/// first-frame probe; later ones go to DEBUG so a broken camera cannot flood
/// the journal at frame rate.
pub const MAX_DETAILED_REJECTION_LOGS: u32 = 5;

/// Marker constants used by the JPEG walk.
mod marker {
    pub const PREFIX: u8 = 0xFF;
    pub const STUFFED_ZERO: u8 = 0x00;
    pub const TEM: u8 = 0x01;
    pub const RST_FIRST: u8 = 0xD0;
    pub const RST_LAST: u8 = 0xD7;
    pub const SOI: u8 = 0xD8;
    pub const EOI: u8 = 0xD9;
    pub const SOS: u8 = 0xDA;
}

/// Why a candidate buffer was not used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameRejection {
    /// Shorter than the smallest possible frame.
    TooShort { len: usize },
    /// MJPEG buffer that does not start with FF D8.
    MissingSoi,
    /// MJPEG buffer without a complete marker structure up to EOI.
    Truncated,
    /// Marker structure that cannot be a JPEG (offset of the offence).
    Malformed { offset: usize },
    /// Raw buffer shorter than `width * height * bytes_per_pixel`.
    ShortRaw { len: usize, need: usize },
    /// The decoder returned an error or panicked.
    Decoder(String),
}

impl fmt::Display for FrameRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { len } => write!(f, "buffer too short ({len} bytes)"),
            Self::MissingSoi => write!(f, "does not start with the JPEG SOI marker FF D8"),
            Self::Truncated => write!(f, "truncated JPEG (no EOI marker FF D9 reached)"),
            Self::Malformed { offset } => write!(f, "malformed JPEG marker at offset {offset}"),
            Self::ShortRaw { len, need } => {
                write!(f, "raw frame too short ({len} bytes, need {need})")
            }
            Self::Decoder(e) => write!(f, "decoder rejected the frame: {e}"),
        }
    }
}

/// Negotiated stream parameters, for validation and for log lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameContext {
    pub fourcc: CameraFourcc,
    pub width: u32,
    pub height: u32,
}

/// One buffer as delivered by the driver layer.
#[derive(Debug, Clone, Copy)]
pub struct RawFrame<'a> {
    pub bytes: &'a [u8],
    /// `bytesused` of the V4L2 buffer when the layer exposes it.
    pub bytesused: Option<usize>,
}

/// Source of raw capture buffers (nokhwa in production, a fake in tests).
pub trait FrameSource {
    /// Block for the next buffer.
    fn next_frame(&mut self) -> Result<RawFrame<'_>>;
}

/// Hex of the first [`HEX_PREVIEW_BYTES`] bytes: the only image data ever logged.
pub fn hex_prefix(buf: &[u8]) -> String {
    buf.iter()
        .take(HEX_PREVIEW_BYTES)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Logs a rejected buffer at WARN (`detailed`) or DEBUG.
pub fn log_rejection(
    ctx: &FrameContext,
    frame_index: u64,
    frame: &RawFrame<'_>,
    why: &FrameRejection,
    detailed: bool,
) {
    let bytesused = frame
        .bytesused
        .map_or_else(|| "unavailable".to_string(), |b| b.to_string());
    let msg = format!(
        "native frame rejected: {why}; first {HEX_PREVIEW_BYTES} bytes = [{}], buffer len = {}, \
         bytesused = {bytesused}, negotiated fourcc = {:?}, resolution = {}x{}, frame index = {frame_index}",
        hex_prefix(frame.bytes),
        frame.bytes.len(),
        ctx.fourcc,
        ctx.width,
        ctx.height,
    );
    if detailed {
        warn!("{msg}");
    } else {
        debug!("{msg}");
    }
}

/// Returns the bytes of the JPEG at the start of `buf`, up to and including EOI.
fn jpeg_span(buf: &[u8]) -> Result<&[u8], FrameRejection> {
    if buf.len() < MIN_JPEG_LEN {
        return Err(FrameRejection::TooShort { len: buf.len() });
    }
    if buf[..2] != JPEG_SOI {
        return Err(FrameRejection::MissingSoi);
    }
    let mut i = 2;
    loop {
        // At a marker boundary: FF, optional FF fill bytes, then the code.
        if buf.get(i) != Some(&marker::PREFIX) {
            return if i >= buf.len() {
                Err(FrameRejection::Truncated)
            } else {
                Err(FrameRejection::Malformed { offset: i })
            };
        }
        while buf.get(i) == Some(&marker::PREFIX) {
            i += 1;
        }
        let code = *buf.get(i).ok_or(FrameRejection::Truncated)?;
        i += 1;
        match code {
            marker::EOI => return Ok(&buf[..i]),
            marker::STUFFED_ZERO => return Err(FrameRejection::Malformed { offset: i - 1 }),
            marker::TEM | marker::SOI | marker::RST_FIRST..=marker::RST_LAST => {}
            _ => {
                let len_bytes = buf.get(i..i + 2).ok_or(FrameRejection::Truncated)?;
                let seg_len = usize::from(u16::from_be_bytes([len_bytes[0], len_bytes[1]]));
                if seg_len < 2 {
                    return Err(FrameRejection::Malformed { offset: i });
                }
                i += seg_len;
                if i > buf.len() {
                    return Err(FrameRejection::Truncated);
                }
                if code == marker::SOS {
                    i = skip_entropy_data(buf, i)?;
                }
            }
        }
    }
}

/// From `start` (first entropy byte after an SOS header) to the next real marker.
fn skip_entropy_data(buf: &[u8], start: usize) -> Result<usize, FrameRejection> {
    let mut i = start;
    loop {
        let off = buf[i..]
            .iter()
            .position(|&b| b == marker::PREFIX)
            .ok_or(FrameRejection::Truncated)?;
        let ff = i + off;
        let next = *buf.get(ff + 1).ok_or(FrameRejection::Truncated)?;
        match next {
            // Stuffed data byte or restart marker: still entropy data.
            marker::STUFFED_ZERO | marker::RST_FIRST..=marker::RST_LAST => i = ff + 2,
            // Fill byte: the real code, if any, follows.
            marker::PREFIX => i = ff + 1,
            _ => return Ok(ff),
        }
    }
}

/// The part of `frame` that may be handed to the decoder, or why it may not.
///
/// MJPEG: SOI check + marker walk to EOI (the bytes after EOI are driver
/// buffer slack and are cut off). YUYV: exactly `width * height * 2` bytes.
pub fn validate_frame<'a>(
    ctx: &FrameContext,
    frame: &RawFrame<'a>,
) -> Result<&'a [u8], FrameRejection> {
    let used = match frame.bytesused {
        Some(n) if n < frame.bytes.len() => &frame.bytes[..n],
        _ => frame.bytes,
    };
    match ctx.fourcc {
        CameraFourcc::Mjpeg => jpeg_span(used),
        CameraFourcc::Yuyv => {
            let need = ctx.width as usize * ctx.height as usize * YUYV_BYTES_PER_PIXEL;
            if used.len() < need {
                Err(FrameRejection::ShortRaw {
                    len: used.len(),
                    need,
                })
            } else {
                Ok(&used[..need])
            }
        }
    }
}

/// Runs `decode` on validated bytes, turning an error or a panic into a rejection.
pub fn decode_guarded<T>(
    span: &[u8],
    decode: &mut impl FnMut(&[u8]) -> Result<T>,
) -> Result<T, FrameRejection> {
    match catch_unwind(AssertUnwindSafe(|| decode(span))) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(FrameRejection::Decoder(format!("{e:#}"))),
        Err(p) => Err(FrameRejection::Decoder(format!(
            "panicked: {}",
            panic_message(p.as_ref())
        ))),
    }
}

/// Time the frame loop may use out of the first-frame `timeout`: slightly less,
/// so its own report (with the last rejection reason) beats the outer timeout.
pub fn loop_budget(timeout: Duration) -> Duration {
    timeout.saturating_sub((timeout / LOOP_BUDGET_MARGIN_DIV).min(LOOP_BUDGET_MARGIN_MAX))
}

/// Outcome of a successful first-frame acquisition.
#[derive(Debug)]
pub struct FirstFrame<T> {
    /// The decoded frame (the production probe only needs the counters).
    #[cfg_attr(not(test), allow(dead_code))]
    pub value: T,
    /// Buffers read in total (warm-up + candidates).
    pub frames_read: u32,
    /// Candidates rejected before the accepted one.
    pub rejected: u32,
}

/// Reads buffers until one validates and decodes.
///
/// The first `warmup` buffers are read and dropped unexamined (UVC junk right
/// after STREAMON). Every later buffer is validated ([`validate_frame`]) and
/// decoded under `catch_unwind`; a rejection is logged at WARN and the loop
/// continues until `budget` is spent. A read error is returned at once.
pub fn first_usable_frame<S, T>(
    source: &mut S,
    ctx: &FrameContext,
    warmup: u32,
    budget: Duration,
    mut decode: impl FnMut(&[u8]) -> Result<T>,
) -> Result<FirstFrame<T>>
where
    S: FrameSource,
{
    let started = Instant::now();
    let mut frames_read: u32 = 0;
    let mut rejected: u32 = 0;
    let mut last: Option<FrameRejection> = None;
    while started.elapsed() < budget {
        let index = u64::from(frames_read);
        let frame = source
            .next_frame()
            .map_err(|e| anyhow!("reading frame {index} failed: {e:#}"))?;
        frames_read += 1;
        if index < u64::from(warmup) {
            debug!("native warm-up: dropped buffer {index} of {warmup}");
            continue;
        }
        let outcome =
            validate_frame(ctx, &frame).and_then(|span| decode_guarded(span, &mut decode));
        match outcome {
            Ok(value) => {
                return Ok(FirstFrame {
                    value,
                    frames_read,
                    rejected,
                })
            }
            Err(why) => {
                log_rejection(ctx, index, &frame, &why, true);
                rejected += 1;
                last = Some(why);
            }
        }
    }
    Err(anyhow!(
        "no usable frame within {} ms: {frames_read} buffers read ({warmup} warm-up dropped), \
         {rejected} rejected{}",
        budget.as_millis(),
        last.map(|w| format!("; last rejection: {w}"))
            .unwrap_or_default()
    ))
}

#[cfg(test)]
pub mod testing {
    //! Synthetic frames and a scripted source, shared with the backend tests.
    use super::*;
    use image::{codecs::jpeg::JpegEncoder, Rgb, RgbImage};
    use std::collections::VecDeque;

    pub const TEST_WIDTH: u32 = 16;
    pub const TEST_HEIGHT: u32 = 16;
    /// A buffer the way the driver layer hands it out: JPEG + stale slack.
    pub const TEST_SLACK: usize = 64;

    pub fn ctx() -> FrameContext {
        FrameContext {
            fourcc: CameraFourcc::Mjpeg,
            width: TEST_WIDTH,
            height: TEST_HEIGHT,
        }
    }

    /// Tiny synthetic gradient as a JPEG (no real image content).
    pub fn jpeg() -> Vec<u8> {
        let img = RgbImage::from_fn(TEST_WIDTH, TEST_HEIGHT, |x, y| {
            Rgb([(x * 16) as u8, (y * 16) as u8, ((x ^ y) * 16) as u8])
        });
        let mut out = Vec::new();
        JpegEncoder::new_with_quality(&mut out, 90)
            .encode_image(&img)
            .unwrap();
        out
    }

    pub fn junk() -> Vec<u8> {
        let mut v = vec![0x61, 0xe8];
        v.extend((0..256u32).map(|i| (i * 7 % 251) as u8));
        v
    }

    pub fn decode_with_image_crate(span: &[u8]) -> Result<RgbImage> {
        Ok(image::load_from_memory_with_format(span, image::ImageFormat::Jpeg)?.to_rgb8())
    }

    /// Source replaying a script; after it ends it keeps returning `fallback`.
    pub struct ScriptedSource {
        script: VecDeque<Vec<u8>>,
        fallback: Vec<u8>,
        current: Vec<u8>,
        pub reads: u32,
    }

    impl ScriptedSource {
        pub fn new(script: Vec<Vec<u8>>, fallback: Vec<u8>) -> Self {
            Self {
                script: script.into(),
                fallback,
                current: Vec::new(),
                reads: 0,
            }
        }
    }

    impl FrameSource for ScriptedSource {
        fn next_frame(&mut self) -> Result<RawFrame<'_>> {
            self.reads += 1;
            self.current = self
                .script
                .pop_front()
                .unwrap_or_else(|| self.fallback.clone());
            Ok(RawFrame {
                bytes: &self.current,
                bytesused: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    const BUDGET: Duration = Duration::from_millis(300);

    fn with_slack(mut v: Vec<u8>) -> Vec<u8> {
        v.extend(std::iter::repeat_n(0x5A, TEST_SLACK));
        v
    }

    #[test]
    fn valid_jpeg_with_driver_slack_is_trimmed_to_eoi() {
        let j = jpeg();
        let buf = with_slack(j.clone());
        let frame = RawFrame {
            bytes: &buf,
            bytesused: None,
        };
        assert_eq!(validate_frame(&ctx(), &frame).unwrap(), &j[..]);
    }

    #[test]
    fn bytesused_shorter_than_buffer_is_honoured() {
        let j = jpeg();
        let buf = with_slack(j.clone());
        let frame = RawFrame {
            bytes: &buf,
            bytesused: Some(j.len()),
        };
        assert_eq!(validate_frame(&ctx(), &frame).unwrap(), &j[..]);
        let cut = RawFrame {
            bytes: &buf,
            bytesused: Some(j.len() - 10),
        };
        assert_eq!(
            validate_frame(&ctx(), &cut).unwrap_err(),
            FrameRejection::Truncated
        );
    }

    #[test]
    fn junk_and_short_buffers_are_rejected_before_the_decoder() {
        let j = junk();
        let frame = RawFrame {
            bytes: &j,
            bytesused: None,
        };
        assert_eq!(
            validate_frame(&ctx(), &frame).unwrap_err(),
            FrameRejection::MissingSoi
        );
        for short in [&[][..], &[0xFF][..], &[0xFF, 0xD8, 0xFF][..]] {
            let frame = RawFrame {
                bytes: short,
                bytesused: None,
            };
            assert!(matches!(
                validate_frame(&ctx(), &frame).unwrap_err(),
                FrameRejection::TooShort { .. }
            ));
        }
    }

    #[test]
    fn truncated_jpeg_is_rejected_at_every_cut() {
        let j = jpeg();
        for cut in (MIN_JPEG_LEN..j.len() - 2).step_by(7) {
            let frame = RawFrame {
                bytes: &j[..cut],
                bytesused: None,
            };
            assert!(
                validate_frame(&ctx(), &frame).is_err(),
                "cut at {cut} of {} accepted",
                j.len()
            );
        }
    }

    #[test]
    fn yuyv_needs_a_full_frame_and_is_trimmed() {
        let yuyv = FrameContext {
            fourcc: CameraFourcc::Yuyv,
            ..ctx()
        };
        let need = (TEST_WIDTH * TEST_HEIGHT * 2) as usize;
        let full = vec![0u8; need + 10];
        let ok = validate_frame(
            &yuyv,
            &RawFrame {
                bytes: &full,
                bytesused: None,
            },
        )
        .unwrap();
        assert_eq!(ok.len(), need);
        let short = vec![0u8; need - 1];
        assert!(matches!(
            validate_frame(
                &yuyv,
                &RawFrame {
                    bytes: &short,
                    bytesused: None
                }
            )
            .unwrap_err(),
            FrameRejection::ShortRaw { .. }
        ));
    }

    #[test]
    fn hex_prefix_never_exceeds_eight_bytes() {
        let j = jpeg();
        assert_eq!(hex_prefix(&j).split(' ').count(), HEX_PREVIEW_BYTES);
        assert_eq!(hex_prefix(&[0x61, 0xe8]), "61 e8");
    }

    #[test]
    fn junk_first_buffer_then_valid_jpeg_succeeds_after_discarding() {
        // warm-up 0: the junk buffer is a candidate and is rejected, the next is used.
        let mut src = ScriptedSource::new(vec![junk(), with_slack(jpeg())], jpeg());
        let got = first_usable_frame(&mut src, &ctx(), 0, BUDGET, decode_with_image_crate).unwrap();
        assert_eq!(got.rejected, 1);
        assert_eq!(got.frames_read, 2);
        assert_eq!(got.value.dimensions(), (TEST_WIDTH, TEST_HEIGHT));
    }

    #[test]
    fn warmup_buffers_are_dropped_unexamined() {
        let mut src = ScriptedSource::new(vec![junk(), junk(), junk(), jpeg()], jpeg());
        let got = first_usable_frame(&mut src, &ctx(), 3, BUDGET, decode_with_image_crate).unwrap();
        assert_eq!((got.frames_read, got.rejected), (4, 0));
    }

    #[test]
    fn truncated_jpeg_is_rejected_and_the_next_frame_used() {
        let j = jpeg();
        let truncated = j[..j.len() / 2].to_vec();
        let mut src = ScriptedSource::new(vec![truncated, with_slack(j)], jpeg());
        let got = first_usable_frame(&mut src, &ctx(), 0, BUDGET, decode_with_image_crate).unwrap();
        assert_eq!((got.frames_read, got.rejected), (2, 1));
    }

    #[test]
    fn all_junk_ends_at_the_budget_with_the_last_reason() {
        let mut src = ScriptedSource::new(vec![], junk());
        let started = Instant::now();
        let err = first_usable_frame(&mut src, &ctx(), 1, BUDGET, decode_with_image_crate)
            .unwrap_err()
            .to_string();
        assert!(started.elapsed() < BUDGET * 3, "must not hang");
        assert!(
            err.contains("no usable frame") && err.contains("SOI") && err.contains("1 warm-up"),
            "{err}"
        );
    }

    #[test]
    fn decoder_panic_is_a_rejection_not_a_crash() {
        let mut calls = 0;
        let mut src = ScriptedSource::new(vec![jpeg(), jpeg()], jpeg());
        let got = first_usable_frame(&mut src, &ctx(), 0, BUDGET, |span| {
            calls += 1;
            if calls == 1 {
                panic!("libjpeg fatal error: boom");
            }
            decode_with_image_crate(span)
        })
        .unwrap();
        assert_eq!(got.rejected, 1);
    }

    #[test]
    fn read_error_fails_fast() {
        struct Broken;
        impl FrameSource for Broken {
            fn next_frame(&mut self) -> Result<RawFrame<'_>> {
                Err(anyhow!("ENODEV"))
            }
        }
        let err = first_usable_frame(&mut Broken, &ctx(), 0, BUDGET, decode_with_image_crate)
            .unwrap_err()
            .to_string();
        assert!(err.contains("ENODEV") && err.contains("frame 0"), "{err}");
    }

    #[test]
    fn loop_budget_leaves_slack_below_the_timeout() {
        let t = Duration::from_millis(1500);
        assert_eq!(loop_budget(t), t - LOOP_BUDGET_MARGIN_MAX);
        let small = Duration::from_millis(100);
        assert_eq!(loop_budget(small), small - small / LOOP_BUDGET_MARGIN_DIV);
    }

    #[test]
    fn all_junk_falls_through_within_the_first_frame_timeout() {
        use super::super::probe::validate_first_frame;
        const TIMEOUT: Duration = Duration::from_millis(400);
        let src = ScriptedSource::new(vec![], junk());
        let started = Instant::now();
        let err = validate_first_frame(src, TIMEOUT, |cam| {
            first_usable_frame(
                cam,
                &ctx(),
                3,
                loop_budget(TIMEOUT),
                decode_with_image_crate,
            )
            .map(|_| ())
        })
        .err()
        .expect("must reject")
        .to_string();
        assert!(started.elapsed() < TIMEOUT + Duration::from_millis(500));
        // the loop's own reason, not the bare outer timeout
        assert!(
            err.contains("last rejection") && err.contains("SOI"),
            "{err}"
        );
    }
}
