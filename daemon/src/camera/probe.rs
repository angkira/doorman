//! First-frame validation for the native camera backend.
//!
//! Opening a V4L2 stream can succeed on a device that then never delivers a
//! decodable frame (or whose capture path panics). The native backend is only
//! declared usable after one real frame was read within a timeout; anything
//! else closes the camera and reports the cause so the caller can fall back.

use anyhow::{anyhow, Result};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::time::Duration;

/// Name of the helper thread that reads the probe frame.
const PROBE_THREAD_NAME: &str = "camera-first-frame";

/// Best-effort text of a panic payload.
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Drop `value`, containing a panic raised by its destructor (nokhwa/v4l
/// destructors `unwrap()`/`panic!` on a failed STREAMOFF).
pub fn drop_contained<T>(value: T) -> Option<String> {
    catch_unwind(AssertUnwindSafe(move || drop(value)))
        .err()
        .map(|p| panic_message(p.as_ref()))
}

/// Read one frame from `camera` with `read_frame` and return the camera only if
/// that succeeded within `timeout`.
///
/// The read runs on a helper thread so a blocking `read_frame` cannot stall the
/// caller. On error or panic the camera is dropped (closed) on the helper
/// thread *before* the failure is reported, so the device is free for the next
/// backend. On timeout the helper still owns the camera and drops it whenever
/// the blocked read returns; the caller is told about the timeout.
pub fn validate_first_frame<C, F>(camera: C, timeout: Duration, read_frame: F) -> Result<C>
where
    C: Send + 'static,
    F: FnOnce(&mut C) -> Result<()> + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<Result<C, String>>();
    std::thread::Builder::new()
        .name(PROBE_THREAD_NAME.to_string())
        .spawn(move || {
            let mut camera = camera;
            let failure = match catch_unwind(AssertUnwindSafe(|| read_frame(&mut camera))) {
                Ok(Ok(())) => {
                    let _ = tx.send(Ok(camera));
                    return;
                }
                Ok(Err(e)) => format!("first frame failed: {e:#}"),
                Err(p) => format!("first frame read panicked: {}", panic_message(p.as_ref())),
            };
            // Close the device before reporting so the next backend can open it.
            let close_note = drop_contained(camera)
                .map(|m| format!(" (closing the camera also panicked: {m})"))
                .unwrap_or_default();
            let _ = tx.send(Err(format!("{failure}{close_note}")));
        })
        .map_err(|e| anyhow!("cannot start first-frame probe thread: {e}"))?;

    match rx.recv_timeout(timeout) {
        Ok(Ok(camera)) => Ok(camera),
        Ok(Err(msg)) => Err(anyhow!(msg)),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(anyhow!(
            "no decoded frame within {} ms",
            timeout.as_millis()
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err(anyhow!("first-frame probe thread ended without a result"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    const TEST_TIMEOUT: Duration = Duration::from_millis(200);
    /// Longer than `TEST_TIMEOUT`: a read that "never" returns for the test.
    const STALL: Duration = Duration::from_millis(1500);

    /// Fake native camera: records when it is closed.
    struct FakeCamera {
        closed: Arc<AtomicBool>,
    }
    impl Drop for FakeCamera {
        fn drop(&mut self) {
            self.closed.store(true, Ordering::SeqCst);
        }
    }

    fn fake() -> (FakeCamera, Arc<AtomicBool>) {
        let closed = Arc::new(AtomicBool::new(false));
        (
            FakeCamera {
                closed: closed.clone(),
            },
            closed,
        )
    }

    #[test]
    fn real_first_frame_keeps_the_camera_open() {
        let (cam, closed) = fake();
        let cam = validate_first_frame(cam, TEST_TIMEOUT, |_| Ok(())).unwrap();
        assert!(!closed.load(Ordering::SeqCst));
        drop(cam);
    }

    #[test]
    fn frame_error_rejects_and_closes_the_camera() {
        let (cam, closed) = fake();
        let err = validate_first_frame(cam, TEST_TIMEOUT, |_| Err(anyhow!("decode failed")))
            .err()
            .expect("must reject")
            .to_string();
        assert!(err.contains("decode failed"), "{err}");
        assert!(closed.load(Ordering::SeqCst), "device closed before report");
    }

    #[test]
    fn frame_panic_rejects_with_the_panic_text_and_closes_the_camera() {
        let (cam, closed) = fake();
        let err = validate_first_frame(cam, TEST_TIMEOUT, |_| -> Result<()> {
            panic!("boom in mozjpeg")
        })
        .err()
        .expect("must reject")
        .to_string();
        assert!(
            err.contains("panicked") && err.contains("boom in mozjpeg"),
            "{err}"
        );
        assert!(closed.load(Ordering::SeqCst), "device closed before report");
    }

    #[test]
    fn no_frame_within_timeout_rejects() {
        let (cam, closed) = fake();
        let err = validate_first_frame(cam, TEST_TIMEOUT, |_| {
            std::thread::sleep(STALL);
            Ok(())
        })
        .err()
        .expect("must reject")
        .to_string();
        assert!(err.contains("no decoded frame within 200 ms"), "{err}");
        // The stalled helper closes the camera once its read returns.
        let deadline = std::time::Instant::now() + STALL * 3;
        while !closed.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(closed.load(Ordering::SeqCst));
    }

    #[test]
    fn panicking_destructor_is_contained() {
        struct BadDrop;
        impl Drop for BadDrop {
            fn drop(&mut self) {
                panic!("streamoff failed");
            }
        }
        let note = drop_contained(BadDrop).expect("panic reported");
        assert!(note.contains("streamoff failed"));
    }
}
