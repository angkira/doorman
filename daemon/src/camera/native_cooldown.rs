//! Negative cache for the native camera backend under `backend = "auto"`.
//!
//! A native open that fails (no usable first frame, decoder error, format
//! mismatch, ...) will usually fail the same way a moment later, and every
//! attempt costs up to `native_first_frame_timeout_ms` before ffmpeg gets the
//! device. After a failure this process remembers it per (device, requested
//! mode) for `native_failure_cooldown_secs` and goes straight to ffmpeg; once
//! the cooldown has passed the native backend is probed again. The memory
//! lives only in the daemon process (a restart re-probes) and is bypassed by
//! `backend = "native"`, which has no fallback to protect.

// Production callers exist only in builds with a native backend; test builds
// without one compile this module for its tests alone.
#![cfg_attr(
    not(any(feature = "camera-v4l2", feature = "camera-nokhwa")),
    allow(dead_code)
)]

use doorman_shared::config::{CameraBackendChoice, CameraConfig};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Time source, injectable so cooldown expiry is testable without sleeping.
pub trait Clock {
    fn now(&self) -> Instant;
}

/// The real monotonic clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// What a failure is remembered for: the device and the mode the config asks for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NativeKey {
    device: String,
    width: u32,
    height: u32,
    fps: u32,
    fourcc: Option<String>,
}

impl NativeKey {
    pub fn from_config(cam: &CameraConfig) -> Self {
        Self {
            device: cam
                .device_path
                .clone()
                .unwrap_or_else(|| format!("index:{}", cam.device_index)),
            width: cam.width,
            height: cam.height,
            fps: cam.fps,
            fourcc: cam.fourcc.as_ref().map(|f| f.to_ascii_uppercase()),
        }
    }
}

/// A remembered failure that is still inside its cooldown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedFailure {
    pub age: Duration,
    pub reason: String,
}

#[derive(Debug)]
struct Entry {
    at: Instant,
    reason: String,
}

/// Failures of the native backend, keyed by [`NativeKey`].
pub struct FailureCache<C: Clock> {
    clock: C,
    entries: Mutex<HashMap<NativeKey, Entry>>,
}

impl<C: Clock> FailureCache<C> {
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            entries: Mutex::new(HashMap::new()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<NativeKey, Entry>> {
        // A poisoned lock only means another thread panicked while holding it;
        // the map itself is always consistent.
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Remembers that the native backend failed for `key`.
    pub fn record_failure(&self, key: NativeKey, reason: String) {
        let at = self.clock.now();
        self.lock().insert(key, Entry { at, reason });
    }

    /// Forgets `key` (the native backend worked).
    pub fn clear(&self, key: &NativeKey) {
        self.lock().remove(key);
    }

    /// The failure of `key` if it happened less than `cooldown` ago. An expired
    /// entry is dropped, so the next open probes native again.
    pub fn recent_failure(&self, key: &NativeKey, cooldown: Duration) -> Option<CachedFailure> {
        let now = self.clock.now();
        let mut entries = self.lock();
        let age = now.saturating_duration_since(entries.get(key)?.at);
        if age >= cooldown {
            entries.remove(key);
            return None;
        }
        entries.get(key).map(|e| CachedFailure {
            age,
            reason: e.reason.clone(),
        })
    }
}

/// The process-wide cache used by `Camera::new_with_config`.
pub fn global() -> &'static FailureCache<SystemClock> {
    static CACHE: OnceLock<FailureCache<SystemClock>> = OnceLock::new();
    CACHE.get_or_init(|| FailureCache::new(SystemClock))
}

/// What `Camera::new_with_config` does about the native backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeDecision {
    /// Try the native backend.
    Try,
    /// `backend = "ffmpeg"`: never try it.
    SkipByConfig,
    /// `backend = "auto"` and it failed recently: go straight to ffmpeg.
    SkipCooling(CachedFailure),
}

/// Decides whether to try the native backend for `cam`.
///
/// `native` ignores the cache (there is no fallback to protect), `ffmpeg`
/// never tries native, `auto` consults the cache unless the cooldown is 0.
pub fn decide<C: Clock>(cam: &CameraConfig, cache: &FailureCache<C>) -> NativeDecision {
    match cam.backend {
        CameraBackendChoice::Ffmpeg => NativeDecision::SkipByConfig,
        CameraBackendChoice::Native => NativeDecision::Try,
        CameraBackendChoice::Auto => {
            let cooldown = Duration::from_secs(u64::from(cam.native_failure_cooldown_secs));
            if cooldown.is_zero() {
                return NativeDecision::Try;
            }
            match cache.recent_failure(&NativeKey::from_config(cam), cooldown) {
                Some(f) => NativeDecision::SkipCooling(f),
                None => NativeDecision::Try,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Clock advanced by hand; clones share the offset.
    #[derive(Clone)]
    struct FakeClock {
        base: Instant,
        offset: Arc<Mutex<Duration>>,
    }
    impl FakeClock {
        fn new() -> Self {
            Self {
                base: Instant::now(),
                offset: Arc::new(Mutex::new(Duration::ZERO)),
            }
        }
        fn advance(&self, d: Duration) {
            *self.offset.lock().unwrap() += d;
        }
    }
    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            self.base + *self.offset.lock().unwrap()
        }
    }

    const COOLDOWN_SECS: u32 = 600;

    fn cam(backend: CameraBackendChoice) -> CameraConfig {
        CameraConfig {
            backend,
            native_failure_cooldown_secs: COOLDOWN_SECS,
            ..CameraConfig::default()
        }
    }

    fn fail(cache: &FailureCache<FakeClock>, cam: &CameraConfig) {
        cache.record_failure(NativeKey::from_config(cam), "no usable frame".to_string());
    }

    #[test]
    fn second_open_skips_native_and_reprobes_after_the_cooldown() {
        let clock = FakeClock::new();
        let cache = FailureCache::new(clock.clone());
        let auto = cam(CameraBackendChoice::Auto);

        assert_eq!(decide(&auto, &cache), NativeDecision::Try);
        fail(&cache, &auto);

        clock.advance(Duration::from_secs(30));
        match decide(&auto, &cache) {
            NativeDecision::SkipCooling(f) => {
                assert_eq!(f.age, Duration::from_secs(30));
                assert_eq!(f.reason, "no usable frame");
            }
            other => panic!("expected a cached skip, got {other:?}"),
        }

        clock.advance(Duration::from_secs(u64::from(COOLDOWN_SECS)));
        assert_eq!(decide(&auto, &cache), NativeDecision::Try, "expired");
        // The expired entry is gone: a later open still probes.
        assert_eq!(decide(&auto, &cache), NativeDecision::Try);
    }

    #[test]
    fn cache_is_per_device_and_mode() {
        let cache = FailureCache::new(FakeClock::new());
        let auto = cam(CameraBackendChoice::Auto);
        fail(&cache, &auto);
        let other_mode = CameraConfig {
            width: auto.width / 2,
            ..auto.clone()
        };
        let other_device = CameraConfig {
            device_index: auto.device_index + 1,
            ..auto.clone()
        };
        let pinned = CameraConfig {
            device_path: Some("/dev/v4l/by-id/x".to_string()),
            ..auto.clone()
        };
        for other in [other_mode, other_device, pinned] {
            assert_eq!(decide(&other, &cache), NativeDecision::Try);
        }
        assert!(matches!(
            decide(&auto, &cache),
            NativeDecision::SkipCooling(_)
        ));
    }

    #[test]
    fn success_clears_the_failure() {
        let cache = FailureCache::new(FakeClock::new());
        let auto = cam(CameraBackendChoice::Auto);
        fail(&cache, &auto);
        cache.clear(&NativeKey::from_config(&auto));
        assert_eq!(decide(&auto, &cache), NativeDecision::Try);
    }

    #[test]
    fn forced_backends_behave_as_documented() {
        let cache = FailureCache::new(FakeClock::new());
        let native = cam(CameraBackendChoice::Native);
        let ffmpeg = cam(CameraBackendChoice::Ffmpeg);
        let auto = cam(CameraBackendChoice::Auto);
        // A recorded failure (same device and mode for all three).
        fail(&cache, &auto);
        assert!(matches!(
            decide(&auto, &cache),
            NativeDecision::SkipCooling(_)
        ));
        // native ignores the cache; ffmpeg never tries native.
        assert_eq!(decide(&native, &cache), NativeDecision::Try);
        assert_eq!(decide(&ffmpeg, &cache), NativeDecision::SkipByConfig);
    }

    #[test]
    fn zero_cooldown_disables_the_memory() {
        let cache = FailureCache::new(FakeClock::new());
        let auto = CameraConfig {
            native_failure_cooldown_secs: 0,
            ..cam(CameraBackendChoice::Auto)
        };
        fail(&cache, &auto);
        assert_eq!(decide(&auto, &cache), NativeDecision::Try);
    }
}
