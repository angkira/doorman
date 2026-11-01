//! Pure camera gating and warm-up logic.
//!
//! Kept free of I/O so the privacy invariant (camera closed whenever the
//! session is unlocked) is covered by exhaustive table tests.

/// Inputs to the capture gate, sampled from the lock supervisor and IPC state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GateInputs {
    /// logind `LockedHint`: the screen is locked.
    pub locked: bool,
    /// logind `IdleHint`: the display is idle/blanked.
    pub idle: bool,
    /// `[camera] prearm_on_lock`: open on lock even while the display is idle.
    pub prearm_on_lock: bool,
    /// `[camera] always_track`: explicit opt-in to capture while unlocked.
    pub always_track: bool,
    /// Preview/debug stream subscribers (explicit opt-in to capture).
    pub stream_subscribers: usize,
    /// Live on-demand capture leases (enroll / `test` auth).
    pub capture_leases: usize,
}

/// Should the live camera be open?
///
/// The lock-driven part (`lock_wanted`) is what unattended unlock relies on:
/// it is true only while the session is locked, and while locked it is true
/// when the display is not idle (`armed`) or when `prearm_on_lock` is set.
/// `idle` and `prearm_on_lock` can therefore never open the camera on an
/// unlocked session. The remaining terms are explicit, user-initiated reasons
/// (always_track config, a connected preview client, an enroll/test lease) that
/// already existed before prearm.
pub fn camera_wanted(i: GateInputs) -> bool {
    lock_wanted(i) || i.always_track || i.stream_subscribers > 0 || i.capture_leases > 0
}

/// The part of the gate driven only by the lock state.
pub fn lock_wanted(i: GateInputs) -> bool {
    i.locked && (!i.idle || i.prearm_on_lock)
}

/// Discards the first N frames after a cold open (auto-exposure settling).
/// A fresh gate is created on every open, so a camera that is already
/// streaming is never subject to a discard.
#[derive(Debug, Clone, Copy)]
pub struct WarmupGate {
    remaining: u32,
}

impl WarmupGate {
    pub fn new(warmup_frames: u32) -> Self {
        Self {
            remaining: warmup_frames,
        }
    }

    /// Returns `true` if the frame must be dropped.
    pub fn discard_next(&mut self) -> bool {
        if self.remaining == 0 {
            false
        } else {
            self.remaining -= 1;
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOOLS: [bool; 2] = [false, true];

    fn inputs(locked: bool, idle: bool, prearm: bool, lease: bool) -> GateInputs {
        GateInputs {
            locked,
            idle,
            prearm_on_lock: prearm,
            capture_leases: usize::from(lease),
            ..GateInputs::default()
        }
    }

    #[test]
    fn table_locked_idle_prearm_lease() {
        // (locked, idle, prearm, lease) -> open?
        let table = [
            (false, false, false, false, false),
            (false, false, true, false, false),
            (false, true, false, false, false),
            (false, true, true, false, false),
            (true, false, false, false, true),
            (true, false, true, false, true),
            (true, true, false, false, false),
            (true, true, true, false, true),
            // An on-demand lease opens the camera in every state.
            (false, false, false, true, true),
            (false, true, true, true, true),
            (true, true, false, true, true),
        ];
        for (locked, idle, prearm, lease, want) in table {
            assert_eq!(
                camera_wanted(inputs(locked, idle, prearm, lease)),
                want,
                "locked={locked} idle={idle} prearm={prearm} lease={lease}"
            );
        }
    }

    #[test]
    fn privacy_unlocked_means_closed_for_every_lock_derived_combination() {
        // No explicit opt-in (always_track / preview / lease): an unlocked
        // session never opens the camera, whatever idle and prearm say.
        for idle in BOOLS {
            for prearm in BOOLS {
                let i = inputs(false, idle, prearm, false);
                assert!(!camera_wanted(i), "idle={idle} prearm={prearm}");
                assert!(!lock_wanted(i), "idle={idle} prearm={prearm}");
            }
        }
    }

    #[test]
    fn lock_driven_gate_is_closed_when_unlocked_even_with_opt_ins() {
        for idle in BOOLS {
            for prearm in BOOLS {
                for always_track in BOOLS {
                    let i = GateInputs {
                        locked: false,
                        idle,
                        prearm_on_lock: prearm,
                        always_track,
                        stream_subscribers: 3,
                        capture_leases: 2,
                    };
                    assert!(!lock_wanted(i));
                }
            }
        }
    }

    #[test]
    fn prearm_only_adds_the_locked_idle_case() {
        for locked in BOOLS {
            for idle in BOOLS {
                let off = camera_wanted(inputs(locked, idle, false, false));
                let on = camera_wanted(inputs(locked, idle, true, false));
                if locked && idle {
                    assert!(!off && on);
                } else {
                    assert_eq!(off, on, "locked={locked} idle={idle}");
                }
            }
        }
    }

    #[test]
    fn always_track_and_preview_keep_existing_behaviour() {
        let mut i = inputs(false, false, false, false);
        i.always_track = true;
        assert!(camera_wanted(i));
        i.always_track = false;
        i.stream_subscribers = 1;
        assert!(camera_wanted(i));
    }

    #[test]
    fn warmup_discards_exactly_n_then_passes() {
        let mut w = WarmupGate::new(3);
        assert_eq!(
            (0..6).map(|_| w.discard_next()).collect::<Vec<_>>(),
            [true, true, true, false, false, false]
        );
    }

    #[test]
    fn warmup_zero_never_discards() {
        let mut w = WarmupGate::new(0);
        assert!(!w.discard_next());
    }
}
