//! Typed result of one depth-liveness evaluation.
//!
//! The relief value is carried so callers can log it, put it in the timing
//! record and calibrate the threshold from it (docs/decision-and-liveness.md).

/// Outcome of evaluating liveness on one frame. A gate that cannot run is a
/// distinct, fail-closed outcome so "model unavailable" is never confused with
/// "spoof suspected".
#[derive(Debug, Clone, PartialEq)]
pub enum LivenessOutcome {
    /// Relief at or above the threshold.
    Pass { relief: f32 },
    /// Relief below the threshold: suspected spoof.
    Fail { relief: f32 },
    /// The check could not run (model missing, inference error). Fail closed.
    Unavailable { reason: String },
}

impl LivenessOutcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass { .. })
    }

    /// The relief value, when the check produced one.
    pub fn relief(&self) -> Option<f32> {
        match self {
            Self::Pass { relief } | Self::Fail { relief } => Some(*relief),
            Self::Unavailable { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_pass_is_pass_and_relief_is_exposed() {
        assert!(LivenessOutcome::Pass { relief: 0.2 }.is_pass());
        assert!(!LivenessOutcome::Fail { relief: 0.01 }.is_pass());
        let u = LivenessOutcome::Unavailable { reason: "x".into() };
        assert!(!u.is_pass());
        assert_eq!(u.relief(), None);
        assert_eq!(LivenessOutcome::Fail { relief: 0.01 }.relief(), Some(0.01));
    }
}
