//! Pure metric functions. No I/O, no clocks, no randomness.
//!
//! Conventions (all documented here because they change the numbers):
//!
//! * **Percentile**: linear interpolation between closest ranks (the "R-7"
//!   definition, numpy's default). For `n` sorted values and percentile `p`
//!   the fractional rank is `p/100 * (n-1)` (0-based); the result is
//!   interpolated between the two neighbouring order statistics.
//! * **Accuracy** (higher score = more likely the enrolled person): a probe is
//!   accepted iff `score > threshold` (strict). `FAR(t)` is the fraction of
//!   impostor scores accepted, `FRR(t)` the fraction of genuine scores
//!   rejected, `TAR = 1 - FRR`.
//! * **TAR@FAR=f**: the threshold is the `(floor(f*N)+1)`-th highest impostor
//!   score (N impostor scores). That is the lowest threshold for which at most
//!   `floor(f*N)` impostors are accepted. When `f*N < 1` the point is not
//!   statistically resolvable (it is the maximum impostor score, FAR = 0) and
//!   is flagged `resolvable = false`.
//! * **EER**: over every distinct score (plus minus infinity) pick the
//!   threshold minimising `|FAR - FRR|` (ties: smaller `max(FAR, FRR)`, then
//!   lower threshold); `EER = (FAR + FRR) / 2` there.
//! * **Liveness** (higher score = more likely live, `relief` in the daemon): a
//!   sample is accepted as live iff `score >= threshold` (the daemon rejects
//!   `relief < threshold`). `APCER` = fraction of attacks accepted,
//!   `BPCER` = fraction of bona fide samples rejected.

use std::cmp::Ordering;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum MetricsError {
    #[error("{0} must contain at least one score")]
    Empty(&'static str),
    #[error("{0} contains a non-finite score (NaN or infinity)")]
    NonFinite(&'static str),
    #[error("{name} must be in the open interval (0, 1), got {value}")]
    BadRate { name: &'static str, value: f64 },
    #[error("histogram needs at least one bin")]
    NoBins,
}

pub type Result<T> = std::result::Result<T, MetricsError>;

fn check(name: &'static str, v: &[f64]) -> Result<()> {
    if v.is_empty() {
        return Err(MetricsError::Empty(name));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(MetricsError::NonFinite(name));
    }
    Ok(())
}

fn check_rate(name: &'static str, value: f64) -> Result<()> {
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(MetricsError::BadRate { name, value })
    }
}

/// Ascending sorted copy.
pub fn sorted(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s
}

/// Percentile `p` in `[0, 100]` of ascending-sorted `sorted_vals`
/// (linear interpolation, R-7). `p` is clamped to `[0, 100]`.
pub fn percentile(sorted_vals: &[f64], p: f64) -> Result<f64> {
    check("percentile input", sorted_vals)?;
    let n = sorted_vals.len();
    let rank = p.clamp(0.0, 100.0) / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    Ok(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))
}

/// Number of values in ascending `s` strictly greater than `t`.
fn count_gt(s: &[f64], t: f64) -> usize {
    s.len() - s.partition_point(|x| *x <= t)
}

/// Number of values in ascending `s` strictly less than `t`.
fn count_lt(s: &[f64], t: f64) -> usize {
    s.partition_point(|x| *x < t)
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperatingPoint {
    pub far_target: f64,
    /// Accept iff `score > threshold`.
    pub threshold: f64,
    pub far: f64,
    pub tar: f64,
    /// `false` when `far_target * N_impostor < 1`: too few impostor scores to
    /// resolve this FAR.
    pub resolvable: bool,
}

/// TAR at the threshold that allows at most `far_target` false accepts.
pub fn tar_at_far(genuine: &[f64], impostor: &[f64], far_target: f64) -> Result<OperatingPoint> {
    check("genuine scores", genuine)?;
    check("impostor scores", impostor)?;
    check_rate("far_target", far_target)?;
    let g = sorted(genuine);
    let i = sorted(impostor);
    let n = i.len();
    // Small epsilon so that e.g. 1e-3 * 1000 does not floor to 0.
    let allowed = ((far_target * n as f64) + 1e-9).floor() as usize;
    let k = allowed.min(n - 1);
    let threshold = i[n - 1 - k];
    Ok(OperatingPoint {
        far_target,
        threshold,
        far: count_gt(&i, threshold) as f64 / n as f64,
        tar: count_gt(&g, threshold) as f64 / g.len() as f64,
        resolvable: allowed >= 1,
    })
}

/// TAR of `genuine` at a fixed `threshold` (accept iff `score > threshold`).
pub fn tar_at_threshold(genuine: &[f64], threshold: f64) -> Result<f64> {
    check("genuine scores", genuine)?;
    Ok(count_gt(&sorted(genuine), threshold) as f64 / genuine.len() as f64)
}

#[derive(Debug, Clone, PartialEq)]
pub struct EerPoint {
    pub eer: f64,
    pub threshold: f64,
    pub far: f64,
    pub frr: f64,
}

pub fn eer(genuine: &[f64], impostor: &[f64]) -> Result<EerPoint> {
    check("genuine scores", genuine)?;
    check("impostor scores", impostor)?;
    let g = sorted(genuine);
    let i = sorted(impostor);
    let mut candidates: Vec<f64> = g.iter().chain(i.iter()).copied().collect();
    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup();
    candidates.insert(0, f64::NEG_INFINITY);

    let mut best: Option<EerPoint> = None;
    for &t in &candidates {
        let far = count_gt(&i, t) as f64 / i.len() as f64;
        let frr = 1.0 - count_gt(&g, t) as f64 / g.len() as f64;
        let cand = EerPoint {
            eer: (far + frr) / 2.0,
            threshold: t,
            far,
            frr,
        };
        best = Some(match best {
            None => cand,
            Some(b) => {
                let key = |p: &EerPoint| ((p.far - p.frr).abs(), p.far.max(p.frr));
                let (kb, kc) = (key(&b), key(&cand));
                let ord =
                    kc.0.partial_cmp(&kb.0)
                        .unwrap_or(Ordering::Equal)
                        .then(kc.1.partial_cmp(&kb.1).unwrap_or(Ordering::Equal));
                // Strictly better only: ties keep the earlier (lower) threshold.
                if ord == Ordering::Less {
                    cand
                } else {
                    b
                }
            }
        });
    }
    Ok(best.expect("candidates is never empty"))
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistBin {
    pub lo: f64,
    pub hi: f64,
    pub genuine: usize,
    pub impostor: usize,
}

/// Equal-width histogram over `[min, max]` of both score sets. The last bin
/// includes the upper edge.
pub fn histogram(genuine: &[f64], impostor: &[f64], bins: usize) -> Result<Vec<HistBin>> {
    check("genuine scores", genuine)?;
    check("impostor scores", impostor)?;
    if bins == 0 {
        return Err(MetricsError::NoBins);
    }
    let all = genuine.iter().chain(impostor.iter());
    let lo = all.clone().copied().fold(f64::INFINITY, f64::min);
    let hi = all.copied().fold(f64::NEG_INFINITY, f64::max);
    let width = (hi - lo) / bins as f64;
    let index = |v: f64| -> usize {
        if width == 0.0 {
            0
        } else {
            (((v - lo) / width) as usize).min(bins - 1)
        }
    };
    let mut out: Vec<HistBin> = (0..bins)
        .map(|b| HistBin {
            lo: lo + width * b as f64,
            hi: lo + width * (b + 1) as f64,
            genuine: 0,
            impostor: 0,
        })
        .collect();
    for &v in genuine {
        out[index(v)].genuine += 1;
    }
    for &v in impostor {
        out[index(v)].impostor += 1;
    }
    Ok(out)
}

#[derive(Debug, Clone, PartialEq)]
pub struct LivenessPoint {
    /// Accept as live iff `score >= threshold`.
    pub threshold: f64,
    pub apcer: f64,
    pub bpcer: f64,
}

pub fn liveness_at(bona_fide: &[f64], attack: &[f64], threshold: f64) -> Result<LivenessPoint> {
    check("bona fide scores", bona_fide)?;
    check("attack scores", attack)?;
    let b = sorted(bona_fide);
    let a = sorted(attack);
    Ok(LivenessPoint {
        threshold,
        apcer: (a.len() - count_lt(&a, threshold)) as f64 / a.len() as f64,
        bpcer: count_lt(&b, threshold) as f64 / b.len() as f64,
    })
}

/// ROC points over every distinct score (ascending threshold) plus `+inf`
/// (reject everything).
pub fn liveness_roc(bona_fide: &[f64], attack: &[f64]) -> Result<Vec<LivenessPoint>> {
    check("bona fide scores", bona_fide)?;
    check("attack scores", attack)?;
    let mut ts: Vec<f64> = bona_fide.iter().chain(attack.iter()).copied().collect();
    ts.sort_by(|a, b| a.total_cmp(b));
    ts.dedup();
    ts.push(f64::INFINITY);
    ts.iter()
        .map(|&t| liveness_at(bona_fide, attack, t))
        .collect()
}

/// Highest threshold (i.e. lowest APCER) whose BPCER does not exceed
/// `target_bpcer`. Candidates are the bona fide scores themselves.
pub fn threshold_for_bpcer(
    bona_fide: &[f64],
    attack: &[f64],
    target_bpcer: f64,
) -> Result<LivenessPoint> {
    check("bona fide scores", bona_fide)?;
    check("attack scores", attack)?;
    check_rate("target_bpcer", target_bpcer)?;
    let b = sorted(bona_fide);
    let allowed = ((target_bpcer * b.len() as f64) + 1e-9).floor() as usize;
    let mut chosen = b[0];
    for &v in &b {
        if count_lt(&b, v) <= allowed {
            chosen = v;
        } else {
            break;
        }
    }
    liveness_at(bona_fide, attack, chosen)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn percentile_r7_hand_computed() {
        let v = [1.0, 2.0, 3.0, 4.0];
        close(percentile(&v, 0.0).unwrap(), 1.0);
        close(percentile(&v, 50.0).unwrap(), 2.5); // rank 1.5
        close(percentile(&v, 95.0).unwrap(), 3.85); // rank 2.85
        close(percentile(&v, 99.0).unwrap(), 3.97); // rank 2.97
        close(percentile(&v, 100.0).unwrap(), 4.0);
        close(percentile(&[7.0], 99.0).unwrap(), 7.0);
        assert_eq!(
            percentile(&[], 50.0),
            Err(MetricsError::Empty("percentile input"))
        );
    }

    #[test]
    fn tar_at_far_hand_computed() {
        let imp: Vec<f64> = (1..=10).map(|x| x as f64 / 10.0).collect();
        let gen = [0.85, 0.95, 0.99, 0.5];
        // FAR 0.1 of N=10 -> one false accept allowed -> threshold = 2nd highest = 0.9.
        let p = tar_at_far(&gen, &imp, 0.1).unwrap();
        close(p.threshold, 0.9);
        close(p.far, 0.1);
        close(p.tar, 0.5);
        assert!(p.resolvable);
        // FAR 0.01 < 1/N -> threshold = max impostor, FAR 0, flagged unresolvable.
        let p = tar_at_far(&gen, &imp, 0.01).unwrap();
        close(p.threshold, 1.0);
        close(p.far, 0.0);
        close(p.tar, 0.0);
        assert!(!p.resolvable);
        assert!(tar_at_far(&gen, &imp, 0.0).is_err());
        assert!(tar_at_far(&gen, &imp, 1.0).is_err());
    }

    #[test]
    fn tar_at_far_handles_ties() {
        // Four impostors tied at 0.5, allowed = floor(0.5*4) = 2 -> t = 0.5 -> FAR 0 (strict >).
        let p = tar_at_far(&[0.6], &[0.5, 0.5, 0.5, 0.5], 0.5).unwrap();
        close(p.threshold, 0.5);
        close(p.far, 0.0);
        close(p.tar, 1.0);
    }

    #[test]
    fn eer_hand_computed() {
        let gen = [0.6, 0.7, 0.8, 0.9];
        let imp = [0.1, 0.2, 0.3, 0.65];
        // t = 0.6: FAR = 1/4 (0.65), FRR = 1/4 (0.6 not > 0.6).
        let e = eer(&gen, &imp).unwrap();
        close(e.eer, 0.25);
        close(e.threshold, 0.6);
        close(e.far, 0.25);
        close(e.frr, 0.25);
    }

    #[test]
    fn eer_perfect_separation_is_zero() {
        let e = eer(&[0.9, 0.8], &[0.1, 0.2]).unwrap();
        close(e.eer, 0.0);
        close(e.far, 0.0);
        close(e.frr, 0.0);
    }

    #[test]
    fn histogram_hand_computed() {
        let h = histogram(&[0.0, 1.0], &[0.5], 2).unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!((h[0].genuine, h[0].impostor), (1, 0));
        assert_eq!((h[1].genuine, h[1].impostor), (1, 1)); // 0.5 and the 1.0 edge
        close(h[1].lo, 0.5);
        close(h[1].hi, 1.0);
        assert_eq!(histogram(&[1.0], &[1.0], 0), Err(MetricsError::NoBins));
        // Degenerate range: everything lands in bin 0.
        let h = histogram(&[1.0], &[1.0], 3).unwrap();
        assert_eq!((h[0].genuine, h[0].impostor), (1, 1));
    }

    #[test]
    fn liveness_hand_computed() {
        let bona = [0.2, 0.3, 0.4, 0.5];
        let attack = [0.0, 0.1, 0.25, 0.35];
        let p = liveness_at(&bona, &attack, 0.3).unwrap();
        close(p.apcer, 0.25); // 0.35
        close(p.bpcer, 0.25); // 0.2
                              // Target 25% BPCER -> allowed 1 -> highest threshold with <=1 bona fide below it = 0.3.
        let r = threshold_for_bpcer(&bona, &attack, 0.25).unwrap();
        close(r.threshold, 0.3);
        // Default-style 1% on 4 samples allows 0 rejects -> min bona fide score.
        let r = threshold_for_bpcer(&bona, &attack, 0.01).unwrap();
        close(r.threshold, 0.2);
        close(r.bpcer, 0.0);
        close(r.apcer, 0.5); // 0.25 and 0.35
    }

    #[test]
    fn liveness_roc_endpoints() {
        let roc = liveness_roc(&[0.2, 0.4], &[0.1, 0.3]).unwrap();
        let first = &roc[0];
        close(first.threshold, 0.1);
        close(first.apcer, 1.0);
        close(first.bpcer, 0.0);
        let last = roc.last().unwrap();
        assert!(last.threshold.is_infinite());
        close(last.apcer, 0.0);
        close(last.bpcer, 1.0);
        assert_eq!(roc.len(), 5);
    }

    #[test]
    fn rejects_non_finite_scores() {
        assert_eq!(
            tar_at_far(&[f64::NAN], &[0.1], 0.1),
            Err(MetricsError::NonFinite("genuine scores"))
        );
    }
}
