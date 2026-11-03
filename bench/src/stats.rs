//! Exact binomial confidence bounds and test-size reasoning.
//!
//! With `k` errors in `n` independent trials the one-sided Clopper-Pearson
//! upper bound at confidence `c = 1 - alpha` is the `p` for which
//! `P(X <= k | n, p) = alpha`. Equivalently `p` solves
//! `I_p(k + 1, n - k) = c`, where `I` is the regularized incomplete beta
//! function (the beta quantile). It is computed here by bisection on `I`, which
//! is evaluated with the continued fraction of Numerical Recipes (`betacf`,
//! modified Lentz) and a Lanczos `ln Gamma`. No dependency is needed.
//!
//! Special cases: `k = 0` has the closed form `1 - alpha^(1/n)` (about `3/n`
//! at 95%, the "rule of 3"); `k = n` gives 1.
//!
//! The bound assumes independent trials. Repeated presentations by one subject,
//! one instrument or one session are correlated, so the bound is optimistic
//! for them; the reports say so.

/// Default one-sided confidence level of every reported bound.
pub const DEFAULT_CONFIDENCE: f64 = 0.95;

/// Stop bisection when the bracket is narrower than this (absolute, on `p`).
const BISECTION_TOLERANCE: f64 = 1e-15;
/// Hard cap on bisection steps (2^-200 is far below f64 resolution).
const BISECTION_MAX_STEPS: usize = 200;
/// Continued-fraction convergence (relative) and iteration cap.
const CF_EPSILON: f64 = 1e-15;
const CF_MAX_ITERATIONS: usize = 100_000;
/// Lentz underflow guard.
const CF_TINY: f64 = 1e-300;
/// Search cap for `required_trials_with_errors`.
const MAX_TRIALS_SEARCH: u64 = 1 << 40;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StatsError {
    #[error("number of trials must be at least 1")]
    NoTrials,
    #[error("{errors} errors cannot exceed {trials} trials")]
    ErrorsExceedTrials { errors: u64, trials: u64 },
    #[error("{name} must be in the open interval (0, 1), got {value}")]
    BadProbability { name: &'static str, value: f64 },
    #[error("incomplete beta continued fraction did not converge (a={a}, b={b}, x={x})")]
    NoConvergence { a: f64, b: f64, x: f64 },
    #[error("no trial count up to {MAX_TRIALS_SEARCH} reaches the target bound")]
    Unreachable,
}

type Result<T> = std::result::Result<T, StatsError>;

fn check_open_unit(name: &'static str, value: f64) -> Result<()> {
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(StatsError::BadProbability { name, value })
    }
}

/// Lanczos approximation of `ln Gamma(x)` for `x > 0` (g = 5, n = 6, NR 6.1).
fn ln_gamma(x: f64) -> f64 {
    const COEFFS: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut y = x;
    let mut ser = 1.000_000_000_190_015;
    for c in COEFFS {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

fn guard_tiny(v: f64) -> f64 {
    if v.abs() < CF_TINY {
        CF_TINY
    } else {
        v
    }
}

/// Continued fraction of the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> Result<f64> {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 / guard_tiny(1.0 - qab * x / qap);
    let mut h = d;
    for m in 1..=CF_MAX_ITERATIONS {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 / guard_tiny(1.0 + aa * d);
        c = guard_tiny(1.0 + aa / c);
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 / guard_tiny(1.0 + aa * d);
        c = guard_tiny(1.0 + aa / c);
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < CF_EPSILON {
            return Ok(h);
        }
    }
    Err(StatsError::NoConvergence { a, b, x })
}

/// Regularized incomplete beta function `I_x(a, b)` for `a, b > 0`, `x` in `[0, 1]`.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> Result<f64> {
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x >= 1.0 {
        return Ok(1.0);
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(front * beta_cf(a, b, x)? / a)
    } else {
        Ok(1.0 - front * beta_cf(b, a, 1.0 - x)? / b)
    }
}

fn check_trials(errors: u64, trials: u64) -> Result<()> {
    if trials == 0 {
        return Err(StatsError::NoTrials);
    }
    if errors > trials {
        return Err(StatsError::ErrorsExceedTrials { errors, trials });
    }
    Ok(())
}

/// One-sided Clopper-Pearson upper bound on the error probability after
/// `errors` errors in `trials` independent trials, at `confidence` (e.g. 0.95).
pub fn upper_bound(errors: u64, trials: u64, confidence: f64) -> Result<f64> {
    check_trials(errors, trials)?;
    check_open_unit("confidence", confidence)?;
    if errors == trials {
        return Ok(1.0);
    }
    let alpha = 1.0 - confidence;
    if errors == 0 {
        return Ok(-(alpha.ln() / trials as f64).exp_m1());
    }
    let (a, b) = ((errors + 1) as f64, (trials - errors) as f64);
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..BISECTION_MAX_STEPS {
        let mid = 0.5 * (lo + hi);
        if regularized_incomplete_beta(a, b, mid)? < confidence {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < BISECTION_TOLERANCE {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Trials needed so that `errors = 0` gives an upper bound `<= target`:
/// `ceil(ln(alpha) / ln(1 - target))`.
pub fn required_trials(target: f64, confidence: f64) -> Result<u64> {
    check_open_unit("target", target)?;
    check_open_unit("confidence", confidence)?;
    let n = ((1.0 - confidence).ln() / (-target).ln_1p()).ceil();
    Ok(n as u64)
}

/// Smallest `n >= errors` for which `upper_bound(errors, n, confidence) <= target`
/// (the bound shrinks as `n` grows with `errors` fixed). Assumes no further errors
/// beyond the `errors` already seen. Exponential then binary search.
pub fn required_trials_with_errors(errors: u64, target: f64, confidence: f64) -> Result<u64> {
    if errors == 0 {
        return required_trials(target, confidence);
    }
    check_open_unit("target", target)?;
    let ok = |n: u64| -> Result<bool> { Ok(upper_bound(errors, n, confidence)? <= target) };
    let mut hi = errors + 1;
    while !ok(hi)? {
        if hi >= MAX_TRIALS_SEARCH {
            return Err(StatsError::Unreachable);
        }
        hi *= 2;
    }
    let mut lo = errors; // upper_bound(errors, errors) = 1 > target
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if ok(mid)? {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok(hi)
}

/// Rates below this (as a fraction) are printed in scientific notation: a
/// percentage with three decimals would round FMR-sized values to nothing.
const SCIENTIFIC_BELOW: f64 = 1e-4;

/// A fraction as a percentage with three decimals (`0.0466` -> `4.656%`);
/// tiny non-zero values are printed as a scientific fraction.
pub fn format_percent(x: f64) -> String {
    if x == 0.0 {
        "0%".to_string()
    } else if x.abs() < SCIENTIFIC_BELOW {
        format!("{x:.2e}")
    } else {
        format!("{:.3}%", x * 100.0)
    }
}

/// A confidence level such as `0.95` as `95.0%`.
pub fn format_confidence(c: f64) -> String {
    format!("{:.1}%", c * 100.0)
}

/// An observed error rate with its one-sided upper confidence bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rate {
    pub errors: u64,
    pub trials: u64,
    pub point: f64,
    pub upper: f64,
}

impl Rate {
    /// `None` when there are no trials (a rate is undefined, not zero).
    pub fn new(errors: u64, trials: u64, confidence: f64) -> Result<Option<Self>> {
        if trials == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            errors,
            trials,
            point: errors as f64 / trials as f64,
            upper: upper_bound(errors, trials, confidence)?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_pairs_point_and_bound() {
        assert_eq!(Rate::new(0, 0, 0.95), Ok(None));
        let r = Rate::new(1, 100, 0.95).unwrap().unwrap();
        assert_eq!((r.errors, r.trials, r.point), (1, 100, 0.01));
        close(r.upper, 0.046_559_811_453_538_755, 1e-9);
    }

    fn close(a: f64, b: f64, rel: f64) {
        assert!(
            (a - b).abs() <= rel * b.abs().max(f64::MIN_POSITIVE),
            "{a} vs {b}"
        );
    }

    #[test]
    fn ln_gamma_matches_factorials() {
        // Gamma(5) = 24, Gamma(0.5) = sqrt(pi).
        close(ln_gamma(5.0), 24.0_f64.ln(), 1e-12);
        close(ln_gamma(0.5), std::f64::consts::PI.sqrt().ln(), 1e-12);
    }

    #[test]
    fn incomplete_beta_closed_forms() {
        // I_x(1, 1) = x; I_x(2, 1) = x^2; I_x(1, 2) = 1 - (1-x)^2.
        close(
            regularized_incomplete_beta(1.0, 1.0, 0.3).unwrap(),
            0.3,
            1e-12,
        );
        close(
            regularized_incomplete_beta(2.0, 1.0, 0.3).unwrap(),
            0.09,
            1e-12,
        );
        close(
            regularized_incomplete_beta(1.0, 2.0, 0.3).unwrap(),
            0.51,
            1e-12,
        );
        // Symmetric case at the midpoint.
        close(
            regularized_incomplete_beta(4.0, 4.0, 0.5).unwrap(),
            0.5,
            1e-12,
        );
    }

    #[test]
    fn zero_errors_closed_form() {
        // 1 - 0.05^(1/n): hand value for n = 300 is 0.0099361 (rule of 3: 0.01).
        close(
            upper_bound(0, 300, 0.95).unwrap(),
            0.009_936_081_944_457_71,
            1e-12,
        );
        close(
            upper_bound(0, 29_957, 0.95).unwrap(),
            9.999_607_738_8e-5,
            1e-9,
        );
        // n = 1: the bound is the confidence itself.
        close(upper_bound(0, 1, 0.95).unwrap(), 0.95, 1e-12);
    }

    #[test]
    fn matches_independent_python_bisection() {
        // Reference values: bisection on the exact binomial CDF in Python
        // (math.lgamma terms), P(X <= k | n, p) = 0.05.
        let cases: [(u64, u64, f64); 7] = [
            (1, 100, 0.046_559_811_453_538_755),
            (2, 100, 0.061_619_200_396_045_38),
            (5, 1000, 0.010_484_076_911_415_277),
            (3, 50, 0.147_837_176_361_954_18),
            (10, 20, 0.698_046_088_713_506_5),
            (1, 2, 0.974_679_434_480_896_3),
            (0, 10, 0.258_865_550_893_050_9),
        ];
        for (k, n, want) in cases {
            close(upper_bound(k, n, 0.95).unwrap(), want, 1e-9);
        }
    }

    #[test]
    fn all_errors_gives_one_and_inputs_are_validated() {
        assert_eq!(upper_bound(7, 7, 0.95).unwrap(), 1.0);
        assert_eq!(upper_bound(0, 0, 0.95), Err(StatsError::NoTrials));
        assert!(matches!(
            upper_bound(3, 2, 0.95),
            Err(StatsError::ErrorsExceedTrials { .. })
        ));
        assert!(upper_bound(0, 10, 1.0).is_err());
        assert!(required_trials(0.0, 0.95).is_err());
    }

    #[test]
    fn bound_is_monotone_in_errors_and_trials() {
        let b = |k, n| upper_bound(k, n, 0.95).unwrap();
        assert!(b(0, 100) < b(1, 100) && b(1, 100) < b(2, 100));
        assert!(b(1, 200) < b(1, 100));
    }

    #[test]
    fn large_n_converges() {
        let b = upper_bound(3, 1_000_000, 0.95).unwrap();
        // Poisson limit: 3 errors, 95% upper mean 7.754 -> 7.754e-6.
        close(b, 7.754e-6, 1e-3);
    }

    #[test]
    fn required_trials_known_values() {
        // ln(0.05)/ln(0.99) = 298.07 -> 299 (the "300" rule of thumb);
        // /ln(0.93) = 41.28 -> 42; /ln(1-1e-4) = 29955.8 -> 29956;
        // /ln(1-2e-5) = 149785.1 -> 149786.
        assert_eq!(required_trials(0.01, 0.95).unwrap(), 299);
        assert_eq!(required_trials(0.07, 0.95).unwrap(), 42);
        assert_eq!(required_trials(1e-4, 0.95).unwrap(), 29_956);
        assert_eq!(required_trials(2e-5, 0.95).unwrap(), 149_786);
        // Consistency: n reaches the target, n-1 does not.
        for t in [0.01, 0.07, 1e-4] {
            let n = required_trials(t, 0.95).unwrap();
            assert!(upper_bound(0, n, 0.95).unwrap() <= t);
            assert!(upper_bound(0, n - 1, 0.95).unwrap() > t);
        }
    }

    #[test]
    fn required_trials_with_errors_is_minimal() {
        assert_eq!(required_trials_with_errors(0, 0.01, 0.95).unwrap(), 299);
        // One error: the 95% bound is -ln-ish 4.74/n, so about 474 trials.
        let n = required_trials_with_errors(1, 0.01, 0.95).unwrap();
        assert!(upper_bound(1, n, 0.95).unwrap() <= 0.01);
        assert!(upper_bound(1, n - 1, 0.95).unwrap() > 0.01);
        assert!((470..=480).contains(&n), "{n}");
    }
}
