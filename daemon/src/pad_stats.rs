//! Exact (Clopper-Pearson) binomial confidence bounds for the presentation
//! attack screening (`doorman pad run`).
//!
//! The bound of `k` events in `n` trials at confidence `c` is the `p` where the
//! binomial tail equals the error level: upper `P(X <= k | n, p) = alpha`,
//! lower `P(X >= k | n, p) = alpha`. This equals the beta quantile
//! `Beta(1 - alpha; k + 1, n - k)` (upper) and `Beta(alpha; k, n - k + 1)`
//! (lower); the unit tests pin both to values computed with a reference beta
//! quantile function. The tail is summed in log space and the root found by
//! bisection, so no special functions are needed.
//!
//! One-sided means `alpha = 1 - confidence` (what a "no more than X %" claim
//! needs); two-sided puts `alpha / 2` in each tail (the wider textbook
//! interval).

/// The confidence level every bound of the screening report uses.
pub const CONFIDENCE: f64 = 0.95;
const BISECTION_STEPS: usize = 200;

/// `ln(n!)` for `0..=n`.
fn ln_factorials(n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n + 1);
    out.push(0.0);
    for i in 1..=n {
        out.push(out[i - 1] + (i as f64).ln());
    }
    out
}

/// `P(X <= k)` for `X ~ Binomial(n, p)`.
pub fn binomial_cdf(k: usize, n: usize, p: f64) -> f64 {
    if k >= n || p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let lf = ln_factorials(n);
    let (ln_p, ln_q) = (p.ln(), (-p).ln_1p());
    let total: f64 = (0..=k)
        .map(|i| (lf[n] - lf[i] - lf[n - i] + i as f64 * ln_p + (n - i) as f64 * ln_q).exp())
        .sum();
    total.min(1.0)
}

/// Root of a function that is decreasing in `p` on `[0, 1]`.
fn bisect(f: impl Fn(f64) -> f64) -> f64 {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..BISECTION_STEPS {
        let mid = (lo + hi) / 2.0;
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

fn alpha(confidence: f64, two_sided: bool) -> f64 {
    (1.0 - confidence) / if two_sided { 2.0 } else { 1.0 }
}

/// Upper confidence bound of `k` events in `n` trials. `n = 0` carries no
/// information: the bound is 1. Panics if `k > n` (a caller bug).
pub fn upper(k: usize, n: usize, confidence: f64, two_sided: bool) -> f64 {
    assert!(k <= n, "k = {k} exceeds n = {n}");
    if n == 0 || k == n {
        return 1.0;
    }
    let a = alpha(confidence, two_sided);
    bisect(|p| binomial_cdf(k, n, p) - a)
}

/// Lower confidence bound of `k` events in `n` trials.
pub fn lower(k: usize, n: usize, confidence: f64, two_sided: bool) -> f64 {
    assert!(k <= n, "k = {k} exceeds n = {n}");
    if n == 0 || k == 0 {
        return 0.0;
    }
    let a = alpha(confidence, two_sided);
    // P(X >= k) = 1 - P(X <= k-1) increases in p; bisect on its negation.
    bisect(|p| a - (1.0 - binomial_cdf(k - 1, n, p)))
}

/// One-sided 95 % upper bound, the figure the screening report prints.
pub fn upper_95(k: usize, n: usize) -> f64 {
    upper(k, n, CONFIDENCE, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-8, "{a} vs {b}");
    }

    /// (k, n, one-sided upper, two-sided upper, one-sided lower, two-sided
    /// lower): reference beta quantiles `Beta(1 - a; k + 1, n - k)` and
    /// `Beta(a; k, n - k + 1)`.
    const REFERENCE: [(usize, usize, f64, f64, f64, f64); 9] = [
        (0, 10, 0.2588655509, 0.3084971078, 0.0, 0.0),
        (0, 30, 0.0950338529, 0.1157033082, 0.0, 0.0),
        (0, 100, 0.0295130496, 0.0362166926, 0.0, 0.0),
        (
            1,
            10,
            0.3941633024,
            0.4450161170,
            0.0051161969,
            0.0025285785,
        ),
        (
            5,
            10,
            0.7775588990,
            0.8129139716,
            0.2224411010,
            0.1870860284,
        ),
        (
            2,
            50,
            0.1206141554,
            0.1371376256,
            0.0071537195,
            0.0048814334,
        ),
        (
            3,
            20,
            0.3436638043,
            0.3789268265,
            0.0421694079,
            0.0320709372,
        ),
        (
            7,
            40,
            0.3037056670,
            0.3277901334,
            0.0851325046,
            0.0733827294,
        ),
        (0, 300, 0.0099360819, 0.0122209747, 0.0, 0.0),
    ];

    #[test]
    fn matches_reference_beta_quantiles() {
        for (k, n, up1, up2, lo1, lo2) in REFERENCE {
            close(upper(k, n, CONFIDENCE, false), up1);
            close(upper(k, n, CONFIDENCE, true), up2);
            close(lower(k, n, CONFIDENCE, false), lo1);
            close(lower(k, n, CONFIDENCE, true), lo2);
        }
    }

    #[test]
    fn the_headline_numbers_of_the_protocol() {
        let pct = |x: f64| (x * 10000.0).round() / 100.0;
        assert_eq!(pct(upper(0, 10, CONFIDENCE, false)), 25.89);
        assert_eq!(pct(upper(0, 10, CONFIDENCE, true)), 30.85);
        assert_eq!(pct(upper(0, 30, CONFIDENCE, false)), 9.50);
        assert_eq!(pct(upper(0, 30, CONFIDENCE, true)), 11.57);
        assert_eq!(pct(upper(0, 100, CONFIDENCE, false)), 2.95);
        assert_eq!(pct(upper(0, 100, CONFIDENCE, true)), 3.62);
        assert_eq!(upper_95(0, 10), upper(0, 10, CONFIDENCE, false));
    }

    #[test]
    fn edges() {
        assert_eq!(
            upper(0, 0, CONFIDENCE, false),
            1.0,
            "no data, no information"
        );
        assert_eq!(lower(0, 0, CONFIDENCE, false), 0.0);
        assert_eq!(upper(10, 10, CONFIDENCE, false), 1.0);
        close(lower(10, 10, CONFIDENCE, false), 0.7411344491);
        close(upper(0, 1, CONFIDENCE, false), 0.95);
    }

    #[test]
    fn bounds_bracket_the_point_estimate_and_widen_two_sided() {
        for k in 0..=20 {
            let (lo, hi) = (
                lower(k, 20, CONFIDENCE, false),
                upper(k, 20, CONFIDENCE, false),
            );
            let p = k as f64 / 20.0;
            assert!(lo <= p && p <= hi, "k={k}: {lo} {p} {hi}");
            assert!(upper(k, 20, CONFIDENCE, true) >= hi);
            assert!(lower(k, 20, CONFIDENCE, true) <= lo);
        }
    }

    #[test]
    fn the_cdf_is_a_cdf() {
        close(binomial_cdf(10, 10, 0.3), 1.0);
        close(binomial_cdf(0, 10, 0.5), 1.0 / 1024.0);
        close(binomial_cdf(5, 10, 0.5), 0.623046875);
    }

    #[test]
    #[should_panic(expected = "exceeds")]
    fn k_above_n_is_a_bug() {
        upper(3, 2, CONFIDENCE, false);
    }
}
