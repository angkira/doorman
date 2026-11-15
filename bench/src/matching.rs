//! Comparison-score report in the style of ISO/IEC 19795-1 (described
//! generically): FMR (impostor comparisons accepted), FNMR (genuine comparisons
//! rejected), failure-to-acquire (FTA), each with a one-sided Clopper-Pearson
//! upper bound ([`crate::stats`]) and the comparison count needed for a target.
//!
//! Impostor comparisons against few subjects are NOT independent: every
//! comparison involving the same person shares that person's face. The report
//! therefore also gives a conservative subject-level bound: with `S` distinct
//! impostor subjects, of which `K` caused at least one false match, the
//! one-sided upper bound on the fraction of subjects that can falsely match is
//! `upper_bound(K, S)`. That is a different, coarser quantity than FMR; it is
//! shown so a huge comparison count from few people cannot look like a
//! population-scale result.
//!
//! Input CSV (header names case-insensitive): `label{genuine,impostor}`,
//! `score`, optional `subject`, optional `response` (`ok`, or `fta` /
//! `no_face` / `timeout` for a failure to acquire; the score is ignored then).
//! Accept rule (same as `accuracy`): `score > threshold`.

use crate::stats::{self, format_confidence, format_percent, Rate, StatsError};
use crate::table::{Table, TableError};
use std::collections::BTreeMap;
use std::fmt::Write as _;

pub const LABEL_GENUINE: &str = "genuine";
pub const LABEL_IMPOSTOR: &str = "impostor";
/// Default FMR target.
pub const DEFAULT_TARGET_FMR: f64 = 1e-4;
/// FMR levels whose required comparison count is always shown.
pub const REFERENCE_FMR_TARGETS: [f64; 2] = [1e-4, 2e-5];
/// Fewer distinct impostor subjects than this triggers the correlation
/// warning. A heuristic for "few people", not a standard's number.
pub const FEW_SUBJECTS: usize = 30;
const RESPONSE_OK: &str = "ok";
const NON_RESPONSES: [&str; 3] = ["fta", "no_face", "timeout"];

#[derive(Debug, thiserror::Error)]
pub enum MatchError {
    #[error(transparent)]
    Table(#[from] TableError),
    #[error(transparent)]
    Stats(#[from] StatsError),
    #[error(
        "line {line}: unknown label `{label}` (expected `{LABEL_GENUINE}` or `{LABEL_IMPOSTOR}`)"
    )]
    BadLabel { line: usize, label: String },
    #[error("line {line}: score `{value}` is not a finite number")]
    BadScore { line: usize, value: String },
    #[error(
        "line {line}: unknown response `{value}` (expected `ok`, `fta`, `no_face` or `timeout`)"
    )]
    BadResponse { line: usize, value: String },
    #[error("{name} must be in the open interval (0, 1), got {value}")]
    BadTarget { name: &'static str, value: f64 },
    #[error("threshold must be finite, got {0}")]
    BadThreshold(f64),
    #[error("the scores CSV has no scored impostor comparisons: FMR is undefined")]
    NoImpostors,
    #[error("the scores CSV has no scored genuine comparisons: FNMR is undefined")]
    NoGenuine,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchRow {
    pub genuine: bool,
    /// `None` for a failure to acquire.
    pub score: Option<f64>,
    pub subject: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchInput {
    pub rows: Vec<MatchRow>,
    pub has_subject_column: bool,
}

pub fn parse_match(text: &str) -> Result<MatchInput, MatchError> {
    let t = Table::parse(text)?;
    let label_c = t.require("label")?;
    let score_c = t.require("score")?;
    let subject_c = t.column("subject");
    let response_c = t.column("response");
    let mut rows = Vec::with_capacity(t.rows.len());
    for r in &t.rows {
        let genuine = match r.get(label_c) {
            LABEL_GENUINE => true,
            LABEL_IMPOSTOR => false,
            other => {
                return Err(MatchError::BadLabel {
                    line: r.line,
                    label: other.to_string(),
                })
            }
        };
        let fta = match r.opt(response_c) {
            None | Some(RESPONSE_OK) => false,
            Some(v) if NON_RESPONSES.contains(&v) => true,
            Some(v) => {
                return Err(MatchError::BadResponse {
                    line: r.line,
                    value: v.to_string(),
                })
            }
        };
        let score = if fta {
            None
        } else {
            let v = r.get(score_c);
            Some(
                v.parse::<f64>()
                    .ok()
                    .filter(|s| s.is_finite())
                    .ok_or_else(|| MatchError::BadScore {
                        line: r.line,
                        value: v.to_string(),
                    })?,
            )
        };
        rows.push(MatchRow {
            genuine,
            score,
            subject: r.opt(subject_c).map(str::to_string),
        });
    }
    Ok(MatchInput {
        rows,
        has_subject_column: subject_c.is_some(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchParams {
    /// Accept iff `score > threshold`.
    pub threshold: f64,
    pub target_fmr: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleNeed {
    pub target: f64,
    /// Comparisons needed at zero false matches.
    pub required: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchReport {
    pub params: MatchParams,
    pub genuine_attempts: u64,
    pub genuine_fta: u64,
    pub impostor_attempts: u64,
    pub impostor_fta: u64,
    pub fnmr: Rate,
    pub fmr: Rate,
    /// FTA over all attempts (genuine and impostor).
    pub fta: Option<Rate>,
    pub n_impostor_subjects: Option<usize>,
    /// Impostor subjects with at least one false match.
    pub n_false_match_subjects: Option<usize>,
    /// Subject-level bound: `(K of S)`.
    pub subject_bound: Option<Rate>,
    /// Comparisons still needed for `target_fmr` assuming no further false match.
    pub required_for_target: Option<u64>,
    pub needs: Vec<SampleNeed>,
    pub warnings: Vec<String>,
}

fn check_open_unit(name: &'static str, v: f64) -> Result<(), MatchError> {
    if v > 0.0 && v < 1.0 {
        Ok(())
    } else {
        Err(MatchError::BadTarget { name, value: v })
    }
}

pub fn analyze_match(input: &MatchInput, params: MatchParams) -> Result<MatchReport, MatchError> {
    if !params.threshold.is_finite() {
        return Err(MatchError::BadThreshold(params.threshold));
    }
    check_open_unit("target FMR", params.target_fmr)?;
    check_open_unit("confidence", params.confidence)?;

    let (mut g_att, mut g_fta, mut g_rej) = (0u64, 0u64, 0u64);
    let (mut i_att, mut i_fta, mut i_acc) = (0u64, 0u64, 0u64);
    // impostor subject -> false matches
    let mut subjects: BTreeMap<&str, u64> = BTreeMap::new();
    for r in &input.rows {
        let accepted = r.score.map(|s| s > params.threshold);
        if r.genuine {
            g_att += 1;
            match accepted {
                None => g_fta += 1,
                Some(a) => g_rej += u64::from(!a),
            }
        } else {
            i_att += 1;
            match accepted {
                None => i_fta += 1,
                Some(a) => i_acc += u64::from(a),
            }
            if let Some(s) = r.subject.as_deref() {
                let e = subjects.entry(s).or_default();
                *e += u64::from(accepted == Some(true));
            }
        }
    }
    let fnmr = Rate::new(g_rej, g_att - g_fta, params.confidence)?.ok_or(MatchError::NoGenuine)?;
    let fmr = Rate::new(i_acc, i_att - i_fta, params.confidence)?.ok_or(MatchError::NoImpostors)?;
    let fta = Rate::new(g_fta + i_fta, g_att + i_att, params.confidence)?;

    let n_subjects = input.has_subject_column.then_some(subjects.len());
    let n_fm_subjects = input
        .has_subject_column
        .then(|| subjects.values().filter(|&&c| c > 0).count());
    let subject_bound = match (n_subjects, n_fm_subjects) {
        (Some(s), Some(k)) => Rate::new(k as u64, s as u64, params.confidence)?,
        _ => None,
    };

    let required_for_target =
        stats::required_trials_with_errors(i_acc, params.target_fmr, params.confidence).ok();
    let mut targets: Vec<f64> = REFERENCE_FMR_TARGETS.to_vec();
    if !targets.contains(&params.target_fmr) {
        targets.push(params.target_fmr);
    }
    let needs = targets
        .into_iter()
        .map(|target| {
            Ok(SampleNeed {
                target,
                required: stats::required_trials(target, params.confidence)?,
            })
        })
        .collect::<Result<Vec<_>, StatsError>>()?;

    let mut warnings = Vec::new();
    match n_subjects {
        None => warnings.push(
            "No subject column: it is unknown how many people the impostor comparisons come from. Comparisons against the same person are correlated, so the FMR bound is optimistic; add a subject column."
                .to_string(),
        ),
        Some(s) if s < FEW_SUBJECTS => warnings.push(format!(
            "Only {s} distinct impostor subjects (fewer than {FEW_SUBJECTS}): comparisons from few subjects are correlated, so the per-comparison FMR bound is optimistic. Read the subject-level bound too."
        )),
        Some(_) => {}
    }
    if let Some(req) = required_for_target {
        if fmr.trials < req {
            warnings.push(format!(
                "{} scored impostor comparisons are fewer than the {req} needed to claim FMR <= {} at {} confidence (assuming no further false match).",
                fmr.trials,
                format_percent(params.target_fmr),
                format_confidence(params.confidence),
            ));
        }
    } else {
        warnings.push("The FMR target is not reachable by adding comparisons.".to_string());
    }

    Ok(MatchReport {
        params,
        genuine_attempts: g_att,
        genuine_fta: g_fta,
        impostor_attempts: i_att,
        impostor_fta: i_fta,
        fnmr,
        fmr,
        fta,
        n_impostor_subjects: n_subjects,
        n_false_match_subjects: n_fm_subjects,
        subject_bound,
        required_for_target,
        needs,
        warnings,
    })
}

impl MatchReport {
    pub fn claim(&self) -> String {
        let p = &self.params;
        let mut s = format!(
            "FMR <= {} ({} one-sided Clopper-Pearson; {} false matches in {} impostor comparisons",
            format_percent(self.fmr.upper),
            format_confidence(p.confidence),
            self.fmr.errors,
            self.fmr.trials,
        );
        match self.n_impostor_subjects {
            Some(n) => {
                let _ = write!(s, " from {n} impostor subjects");
            }
            None => s.push_str(", number of subjects unknown"),
        }
        let _ = write!(s, ") at threshold {}", p.threshold);
        if self.fmr.upper <= p.target_fmr {
            let _ = write!(
                s,
                "; the target FMR <= {} is met",
                format_percent(p.target_fmr)
            );
        } else {
            let _ = write!(
                s,
                "; the target FMR <= {} is NOT demonstrated",
                format_percent(p.target_fmr)
            );
        }
        if let Some(b) = &self.subject_bound {
            let _ = write!(
                s,
                ". Conservative subject-level bound: at most {} of impostor subjects can falsely match ({} of {} did)",
                format_percent(b.upper),
                b.errors,
                b.trials
            );
        }
        s.push('.');
        s
    }

    pub fn markdown(&self) -> String {
        let p = &self.params;
        let mut s = String::from("# Match self-assessment report\n\n");
        let _ = writeln!(
            s,
            "Accept rule: score > {}. Bounds: one-sided {} Clopper-Pearson upper bound. Target FMR: {}.\n",
            p.threshold,
            format_confidence(p.confidence),
            format_percent(p.target_fmr)
        );
        let _ = writeln!(s, "**Claim:** {}\n", self.claim());
        s.push_str("## Error rates\n\n| measure | attempts | failure to acquire | scored (n) | errors (k) | rate | upper bound |\n|---|---|---|---|---|---|---|\n");
        let row = |name: &str, att: u64, fta: u64, r: &Rate| {
            format!(
                "| {name} | {att} | {fta} | {} | {} | {} | {} |\n",
                r.trials,
                r.errors,
                format_percent(r.point),
                format_percent(r.upper)
            )
        };
        s.push_str(&row(
            "FMR (impostor accepted)",
            self.impostor_attempts,
            self.impostor_fta,
            &self.fmr,
        ));
        s.push_str(&row(
            "FNMR (genuine rejected)",
            self.genuine_attempts,
            self.genuine_fta,
            &self.fnmr,
        ));
        if let Some(f) = &self.fta {
            let _ = writeln!(
                s,
                "\nFTA (all attempts): {} of {} ({}, upper bound {}).\n",
                f.errors,
                f.trials,
                format_percent(f.point),
                format_percent(f.upper)
            );
        }
        s.push_str("## Impostor population\n\n");
        match (self.n_impostor_subjects, self.n_false_match_subjects) {
            (Some(n), Some(k)) => {
                let _ = writeln!(
                    s,
                    "* n_subjects (distinct impostor subjects): {n}\n* subjects with at least one false match: {k}"
                );
                if let Some(b) = &self.subject_bound {
                    let _ = writeln!(
                        s,
                        "* conservative bound with n = number of subjects: {} of {} subjects, upper bound {}",
                        b.errors,
                        b.trials,
                        format_percent(b.upper)
                    );
                }
            }
            _ => s.push_str("* n_subjects: unknown (no subject column)\n"),
        }
        s.push_str("\n## Comparisons needed at zero false matches\n\n| target FMR | comparisons needed | additional |\n|---|---|---|\n");
        for n in &self.needs {
            let _ = writeln!(
                s,
                "| {} | {} | {} |",
                format_percent(n.target),
                n.required,
                n.required.saturating_sub(self.fmr.trials)
            );
        }
        s.push('\n');
        if !self.warnings.is_empty() {
            s.push_str("## Warnings\n\n");
            for w in &self.warnings {
                let _ = writeln!(s, "* {w}");
            }
            s.push('\n');
        }
        s.push_str("This is a self-assessment on the owner's own data, not a certification.\n");
        s
    }

    /// `measure,attempts,fta,n,errors,rate,upper_bound`
    pub fn csv(&self) -> String {
        let mut s = String::from("measure,attempts,fta,n,errors,rate,upper_bound\n");
        let mut row = |name: &str, att: u64, fta: u64, r: &Rate| {
            let _ = writeln!(
                s,
                "{name},{att},{fta},{},{},{:.6e},{:.6e}",
                r.trials, r.errors, r.point, r.upper
            );
        };
        row("fmr", self.impostor_attempts, self.impostor_fta, &self.fmr);
        row("fnmr", self.genuine_attempts, self.genuine_fta, &self.fnmr);
        if let Some(b) = &self.subject_bound {
            row("fmr_subject_level", 0, 0, b);
        }
        let _ = writeln!(
            s,
            "n_subjects,{},,,,,",
            self.n_impostor_subjects
                .map_or(String::new(), |n| n.to_string())
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: MatchParams = MatchParams {
        threshold: 0.4,
        target_fmr: 1e-4,
        confidence: 0.95,
    };

    /// 50 genuine (1 below the threshold, 1 FTA), 1000 impostor comparisons
    /// from 10 subjects, 1 false match (subject imp3) and 2 FTA.
    fn sample() -> String {
        let mut s = String::from("label,score,subject,response\n");
        for i in 0..48 {
            let _ = writeln!(s, "genuine,{},me,ok", if i == 0 { 0.3 } else { 0.8 });
        }
        s.push_str("genuine,0.8,me,ok\ngenuine,,me,fta\n");
        for i in 0..1000 {
            let score = if i == 3 { 0.5 } else { 0.1 };
            let _ = writeln!(s, "impostor,{score},imp{},ok", i % 10);
        }
        s.push_str("impostor,,imp0,no_face\nimpostor,,imp1,timeout\n");
        s
    }

    #[test]
    fn counts_and_bounds() {
        let r = analyze_match(&parse_match(&sample()).unwrap(), P).unwrap();
        assert_eq!((r.genuine_attempts, r.genuine_fta), (50, 1));
        assert_eq!((r.impostor_attempts, r.impostor_fta), (1002, 2));
        // FNMR: 1 of 49 scored. FMR: 1 of 1000 scored.
        assert_eq!((r.fnmr.errors, r.fnmr.trials), (1, 49));
        assert_eq!((r.fmr.errors, r.fmr.trials), (1, 1000));
        // k = 1, n = 1000 -> 4.743/1000-ish; exact value from the shared test table
        // scaled: upper_bound(1, 1000).
        assert!((r.fmr.upper - stats::upper_bound(1, 1000, 0.95).unwrap()).abs() < 1e-15);
        assert!(r.fmr.upper > 0.0047 && r.fmr.upper < 0.0048);
        // FTA: 3 of 1052 attempts.
        let f = r.fta.unwrap();
        assert_eq!((f.errors, f.trials), (3, 1052));
    }

    #[test]
    fn subject_level_bound_and_needs() {
        let r = analyze_match(&parse_match(&sample()).unwrap(), P).unwrap();
        assert_eq!(r.n_impostor_subjects, Some(10));
        assert_eq!(r.n_false_match_subjects, Some(1));
        let b = r.subject_bound.unwrap();
        assert_eq!((b.errors, b.trials), (1, 10));
        // 1 of 10: upper bound 0.3942 (one-sided 95%).
        assert!((b.upper - 0.394_17).abs() < 1e-4, "{}", b.upper);
        let need = |t: f64| r.needs.iter().find(|n| n.target == t).unwrap().required;
        assert_eq!(need(1e-4), 29_956);
        assert_eq!(need(2e-5), 149_786);
        assert!(r.warnings.iter().any(|w| w.contains("Only 10 distinct")));
        assert!(r.warnings.iter().any(|w| w.contains("fewer than the")));
        let c = r.claim();
        assert!(c.contains("NOT demonstrated"), "{c}");
        assert!(c.contains("from 10 impostor subjects"), "{c}");
        assert!(c.contains("subject-level"), "{c}");
    }

    #[test]
    fn zero_false_matches_with_enough_comparisons_meets_target() {
        let mut s = String::from("label,score,subject\ngenuine,0.9,me\n");
        for i in 0..30_000 {
            let _ = writeln!(s, "impostor,0.1,imp{}", i % 100);
        }
        let r = analyze_match(&parse_match(&s).unwrap(), P).unwrap();
        assert!(r.fmr.upper <= 1e-4);
        assert!(r.claim().contains("is met"));
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(r.required_for_target, Some(29_956));
    }

    #[test]
    fn no_subject_column_warns() {
        let input = parse_match("label,score\ngenuine,0.9\nimpostor,0.1\n").unwrap();
        let r = analyze_match(&input, P).unwrap();
        assert_eq!(r.n_impostor_subjects, None);
        assert!(r.subject_bound.is_none());
        assert!(r.warnings.iter().any(|w| w.contains("No subject column")));
        assert!(r.markdown().contains("unknown (no subject column)"));
    }

    #[test]
    fn strict_threshold_and_explicit_errors() {
        // score == threshold is rejected (accept iff score > threshold).
        let input = parse_match("label,score\ngenuine,0.4\nimpostor,0.4\n").unwrap();
        let r = analyze_match(&input, P).unwrap();
        assert_eq!((r.fnmr.errors, r.fmr.errors), (1, 0));
        let e = parse_match("label,score\nmaybe,0.4\n").unwrap_err();
        assert!(e.to_string().starts_with("line 2: unknown label"), "{e}");
        let e = parse_match("label,score,response\ngenuine,0.4,huh\n").unwrap_err();
        assert!(e.to_string().contains("unknown response"), "{e}");
        let only_g = parse_match("label,score\ngenuine,0.4\n").unwrap();
        assert!(matches!(
            analyze_match(&only_g, P),
            Err(MatchError::NoImpostors)
        ));
        let only_i = parse_match("label,score\nimpostor,0.4\n").unwrap();
        assert!(matches!(
            analyze_match(&only_i, P),
            Err(MatchError::NoGenuine)
        ));
    }

    #[test]
    fn csv_rows() {
        let r = analyze_match(&parse_match(&sample()).unwrap(), P).unwrap();
        let csv = r.csv();
        assert!(csv.contains("fmr,1002,2,1000,1,1.000000e-3,"), "{csv}");
        assert!(csv.contains("n_subjects,10"), "{csv}");
    }
}
