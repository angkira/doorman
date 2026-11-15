//! Turns score rows into report text/CSV. Output is a pure function of the
//! input rows and parameters (no timestamps, no machine info), so identical
//! input gives byte-identical output.

use crate::metrics::{self, EerPoint, HistBin, LivenessPoint, MetricsError, OperatingPoint};
use crate::scores::ScoreRow;
use std::collections::BTreeMap;
use std::fmt::Write;

/// FAR targets reported by default (spec: 1e-3 and 1e-4).
pub const DEFAULT_FARS: [f64; 2] = [1e-3, 1e-4];
/// Default histogram bin count.
pub const DEFAULT_BINS: usize = 50;
/// Default target BPCER for the recommended liveness threshold (1%).
pub const DEFAULT_TARGET_BPCER: f64 = 0.01;
/// Condition name for genuine rows without a condition column.
pub const NO_CONDITION: &str = "(none)";

fn f6(x: f64) -> String {
    format!("{x:.6}")
}

pub struct ConditionRow {
    pub name: String,
    pub n: usize,
    /// TAR at each global operating point, same order as `AccuracyReport::points`.
    pub tars: Vec<f64>,
}

pub struct AccuracyReport {
    pub n_genuine: usize,
    pub n_impostor: usize,
    pub points: Vec<OperatingPoint>,
    pub eer: EerPoint,
    pub conditions: Vec<ConditionRow>,
    pub hist: Vec<HistBin>,
}

pub fn analyze_accuracy(
    rows: &[ScoreRow],
    fars: &[f64],
    bins: usize,
) -> Result<AccuracyReport, MetricsError> {
    let genuine: Vec<f64> = rows
        .iter()
        .filter(|r| r.positive)
        .map(|r| r.score)
        .collect();
    let impostor: Vec<f64> = rows
        .iter()
        .filter(|r| !r.positive)
        .map(|r| r.score)
        .collect();
    let points = fars
        .iter()
        .map(|&f| metrics::tar_at_far(&genuine, &impostor, f))
        .collect::<Result<Vec<_>, _>>()?;
    let eer = metrics::eer(&genuine, &impostor)?;
    let hist = metrics::histogram(&genuine, &impostor, bins)?;

    let mut by_cond: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.positive) {
        by_cond
            .entry(
                r.condition
                    .clone()
                    .unwrap_or_else(|| NO_CONDITION.to_string()),
            )
            .or_default()
            .push(r.score);
    }
    let conditions = by_cond
        .into_iter()
        .map(|(name, scores)| {
            let tars = points
                .iter()
                .map(|p| metrics::tar_at_threshold(&scores, p.threshold))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ConditionRow {
                name,
                n: scores.len(),
                tars,
            })
        })
        .collect::<Result<Vec<_>, MetricsError>>()?;

    Ok(AccuracyReport {
        n_genuine: genuine.len(),
        n_impostor: impostor.len(),
        points,
        eer,
        conditions,
        hist,
    })
}

impl AccuracyReport {
    /// Worst (lowest) per-condition TAR at operating point `idx`.
    pub fn worst_condition(&self, idx: usize) -> Option<(&str, f64)> {
        self.conditions
            .iter()
            .map(|c| (c.name.as_str(), c.tars[idx]))
            .fold(None, |acc, cur| match acc {
                Some((_, t)) if t <= cur.1 => acc,
                _ => Some(cur),
            })
    }

    pub fn markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Accuracy report\n");
        let _ = writeln!(s, "- genuine scores: {}", self.n_genuine);
        let _ = writeln!(s, "- impostor scores: {}", self.n_impostor);
        let _ = writeln!(
            s,
            "- accept rule: `score > threshold` (higher score = more similar)\n"
        );
        let _ = writeln!(s, "## Operating points\n");
        let _ = writeln!(
            s,
            "| FAR target | threshold | actual FAR | TAR | resolvable |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|");
        for p in &self.points {
            let _ = writeln!(
                s,
                "| {:e} | {} | {} | {} | {} |",
                p.far_target,
                f6(p.threshold),
                f6(p.far),
                f6(p.tar),
                if p.resolvable {
                    "yes"
                } else {
                    "NO (need >= 1/FAR impostor scores)"
                }
            );
        }
        let _ = writeln!(
            s,
            "\nEER: {} at threshold {} (FAR {}, FRR {})\n",
            f6(self.eer.eer),
            f6(self.eer.threshold),
            f6(self.eer.far),
            f6(self.eer.frr)
        );
        let _ = writeln!(s, "## Per-condition TAR at the global thresholds\n");
        let mut header = String::from("| condition | n |");
        let mut sep = String::from("|---|---|");
        for p in &self.points {
            let _ = write!(header, " TAR@FAR {:e} |", p.far_target);
            sep.push_str("---|");
        }
        let _ = writeln!(s, "{header}\n{sep}");
        for c in &self.conditions {
            let _ = write!(s, "| {} | {} |", c.name, c.n);
            for t in &c.tars {
                let _ = write!(s, " {} |", f6(*t));
            }
            let _ = writeln!(s);
        }
        let _ = writeln!(s);
        for (i, p) in self.points.iter().enumerate() {
            if let Some((name, tar)) = self.worst_condition(i) {
                let _ = writeln!(
                    s,
                    "Worst condition at FAR {:e}: `{}` (TAR {})",
                    p.far_target,
                    name,
                    f6(tar)
                );
            }
        }
        s
    }

    /// One row per FAR target plus an `eer` row.
    pub fn summary_csv(&self) -> String {
        let mut s = String::from(
            "metric,far_target,threshold,far,tar,resolvable,worst_condition,worst_condition_tar\n",
        );
        for (i, p) in self.points.iter().enumerate() {
            let (wn, wt) = self
                .worst_condition(i)
                .map(|(n, t)| (n.to_string(), f6(t)))
                .unwrap_or_default();
            let _ = writeln!(
                s,
                "tar_at_far,{:e},{},{},{},{},{},{}",
                p.far_target,
                f6(p.threshold),
                f6(p.far),
                f6(p.tar),
                p.resolvable,
                wn,
                wt
            );
        }
        let _ = writeln!(
            s,
            "eer,,{},{},{},,,",
            f6(self.eer.threshold),
            f6(self.eer.far),
            f6(1.0 - self.eer.frr)
        );
        s
    }

    pub fn histogram_csv(&self) -> String {
        let mut s = String::from("bin_lo,bin_hi,genuine,impostor\n");
        for b in &self.hist {
            let _ = writeln!(s, "{},{},{},{}", f6(b.lo), f6(b.hi), b.genuine, b.impostor);
        }
        s
    }
}

pub struct LivenessReport {
    pub n_bona_fide: usize,
    pub n_attack: usize,
    pub at_thresholds: Vec<LivenessPoint>,
    pub roc: Vec<LivenessPoint>,
    pub target_bpcer: f64,
    pub recommended: LivenessPoint,
}

pub fn analyze_liveness(
    rows: &[ScoreRow],
    thresholds: &[f64],
    target_bpcer: f64,
) -> Result<LivenessReport, MetricsError> {
    let bona: Vec<f64> = rows
        .iter()
        .filter(|r| r.positive)
        .map(|r| r.score)
        .collect();
    let attack: Vec<f64> = rows
        .iter()
        .filter(|r| !r.positive)
        .map(|r| r.score)
        .collect();
    Ok(LivenessReport {
        n_bona_fide: bona.len(),
        n_attack: attack.len(),
        at_thresholds: thresholds
            .iter()
            .map(|&t| metrics::liveness_at(&bona, &attack, t))
            .collect::<Result<Vec<_>, _>>()?,
        roc: metrics::liveness_roc(&bona, &attack)?,
        target_bpcer,
        recommended: metrics::threshold_for_bpcer(&bona, &attack, target_bpcer)?,
    })
}

impl LivenessReport {
    pub fn markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# Liveness report\n");
        let _ = writeln!(s, "- bona fide scores: {}", self.n_bona_fide);
        let _ = writeln!(s, "- attack scores: {}", self.n_attack);
        let _ = writeln!(
            s,
            "- accept rule: `score >= threshold` (same as the daemon's relief gate)\n"
        );
        let _ = writeln!(s, "## Requested thresholds\n");
        let _ = writeln!(s, "| threshold | APCER | BPCER |\n|---|---|---|");
        for p in &self.at_thresholds {
            let _ = writeln!(
                s,
                "| {} | {} | {} |",
                f6(p.threshold),
                f6(p.apcer),
                f6(p.bpcer)
            );
        }
        let _ = writeln!(
            s,
            "\n## Recommended threshold\n\nFor BPCER <= {}: threshold {} gives APCER {} and BPCER {}.\n\
             Small sample sizes make this optimistic: with {} bona fide samples a {} BPCER target allows {} rejections.",
            f6(self.target_bpcer),
            f6(self.recommended.threshold),
            f6(self.recommended.apcer),
            f6(self.recommended.bpcer),
            self.n_bona_fide,
            f6(self.target_bpcer),
            (self.target_bpcer * self.n_bona_fide as f64 + 1e-9).floor() as usize
        );
        s
    }

    pub fn roc_csv(&self) -> String {
        let mut s = String::from("threshold,apcer,bpcer\n");
        for p in &self.roc {
            let _ = writeln!(s, "{},{},{}", f6(p.threshold), f6(p.apcer), f6(p.bpcer));
        }
        s
    }
}
