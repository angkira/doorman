//! Latency statistics and rendering. Model-free so it is unit-testable in CI;
//! the runner that fills it lives in `latency.rs` (needs the `ort` feature).

use crate::metrics;
use std::fmt::Write;

/// Percentiles reported for every stage.
pub const PERCENTILES: [f64; 3] = [50.0, 95.0, 99.0];

#[derive(Debug, Clone, PartialEq)]
pub struct StageStats {
    pub n: usize,
    pub min: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

/// Summarise millisecond samples. `None` for an empty sample set.
pub fn summarize(samples_ms: &[f64]) -> Option<StageStats> {
    if samples_ms.is_empty() {
        return None;
    }
    let s = metrics::sorted(samples_ms);
    let pct = |p: f64| metrics::percentile(&s, p).expect("non-empty, finite samples");
    Some(StageStats {
        n: s.len(),
        min: s[0],
        mean: s.iter().sum::<f64>() / s.len() as f64,
        p50: pct(PERCENTILES[0]),
        p95: pct(PERCENTILES[1]),
        p99: pct(PERCENTILES[2]),
        max: s[s.len() - 1],
    })
}

#[derive(Debug, Clone)]
pub struct SteadyRow {
    pub threads: i32,
    pub stage: &'static str,
    pub stats: Option<StageStats>,
}

#[derive(Debug, Clone)]
pub struct ColdStart {
    pub threads: i32,
    /// Creating the pipeline: loading every model into ORT sessions.
    pub session_creation_ms: f64,
    pub first_detect_ms: f64,
    pub first_embed_ms: Option<f64>,
    pub first_liveness_ms: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct LatencyReport {
    pub images: usize,
    pub iters: usize,
    pub warmup: usize,
    pub steady: Vec<SteadyRow>,
    pub cold: Vec<ColdStart>,
    pub notes: Vec<String>,
}

fn f3(x: f64) -> String {
    format!("{x:.3}")
}

fn opt(x: Option<f64>) -> String {
    x.map(f3).unwrap_or_default()
}

impl LatencyReport {
    pub fn markdown(&self) -> String {
        let mut s = String::from("# Latency report\n\n");
        let _ = writeln!(
            s,
            "- images: {}\n- measured iterations per thread setting: {}\n- warm-up iterations (discarded): {}\n- all times in milliseconds; percentiles use linear interpolation (R-7)\n",
            self.images, self.iters, self.warmup
        );
        for n in &self.notes {
            let _ = writeln!(s, "> {n}");
        }
        let _ = writeln!(s, "\n## Cold start (session creation + first inference)\n");
        let _ = writeln!(
            s,
            "| intra_op threads | session creation | first detect | first embed | first liveness |\n|---|---|---|---|---|"
        );
        for c in &self.cold {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} |",
                c.threads,
                f3(c.session_creation_ms),
                f3(c.first_detect_ms),
                opt(c.first_embed_ms),
                opt(c.first_liveness_ms)
            );
        }
        let _ = writeln!(s, "\n## Steady state\n");
        let _ = writeln!(
            s,
            "| intra_op threads | stage | n | min | mean | p50 | p95 | p99 | max |\n|---|---|---|---|---|---|---|---|---|"
        );
        for r in &self.steady {
            match &r.stats {
                Some(t) => {
                    let _ = writeln!(
                        s,
                        "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                        r.threads,
                        r.stage,
                        t.n,
                        f3(t.min),
                        f3(t.mean),
                        f3(t.p50),
                        f3(t.p95),
                        f3(t.p99),
                        f3(t.max)
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        "| {} | {} | 0 | - | - | - | - | - | - |",
                        r.threads, r.stage
                    );
                }
            }
        }
        s
    }

    pub fn steady_csv(&self) -> String {
        let mut s = String::from("threads,stage,n,min_ms,mean_ms,p50_ms,p95_ms,p99_ms,max_ms\n");
        for r in &self.steady {
            match &r.stats {
                Some(t) => {
                    let _ = writeln!(
                        s,
                        "{},{},{},{},{},{},{},{},{}",
                        r.threads,
                        r.stage,
                        t.n,
                        f3(t.min),
                        f3(t.mean),
                        f3(t.p50),
                        f3(t.p95),
                        f3(t.p99),
                        f3(t.max)
                    );
                }
                None => {
                    let _ = writeln!(s, "{},{},0,,,,,,", r.threads, r.stage);
                }
            }
        }
        s
    }

    pub fn cold_csv(&self) -> String {
        let mut s = String::from(
            "threads,session_creation_ms,first_detect_ms,first_embed_ms,first_liveness_ms\n",
        );
        for c in &self.cold {
            let _ = writeln!(
                s,
                "{},{},{},{},{}",
                c.threads,
                f3(c.session_creation_ms),
                f3(c.first_detect_ms),
                opt(c.first_embed_ms),
                opt(c.first_liveness_ms)
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_hand_computed() {
        let st = summarize(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(st.n, 4);
        assert_eq!(st.min, 1.0);
        assert_eq!(st.max, 4.0);
        assert!((st.mean - 2.5).abs() < 1e-12);
        assert!((st.p50 - 2.5).abs() < 1e-12);
        assert!((st.p95 - 3.85).abs() < 1e-12);
        assert!((st.p99 - 3.97).abs() < 1e-12);
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn renders_rows_and_empty_stage() {
        let r = LatencyReport {
            images: 1,
            iters: 4,
            warmup: 1,
            steady: vec![
                SteadyRow {
                    threads: 2,
                    stage: "detect",
                    stats: summarize(&[1.0, 2.0]),
                },
                SteadyRow {
                    threads: 2,
                    stage: "embed",
                    stats: None,
                },
            ],
            cold: vec![ColdStart {
                threads: 2,
                session_creation_ms: 10.0,
                first_detect_ms: 5.0,
                first_embed_ms: None,
                first_liveness_ms: None,
            }],
            notes: vec![],
        };
        let csv = r.steady_csv();
        assert!(
            csv.contains("2,detect,2,1.000,1.500,1.500,1.950,1.990,2.000"),
            "{csv}"
        );
        assert!(csv.contains("2,embed,0,,,,,,"));
        assert!(r.cold_csv().contains("2,10.000,5.000,,"));
        assert!(r.markdown().contains("## Cold start"));
    }
}
