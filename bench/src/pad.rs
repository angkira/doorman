//! Presentation-attack-detection report in the style of ISO/IEC 30107-3
//! (described generically; clause text is not reproduced here):
//!
//! * APCER is computed PER attack species (one attack instrument type each)
//!   and the worst species is reported; pooling species would hide a weak one.
//! * BPCER is the fraction of bona fide presentations rejected.
//! * Presentations the system gave no response to (no face found, timeout) are
//!   reported separately as non-response rates and are NOT part of the APCER or
//!   BPCER denominators; a missing response never unlocks, so for attacks it is
//!   a safe outcome, but it is still shown so it cannot hide.
//! * Every rate carries a one-sided Clopper-Pearson upper bound
//!   ([`crate::stats`]) and the number of attacks still needed to claim a target.
//!
//! Input CSV (header names are case-insensitive, column order is free):
//! `label{bona_fide,attack}`, `score`, optional `species`, `response`
//! (`ok`, `no_face`, `timeout`), `session`, `subject`. Accept rule (liveness
//! relief, as in the daemon): `score >= threshold`. Output is a pure function of
//! the input.

use crate::stats::{self, format_confidence, format_percent, Rate, StatsError};
use crate::table::{Table, TableError};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

pub const LABEL_BONA_FIDE: &str = "bona_fide";
pub const LABEL_ATTACK: &str = "attack";
/// Default name of the attack-species column.
pub const DEFAULT_SPECIES_COLUMN: &str = "species";
/// Default APCER target (1%), the figure the self-assessment protocol aims at.
pub const DEFAULT_TARGET_APCER: f64 = 0.01;
/// Response column values.
pub const RESPONSE_OK: &str = "ok";
pub const RESPONSE_NO_FACE: &str = "no_face";
pub const RESPONSE_TIMEOUT: &str = "timeout";
/// Species used when the CSV has no species column.
pub const UNSPECIFIED_SPECIES: &str = "(all attacks, no species column)";
/// Species value that marks "not an attack" and is invalid on attack rows.
const NO_SPECIES: &str = "none";
/// The species of the self-assessment protocol (docs/bench/SELF-ASSESSMENT.md).
/// A species from this list that has no scored attack is reported as untested.
pub const PROTOCOL_SPECIES: [&str; 7] = [
    "print_matte",
    "print_glossy",
    "print_curved",
    "phone_photo",
    "phone_replay",
    "screen_replay",
    "paper_mask",
];
const SESSION_COLUMN: &str = "session";
const SUBJECT_COLUMN: &str = "subject";
const RESPONSE_COLUMN: &str = "response";

#[derive(Debug, thiserror::Error)]
pub enum PadError {
    #[error(transparent)]
    Table(#[from] TableError),
    #[error(transparent)]
    Stats(#[from] StatsError),
    #[error(
        "line {line}: unknown label `{label}` (expected `{LABEL_BONA_FIDE}` or `{LABEL_ATTACK}`)"
    )]
    BadLabel { line: usize, label: String },
    #[error("line {line}: score `{value}` is not a finite number")]
    BadScore { line: usize, value: String },
    #[error("line {line}: unknown response `{value}` (expected `{RESPONSE_OK}`, `{RESPONSE_NO_FACE}` or `{RESPONSE_TIMEOUT}`)")]
    BadResponse { line: usize, value: String },
    #[error("line {line}: attack row has no species (empty or `none`); every attack needs its instrument species")]
    AttackWithoutSpecies { line: usize },
    #[error("{name} must be in the open interval (0, 1), got {value}")]
    BadTarget { name: &'static str, value: f64 },
    #[error("threshold must be finite, got {0}")]
    BadThreshold(f64),
    #[error("the scores CSV has no attack rows with a response: APCER is undefined")]
    NoAttacks,
    #[error("the scores CSV has no bona fide rows: BPCER is undefined")]
    NoBonaFide,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PadRow {
    pub attack: bool,
    pub species: String,
    /// `None` for a non-response.
    pub score: Option<f64>,
    pub non_response: Option<&'static str>,
    pub session: Option<String>,
    pub subject: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PadInput {
    pub rows: Vec<PadRow>,
    pub has_species_column: bool,
    pub has_session_column: bool,
    pub has_subject_column: bool,
}

pub fn parse_pad(text: &str, species_column: &str) -> Result<PadInput, PadError> {
    let t = Table::parse(text)?;
    let label_c = t.require("label")?;
    let score_c = t.require("score")?;
    let species_c = t.column(species_column);
    let response_c = t.column(RESPONSE_COLUMN);
    let session_c = t.column(SESSION_COLUMN);
    let subject_c = t.column(SUBJECT_COLUMN);
    let mut rows = Vec::with_capacity(t.rows.len());
    for r in &t.rows {
        let attack = match r.get(label_c) {
            LABEL_ATTACK => true,
            LABEL_BONA_FIDE => false,
            other => {
                return Err(PadError::BadLabel {
                    line: r.line,
                    label: other.to_string(),
                })
            }
        };
        let non_response = match r.opt(response_c) {
            None | Some(RESPONSE_OK) => None,
            Some(RESPONSE_NO_FACE) => Some(RESPONSE_NO_FACE),
            Some(RESPONSE_TIMEOUT) => Some(RESPONSE_TIMEOUT),
            Some(other) => {
                return Err(PadError::BadResponse {
                    line: r.line,
                    value: other.to_string(),
                })
            }
        };
        let score = if non_response.is_some() {
            None
        } else {
            let v = r.get(score_c);
            Some(
                v.parse::<f64>()
                    .ok()
                    .filter(|s| s.is_finite())
                    .ok_or_else(|| PadError::BadScore {
                        line: r.line,
                        value: v.to_string(),
                    })?,
            )
        };
        let species = match (attack, species_c) {
            (true, Some(c)) => match r.get(c) {
                "" | NO_SPECIES => return Err(PadError::AttackWithoutSpecies { line: r.line }),
                s => s.to_string(),
            },
            (true, None) => UNSPECIFIED_SPECIES.to_string(),
            (false, _) => NO_SPECIES.to_string(),
        };
        rows.push(PadRow {
            attack,
            species,
            score,
            non_response,
            session: r.opt(session_c).map(str::to_string),
            subject: r.opt(subject_c).map(str::to_string),
        });
    }
    Ok(PadInput {
        rows,
        has_species_column: species_c.is_some(),
        has_session_column: session_c.is_some(),
        has_subject_column: subject_c.is_some(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PadParams {
    /// Accept as live iff `score >= threshold`.
    pub threshold: f64,
    pub target_apcer: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeciesRow {
    pub name: String,
    pub presented: u64,
    pub non_responses: u64,
    /// Scored presentations (`presented - non_responses`), accepted = errors.
    pub rate: Option<Rate>,
    /// Attacks needed for the target assuming no further accepted attack.
    pub required: Option<u64>,
    pub additional: Option<u64>,
}

impl SpeciesRow {
    fn scored(&self) -> u64 {
        self.presented - self.non_responses
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PadReport {
    pub params: PadParams,
    pub species: Vec<SpeciesRow>,
    /// Index into `species` of the largest upper bound among scored species.
    pub worst: Option<usize>,
    pub bona_fide_presented: u64,
    pub bona_fide_non_responses: u64,
    pub bpcer: Option<Rate>,
    pub attack_presented: u64,
    pub attack_non_responses: u64,
    pub untested: Vec<String>,
    pub n_sessions: Option<usize>,
    pub n_subjects: Option<usize>,
    pub warnings: Vec<String>,
}

fn check_target(name: &'static str, v: f64) -> Result<(), PadError> {
    if v > 0.0 && v < 1.0 {
        Ok(())
    } else {
        Err(PadError::BadTarget { name, value: v })
    }
}

pub fn analyze_pad(input: &PadInput, params: PadParams) -> Result<PadReport, PadError> {
    if !params.threshold.is_finite() {
        return Err(PadError::BadThreshold(params.threshold));
    }
    check_target("target APCER", params.target_apcer)?;
    check_target("confidence", params.confidence)?;
    let accepted = |s: f64| s >= params.threshold;

    // species -> (presented, non_responses, accepted)
    let mut by_species: BTreeMap<&str, (u64, u64, u64)> = BTreeMap::new();
    let (mut bf_presented, mut bf_nr, mut bf_rejected) = (0u64, 0u64, 0u64);
    for r in &input.rows {
        if r.attack {
            let e = by_species.entry(r.species.as_str()).or_default();
            e.0 += 1;
            match r.score {
                None => e.1 += 1,
                Some(s) => e.2 += u64::from(accepted(s)),
            }
        } else {
            bf_presented += 1;
            match r.score {
                None => bf_nr += 1,
                Some(s) => bf_rejected += u64::from(!accepted(s)),
            }
        }
    }
    if by_species.is_empty() {
        return Err(PadError::NoAttacks);
    }
    if bf_presented == 0 {
        return Err(PadError::NoBonaFide);
    }

    let mut species = Vec::new();
    for (name, (presented, nr, acc)) in by_species {
        let scored = presented - nr;
        let rate = Rate::new(acc, scored, params.confidence)?;
        let required =
            stats::required_trials_with_errors(acc, params.target_apcer, params.confidence).ok();
        species.push(SpeciesRow {
            name: name.to_string(),
            presented,
            non_responses: nr,
            rate,
            required,
            additional: required.map(|r| r.saturating_sub(scored)),
        });
    }
    let worst = species
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.rate.map(|r| (i, r.upper)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i);

    let bpcer = Rate::new(bf_rejected, bf_presented - bf_nr, params.confidence)?;
    let attack_presented = species.iter().map(|s| s.presented).sum();
    let attack_non_responses = species.iter().map(|s| s.non_responses).sum();

    let tested: BTreeSet<&str> = species
        .iter()
        .filter(|s| s.scored() > 0)
        .map(|s| s.name.as_str())
        .collect();
    let untested: Vec<String> = if input.has_species_column {
        PROTOCOL_SPECIES
            .iter()
            .filter(|p| !tested.contains(*p))
            .map(|p| p.to_string())
            .collect()
    } else {
        Vec::new()
    };

    let distinct = |has: bool, f: fn(&PadRow) -> &Option<String>| -> Option<usize> {
        has.then(|| {
            input
                .rows
                .iter()
                .filter_map(|r| f(r).as_deref())
                .collect::<BTreeSet<_>>()
                .len()
        })
    };
    let n_sessions = distinct(input.has_session_column, |r| &r.session);
    let n_subjects = distinct(input.has_subject_column, |r| &r.subject);

    let mut warnings = Vec::new();
    if !input.has_species_column {
        warnings.push(
            "No species column: all attacks are pooled into one row. A pooled APCER can hide a weak species; label every attack with its instrument species."
                .to_string(),
        );
    }
    for s in &species {
        if let (Some(req), true) = (s.required, s.scored() < s.required.unwrap_or(0)) {
            warnings.push(format!(
                "Species `{}`: {} scored attacks, fewer than the {} needed to claim APCER <= {} at {} confidence (assuming no further accepted attack).",
                s.name,
                s.scored(),
                req,
                format_percent(params.target_apcer),
                format_confidence(params.confidence),
            ));
        }
        if s.required.is_none() {
            warnings.push(format!(
                "Species `{}`: the target is not reachable by adding attacks.",
                s.name
            ));
        }
    }
    match n_subjects {
        Some(n) if n > 1 => {}
        _ => warnings.push(
            "One subject (or no subject column): results describe a single face and presenter, not a population."
                .to_string(),
        ),
    }
    match n_sessions {
        Some(n) if n > 1 => {}
        _ => warnings.push(
            "One session (or no session column): presentations within a session are correlated, so the independence assumption behind the bounds is optimistic."
                .to_string(),
        ),
    }

    Ok(PadReport {
        params,
        species,
        worst,
        bona_fide_presented: bf_presented,
        bona_fide_non_responses: bf_nr,
        bpcer,
        attack_presented,
        attack_non_responses,
        untested,
        n_sessions,
        n_subjects,
        warnings,
    })
}

fn opt_pct(r: Option<&Rate>, f: fn(&Rate) -> f64) -> String {
    r.map_or("n/a".to_string(), |r| format_percent(f(r)))
}

impl PadReport {
    /// The honest one-paragraph claim.
    pub fn claim(&self) -> String {
        let conf = format_confidence(self.params.confidence);
        let Some(w) = self.worst.map(|i| &self.species[i]) else {
            return "No claim: no attack presentation produced a scored response.".to_string();
        };
        let r = w.rate.as_ref().expect("worst species has a rate");
        let mut s = format!(
            "APCER <= {} ({} one-sided Clopper-Pearson) for the worst tested species `{}` ({} accepted of {} scored attacks) at threshold {}",
            format_percent(r.upper),
            conf,
            w.name,
            r.errors,
            r.trials,
            self.params.threshold,
        );
        if r.upper <= self.params.target_apcer {
            let _ = write!(
                s,
                "; the target APCER <= {} is met for every tested species",
                format_percent(self.params.target_apcer)
            );
        } else {
            let _ = write!(
                s,
                "; the target APCER <= {} is NOT demonstrated",
                format_percent(self.params.target_apcer)
            );
        }
        if self.species.len() > 1 {
            s.push_str(" (each species bounded separately, no multiplicity correction)");
        }
        if self.untested.is_empty() {
            s.push('.');
        } else {
            let _ = write!(s, "; untested species: {}.", self.untested.join(", "));
        }
        s
    }

    pub fn markdown(&self) -> String {
        let p = &self.params;
        let mut s = String::from("# PAD self-assessment report\n\n");
        let _ = writeln!(
            s,
            "Accept rule: score >= {} (liveness relief). Bounds: one-sided {} Clopper-Pearson upper bound. Target APCER: {}.\n",
            p.threshold,
            format_confidence(p.confidence),
            format_percent(p.target_apcer)
        );
        let _ = writeln!(s, "**Claim:** {}\n", self.claim());
        s.push_str("## APCER per attack species\n\n");
        s.push_str("| species | presented | no response | scored (n) | accepted (k) | APCER | upper bound | attacks needed for target | additional needed |\n");
        s.push_str("|---|---|---|---|---|---|---|---|---|\n");
        for (i, sp) in self.species.iter().enumerate() {
            let _ = writeln!(
                s,
                "| {}{} | {} | {} | {} | {} | {} | {} | {} | {} |",
                sp.name,
                if self.worst == Some(i) {
                    " (worst)"
                } else {
                    ""
                },
                sp.presented,
                sp.non_responses,
                sp.scored(),
                sp.rate.map_or(0, |r| r.errors),
                opt_pct(sp.rate.as_ref(), |r| r.point),
                opt_pct(sp.rate.as_ref(), |r| r.upper),
                sp.required
                    .map_or("unreachable".to_string(), |v| v.to_string()),
                sp.additional.map_or("n/a".to_string(), |v| v.to_string()),
            );
        }
        s.push_str("\nWorst species = largest upper bound. \"Needed\" assumes no further accepted attack.\n\n");
        s.push_str("## BPCER (bona fide)\n\n");
        let _ = writeln!(
            s,
            "Bona fide presented {}, no response {}, scored {}, rejected {}: BPCER {} (upper bound {}).\n",
            self.bona_fide_presented,
            self.bona_fide_non_responses,
            self.bona_fide_presented - self.bona_fide_non_responses,
            self.bpcer.map_or(0, |r| r.errors),
            opt_pct(self.bpcer.as_ref(), |r| r.point),
            opt_pct(self.bpcer.as_ref(), |r| r.upper),
        );
        s.push_str("## Non-response rates\n\n");
        let rate = |nr: u64, n: u64| format_percent(nr as f64 / n as f64);
        let _ = writeln!(
            s,
            "* Attack non-response (APNRR-style): {} of {} ({})",
            self.attack_non_responses,
            self.attack_presented,
            rate(self.attack_non_responses, self.attack_presented)
        );
        let _ = writeln!(
            s,
            "* Bona fide non-response (BPNRR-style): {} of {} ({})\n",
            self.bona_fide_non_responses,
            self.bona_fide_presented,
            rate(self.bona_fide_non_responses, self.bona_fide_presented)
        );
        s.push_str("Non-responses are excluded from the APCER and BPCER denominators.\n\n");
        s.push_str("## Test population\n\n");
        let cnt = |v: Option<usize>| v.map_or("unknown (no column)".to_string(), |n| n.to_string());
        let _ = writeln!(
            s,
            "* Distinct subjects: {}\n* Distinct sessions: {}\n",
            cnt(self.n_subjects),
            cnt(self.n_sessions)
        );
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

    /// `kind,species,presented,non_responses,n,errors,rate,upper95,required_n,additional_needed`
    pub fn csv(&self) -> String {
        let mut s = String::from(
            "kind,species,presented,non_responses,n,errors,rate,upper_bound,required_n,additional_needed\n",
        );
        let f = |v: Option<f64>| v.map_or(String::new(), |x| format!("{x:.6}"));
        for sp in &self.species {
            let _ = writeln!(
                s,
                "apcer,{},{},{},{},{},{},{},{},{}",
                sp.name,
                sp.presented,
                sp.non_responses,
                sp.scored(),
                sp.rate.map_or(0, |r| r.errors),
                f(sp.rate.map(|r| r.point)),
                f(sp.rate.map(|r| r.upper)),
                sp.required.map_or(String::new(), |v| v.to_string()),
                sp.additional.map_or(String::new(), |v| v.to_string()),
            );
        }
        let _ = writeln!(
            s,
            "bpcer,none,{},{},{},{},{},{},,",
            self.bona_fide_presented,
            self.bona_fide_non_responses,
            self.bona_fide_presented - self.bona_fide_non_responses,
            self.bpcer.map_or(0, |r| r.errors),
            f(self.bpcer.map(|r| r.point)),
            f(self.bpcer.map(|r| r.upper)),
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: PadParams = PadParams {
        threshold: 0.05,
        target_apcer: 0.01,
        confidence: 0.95,
    };

    /// 100 bona fide (2 below 0.05), print_matte 300 attacks all rejected,
    /// phone_replay 100 attacks of which 1 accepted, plus 2 attack and 1 bona
    /// fide non-responses.
    fn sample() -> String {
        let mut s = String::from("label,species,score,response,session,subject\n");
        for i in 0..100 {
            let score = if i < 2 { 0.01 } else { 0.2 };
            let _ = writeln!(s, "bona_fide,none,{score},ok,s{},me", i % 2);
        }
        let _ = writeln!(s, "bona_fide,none,,no_face,s0,me");
        for _ in 0..300 {
            s.push_str("attack,print_matte,0.0,ok,s0,me\n");
        }
        for i in 0..100 {
            let score = if i == 0 { 0.06 } else { 0.0 };
            let _ = writeln!(s, "attack,phone_replay,{score},ok,s1,me");
        }
        s.push_str("attack,phone_replay,,timeout,s1,me\n");
        s.push_str("attack,phone_replay,,no_face,s1,me\n");
        s
    }

    #[test]
    fn per_species_counts_and_bounds() {
        let input = parse_pad(&sample(), DEFAULT_SPECIES_COLUMN).unwrap();
        let r = analyze_pad(&input, P).unwrap();
        let print = r.species.iter().find(|s| s.name == "print_matte").unwrap();
        let phone = r.species.iter().find(|s| s.name == "phone_replay").unwrap();
        // k = 0, n = 300: 1 - 0.05^(1/300) = 0.0099361; 299 needed, 0 more.
        assert_eq!(print.rate.unwrap().errors, 0);
        assert!((print.rate.unwrap().upper - 0.009_936_081_9).abs() < 1e-9);
        assert_eq!((print.required, print.additional), (Some(299), Some(0)));
        // k = 1, n = 100 (2 non-responses excluded): bound 0.04656.
        assert_eq!((phone.presented, phone.non_responses), (102, 2));
        assert_eq!(phone.rate.unwrap().trials, 100);
        assert!((phone.rate.unwrap().upper - 0.046_559_811).abs() < 1e-8);
        assert!(phone.additional.unwrap() > 300);
        // Worst species is the one with the larger bound.
        assert_eq!(r.species[r.worst.unwrap()].name, "phone_replay");
        // BPCER: 2 of 100 scored (1 non-response excluded).
        let b = r.bpcer.unwrap();
        assert_eq!((b.errors, b.trials), (2, 100));
        assert!((b.upper - 0.061_619_200).abs() < 1e-8);
        assert_eq!(r.bona_fide_non_responses, 1);
        assert_eq!((r.attack_presented, r.attack_non_responses), (402, 2));
    }

    #[test]
    fn claim_untested_and_warnings() {
        let input = parse_pad(&sample(), DEFAULT_SPECIES_COLUMN).unwrap();
        let r = analyze_pad(&input, P).unwrap();
        let claim = r.claim();
        assert!(claim.contains("APCER <= 4.656%"), "{claim}");
        assert!(claim.contains("`phone_replay`"), "{claim}");
        assert!(claim.contains("NOT demonstrated"), "{claim}");
        assert!(claim.contains("untested species:"), "{claim}");
        assert!(claim.contains("screen_replay"), "{claim}");
        assert!(!r.untested.contains(&"print_matte".to_string()));
        assert!(r.warnings.iter().any(|w| w.contains("phone_replay")));
        assert!(r.warnings.iter().any(|w| w.contains("One subject")));
        assert_eq!(r.n_sessions, Some(2));
        assert_eq!(r.n_subjects, Some(1));
        let md = r.markdown();
        assert!(md.contains("phone_replay (worst)"));
        assert!(md.contains("not a certification"));
    }

    #[test]
    fn missing_species_column_pools_and_warns() {
        let input = parse_pad(
            "label,score\nbona_fide,0.2\nattack,0.0\nattack,0.0\n",
            "species",
        )
        .unwrap();
        let r = analyze_pad(&input, P).unwrap();
        assert_eq!(r.species.len(), 1);
        assert_eq!(r.species[0].name, UNSPECIFIED_SPECIES);
        assert!(r.warnings.iter().any(|w| w.contains("No species column")));
        assert!(r.untested.is_empty());
    }

    #[test]
    fn custom_species_column_name() {
        let input = parse_pad(
            "label,instrument,score\nbona_fide,none,0.2\nattack,x,0.0\n",
            "instrument",
        )
        .unwrap();
        assert_eq!(input.rows[1].species, "x");
    }

    #[test]
    fn explicit_errors() {
        let e = parse_pad("label,species,score\nattack,,0.1\n", "species").unwrap_err();
        assert!(e.to_string().contains("no species"), "{e}");
        let e = parse_pad("label,score\nmaybe,0.1\n", "species").unwrap_err();
        assert!(e.to_string().starts_with("line 2: unknown label"), "{e}");
        let e = parse_pad("label,score,response\nattack,0.1,meh\n", "species").unwrap_err();
        assert!(e.to_string().contains("unknown response"), "{e}");
        let e = parse_pad("label,score\nattack,abc\n", "species").unwrap_err();
        assert!(matches!(e, PadError::BadScore { .. }));
        let only_attacks = parse_pad("label,score\nattack,0.1\n", "species").unwrap();
        assert!(matches!(
            analyze_pad(&only_attacks, P),
            Err(PadError::NoBonaFide)
        ));
        let only_bf = parse_pad("label,score\nbona_fide,0.1\n", "species").unwrap();
        assert!(matches!(analyze_pad(&only_bf, P), Err(PadError::NoAttacks)));
        let ok = parse_pad("label,score\nbona_fide,0.1\nattack,0.1\n", "species").unwrap();
        assert!(analyze_pad(
            &ok,
            PadParams {
                target_apcer: 0.0,
                ..P
            }
        )
        .is_err());
    }

    #[test]
    fn threshold_boundary_is_accept() {
        // score == threshold is accepted (relief >= threshold passes liveness).
        let input = parse_pad("label,score\nbona_fide,0.05\nattack,0.05\n", "species").unwrap();
        let r = analyze_pad(&input, P).unwrap();
        assert_eq!(r.bpcer.unwrap().errors, 0);
        assert_eq!(r.species[0].rate.unwrap().errors, 1);
    }

    #[test]
    fn csv_has_one_row_per_species_plus_bpcer() {
        let input = parse_pad(&sample(), DEFAULT_SPECIES_COLUMN).unwrap();
        let csv = analyze_pad(&input, P).unwrap().csv();
        assert_eq!(csv.lines().count(), 1 + 2 + 1);
        assert!(csv.contains("apcer,print_matte,300,0,300,0,0.000000,0.009936,299,0"));
        assert!(csv.contains("bpcer,none,101,1,100,2,0.020000,0.061619,,"));
    }
}
