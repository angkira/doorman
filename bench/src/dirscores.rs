//! Plain data and CSV writers for the directory mode of `accuracy`, kept free of
//! the ML backend so they are unit-tested everywhere. The directory mode scores
//! every probe image against the enrolled templates; a probe without a detected
//! face is a failure to acquire (FTA), kept as a row with no score.
//!
//! Output `match_scores.csv` (`label,score,subject,response`) is the input of
//! `doorman-bench match-report`.

use crate::scores::ScoreRow;
use std::fmt::Write as _;
use std::path::Path;

/// Subject id given to every genuine probe (the enrolled owner).
pub const OWNER_SUBJECT: &str = "owner";
/// Subject id of an impostor image that sits directly in the impostor root and
/// has no readable file stem.
pub const UNKNOWN_SUBJECT: &str = "(unknown)";

const LABEL_GENUINE: &str = "genuine";
const LABEL_IMPOSTOR: &str = "impostor";
const RESPONSE_OK: &str = "ok";
const RESPONSE_FTA: &str = "fta";

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeScore {
    pub genuine: bool,
    /// `None`: no face detected (failure to acquire).
    pub score: Option<f64>,
    pub condition: Option<String>,
    pub subject: Option<String>,
}

/// Impostor identity of `path` below `root`: the first directory under the root
/// (one directory per identity, like LFW), or the file stem when the image lies
/// directly in the root (one image per identity, like FFHQ).
pub fn impostor_subject(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let mut comps = rel.components();
    let first = comps
        .next()
        .map(|c| c.as_os_str().to_string_lossy().into_owned());
    if comps.next().is_some() {
        if let Some(dir) = first {
            return dir;
        }
    }
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| UNKNOWN_SUBJECT.to_string())
}

/// Scored probes as accuracy rows (failures to acquire are dropped, as before).
pub fn to_score_rows(probes: &[ProbeScore]) -> Vec<ScoreRow> {
    probes
        .iter()
        .filter_map(|p| {
            p.score.map(|score| ScoreRow {
                positive: p.genuine,
                score,
                condition: p.condition.clone(),
            })
        })
        .collect()
}

/// `label,score,subject,response` for `match-report`. Failures to acquire have an
/// empty score and response `fta`.
pub fn match_csv(probes: &[ProbeScore]) -> String {
    let mut s = String::from("label,score,subject,response\n");
    for p in probes {
        let label = if p.genuine {
            LABEL_GENUINE
        } else {
            LABEL_IMPOSTOR
        };
        let subject = p.subject.as_deref().unwrap_or("");
        match p.score {
            Some(v) => {
                let _ = writeln!(s, "{label},{v:.6},{subject},{RESPONSE_OK}");
            }
            None => {
                let _ = writeln!(s, "{label},,{subject},{RESPONSE_FTA}");
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn subject_is_first_directory_or_file_stem() {
        let root = PathBuf::from("/data/impostor");
        assert_eq!(
            impostor_subject(&root, &root.join("Aaron_Eckhart/Aaron_Eckhart_0001.jpg")),
            "Aaron_Eckhart"
        );
        assert_eq!(
            impostor_subject(&root, &root.join("a/b/c.png")),
            "a",
            "deeper levels belong to the same identity"
        );
        assert_eq!(impostor_subject(&root, &root.join("00042.png")), "00042");
    }

    #[test]
    fn csv_and_rows_from_probes() {
        let probes = vec![
            ProbeScore {
                genuine: true,
                score: Some(0.81),
                condition: Some("day".into()),
                subject: Some(OWNER_SUBJECT.into()),
            },
            ProbeScore {
                genuine: false,
                score: Some(0.12),
                condition: None,
                subject: Some("p1".into()),
            },
            ProbeScore {
                genuine: false,
                score: None,
                condition: None,
                subject: Some("p2".into()),
            },
        ];
        assert_eq!(
            match_csv(&probes),
            "label,score,subject,response\n\
             genuine,0.810000,owner,ok\n\
             impostor,0.120000,p1,ok\n\
             impostor,,p2,fta\n"
        );
        let rows = to_score_rows(&probes);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].positive && !rows[1].positive);
        assert_eq!(rows[0].condition.as_deref(), Some("day"));
    }
}
