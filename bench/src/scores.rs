//! Scores CSV: `label,score[,condition]`, one header line, `#` comments allowed.

use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum ScoresError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("scores CSV is empty (expected a `label,score[,condition]` header)")]
    Empty,
    #[error(
        "line {line}: header must start with `label,score` (optionally `,condition`), got `{got}`"
    )]
    BadHeader { line: usize, got: String },
    #[error("line {line}: expected at least 2 columns (label,score), got `{got}`")]
    TooFewColumns { line: usize, got: String },
    #[error("line {line}: unknown label `{label}` (expected `{positive}` or `{negative}`)")]
    BadLabel {
        line: usize,
        label: String,
        positive: String,
        negative: String,
    },
    #[error("line {line}: score `{value}` is not a finite number")]
    BadScore { line: usize, value: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreRow {
    /// `true` for the positive label (genuine / bona_fide).
    pub positive: bool,
    pub score: f64,
    pub condition: Option<String>,
}

/// Parse scores text. `positive`/`negative` are the two accepted labels.
pub fn parse_scores(
    text: &str,
    positive: &str,
    negative: &str,
) -> Result<Vec<ScoreRow>, ScoresError> {
    let mut lines = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.trim()))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'));

    let (hline, header) = lines.next().ok_or(ScoresError::Empty)?;
    let cols: Vec<String> = header.split(',').map(|c| c.trim().to_lowercase()).collect();
    if cols.len() < 2 || cols[0] != "label" || cols[1] != "score" {
        return Err(ScoresError::BadHeader {
            line: hline,
            got: header.to_string(),
        });
    }

    let mut rows = Vec::new();
    for (line, l) in lines {
        let f: Vec<&str> = l.split(',').map(str::trim).collect();
        if f.len() < 2 {
            return Err(ScoresError::TooFewColumns {
                line,
                got: l.to_string(),
            });
        }
        let positive_row = if f[0] == positive {
            true
        } else if f[0] == negative {
            false
        } else {
            return Err(ScoresError::BadLabel {
                line,
                label: f[0].to_string(),
                positive: positive.to_string(),
                negative: negative.to_string(),
            });
        };
        let score: f64 = f[1]
            .parse()
            .ok()
            .filter(|s: &f64| s.is_finite())
            .ok_or_else(|| ScoresError::BadScore {
                line,
                value: f[1].to_string(),
            })?;
        let condition = f.get(2).filter(|c| !c.is_empty()).map(|c| c.to_string());
        rows.push(ScoreRow {
            positive: positive_row,
            score,
            condition,
        });
    }
    Ok(rows)
}

pub fn read_scores(
    path: &Path,
    positive: &str,
    negative: &str,
) -> Result<Vec<ScoreRow>, ScoresError> {
    let text = std::fs::read_to_string(path).map_err(|source| ScoresError::Read {
        path: path.display().to_string(),
        source,
    })?;
    parse_scores(&text, positive, negative)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_with_and_without_condition() {
        let rows = parse_scores(
            "label,score,condition\n# c\ngenuine,0.5,day\nimpostor,0.1\n",
            "genuine",
            "impostor",
        )
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows[0].positive);
        assert_eq!(rows[0].condition.as_deref(), Some("day"));
        assert!(!rows[1].positive);
        assert_eq!(rows[1].condition, None);
    }

    #[test]
    fn reports_line_numbers() {
        let e = parse_scores("label,score\ngenuine,abc\n", "genuine", "impostor").unwrap_err();
        assert!(e.to_string().starts_with("line 2:"), "{e}");
        let e = parse_scores("label,score\nfoo,1\n", "genuine", "impostor").unwrap_err();
        assert!(e.to_string().contains("unknown label"), "{e}");
        let e = parse_scores("a,b\n", "genuine", "impostor").unwrap_err();
        assert!(matches!(e, ScoresError::BadHeader { .. }));
        let e = parse_scores("", "genuine", "impostor").unwrap_err();
        assert!(matches!(e, ScoresError::Empty));
        let e = parse_scores("label,score\ngenuine,NaN\n", "genuine", "impostor").unwrap_err();
        assert!(matches!(e, ScoresError::BadScore { .. }));
    }
}
