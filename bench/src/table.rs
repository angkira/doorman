//! Minimal header-driven CSV reader for the report inputs (`pad-report`,
//! `match-report`). Columns are found by name (case-insensitive) so optional
//! columns can be present or absent. Fields are plain: no quoting, no embedded
//! commas. `#` comment lines and blank lines are skipped.

#[derive(Debug, thiserror::Error)]
pub enum TableError {
    #[error("CSV is empty (expected a header line)")]
    Empty,
    #[error("missing required column `{0}` in the header")]
    MissingColumn(String),
    #[error("line {line}: expected {expected} columns like the header, got {got}")]
    RowWidth {
        line: usize,
        expected: usize,
        got: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// 1-based line number in the file, for error messages.
    pub line: usize,
    fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    header: Vec<String>,
    pub rows: Vec<Row>,
}

impl Table {
    pub fn parse(text: &str) -> Result<Self, TableError> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(i, l)| (i + 1, l.trim()))
            .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'));
        let (_, head) = lines.next().ok_or(TableError::Empty)?;
        let header: Vec<String> = head.split(',').map(|c| c.trim().to_lowercase()).collect();
        let mut rows = Vec::new();
        for (line, l) in lines {
            let fields: Vec<String> = l.split(',').map(|f| f.trim().to_string()).collect();
            if fields.len() != header.len() {
                return Err(TableError::RowWidth {
                    line,
                    expected: header.len(),
                    got: fields.len(),
                });
            }
            rows.push(Row { line, fields });
        }
        Ok(Self { header, rows })
    }

    /// Index of column `name` (case-insensitive), if present.
    pub fn column(&self, name: &str) -> Option<usize> {
        let name = name.to_lowercase();
        self.header.iter().position(|h| *h == name)
    }

    pub fn require(&self, name: &str) -> Result<usize, TableError> {
        self.column(name)
            .ok_or_else(|| TableError::MissingColumn(name.to_string()))
    }
}

impl Row {
    pub fn get(&self, col: usize) -> &str {
        &self.fields[col]
    }

    /// Cell of an optional column; `None` if the column is absent or the cell empty.
    pub fn opt(&self, col: Option<usize>) -> Option<&str> {
        col.map(|c| self.fields[c].as_str())
            .filter(|s| !s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_by_name_and_skips_comments() {
        let t = Table::parse("Label, Score ,extra\n# c\n\na,1,x\nb,2,\n").unwrap();
        assert_eq!(t.column("score"), Some(1));
        assert_eq!(t.column("nope"), None);
        assert_eq!(t.rows.len(), 2);
        assert_eq!(t.rows[0].get(0), "a");
        assert_eq!(t.rows[1].opt(t.column("extra")), None);
        assert_eq!(t.rows[0].opt(t.column("extra")), Some("x"));
        assert_eq!(t.rows[1].line, 5);
    }

    #[test]
    fn explicit_errors() {
        assert!(matches!(Table::parse(""), Err(TableError::Empty)));
        let e = Table::parse("a,b\n1\n").unwrap_err();
        assert!(e.to_string().starts_with("line 2:"), "{e}");
        let t = Table::parse("a,b\n").unwrap();
        assert!(matches!(t.require("c"), Err(TableError::MissingColumn(_))));
    }
}
