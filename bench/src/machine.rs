//! Machine/run information kept OUT of the deterministic report files (it
//! contains a timestamp), in `machine.md`.

use std::fmt::Write;

#[derive(Debug, Clone)]
pub struct MachineInfo {
    pub cpu_model: String,
    pub logical_cores: usize,
    pub os: String,
    pub ort: String,
    pub git_rev: String,
    pub timestamp: String,
    pub command: String,
}

/// First `model name` line of /proc/cpuinfo text.
pub fn parse_cpu_model(cpuinfo: &str) -> Option<String> {
    cpuinfo.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim() == "model name").then(|| v.trim().to_string())
    })
}

#[cfg(feature = "ort")]
fn ort_description() -> String {
    doormand::ml::ort_backend::runtime_info()
}

#[cfg(not(feature = "ort"))]
fn ort_description() -> String {
    "not available (doorman-bench built without the `ort` feature)".to_string()
}

fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

pub fn collect() -> MachineInfo {
    MachineInfo {
        cpu_model: std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|t| parse_cpu_model(&t))
            .unwrap_or_else(|| "unknown".to_string()),
        logical_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        ort: ort_description(),
        git_rev: git_rev(),
        timestamp: chrono::Local::now().to_rfc3339(),
        command: std::env::args().collect::<Vec<_>>().join(" "),
    }
}

impl MachineInfo {
    /// `extra` lines are appended verbatim as `- key: value` (e.g. thread list).
    pub fn markdown(&self, extra: &[(&str, String)]) -> String {
        let mut s = String::from("# Machine and run info\n\n");
        let _ = writeln!(s, "- timestamp: {}", self.timestamp);
        let _ = writeln!(s, "- git rev: {}", self.git_rev);
        let _ = writeln!(s, "- cpu: {}", self.cpu_model);
        let _ = writeln!(s, "- logical cores: {}", self.logical_cores);
        let _ = writeln!(s, "- os: {}", self.os);
        let _ = writeln!(s, "- onnx runtime: {}", self.ort);
        for (k, v) in extra {
            let _ = writeln!(s, "- {k}: {v}");
        }
        let _ = writeln!(s, "- command: `{}`", self.command);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpu_model() {
        let t = "processor\t: 0\nvendor_id\t: X\nmodel name\t: AMD Ryzen 7 8700G\nmodel name\t: other\n";
        assert_eq!(parse_cpu_model(t).as_deref(), Some("AMD Ryzen 7 8700G"));
        assert_eq!(parse_cpu_model("nothing here"), None);
    }
}
