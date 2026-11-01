//! CLI smoke and determinism tests on synthetic data (no models, no camera).

use std::path::Path;
use std::process::{Command, Output};

fn bench(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_doorman-bench"))
        .args(args)
        .output()
        .expect("run doorman-bench")
}

fn ok(args: &[&str]) -> String {
    let o = bench(args);
    assert!(
        o.status.success(),
        "{args:?} failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn synth(dir: &Path, seed: &str) {
    ok(&["synth", "--out", s(dir), "--seed", seed, "--images", "2"]);
}

/// Every report file except machine.md (timestamps), name -> bytes.
fn deterministic_files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| e.file_name() != "machine.md" && e.path().is_file())
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    v.sort();
    v
}

#[test]
fn accuracy_and_liveness_smoke_with_invariants() {
    let d = tempfile::tempdir().unwrap();
    synth(d.path(), "1");
    assert!(d.path().join("images/synth_001.png").is_file());

    let rep = d.path().join("rep");
    ok(&[
        "accuracy",
        "--scores",
        s(&d.path().join("accuracy_scores.csv")),
        "--out",
        s(&rep),
    ]);
    for f in [
        "accuracy.md",
        "accuracy_summary.csv",
        "accuracy_histogram.csv",
        "machine.md",
    ] {
        assert!(rep.join(f).is_file(), "{f}");
    }
    // Algorithmic invariants on the well-separated synthetic data.
    let summary = std::fs::read_to_string(rep.join("accuracy_summary.csv")).unwrap();
    let tar_at = |far: &str| -> f64 {
        summary
            .lines()
            .find(|l| l.starts_with(&format!("tar_at_far,{far},")))
            .unwrap()
            .split(',')
            .nth(4)
            .unwrap()
            .parse()
            .unwrap()
    };
    // Looser FAR can never give a lower TAR.
    assert!(tar_at("1e-3") >= tar_at("1e-4"));
    let eer: f64 = summary
        .lines()
        .find(|l| l.starts_with("eer,"))
        .unwrap()
        .split(',')
        .nth(3)
        .unwrap()
        .parse()
        .unwrap();
    assert!(eer < 0.05, "EER {eer} on separable synthetic data");
    // Histogram counts add up to the generated sample sizes.
    let hist = std::fs::read_to_string(rep.join("accuracy_histogram.csv")).unwrap();
    let (mut g, mut i) = (0usize, 0usize);
    for l in hist.lines().skip(1) {
        let c: Vec<&str> = l.split(',').collect();
        g += c[2].parse::<usize>().unwrap();
        i += c[3].parse::<usize>().unwrap();
    }
    assert_eq!((g, i), (2000, 20_000));

    let lrep = d.path().join("lrep");
    ok(&[
        "liveness",
        "--scores",
        s(&d.path().join("liveness_scores.csv")),
        "--thresholds",
        "0.05,0.1",
        "--out",
        s(&lrep),
    ]);
    let roc = std::fs::read_to_string(lrep.join("liveness_roc.csv")).unwrap();
    let first: Vec<&str> = roc.lines().nth(1).unwrap().split(',').collect();
    let last: Vec<&str> = roc.lines().last().unwrap().split(',').collect();
    assert_eq!((first[1], first[2]), ("1.000000", "0.000000"));
    assert_eq!((last[1], last[2]), ("0.000000", "1.000000"));
}

#[test]
fn reports_are_deterministic_for_a_seed() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    for d in [&a, &b] {
        synth(d.path(), "42");
        ok(&[
            "accuracy",
            "--scores",
            s(&d.path().join("accuracy_scores.csv")),
            "--out",
            s(&d.path().join("rep")),
        ]);
        ok(&[
            "liveness",
            "--scores",
            s(&d.path().join("liveness_scores.csv")),
            "--out",
            s(&d.path().join("rep")),
        ]);
    }
    let fa = deterministic_files(a.path());
    let fb = deterministic_files(b.path());
    assert_eq!(fa, fb, "synthetic inputs differ between runs");
    let ra = deterministic_files(&a.path().join("rep"));
    let rb = deterministic_files(&b.path().join("rep"));
    assert!(!ra.is_empty());
    assert_eq!(ra, rb, "report bytes differ between identical runs");

    // A different seed must change the data.
    let c = tempfile::tempdir().unwrap();
    synth(c.path(), "43");
    assert_ne!(
        std::fs::read(a.path().join("accuracy_scores.csv")).unwrap(),
        std::fs::read(c.path().join("accuracy_scores.csv")).unwrap()
    );
}

#[test]
fn bad_inputs_give_explicit_errors() {
    let d = tempfile::tempdir().unwrap();
    let bad = d.path().join("bad.csv");
    std::fs::write(&bad, "label,score\nmaybe,0.5\n").unwrap();
    let o = bench(&[
        "accuracy",
        "--scores",
        s(&bad),
        "--out",
        s(&d.path().join("o")),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown label"));

    let o = bench(&["accuracy", "--scores", "/nonexistent/x.csv"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("cannot read"));
}

#[test]
fn record_refuses_inside_git_repo_before_touching_camera() {
    let repo = tempfile::tempdir().unwrap();
    std::fs::create_dir(repo.path().join(".git")).unwrap();
    let o = bench(&[
        "record",
        "--out",
        s(&repo.path().join("data")),
        "--user",
        "alice",
        "--condition",
        "day",
        "--yes",
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("git repository"));
}

#[test]
fn record_requires_exactly_one_target() {
    let d = tempfile::tempdir().unwrap();
    let o = bench(&["record", "--out", s(d.path()), "--yes"]);
    assert!(!o.status.success());
}

#[cfg(not(feature = "ort"))]
#[test]
fn latency_without_ort_feature_is_an_explicit_error() {
    let o = bench(&["latency", "--frames", ".", "--iters", "1"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("`ort` feature"));
}

#[cfg(feature = "ort")]
#[test]
fn latency_with_missing_models_is_an_explicit_error() {
    let d = tempfile::tempdir().unwrap();
    synth(d.path(), "1");
    let o = bench(&[
        "latency",
        "--frames",
        s(&d.path().join("images")),
        "--iters",
        "1",
        "--models-dir",
        s(&d.path().join("no-models")),
        "--out",
        s(&d.path().join("o")),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("model files are missing"));
}
