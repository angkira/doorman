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

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn pad_report_end_to_end_on_synthetic_data() {
    let d = tempfile::tempdir().unwrap();
    synth(d.path(), "5");
    let rep = d.path().join("pad");
    let out = ok(&[
        "pad-report",
        "--scores",
        s(&d.path().join("pad_scores.csv")),
        "--threshold",
        "0.05",
        "--out",
        s(&rep),
    ]);
    for f in ["pad_report.md", "pad_report.csv", "machine.md"] {
        assert!(rep.join(f).is_file(), "{f}");
    }
    let md = read(&rep.join("pad_report.md"));
    assert!(out.contains(&md));
    // Synthetic data: 4 species, screen_replay is the weakest, paper_mask never tested.
    assert!(md.contains("screen_replay (worst)"), "{md}");
    assert!(md.contains("untested species:"), "{md}");
    assert!(md.contains("paper_mask"), "{md}");
    assert!(md.contains("not a certification"), "{md}");
    assert!(md.contains("Distinct subjects: 1"), "{md}");
    assert!(md.contains("Distinct sessions: 3"), "{md}");
    assert!(md.contains("One subject"), "{md}");

    let csv = read(&rep.join("pad_report.csv"));
    let rows: Vec<Vec<&str>> = csv
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect())
        .collect();
    // 4 species rows + bpcer. 400 attacks over 4 species, 100 each; the 8
    // non-responses (i = 49, 99, ..., 399) fall on the two odd species, 4 each.
    assert_eq!(rows.len(), 5);
    for r in &rows[..4] {
        assert_eq!((r[0], r[2]), ("apcer", "100"), "{r:?}");
    }
    let nr: Vec<&str> = rows[..4].iter().map(|r| r[3]).collect();
    assert_eq!(nr.iter().map(|v| v.parse::<u32>().unwrap()).sum::<u32>(), 8);
    // BPCER row: 400 bona fide, 8 non-responses (i = 49, 99, ..., 399).
    assert_eq!(&rows[4][..5], ["bpcer", "none", "400", "8", "392"]);
    // Upper bound is never below the point estimate, and print_matte (mean 0.01,
    // std 0.015, threshold 0.05: ~2.7 sigma) is stronger than screen_replay.
    let rate = |name: &str, col: usize| -> f64 {
        rows.iter().find(|r| r[1] == name).unwrap()[col]
            .parse()
            .unwrap()
    };
    assert!(rate("print_matte", 7) >= rate("print_matte", 6));
    assert!(rate("screen_replay", 6) > rate("print_matte", 6));

    // Deterministic bytes for identical input.
    let rep2 = d.path().join("pad2");
    ok(&[
        "pad-report",
        "--scores",
        s(&d.path().join("pad_scores.csv")),
        "--threshold",
        "0.05",
        "--out",
        s(&rep2),
    ]);
    assert_eq!(deterministic_files(&rep), deterministic_files(&rep2));
}

#[test]
fn pad_report_target_and_species_column_options() {
    let d = tempfile::tempdir().unwrap();
    synth(d.path(), "5");
    let renamed = d.path().join("renamed.csv");
    std::fs::write(
        &renamed,
        read(&d.path().join("pad_scores.csv")).replacen("label,species,", "label,instrument,", 1),
    )
    .unwrap();
    let out = ok(&[
        "pad-report",
        "--scores",
        s(&renamed),
        "--species-column",
        "instrument",
        "--threshold",
        "0.05",
        "--target-apcer",
        "0.07",
        "--out",
        s(&d.path().join("r")),
    ]);
    assert!(out.contains("Target APCER: 7.000%"), "{out}");
    // Without the option the renamed column is not found: pooled, with a warning.
    let out = ok(&[
        "pad-report",
        "--scores",
        s(&renamed),
        "--threshold",
        "0.05",
        "--out",
        s(&d.path().join("r2")),
    ]);
    assert!(out.contains("No species column"), "{out}");
}

#[test]
fn match_report_end_to_end_on_synthetic_data() {
    let d = tempfile::tempdir().unwrap();
    synth(d.path(), "9");
    let rep = d.path().join("match");
    let out = ok(&[
        "match-report",
        "--scores",
        s(&d.path().join("match_scores.csv")),
        "--threshold",
        "0.4",
        "--out",
        s(&rep),
    ]);
    for f in ["match_report.md", "match_report.csv", "machine.md"] {
        assert!(rep.join(f).is_file(), "{f}");
    }
    assert!(out.contains(&read(&rep.join("match_report.md"))));
    // 20000 impostor comparisons, every 200th (i = 199, ...) is an FTA: 100.
    // 2000 genuine, every 200th an FTA: 10.
    let csv = read(&rep.join("match_report.csv"));
    let line = |k: &str| -> Vec<String> {
        csv.lines()
            .find(|l| l.starts_with(&format!("{k},")))
            .unwrap()
            .split(',')
            .map(String::from)
            .collect()
    };
    assert_eq!(&line("fmr")[1..4], ["20000", "100", "19900"]);
    assert_eq!(&line("fnmr")[1..4], ["2000", "10", "1990"]);
    // Impostor mean 0.05, std 0.10: threshold 0.4 is 3.5 sigma, a handful of false matches at most.
    let fmr_upper: f64 = line("fmr")[6].parse().unwrap();
    assert!(fmr_upper < 1e-2, "{fmr_upper}");
    assert!(line("n_subjects")[1] == "50", "{csv}");
    assert!(
        out.contains("n_subjects (distinct impostor subjects): 50"),
        "{out}"
    );
    assert!(
        out.contains("Comparisons needed at zero false matches"),
        "{out}"
    );
    assert!(out.contains("29956"), "{out}");
    assert!(out.contains("149786"), "{out}");
    // 19900 scored comparisons with 4 false matches cannot show 1e-4.
    assert!(
        out.contains("19900 scored impostor comparisons are fewer than"),
        "{out}"
    );
    assert!(out.contains("NOT demonstrated"), "{out}");

    // Identical input gives identical report bytes.
    let rep2 = d.path().join("match2");
    ok(&[
        "match-report",
        "--scores",
        s(&d.path().join("match_scores.csv")),
        "--threshold",
        "0.4",
        "--out",
        s(&rep2),
    ]);
    assert_eq!(deterministic_files(&rep), deterministic_files(&rep2));
}

#[test]
fn pad_and_match_report_reject_wrong_csv_explicitly() {
    let d = tempfile::tempdir().unwrap();
    synth(d.path(), "1");
    // Accuracy labels are not PAD labels and vice versa.
    let o = bench(&[
        "pad-report",
        "--scores",
        s(&d.path().join("accuracy_scores.csv")),
        "--threshold",
        "0.05",
        "--out",
        s(&d.path().join("o")),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown label"));
    let o = bench(&[
        "match-report",
        "--scores",
        s(&d.path().join("pad_scores.csv")),
        "--threshold",
        "0.4",
        "--out",
        s(&d.path().join("o")),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown label"));
    let o = bench(&[
        "pad-report",
        "--scores",
        "/nonexistent/x.csv",
        "--threshold",
        "0.1",
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("cannot read"));
    let o = bench(&["synth", "--out", s(d.path()), "--subjects", "0"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--subjects"));
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
fn accuracy_enrol_source_flags_are_validated() {
    let d = tempfile::tempdir().unwrap();
    let dir = s(d.path());
    // --enroll-store needs --enroll-user.
    let o = bench(&[
        "accuracy",
        "--enroll-store",
        dir,
        "--genuine",
        dir,
        "--impostor",
        dir,
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--enroll-user"));
    // --enroll and --enroll-store are alternatives.
    let o = bench(&[
        "accuracy",
        "--enroll",
        dir,
        "--enroll-store",
        dir,
        "--enroll-user",
        "me",
        "--genuine",
        dir,
        "--impostor",
        dir,
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("cannot be used with"));
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
