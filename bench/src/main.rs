//! doorman-bench: benchmark and evaluation harness for the doorman daemon.
//!
//! Subcommands: `synth`, `accuracy`, `liveness`, `pad-report`, `match-report`,
//! `latency`, `record`.
//! See docs/bench/README.md for dataset layout and command lines.

mod analysis;
#[cfg(feature = "ort")]
mod backend;
#[cfg_attr(not(feature = "ort"), allow(dead_code))]
mod dataset;
#[cfg_attr(not(feature = "ort"), allow(dead_code))]
mod dirscores;
#[cfg_attr(not(feature = "ort"), allow(dead_code))]
mod latency_report;
mod machine;
mod matching;
mod metrics;
mod pad;
mod record;
mod scores;
mod stats;
mod synth;
mod table;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Label pair of the accuracy scores CSV.
const LABEL_GENUINE: &str = "genuine";
const LABEL_IMPOSTOR: &str = "impostor";
/// Label pair of the liveness scores CSV.
const LABEL_BONA_FIDE: &str = "bona_fide";
const LABEL_ATTACK: &str = "attack";
/// Defaults for `latency`.
const DEFAULT_WARMUP: usize = 5;
const DEFAULT_THREADS: &str = "0";
/// Defaults for `record`.
const DEFAULT_RECORD_FRAMES: usize = 30;
const DEFAULT_RECORD_INTERVAL_MS: u64 = 200;
/// Liveness thresholds reported when `--thresholds` is not given (daemon default relief gate).
const DEFAULT_LIVENESS_THRESHOLDS: &str = "0.05";

#[derive(Parser)]
#[command(
    name = "doorman-bench",
    version,
    about = "Benchmark and evaluation harness for doorman"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate seeded synthetic score CSVs and textured images (no real faces).
    Synth(SynthArgs),
    /// Accuracy metrics from a scores CSV, or by embedding image directories.
    Accuracy(AccuracyArgs),
    /// APCER/BPCER, ROC and a recommended threshold from a liveness scores CSV.
    Liveness(LivenessArgs),
    /// ISO/IEC 30107-3 style PAD report: APCER per attack species, BPCER, bounds.
    PadReport(PadReportArgs),
    /// ISO/IEC 19795-1 style FMR/FNMR/FTA report with exact bounds from comparison scores.
    MatchReport(MatchReportArgs),
    /// Per-stage latency (p50/p95/p99), cold start separate; needs models + `ort` feature.
    Latency(LatencyArgs),
    /// Guided capture of a labelled dataset from the camera.
    Record(RecordCliArgs),
}

#[derive(Args)]
struct OutArg {
    /// Report directory (default: bench/results/<date>/).
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct SynthArgs {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = synth::SynthParams::default().genuine)]
    genuine: usize,
    #[arg(long, default_value_t = synth::SynthParams::default().impostor)]
    impostor: usize,
    #[arg(long, default_value_t = synth::SynthParams::default().bona_fide)]
    bona_fide: usize,
    #[arg(long, default_value_t = synth::SynthParams::default().attack)]
    attack: usize,
    /// Number of random textured PNGs written to <out>/images/.
    #[arg(long, default_value_t = synth::SynthParams::default().images)]
    images: usize,
    /// Distinct impostor subjects in match_scores.csv.
    #[arg(long, default_value_t = synth::SynthParams::default().subjects)]
    subjects: usize,
}

#[derive(Args)]
struct AccuracyArgs {
    /// Scores CSV: label{genuine,impostor},score[,condition].
    #[arg(long, conflicts_with_all = ["enroll", "enroll_store", "genuine", "impostor"])]
    scores: Option<PathBuf>,
    /// Directory mode: enrolment images.
    #[arg(long, conflicts_with = "enroll_store", requires_all = ["genuine", "impostor"])]
    enroll: Option<PathBuf>,
    /// Directory mode: directory holding a COPY of the daemon's `embeddings.bin`
    /// (the loader tightens the directory mode). Needs --enroll-user.
    #[arg(long, requires_all = ["enroll_user", "genuine", "impostor"])]
    enroll_store: Option<PathBuf>,
    /// Directory mode: user whose templates in --enroll-store are the gallery.
    #[arg(long, requires = "enroll_store")]
    enroll_user: Option<String>,
    /// Directory mode: genuine probe images (`<condition>/*.jpg` or flat).
    #[arg(long)]
    genuine: Option<PathBuf>,
    /// Directory mode: impostor images (any depth).
    #[arg(long)]
    impostor: Option<PathBuf>,
    /// Daemon config used for the models (directory mode).
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Override `ml.models_dir` (directory mode).
    #[arg(long)]
    models_dir: Option<PathBuf>,
    /// FAR targets, comma separated.
    #[arg(long, value_delimiter = ',', default_values_t = analysis::DEFAULT_FARS)]
    far: Vec<f64>,
    #[arg(long, default_value_t = analysis::DEFAULT_BINS)]
    bins: usize,
    #[command(flatten)]
    out: OutArg,
}

#[derive(Args)]
struct LivenessArgs {
    /// Scores CSV: label{bona_fide,attack},score.
    #[arg(long)]
    scores: PathBuf,
    /// Thresholds to report APCER/BPCER at, comma separated.
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_LIVENESS_THRESHOLDS)]
    thresholds: Vec<f64>,
    /// Target BPCER for the recommended threshold.
    #[arg(long, default_value_t = analysis::DEFAULT_TARGET_BPCER)]
    target_bpcer: f64,
    #[command(flatten)]
    out: OutArg,
}

#[derive(Args)]
struct PadReportArgs {
    /// Scores CSV: label{bona_fide,attack},species,score[,response,session,subject].
    #[arg(long)]
    scores: PathBuf,
    /// Accept as live iff score >= threshold (liveness relief).
    #[arg(long)]
    threshold: f64,
    /// Name of the attack-species column.
    #[arg(long, default_value = pad::DEFAULT_SPECIES_COLUMN)]
    species_column: String,
    /// APCER the claim is measured against.
    #[arg(long, default_value_t = pad::DEFAULT_TARGET_APCER)]
    target_apcer: f64,
    #[command(flatten)]
    out: OutArg,
}

#[derive(Args)]
struct MatchReportArgs {
    /// Scores CSV: label{genuine,impostor},score[,subject,response].
    #[arg(long)]
    scores: PathBuf,
    /// Accept iff score > threshold (cosine similarity).
    #[arg(long)]
    threshold: f64,
    /// FMR the claim is measured against.
    #[arg(long, default_value_t = matching::DEFAULT_TARGET_FMR)]
    target_fmr: f64,
    #[command(flatten)]
    out: OutArg,
}

#[derive(Args)]
struct LatencyArgs {
    /// Directory of image files (jpg/png).
    #[arg(long)]
    frames: PathBuf,
    #[arg(long)]
    iters: usize,
    #[arg(long, default_value_t = DEFAULT_WARMUP)]
    warmup: usize,
    /// intra_op thread counts to sweep (0 = backend default), comma separated.
    #[arg(long, value_delimiter = ',', default_value = DEFAULT_THREADS)]
    threads: Vec<i32>,
    /// Daemon config providing models_dir and device settings.
    #[arg(long)]
    profile: Option<PathBuf>,
    #[arg(long)]
    models_dir: Option<PathBuf>,
    #[command(flatten)]
    out: OutArg,
}

#[derive(Args)]
struct RecordCliArgs {
    /// Dataset root (must be outside any git repository).
    #[arg(long)]
    out: PathBuf,
    /// Record genuine frames for this user (requires --condition).
    #[arg(long, requires = "condition", conflicts_with_all = ["impostor", "attack"])]
    user: Option<String>,
    #[arg(long)]
    condition: Option<String>,
    /// Record an impostor identity.
    #[arg(long, conflicts_with = "attack")]
    impostor: Option<String>,
    /// Record a presentation attack of this type (e.g. phone_replay, screen_replay, printout).
    #[arg(long)]
    attack: Option<String>,
    #[arg(long, default_value_t = DEFAULT_RECORD_FRAMES)]
    frames: usize,
    #[arg(long, default_value_t = DEFAULT_RECORD_INTERVAL_MS)]
    interval_ms: u64,
    /// Daemon config used to open the camera.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Accept the consent notice without the interactive prompt.
    #[arg(long)]
    yes: bool,
}

fn out_dir(arg: &OutArg) -> PathBuf {
    arg.out.clone().unwrap_or_else(|| {
        Path::new("bench/results").join(chrono::Local::now().format("%Y-%m-%d").to_string())
    })
}

fn write(dir: &Path, name: &str, content: &str) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let p = dir.join(name);
    std::fs::write(&p, content).with_context(|| format!("cannot write {}", p.display()))?;
    println!("wrote {}", p.display());
    Ok(())
}

fn write_machine(dir: &Path, extra: &[(&str, String)]) -> Result<()> {
    write(dir, "machine.md", &machine::collect().markdown(extra))
}

fn cmd_synth(a: SynthArgs) -> Result<()> {
    if a.subjects == 0 {
        bail!("--subjects must be at least 1");
    }
    let p = synth::SynthParams {
        seed: a.seed,
        genuine: a.genuine,
        impostor: a.impostor,
        bona_fide: a.bona_fide,
        attack: a.attack,
        images: a.images,
        subjects: a.subjects,
    };
    write(&a.out, "accuracy_scores.csv", &synth::accuracy_csv(&p))?;
    write(&a.out, "liveness_scores.csv", &synth::liveness_csv(&p))?;
    write(&a.out, "pad_scores.csv", &synth::pad_csv(&p))?;
    write(&a.out, "match_scores.csv", &synth::match_csv(&p))?;
    if p.images > 0 {
        let names = synth::write_images(&a.out.join("images"), &p)?;
        println!(
            "wrote {} images to {}",
            names.len(),
            a.out.join("images").display()
        );
    }
    Ok(())
}

fn report_accuracy(rows: &[scores::ScoreRow], a: &AccuracyArgs, dir: &Path) -> Result<()> {
    let rep = analysis::analyze_accuracy(rows, &a.far, a.bins)?;
    write(dir, "accuracy.md", &rep.markdown())?;
    write(dir, "accuracy_summary.csv", &rep.summary_csv())?;
    write(dir, "accuracy_histogram.csv", &rep.histogram_csv())?;
    print!("{}", rep.markdown());
    Ok(())
}

#[cfg_attr(not(feature = "ort"), allow(dead_code))]
fn rows_to_csv(rows: &[scores::ScoreRow]) -> String {
    let mut s = String::from("label,score,condition\n");
    for r in rows {
        let _ = writeln!(
            s,
            "{},{:.6},{}",
            if r.positive {
                LABEL_GENUINE
            } else {
                LABEL_IMPOSTOR
            },
            r.score,
            r.condition.as_deref().unwrap_or("")
        );
    }
    s
}

#[cfg(feature = "ort")]
async fn cmd_accuracy_dirs(a: &AccuracyArgs, dir: &Path) -> Result<()> {
    let (Some(genuine), Some(impostor)) = (&a.genuine, &a.impostor) else {
        bail!("directory mode needs --genuine and --impostor");
    };
    let enrol = match (&a.enroll, &a.enroll_store, &a.enroll_user) {
        (Some(images), None, None) => backend::EnrolSource::Images(images),
        (None, Some(store), Some(user)) => backend::EnrolSource::Store { dir: store, user },
        _ => bail!("directory mode needs --enroll, or --enroll-store with --enroll-user"),
    };
    let config = backend::load_config(a.profile.as_deref(), a.models_dir.as_deref())?;
    let (probes, st) = backend::score_directories(&config, enrol, genuine, impostor).await?;
    let rows = dirscores::to_score_rows(&probes);
    write(dir, "scores.csv", &rows_to_csv(&rows))?;
    write(dir, "match_scores.csv", &dirscores::match_csv(&probes))?;
    let mut notes = String::from("# Dataset notes\n\n");
    let _ = writeln!(notes, "{st:#?}");
    write(dir, "dataset_notes.md", &notes)?;
    report_accuracy(&rows, a, dir)
}

#[cfg(not(feature = "ort"))]
async fn cmd_accuracy_dirs(_a: &AccuracyArgs, _dir: &Path) -> Result<()> {
    bail!(
        "directory mode needs the daemon ML backend, but doorman-bench was built without the `ort` feature. \
         Rebuild with `--features ort`, or compute scores elsewhere and use `accuracy --scores <csv>`"
    )
}

async fn cmd_accuracy(a: AccuracyArgs) -> Result<()> {
    let dir = out_dir(&a.out);
    if let Some(csv) = &a.scores {
        let rows = scores::read_scores(csv, LABEL_GENUINE, LABEL_IMPOSTOR)?;
        report_accuracy(&rows, &a, &dir)?;
        write_machine(&dir, &[("scores file", csv.display().to_string())])
    } else if a.enroll.is_some() || a.enroll_store.is_some() {
        cmd_accuracy_dirs(&a, &dir).await?;
        write_machine(&dir, &[("mode", "directories".to_string())])
    } else {
        bail!("give either --scores <csv> or --enroll (or --enroll-store/--enroll-user) with --genuine/--impostor directories")
    }
}

fn cmd_liveness(a: LivenessArgs) -> Result<()> {
    let dir = out_dir(&a.out);
    let rows = scores::read_scores(&a.scores, LABEL_BONA_FIDE, LABEL_ATTACK)?;
    let rep = analysis::analyze_liveness(&rows, &a.thresholds, a.target_bpcer)?;
    write(&dir, "liveness.md", &rep.markdown())?;
    write(&dir, "liveness_roc.csv", &rep.roc_csv())?;
    print!("{}", rep.markdown());
    write_machine(&dir, &[("scores file", a.scores.display().to_string())])
}

fn cmd_pad_report(a: PadReportArgs) -> Result<()> {
    let dir = out_dir(&a.out);
    let text = std::fs::read_to_string(&a.scores)
        .with_context(|| format!("cannot read {}", a.scores.display()))?;
    let input = pad::parse_pad(&text, &a.species_column)?;
    let rep = pad::analyze_pad(
        &input,
        pad::PadParams {
            threshold: a.threshold,
            target_apcer: a.target_apcer,
            confidence: stats::DEFAULT_CONFIDENCE,
        },
    )?;
    write(&dir, "pad_report.md", &rep.markdown())?;
    write(&dir, "pad_report.csv", &rep.csv())?;
    print!("{}", rep.markdown());
    write_machine(&dir, &[("scores file", a.scores.display().to_string())])
}

fn cmd_match_report(a: MatchReportArgs) -> Result<()> {
    let dir = out_dir(&a.out);
    let text = std::fs::read_to_string(&a.scores)
        .with_context(|| format!("cannot read {}", a.scores.display()))?;
    let input = matching::parse_match(&text)?;
    let rep = matching::analyze_match(
        &input,
        matching::MatchParams {
            threshold: a.threshold,
            target_fmr: a.target_fmr,
            confidence: stats::DEFAULT_CONFIDENCE,
        },
    )?;
    write(&dir, "match_report.md", &rep.markdown())?;
    write(&dir, "match_report.csv", &rep.csv())?;
    print!("{}", rep.markdown());
    write_machine(&dir, &[("scores file", a.scores.display().to_string())])
}

#[cfg(feature = "ort")]
async fn cmd_latency(a: LatencyArgs) -> Result<()> {
    let dir = out_dir(&a.out);
    let config = backend::load_config(a.profile.as_deref(), a.models_dir.as_deref())?;
    let threads_note = a
        .threads
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let rep = backend::latency(backend::LatencyArgs {
        frames: a.frames,
        iters: a.iters,
        warmup: a.warmup,
        threads: a.threads,
        config,
    })
    .await?;
    write(&dir, "latency.md", &rep.markdown())?;
    write(&dir, "latency_steady.csv", &rep.steady_csv())?;
    write(&dir, "latency_cold.csv", &rep.cold_csv())?;
    print!("{}", rep.markdown());
    write_machine(&dir, &[("intra_op threads swept", threads_note)])
}

#[cfg(not(feature = "ort"))]
async fn cmd_latency(_a: LatencyArgs) -> Result<()> {
    bail!(
        "`latency` needs the daemon ML backend, but doorman-bench was built without the `ort` feature. \
         Rebuild with `--features ort` (and provide the model files)"
    )
}

async fn cmd_record(a: RecordCliArgs) -> Result<()> {
    let target = match (a.user, a.impostor, a.attack) {
        (Some(user), None, None) => dataset::RecordTarget::Genuine {
            user,
            condition: a.condition.context("--user requires --condition")?,
        },
        (None, Some(identity), None) => dataset::RecordTarget::Impostor { identity },
        (None, None, Some(kind)) => dataset::RecordTarget::Attack { kind },
        _ => bail!("give exactly one of --user (with --condition), --impostor or --attack"),
    };
    record::run(record::RecordArgs {
        out: a.out,
        target,
        frames: a.frames,
        interval: Duration::from_millis(a.interval_ms),
        profile: a.profile,
        accept_notice: a.yes,
    })
    .await
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Cmd::Synth(a) => cmd_synth(a),
        Cmd::Accuracy(a) => cmd_accuracy(a).await,
        Cmd::Liveness(a) => cmd_liveness(a),
        Cmd::PadReport(a) => cmd_pad_report(a),
        Cmd::MatchReport(a) => cmd_match_report(a),
        Cmd::Latency(a) => cmd_latency(a).await,
        Cmd::Record(a) => cmd_record(a).await,
    }
}
