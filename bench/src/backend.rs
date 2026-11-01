//! Model-dependent runners (need the daemon's ONNX Runtime backend).
//!
//! Compiled only with the `ort` feature; `main.rs` provides explicit-error
//! stand-ins otherwise. NOTE: stage granularity is whatever the daemon's
//! `MLPipeline` exposes: face alignment happens inside `extract_embedding`, so
//! "align" cannot be timed separately and is reported as part of `embed`.

use crate::dataset;
use crate::latency_report::{summarize, ColdStart, LatencyReport, SteadyRow};
use crate::scores::ScoreRow;
use anyhow::{bail, Context, Result};
use doorman_shared::Config;
use doormand::ml::{DepthPadConfig, DetectorConfig, MLPipeline, RecognizerConfig};
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Stage labels used in reports.
pub const STAGE_DETECT: &str = "detect";
pub const STAGE_EMBED: &str = "embed (incl. align)";
pub const STAGE_LIVENESS: &str = "liveness (depth gate)";

pub fn load_config(profile: Option<&Path>, models_dir: Option<&Path>) -> Result<Config> {
    let mut config = match profile {
        Some(p) => {
            Config::load_from(p).with_context(|| format!("cannot load profile {}", p.display()))?
        }
        None => Config::default(),
    };
    if let Some(d) = models_dir {
        config.ml.models_dir = d.display().to_string();
    }
    Ok(config)
}

/// Explicit error listing every missing required model file.
/// The depth model is only required when `need_liveness`.
pub fn check_models(config: &Config, need_liveness: bool) -> Result<()> {
    let dir = PathBuf::from(&config.ml.models_dir);
    let mut required = vec![
        DetectorConfig::YUNET.model_file,
        RecognizerConfig::EDGEFACE.model_file,
    ];
    if need_liveness {
        required.push(DepthPadConfig::DEPTH_ANYTHING_V2.model_file);
    }
    let missing: Vec<String> = required
        .iter()
        .map(|f| dir.join(f))
        .filter(|p| !p.is_file())
        .map(|p| p.display().to_string())
        .collect();
    if !missing.is_empty() {
        bail!(
            "required model files are missing: {}. Set `--models-dir`/`--profile` to a directory containing them (see MODELS.md / scripts/fetch_models.sh)",
            missing.join(", ")
        );
    }
    Ok(())
}

fn load_images(dir: &Path) -> Result<Vec<DynamicImage>> {
    let paths = dataset::list_images(dir)?;
    if paths.is_empty() {
        bail!("no images (.jpg/.jpeg/.png) found under {}", dir.display());
    }
    paths
        .iter()
        .map(|p| image::open(p).with_context(|| format!("cannot decode {}", p.display())))
        .collect()
}

pub struct LatencyArgs {
    pub frames: PathBuf,
    pub iters: usize,
    pub warmup: usize,
    pub threads: Vec<i32>,
    pub config: Config,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

pub async fn latency(args: LatencyArgs) -> Result<LatencyReport> {
    if args.iters == 0 {
        bail!("--iters must be at least 1");
    }
    let mut cfg = args.config.clone();
    // The liveness stage is only meaningful with the gate enabled.
    cfg.authentication.liveness_enabled = true;
    let have_depth = check_models(&cfg, true).is_ok();
    if !have_depth {
        check_models(&cfg, false)?;
        cfg.authentication.liveness_enabled = false;
    }
    let images = load_images(&args.frames)?;

    let mut report = LatencyReport {
        images: images.len(),
        iters: args.iters,
        warmup: args.warmup,
        ..LatencyReport::default()
    };
    report.notes.push(
        "intra_op thread count is `ml.cpu_threads` (0 = backend default). ORT spinning, graph optimisation level and pre-optimised model files are not exposed by the daemon yet (TODO)."
            .to_string(),
    );
    if !have_depth {
        report.notes.push(format!(
            "depth model {} not found: liveness stage skipped.",
            DepthPadConfig::DEPTH_ANYTHING_V2.model_file
        ));
    }

    for &threads in &args.threads {
        let mut c = cfg.clone();
        c.ml.cpu_threads = threads;

        let t = Instant::now();
        let pipeline = MLPipeline::new(&c)
            .await
            .context("cannot create ML pipeline")?;
        let session_creation_ms = ms(t);
        if !pipeline.models_loaded() {
            bail!("ML backend reports not ready after creation (model failed to load; check logs)");
        }

        // Cold first inference on the first image that contains a face.
        let mut cold = ColdStart {
            threads,
            session_creation_ms,
            first_detect_ms: 0.0,
            first_embed_ms: None,
            first_liveness_ms: None,
        };
        let t = Instant::now();
        let first = pipeline.detect_face(&images[0]).await?;
        cold.first_detect_ms = ms(t);
        if let Some(face) = first {
            let t = Instant::now();
            pipeline.extract_embedding(&images[0], &face).await?;
            cold.first_embed_ms = Some(ms(t));
            if have_depth {
                let t = Instant::now();
                pipeline.liveness_gate(&images[0], &face).await?;
                cold.first_liveness_ms = Some(ms(t));
            }
        }
        report.cold.push(cold);

        let (mut detect, mut embed, mut live) = (Vec::new(), Vec::new(), Vec::new());
        let mut no_face = 0usize;
        for i in 0..(args.warmup + args.iters) {
            let img = &images[i % images.len()];
            let measuring = i >= args.warmup;
            let t = Instant::now();
            let face = pipeline.detect_face(img).await?;
            let d = ms(t);
            let Some(face) = face else {
                if measuring {
                    detect.push(d);
                    no_face += 1;
                }
                continue;
            };
            let t = Instant::now();
            pipeline.extract_embedding(img, &face).await?;
            let e = ms(t);
            let l = if have_depth {
                let t = Instant::now();
                pipeline.liveness_gate(img, &face).await?;
                Some(ms(t))
            } else {
                None
            };
            if measuring {
                detect.push(d);
                embed.push(e);
                if let Some(l) = l {
                    live.push(l);
                }
            }
        }
        if no_face > 0 {
            report.notes.push(format!(
                "threads={threads}: {no_face} of {} measured iterations found no face; embed/liveness only run on detections.",
                args.iters
            ));
        }
        for (stage, v) in [
            (STAGE_DETECT, &detect),
            (STAGE_EMBED, &embed),
            (STAGE_LIVENESS, &live),
        ] {
            report.steady.push(SteadyRow {
                threads,
                stage,
                stats: summarize(v),
            });
        }
    }
    Ok(report)
}

#[derive(Debug, Default)]
pub struct EmbedStats {
    pub enroll_images: usize,
    pub enroll_used: usize,
    pub genuine_images: usize,
    pub genuine_no_face: usize,
    pub impostor_images: usize,
    pub impostor_no_face: usize,
}

async fn embed_image(pipeline: &MLPipeline, path: &Path) -> Result<Option<Vec<f32>>> {
    let img = image::open(path).with_context(|| format!("cannot decode {}", path.display()))?;
    match pipeline.detect_face(&img).await? {
        Some(face) => Ok(Some(pipeline.extract_embedding(&img, &face).await?)),
        None => Ok(None),
    }
}

/// Embed enrolment, genuine and impostor images with the daemon backend and
/// score each probe as the maximum cosine similarity against the enrolment
/// embeddings (the gallery rule of the daemon's auth path). Images without a
/// detected face are skipped and counted.
pub async fn score_directories(
    config: &Config,
    enroll: &Path,
    genuine: &Path,
    impostor: &Path,
) -> Result<(Vec<ScoreRow>, EmbedStats)> {
    check_models(config, false)?;
    let pipeline = MLPipeline::new(config)
        .await
        .context("cannot create ML pipeline")?;
    if !pipeline.models_loaded() {
        bail!("ML backend reports not ready (model failed to load; check logs)");
    }
    let mut stats = EmbedStats::default();

    let enroll_paths = dataset::list_images(enroll)?;
    stats.enroll_images = enroll_paths.len();
    let mut gallery = Vec::new();
    for p in &enroll_paths {
        if let Some(e) = embed_image(&pipeline, p).await? {
            gallery.push(e);
        }
    }
    stats.enroll_used = gallery.len();
    if gallery.is_empty() {
        bail!(
            "no usable enrolment embeddings: {} images under {}, none with a detected face",
            stats.enroll_images,
            enroll.display()
        );
    }
    let score = |e: &[f32]| -> f64 {
        gallery
            .iter()
            .map(|g| doormand::ml::cosine_similarity(e, g) as f64)
            .fold(f64::NEG_INFINITY, f64::max)
    };

    let mut rows = Vec::new();
    for (cond, paths) in dataset::list_images_by_condition(genuine)? {
        for p in paths {
            stats.genuine_images += 1;
            match embed_image(&pipeline, &p).await? {
                Some(e) => rows.push(ScoreRow {
                    positive: true,
                    score: score(&e),
                    condition: Some(cond.clone()),
                }),
                None => stats.genuine_no_face += 1,
            }
        }
    }
    for p in dataset::list_images(impostor)? {
        stats.impostor_images += 1;
        match embed_image(&pipeline, &p).await? {
            Some(e) => rows.push(ScoreRow {
                positive: false,
                score: score(&e),
                condition: None,
            }),
            None => stats.impostor_no_face += 1,
        }
    }
    Ok((rows, stats))
}
