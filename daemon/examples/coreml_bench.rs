//! CoreML vs CPU honest benchmark for the doorman ONNX models.
//!
//! Builds an ORT `Session` for each of the three models (YuNet detector,
//! EdgeFace recognizer, MiniFASNet liveness) twice — once on the plain CPU EP
//! and once on the CoreML EP (compute units = ALL, MLProgram) — then:
//!
//!  1. Captures ORT's VERBOSE log during CoreML session creation to report how
//!     many nodes / subgraphs of each model were placed on the
//!     `CoreMLExecutionProvider` vs left on the CPU EP. This is the proof of
//!     ANE/GPU usage (CoreML partitions the graph and logs the assignment).
//!  2. Times N warm inferences per model per EP and reports mean per-stage
//!     latency + derived FPS.
//!  3. Checks correctness: runs the SAME real input through CPU and CoreML and
//!     reports the max-abs / cosine delta of the raw model outputs (CoreML may
//!     use fp16 internally).
//!
//! Run:
//!   cargo run --release --example coreml_bench --features backend-ort-coreml \
//!       -- [models_dir] [iters]
//!
//! models_dir defaults to ~/.local/share/doorman/models, iters to 100.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ort::execution_providers::coreml::{CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

/// A tracing layer that counts ORT log lines mentioning CoreML node placement.
#[derive(Clone, Default)]
struct PlacementCounter {
    inner: Arc<Mutex<PlacementStats>>,
}

#[derive(Default, Debug, Clone)]
struct PlacementStats {
    lines: Vec<String>,
}

impl<S> tracing_subscriber::Layer<S> for PlacementCounter
where
    S: tracing::Subscriber,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        struct Visitor<'a>(&'a mut String);
        impl tracing::field::Visit for Visitor<'_> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.push_str(&format!("{:?}", value));
                }
            }
        }
        let mut msg = String::new();
        event.record(&mut Visitor(&mut msg));
        let lower = msg.to_lowercase();
        if lower.contains("coreml")
            || lower.contains("number of nodes")
            || lower.contains("nodes placed")
            || lower.contains("nodes in the graph")
            || lower.contains("assigned to")
            || lower.contains("execution provider")
        {
            self.inner.lock().unwrap().lines.push(msg);
        }
    }
}

fn build_session(path: &Path, coreml: bool) -> ort::Result<Session> {
    let mut b = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?;
    if coreml {
        // VERBOSE so ORT prints CoreML GetCapability partition reports (node
        // counts placed on CoreML vs CPU) for every model, including ones that
        // partition cleanly with no warnings.
        b = b
            .with_log_level(ort::logging::LogLevel::Verbose)?
            .with_log_verbosity(1)?;
    }
    let b = if coreml {
        let cache = std::env::temp_dir().join("doorman_coreml_bench_cache");
        let _ = std::fs::create_dir_all(&cache);
        let ep = CoreMLExecutionProvider::default()
            .with_compute_units(CoreMLComputeUnits::All)
            .with_model_format(CoreMLModelFormat::MLProgram)
            .with_static_input_shapes(true)
            .with_profile_compute_plan(true)
            .with_model_cache_dir(cache.to_string_lossy().to_string());
        b.with_execution_providers([ep.build()])?
    } else {
        b
    };
    let bytes = std::fs::read(path).expect("read model");
    b.commit_from_memory(&bytes)
}

fn nchw(size: usize, fill: impl Fn(usize, usize, usize) -> f32) -> Vec<f32> {
    let mut v = vec![0f32; 3 * size * size];
    let n = size * size;
    for c in 0..3 {
        for y in 0..size {
            for x in 0..size {
                v[c * n + y * size + x] = fill(c, y, x);
            }
        }
    }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

struct ModelSpec {
    name: &'static str,
    file: &'static str,
    size: usize,
    input_name_hint: &'static str,
}

fn time_runs(session: &mut Session, input_shape: [usize; 4], data: &[f32], iters: usize) -> (f64, Vec<f32>) {
    // warmup (CoreML compiles on first run)
    let warm = Value::from_array((input_shape, data.to_vec())).unwrap();
    let out = session.run(ort::inputs![warm]).unwrap();
    let (_, first) = out[0].try_extract_tensor::<f32>().unwrap();
    let first: Vec<f32> = first.to_vec();
    drop(out);

    let start = Instant::now();
    for _ in 0..iters {
        let t = Value::from_array((input_shape, data.to_vec())).unwrap();
        let o = session.run(ort::inputs![t]).unwrap();
        std::hint::black_box(&o);
    }
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed / iters as f64 * 1000.0, first) // ms/iter, first-run output
}

fn main() {
    use tracing_subscriber::prelude::*;

    let args: Vec<String> = std::env::args().collect();
    let models_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join(".local/share/doorman/models")
        });
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    // Optional 3rd arg: only benchmark models whose name contains this substring
    // (case-insensitive). Lets each model be isolated in its own process so the
    // ORT-C++-stderr GetCapability report is unambiguous.
    let filter_name = args.get(3).map(|s| s.to_lowercase());

    let counter = PlacementCounter::default();
    // VERBOSE log level so ORT emits its graph-partition / node-placement lines.
    let filter = tracing_subscriber::EnvFilter::new("ort=debug,ort=trace");
    tracing_subscriber::registry()
        .with(counter.clone())
        .with(filter)
        .init();

    let specs = [
        ModelSpec { name: "YuNet detector", file: "face_detection_yunet_2023mar.onnx", size: 640, input_name_hint: "input" },
        ModelSpec { name: "EdgeFace recog", file: "edgeface_s.onnx", size: 112, input_name_hint: "input" },
        ModelSpec { name: "MiniFASNet live", file: "minifasnet_v2se.onnx", size: 128, input_name_hint: "input" },
    ];

    println!("models_dir = {}", models_dir.display());
    println!("iters      = {}\n", iters);

    println!("{:<18} {:>12} {:>12} {:>9} {:>12} {:>12}", "model", "CPU ms", "CoreML ms", "speedup", "cos(out)", "maxabsΔ");
    println!("{}", "-".repeat(80));

    for spec in &specs {
        if let Some(f) = &filter_name {
            if !spec.name.to_lowercase().contains(f.as_str()) {
                continue;
            }
        }
        let path = models_dir.join(spec.file);
        if !path.exists() {
            println!("{:<18} MISSING ({})", spec.name, spec.file);
            continue;
        }
        let shape = [1usize, 3, spec.size, spec.size];
        // Deterministic pseudo-real input (gradient + noise) in a sane range.
        let data = nchw(spec.size, |c, y, x| {
            let v = ((x * 7 + y * 13 + c * 29) % 256) as f32;
            // YuNet wants 0..255; recog/live want roughly [-1,1] or [0,1] —
            // value range doesn't change op placement or relative timing.
            if spec.size == 640 { v } else { v / 255.0 }
        });

        // ---- CPU ----
        let mut cpu = build_session(&path, false).expect("cpu session");
        let (cpu_ms, cpu_out) = time_runs(&mut cpu, shape, &data, iters);

        // ---- CoreML (capture placement log around session creation) ----
        // ORT prints its GetCapability partition report to stderr directly
        // (not via tracing), so emit a stderr banner to delimit each model.
        eprintln!("===== CoreML GetCapability for: {} =====", spec.name);
        counter.inner.lock().unwrap().lines.clear();
        let mut cml = build_session(&path, true).expect("coreml session");
        let placement_lines = counter.inner.lock().unwrap().lines.clone();
        let (cml_ms, cml_out) = time_runs(&mut cml, shape, &data, iters);

        let speedup = cpu_ms / cml_ms;
        let cos = cosine(&cpu_out, &cml_out);
        let maxabs = cpu_out
            .iter()
            .zip(&cml_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        println!(
            "{:<18} {:>12.3} {:>12.3} {:>8.2}x {:>12.5} {:>12.5}",
            spec.name, cpu_ms, cml_ms, speedup, cos, maxabs
        );
        let _ = spec.input_name_hint;

        // Node placement evidence for this model.
        if placement_lines.is_empty() {
            println!("    [placement] no CoreML partition log captured (raise ORT log verbosity)");
        } else {
            for l in placement_lines.iter().take(12) {
                println!("    [placement] {}", l);
            }
        }
        println!();
    }

    println!("FPS (single-stream, 1/latency):");
    println!("  (compute per-stage from the ms columns above: fps = 1000 / ms)");
}
