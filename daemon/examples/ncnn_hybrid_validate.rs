//! THROWAWAY security-validation harness (Phase 4): prove the ncnn-Vulkan iGPU
//! depth path of `NcnnHybridBackend` is correct + safe before it is trusted as
//! the FATAL anti-spoof gate.
//!
//! For each genuine / screen-spoof image it:
//!   1. detects the face (CPU ORT detector — shared by both backends),
//!   2. runs `hybrid.depth_relief` (ncnn depth on the iGPU via Vulkan), and
//!   3. runs the reference CPU `ort.depth_relief` (Depth-Anything-V2 on ORT-CPU),
//! then reports the per-class relief tables, the 0.05-threshold separation
//! verdict, ncnn-vs-ORT agreement, latency, and CPU-offload estimate.
//!
//! Build/run (hybrid features, static libncnn at /tmp/ncnn_install):
//!   DOORMAN_NCNN_DIR=/tmp/ncnn_install \
//!   cargo run --release --no-default-features \
//!     --features backend-ort,backend-ncnn,camera-v4l2,camera-ffmpeg \
//!     --example ncnn_hybrid_validate
//!
//! Env overrides:
//!   DOORMAN_VAL_MODELS   test models dir (ONNX + depth.ncnn.*), default
//!                        /tmp/ncnn_hybrid_validate/models
//!   DOORMAN_VAL_GENUINE  genuine images dir, default ~/datasets/insitu/genuine
//!   DOORMAN_VAL_SPOOF    comma-separated spoof dirs, default
//!                        ~/datasets/insitu/attack_screen,~/datasets/insitu/attack_screen2

#![cfg(all(feature = "_ort", feature = "backend-ncnn"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use doorman_shared::Config;
use doormand::ml::ncnn_hybrid::NcnnHybridBackend;
use doormand::ml::ort_backend::OrtBackend;
use doormand::ml::MLBackend;

struct Stats {
    n: usize,
    mean: f64,
    min: f64,
    max: f64,
}

fn stats(v: &[f64]) -> Stats {
    if v.is_empty() {
        return Stats {
            n: 0,
            mean: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        };
    }
    let n = v.len();
    let sum: f64 = v.iter().sum();
    let mean = sum / n as f64;
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Stats { n, mean, min, max }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return f64::NAN;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma = a.iter().sum::<f64>() / n as f64;
    let mb = b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let xa = a[i] - ma;
        let xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da == 0.0 || db == 0.0 {
        return f64::NAN;
    }
    num / (da.sqrt() * db.sqrt())
}

fn list_images(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                let ext = ext.to_ascii_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp") {
                    v.push(p);
                }
            }
        }
    }
    v.sort();
    v
}

fn gpu_busy(card: &str) -> i64 {
    std::fs::read_to_string(format!("/sys/class/drm/{card}/device/gpu_busy_percent"))
        .ok()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(-1)
}

struct ClassResult {
    name: String,
    ncnn: Vec<f64>,
    ort: Vec<f64>,
    no_face: usize,
    ncnn_lat_ms: Vec<f64>,
    ort_lat_ms: Vec<f64>,
}

async fn run_class(
    name: &str,
    dirs: &[PathBuf],
    hybrid: &NcnnHybridBackend,
    reference: &OrtBackend,
) -> ClassResult {
    let mut res = ClassResult {
        name: name.to_string(),
        ncnn: Vec::new(),
        ort: Vec::new(),
        no_face: 0,
        ncnn_lat_ms: Vec::new(),
        ort_lat_ms: Vec::new(),
    };

    let mut images: Vec<PathBuf> = Vec::new();
    for d in dirs {
        images.extend(list_images(d));
    }
    println!(
        "\n== class '{}' : {} image(s) from {:?} ==",
        name,
        images.len(),
        dirs
    );

    for path in &images {
        let img = match image::open(path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  skip {}: decode error {}", path.display(), e);
                continue;
            }
        };

        // Detect face on the CPU ORT detector (shared by both paths).
        let face = match reference.detect_face(&img).await {
            Ok(Some(f)) => f,
            Ok(None) => {
                res.no_face += 1;
                continue;
            }
            Err(e) => {
                eprintln!("  detect error {}: {}", path.display(), e);
                res.no_face += 1;
                continue;
            }
        };

        // ncnn iGPU depth.
        let t0 = Instant::now();
        let ncnn_relief = match hybrid.depth_relief(&img, &face).await {
            Ok(Some(r)) => r,
            Ok(None) => {
                eprintln!(
                    "  ncnn depth returned None for {} (model not loaded?)",
                    path.display()
                );
                continue;
            }
            Err(e) => {
                eprintln!("  ncnn depth error {}: {}", path.display(), e);
                continue;
            }
        };
        let ncnn_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Reference CPU ORT depth.
        let t1 = Instant::now();
        let ort_relief = match reference.depth_relief(&img, &face).await {
            Ok(Some(r)) => r,
            Ok(None) => {
                eprintln!("  ORT depth returned None for {}", path.display());
                continue;
            }
            Err(e) => {
                eprintln!("  ORT depth error {}: {}", path.display(), e);
                continue;
            }
        };
        let ort_ms = t1.elapsed().as_secs_f64() * 1000.0;

        res.ncnn.push(ncnn_relief as f64);
        res.ort.push(ort_relief as f64);
        res.ncnn_lat_ms.push(ncnn_ms);
        res.ort_lat_ms.push(ort_ms);

        println!(
            "  {:<28} ncnn={:.4}  ort={:.4}  d={:+.4}  (ncnn {:.0}ms / ort {:.0}ms)",
            path.file_name().unwrap().to_string_lossy(),
            ncnn_relief,
            ort_relief,
            ncnn_relief - ort_relief,
            ncnn_ms,
            ort_ms
        );
    }
    res
}

fn print_stats(label: &str, s: &Stats) {
    println!(
        "    {:<22} n={:<3} mean={:.4} min={:.4} max={:.4}",
        label, s.n, s.mean, s.min, s.max
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    const THRESH: f64 = 0.05;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    let models_dir = std::env::var("DOORMAN_VAL_MODELS")
        .unwrap_or_else(|_| "/tmp/ncnn_hybrid_validate/models".into());
    let genuine_dir = std::env::var("DOORMAN_VAL_GENUINE")
        .unwrap_or_else(|_| format!("{home}/datasets/insitu/genuine"));
    let spoof_dirs_env = std::env::var("DOORMAN_VAL_SPOOF").unwrap_or_else(|_| {
        format!("{home}/datasets/insitu/attack_screen,{home}/datasets/insitu/attack_screen2")
    });
    let spoof_dirs: Vec<PathBuf> = spoof_dirs_env.split(',').map(PathBuf::from).collect();

    println!("== ncnn-hybrid depth-PAD validation ==");
    println!("  models_dir = {models_dir}");
    println!("  genuine    = {genuine_dir}");
    println!("  spoof      = {:?}", spoof_dirs);

    // CPU device config; ml.backend=ncnn selects the hybrid wiring, vulkan_device
    // pins the iGPU by name (never the dGPU).
    let mut config = Config::default();
    config.ml.models_dir = models_dir.clone();
    config.ml.backend = "ncnn".to_string();
    config.ml.device = "cpu".to_string();
    config.ml.vulkan_device = Some("RADV PHOENIX".to_string());

    let models_path = PathBuf::from(&models_dir);

    // Hybrid backend: CPU detect/embed/liveness + ncnn iGPU depth.
    let hybrid = NcnnHybridBackend::new(&models_path, &config)?;
    println!("  hybrid backend: {}", hybrid.name());

    // Reference: pure-CPU ORT backend (in-process CPU depth).
    let mut ref_config = config.clone();
    ref_config.ml.backend = "ort".to_string();
    let reference = OrtBackend::new(&models_path, &ref_config)?;
    println!("  reference backend: {}", reference.name());

    // iGPU proof: snapshot card2 (iGPU) and card1 (dGPU, must stay idle) busy %.
    let igpu_before = gpu_busy("card2");
    let dgpu_before = gpu_busy("card1");
    println!(
        "  gpu_busy before: iGPU(card2)={}% dGPU(card1)={}%",
        igpu_before, dgpu_before
    );

    let genuine = run_class(
        "genuine",
        &[PathBuf::from(&genuine_dir)],
        &hybrid,
        &reference,
    )
    .await;

    // Sample the iGPU mid-run (after the genuine batch, which drove many ncnn calls).
    let igpu_mid = gpu_busy("card2");
    let dgpu_mid = gpu_busy("card1");

    let spoof = run_class("screen-spoof", &spoof_dirs, &hybrid, &reference).await;

    let igpu_after = gpu_busy("card2");
    let dgpu_after = gpu_busy("card1");

    // ---- report ----------------------------------------------------------
    println!("\n========================= RESULTS =========================");

    for (cls, c) in [("GENUINE", &genuine), ("SCREEN-SPOOF", &spoof)] {
        println!("\n[{}]  (no-face skipped: {})", cls, c.no_face);
        print_stats("ncnn-iGPU relief", &stats(&c.ncnn));
        print_stats("ORT-CPU  relief", &stats(&c.ort));
        let mut diffs: Vec<f64> = c
            .ncnn
            .iter()
            .zip(c.ort.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        print_stats("|ncnn - ORT|", &stats(&diffs));
        println!("    pearson(ncnn,ORT) = {:.4}", pearson(&c.ncnn, &c.ort));
        print_stats("ncnn latency (ms)", &stats(&c.ncnn_lat_ms));
        print_stats("ORT  latency (ms)", &stats(&c.ort_lat_ms));
    }

    // ---- threshold verdict (ncnn-iGPU path) ------------------------------
    let g = stats(&genuine.ncnn);
    let s = stats(&spoof.ncnn);
    println!("\n[THRESHOLD 0.05 verdict — ncnn-iGPU path]");
    let genuine_ge: Vec<f64> = genuine
        .ncnn
        .iter()
        .cloned()
        .filter(|&x| x >= THRESH)
        .collect();
    let spoof_lt: Vec<f64> = spoof.ncnn.iter().cloned().filter(|&x| x < THRESH).collect();
    let genuine_pass = if g.n > 0 {
        genuine_ge.len() as f64 / g.n as f64
    } else {
        f64::NAN
    };
    let spoof_pass = if s.n > 0 {
        spoof_lt.len() as f64 / s.n as f64
    } else {
        f64::NAN
    };
    println!(
        "  genuine >= 0.05 : {}/{} ({:.1}%)   [min genuine = {:.4}]",
        genuine_ge.len(),
        g.n,
        genuine_pass * 100.0,
        g.min
    );
    println!(
        "  spoof   <  0.05 : {}/{} ({:.1}%)   [max spoof   = {:.4}]",
        spoof_lt.len(),
        s.n,
        spoof_pass * 100.0,
        s.max
    );
    let clean_sep = g.n > 0 && s.n > 0 && g.min >= THRESH && s.max < THRESH;
    if clean_sep {
        println!(
            "  VERDICT: CLEAN SEPARATION at 0.05 on iGPU. margin = min_genuine({:.4}) - max_spoof({:.4}) = {:.4}",
            g.min, s.max, g.min - s.max
        );
    } else {
        // Suggest a midpoint threshold if the classes are still separable.
        println!("  VERDICT: 0.05 does NOT cleanly separate on the iGPU path.");
        if g.n > 0 && s.n > 0 && g.min > s.max {
            let suggested = (g.min + s.max) / 2.0;
            println!(
                "  classes ARE separable: max_spoof={:.4} < min_genuine={:.4}. Suggested recalibrated threshold = {:.4}",
                s.max, g.min, suggested
            );
        } else {
            println!(
                "  classes OVERLAP on iGPU (max_spoof={:.4} >= min_genuine={:.4}) — NOT safe; depth range likely compressed.",
                s.max, g.min
            );
        }
    }

    // ---- iGPU use proof --------------------------------------------------
    println!("\n[iGPU-use proof]");
    println!(
        "  gpu_busy%  iGPU(card2): before={} mid={} after={}",
        igpu_before, igpu_mid, igpu_after
    );
    println!(
        "  gpu_busy%  dGPU(card1): before={} mid={} after={} (MUST stay idle)",
        dgpu_before, dgpu_mid, dgpu_after
    );

    // ---- latency / offload summary --------------------------------------
    let mut all_ncnn: Vec<f64> = genuine.ncnn_lat_ms.clone();
    all_ncnn.extend(spoof.ncnn_lat_ms.clone());
    let mut all_ort: Vec<f64> = genuine.ort_lat_ms.clone();
    all_ort.extend(spoof.ort_lat_ms.clone());
    let ns = stats(&all_ncnn);
    let os = stats(&all_ort);
    println!("\n[latency / CPU-offload]");
    print_stats("ncnn-iGPU depth (ms)", &ns);
    print_stats("ORT-CPU  depth (ms)", &os);
    if ns.n > 0 && os.n > 0 {
        println!(
            "  per-auth CPU depth time moved OFF the daemon CPU: ~{:.0} ms (ORT-CPU mean) -> {:.0} ms iGPU wall, CPU now ~idle during depth.",
            os.mean, ns.mean
        );
    }

    println!("\n== done ==");
    Ok(())
}
