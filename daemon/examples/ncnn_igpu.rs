//! PHASE 1 proof: load converted YuNet + Depth-Anything-V2 ncnn models and run
//! them on the integrated Radeon 780M (RADV PHOENIX) via Vulkan — NOT the dGPU.
//!
//! What it proves:
//!   * the C-FFI shim + Rust wrapper load .param/.bin and run inference,
//!   * the integrated device is selected (printed by name + index), and the
//!     discrete GPU is enumerated but NOT chosen,
//!   * a tight inference loop drives the iGPU (watch card2 gpu_busy_percent),
//!   * latency per model on the iGPU is reported.
//!
//! For correctness vs onnxruntime-CPU, this dumps the exact input and the ncnn
//! output to raw f32 files in $DOORMAN_NCNN_OUT (default /tmp/ncnn_probe) so the
//! companion python script can run ORT on the SAME input and compute cosine.
//!
//! Run:
//!   DOORMAN_NCNN_DIR=/tmp/ncnn_install \
//!   cargo run --release --no-default-features \
//!     --features backend-ncnn,camera-v4l2,camera-ffmpeg --example ncnn_igpu

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use doormand::ml::ncnn_backend::{DeviceSelector, GpuInstance, GpuKind, NcnnNet};

fn write_raw_f32(path: &PathBuf, data: &[f32]) {
    let mut f = std::fs::File::create(path).expect("create raw f32 file");
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes).expect("write raw f32");
}

/// Deterministic pseudo-random input in [0,1) so each run is reproducible and
/// the dumped buffer matches what the python ORT reference reads back.
fn make_input(c: i32, h: i32, w: i32) -> Vec<f32> {
    let n = (c * h * w) as usize;
    let mut v = Vec::with_capacity(n);
    // simple xorshift -> uniform [0,1)
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push((s >> 11) as f32 / (1u64 << 53) as f32);
    }
    v
}

struct ModelSpec {
    name: &'static str,
    param: &'static str,
    bin: &'static str,
    shape: [i32; 3],
    output: &'static str,
}

fn main() -> anyhow::Result<()> {
    let models = std::env::var("DOORMAN_NCNN_MODELS")
        .unwrap_or_else(|_| "/tmp/ncnn_probe/models".to_string());
    let out_dir = PathBuf::from(
        std::env::var("DOORMAN_NCNN_OUT").unwrap_or_else(|_| "/tmp/ncnn_probe".to_string()),
    );

    // ---- enumerate devices + prove iGPU selection -------------------------
    let inst = GpuInstance::create()?;
    println!("== Vulkan devices ncnn sees ==");
    let mut discrete_idx: Option<i32> = None;
    let mut integrated_idx: Option<i32> = None;
    for d in inst.devices()? {
        println!("  [{}] {:?}  '{}'", d.index, d.kind, d.name);
        match d.kind {
            GpuKind::Discrete => discrete_idx = Some(d.index),
            GpuKind::Integrated if integrated_idx.is_none() => integrated_idx = Some(d.index),
            _ => {}
        }
    }
    drop(inst);

    let specs = [
        ModelSpec {
            name: "yunet",
            param: "yunet.ncnn.param",
            bin: "yunet.ncnn.bin",
            shape: [3, 640, 640],
            output: "out0",
        },
        ModelSpec {
            name: "depth",
            param: "depth.ncnn.param",
            bin: "depth.ncnn.bin",
            shape: [3, 518, 518],
            output: "out0",
        },
    ];

    let mut all_ok = true;
    for spec in &specs {
        println!("\n===== {} on iGPU (Vulkan) =====", spec.name.to_uppercase());
        let param = format!("{models}/{}", spec.param);
        let bin = format!("{models}/{}", spec.bin);

        // Select the integrated RADV PHOENIX device by name (never the dGPU).
        let net = NcnnNet::new(&param, &bin, DeviceSelector::NameSubstring("RADV PHOENIX".into()))?;
        println!(
            "  SELECTED device index={} name='{}'",
            net.device_index(),
            net.device_name()
        );
        if let Some(di) = discrete_idx {
            assert_ne!(
                net.device_index(),
                di,
                "FATAL: selected the discrete GPU (index {di})"
            );
            println!("  (discrete GPU index {di} present but NOT selected — good)");
        }

        let [c, h, w] = spec.shape;
        let input = make_input(c, h, w);

        // Warmup + correctness output.
        let out = net.infer("in0", &input, spec.shape, spec.output)?;
        let (data, oshape) = &out[spec.output];
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let (mn, mx) = data.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
        println!(
            "  output '{}' shape=[{},{},{}] len={} mean={:.5} min={:.4} max={:.4}",
            spec.output, oshape[0], oshape[1], oshape[2], data.len(), mean, mn, mx
        );

        // Dump input + output for the python ORT cosine check.
        write_raw_f32(&out_dir.join(format!("{}_rust_in.f32", spec.name)), &input);
        write_raw_f32(&out_dir.join(format!("{}_rust_out.f32", spec.name)), data);
        println!(
            "  wrote {0}/{1}_rust_in.f32 + {0}/{1}_rust_out.f32",
            out_dir.display(),
            spec.name
        );

        // ---- timed loop on the iGPU (watch card2 gpu_busy_percent) --------
        let reps = if spec.name == "depth" { 30 } else { 60 };
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = net.infer("in0", &input, spec.shape, spec.output)?;
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        println!("  iGPU latency = {dt:.2} ms/inference  ({reps} reps)");

        if data.is_empty() {
            all_ok = false;
        }
    }

    println!("\n== summary ==");
    println!("  integrated device idx = {:?}", integrated_idx);
    println!("  discrete device idx   = {:?} (never selected)", discrete_idx);
    if all_ok {
        println!("  ALL MODELS RAN ON iGPU OK");
    } else {
        anyhow::bail!("one or more models produced empty output");
    }
    Ok(())
}
