//! Guided capture of a labelled dataset from the real camera.
//!
//! Needs a camera, so only the path/layout logic (in `dataset.rs`) is unit
//! tested. The session refuses to run on the mock camera: synthetic frames in
//! a "real" dataset would silently poison every later measurement.

use crate::dataset::{self, RecordTarget};
use anyhow::{bail, Context, Result};
use doorman_shared::Config;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Shown before anything is captured.
pub const CONSENT_NOTICE: &str = "\
CONSENT NOTICE
  This command captures face images from the camera and stores them as
  files in the output directory below. Face images are biometric personal
  data. Only record people who have given explicit consent (yourself, or
  people who agreed to be recorded), keep the files out of version control
  and shared storage, and delete them when no longer needed.
";

/// Seconds shown in the get-ready countdown.
pub const COUNTDOWN_SECS: u64 = 3;

pub struct RecordArgs {
    pub out: PathBuf,
    pub target: RecordTarget,
    pub frames: usize,
    pub interval: Duration,
    pub profile: Option<PathBuf>,
    pub accept_notice: bool,
}

fn confirm(accept_notice: bool, dir: &Path) -> Result<()> {
    println!("{CONSENT_NOTICE}");
    println!("  output directory: {}\n", dir.display());
    if accept_notice {
        println!("  --yes given: notice accepted non-interactively.\n");
        return Ok(());
    }
    println!("Type `yes` to confirm that everyone in front of the camera consents:");
    let mut line = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut line)
        .context("cannot read confirmation from stdin")?;
    if line.trim() != "yes" {
        bail!("consent not confirmed; nothing was recorded");
    }
    Ok(())
}

pub async fn run(args: RecordArgs) -> Result<()> {
    // Validate the destination before touching the camera.
    dataset::ensure_outside_git_repo(&args.out)?;
    let dir = args.target.dir(&args.out)?;
    dataset::ensure_outside_git_repo(&dir)?;
    confirm(args.accept_notice, &dir)?;

    let config = match &args.profile {
        Some(p) => Config::load_from(p).with_context(|| format!("cannot load {}", p.display()))?,
        None => Config::default(),
    };
    let mut camera = doormand::camera::Camera::new_with_config(&config)
        .await
        .context("cannot open camera")?;
    if camera.is_mock() {
        bail!(
            "no real camera could be opened (fell back to the synthetic mock camera); \
             nothing was recorded. Check the camera device and the `[camera]` profile."
        );
    }

    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let start = dataset::next_index(&dir);
    for s in (1..=COUNTDOWN_SECS).rev() {
        println!("Recording starts in {s}...");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    for i in 0..args.frames {
        let frame = camera.capture_frame().context("camera capture failed")?;
        let path = dir.join(dataset::frame_file_name(start + i));
        frame
            .to_rgb8()
            .save(&path)
            .with_context(|| format!("cannot write {}", path.display()))?;
        println!("saved {} ({}/{})", path.display(), i + 1, args.frames);
        tokio::time::sleep(args.interval).await;
    }
    println!("Done: {} frames in {}", args.frames, dir.display());
    Ok(())
}
