//! Synthetic data for CI and smoke tests: seeded score CSVs and random textured
//! images. No real faces, no dataset content. The distributions only need to be
//! plausible and well separated enough to exercise the metrics.

use image::{ImageBuffer, Rgb};
use std::fmt::Write;
use std::path::Path;

/// (condition, genuine mean) -- lower means simulate harder lighting.
const GENUINE_CONDITIONS: [(&str, f64); 4] = [
    ("day", 0.72),
    ("evening", 0.66),
    ("dark", 0.58),
    ("backlit", 0.52),
];
const GENUINE_STD: f64 = 0.08;
const IMPOSTOR_MEAN: f64 = 0.05;
const IMPOSTOR_STD: f64 = 0.10;
const BONA_FIDE_MEAN: f64 = 0.15;
const BONA_FIDE_STD: f64 = 0.05;
const ATTACK_MEAN: f64 = 0.02;
const ATTACK_STD: f64 = 0.015;

/// (attack species, accepted-relief mean). `screen_replay` is deliberately the
/// weakest so the worst-species logic has something to find. Species of the
/// protocol that are absent here (e.g. `paper_mask`) stay "untested".
const PAD_SPECIES: [(&str, f64); 4] = [
    ("print_matte", 0.01),
    ("print_glossy", 0.02),
    ("phone_replay", 0.04),
    ("screen_replay", 0.06),
];
/// Sessions the synthetic PAD rows are spread over.
const PAD_SESSIONS: usize = 3;
/// Every n-th synthetic PAD/match row is a non-response / failure to acquire.
const NON_RESPONSE_EVERY: usize = 50;
/// Subject id of the synthetic enrolled person.
const OWNER_SUBJECT: &str = "owner";

/// Image side lengths for synthetic textures.
pub const IMAGE_WIDTH: u32 = 320;
pub const IMAGE_HEIGHT: u32 = 240;

#[derive(Debug, Clone)]
pub struct SynthParams {
    pub seed: u64,
    pub genuine: usize,
    pub impostor: usize,
    pub bona_fide: usize,
    pub attack: usize,
    pub images: usize,
    /// Distinct impostor subjects in `match_scores.csv`.
    pub subjects: usize,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            seed: 0,
            genuine: 2000,
            // >= 1/1e-4 so that FAR 1e-4 is resolvable.
            impostor: 20_000,
            bona_fide: 400,
            attack: 400,
            images: 4,
            subjects: 50,
        }
    }
}

/// SplitMix64: tiny, portable, fully specified.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (0, 1].
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// `label,score,condition` accuracy scores.
pub fn accuracy_csv(p: &SynthParams) -> String {
    let mut rng = Rng::new(p.seed);
    let mut s = String::from("label,score,condition\n");
    for i in 0..p.genuine {
        let (cond, mean) = GENUINE_CONDITIONS[i % GENUINE_CONDITIONS.len()];
        let v = (mean + GENUINE_STD * rng.normal()).clamp(-1.0, 1.0);
        let _ = writeln!(s, "genuine,{v:.6},{cond}");
    }
    for _ in 0..p.impostor {
        let v = (IMPOSTOR_MEAN + IMPOSTOR_STD * rng.normal()).clamp(-1.0, 1.0);
        let _ = writeln!(s, "impostor,{v:.6},");
    }
    s
}

/// `label,score` liveness scores (relief values, non-negative).
pub fn liveness_csv(p: &SynthParams) -> String {
    // Offset the seed so accuracy and liveness streams are independent.
    let mut rng = Rng::new(p.seed ^ 0xA5A5_A5A5_A5A5_A5A5);
    let mut s = String::from("label,score\n");
    for _ in 0..p.bona_fide {
        let v = (BONA_FIDE_MEAN + BONA_FIDE_STD * rng.normal()).max(0.0);
        let _ = writeln!(s, "bona_fide,{v:.6}");
    }
    for _ in 0..p.attack {
        let v = (ATTACK_MEAN + ATTACK_STD * rng.normal()).max(0.0);
        let _ = writeln!(s, "attack,{v:.6}");
    }
    s
}

/// `label,species,score,response,session,subject` PAD scores: bona fide rows
/// (species `none`) and attacks cycling over [`PAD_SPECIES`]. Non-responses
/// have an empty score.
pub fn pad_csv(p: &SynthParams) -> String {
    let mut rng = Rng::new(p.seed ^ 0x1357_9BDF_1357_9BDF);
    let mut s = String::from("label,species,score,response,session,subject\n");
    for i in 0..p.bona_fide {
        let session = i % PAD_SESSIONS;
        if i % NON_RESPONSE_EVERY == NON_RESPONSE_EVERY - 1 {
            let _ = writeln!(s, "bona_fide,none,,no_face,s{session},{OWNER_SUBJECT}");
            continue;
        }
        let v = (BONA_FIDE_MEAN + BONA_FIDE_STD * rng.normal()).max(0.0);
        let _ = writeln!(s, "bona_fide,none,{v:.6},ok,s{session},{OWNER_SUBJECT}");
    }
    for i in 0..p.attack {
        let (species, mean) = PAD_SPECIES[i % PAD_SPECIES.len()];
        let session = i % PAD_SESSIONS;
        if i % NON_RESPONSE_EVERY == NON_RESPONSE_EVERY - 1 {
            let _ = writeln!(s, "attack,{species},,timeout,s{session},{OWNER_SUBJECT}");
            continue;
        }
        let v = (mean + ATTACK_STD * rng.normal()).max(0.0);
        let _ = writeln!(s, "attack,{species},{v:.6},ok,s{session},{OWNER_SUBJECT}");
    }
    s
}

/// `label,score,subject,response` comparison scores: genuine rows for the owner
/// and impostor rows spread round-robin over `p.subjects` impostor subjects.
pub fn match_csv(p: &SynthParams) -> String {
    let mut rng = Rng::new(p.seed ^ 0x2468_ACE0_2468_ACE0);
    let mut s = String::from("label,score,subject,response\n");
    for i in 0..p.genuine {
        if i % (NON_RESPONSE_EVERY * 4) == NON_RESPONSE_EVERY * 4 - 1 {
            let _ = writeln!(s, "genuine,,{OWNER_SUBJECT},fta");
            continue;
        }
        let (_, mean) = GENUINE_CONDITIONS[i % GENUINE_CONDITIONS.len()];
        let v = (mean + GENUINE_STD * rng.normal()).clamp(-1.0, 1.0);
        let _ = writeln!(s, "genuine,{v:.6},{OWNER_SUBJECT},ok");
    }
    for i in 0..p.impostor {
        let subject = i % p.subjects;
        if i % (NON_RESPONSE_EVERY * 4) == NON_RESPONSE_EVERY * 4 - 1 {
            let _ = writeln!(s, "impostor,,imp{subject:03},fta");
            continue;
        }
        let v = (IMPOSTOR_MEAN + IMPOSTOR_STD * rng.normal()).clamp(-1.0, 1.0);
        let _ = writeln!(s, "impostor,{v:.6},imp{subject:03},ok");
    }
    s
}

/// Random textured PNGs: a few random sinusoid gratings plus noise. Not faces.
pub fn write_images(dir: &Path, p: &SynthParams) -> Result<Vec<String>, SynthError> {
    std::fs::create_dir_all(dir).map_err(|source| SynthError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    let mut rng = Rng::new(p.seed ^ 0x5A5A_5A5A_5A5A_5A5A);
    let mut names = Vec::new();
    for n in 0..p.images {
        let waves: Vec<(f64, f64, f64, [f64; 3])> = (0..3)
            .map(|_| {
                (
                    rng.unit() * 0.2,
                    rng.unit() * 0.2,
                    rng.unit() * std::f64::consts::TAU,
                    [rng.unit(), rng.unit(), rng.unit()],
                )
            })
            .collect();
        let mut img = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(IMAGE_WIDTH, IMAGE_HEIGHT);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let mut c = [0.5f64; 3];
            for (fx, fy, ph, w) in &waves {
                let v = (fx * x as f64 + fy * y as f64 + ph).sin() * 0.25;
                for (ch, wk) in c.iter_mut().zip(w) {
                    *ch += v * wk;
                }
            }
            for ch in c.iter_mut() {
                *ch += (rng.unit() - 0.5) * 0.1;
            }
            *px = Rgb(c.map(|ch| (ch.clamp(0.0, 1.0) * 255.0) as u8));
        }
        let name = format!("synth_{n:03}.png");
        let path = dir.join(&name);
        img.save(&path).map_err(|source| SynthError::Image {
            path: path.display().to_string(),
            source,
        })?;
        names.push(name);
    }
    Ok(names)
}

#[derive(Debug, thiserror::Error)]
pub enum SynthError {
    #[error("cannot write {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("cannot encode {path}: {source}")]
    Image {
        path: String,
        source: image::ImageError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_bytes_different_seed_differs() {
        let p = SynthParams {
            seed: 7,
            ..SynthParams::default()
        };
        assert_eq!(accuracy_csv(&p), accuracy_csv(&p));
        assert_eq!(liveness_csv(&p), liveness_csv(&p));
        assert_eq!(pad_csv(&p), pad_csv(&p));
        assert_eq!(match_csv(&p), match_csv(&p));
        let q = SynthParams {
            seed: 8,
            ..SynthParams::default()
        };
        assert_ne!(accuracy_csv(&p), accuracy_csv(&q));
        assert_ne!(pad_csv(&p), pad_csv(&q));
        assert_ne!(match_csv(&p), match_csv(&q));
    }

    #[test]
    fn pad_and_match_csv_shape() {
        let p = SynthParams {
            bona_fide: 100,
            attack: 200,
            genuine: 400,
            impostor: 1000,
            subjects: 7,
            ..SynthParams::default()
        };
        let pad = pad_csv(&p);
        assert_eq!(pad.lines().count(), 1 + 100 + 200);
        // Rows 49 and 99 of each block are non-responses.
        assert_eq!(pad.lines().filter(|l| l.contains(",no_face,")).count(), 2);
        assert_eq!(pad.lines().filter(|l| l.contains(",timeout,")).count(), 4);
        // 200 attacks over 4 species: 50 each.
        assert_eq!(
            pad.lines()
                .filter(|l| l.starts_with("attack,screen_replay,"))
                .count(),
            50
        );
        let m = match_csv(&p);
        assert_eq!(m.lines().count(), 1 + 400 + 1000);
        let subjects: std::collections::BTreeSet<&str> = m
            .lines()
            .filter(|l| l.starts_with("impostor,"))
            .map(|l| l.split(',').nth(2).unwrap())
            .collect();
        assert_eq!(subjects.len(), 7);
        // Rows 199 of the genuine block, and 199/399/.../999 of impostors: fta.
        assert_eq!(m.lines().filter(|l| l.ends_with(",fta")).count(), 2 + 5);
    }

    #[test]
    fn splitmix_reference_vector() {
        // Reference outputs of SplitMix64 seeded with 0.
        let mut r = Rng::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn images_are_deterministic() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let p = SynthParams {
            seed: 3,
            images: 2,
            ..SynthParams::default()
        };
        write_images(a.path(), &p).unwrap();
        write_images(b.path(), &p).unwrap();
        for n in ["synth_000.png", "synth_001.png"] {
            assert_eq!(
                std::fs::read(a.path().join(n)).unwrap(),
                std::fs::read(b.path().join(n)).unwrap()
            );
        }
    }
}
