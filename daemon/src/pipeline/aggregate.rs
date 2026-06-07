//! Phase 1 recognition-stability helpers: multi-frame embedding aggregation and
//! a cheap frame-quality (sharpness) metric.
//!
//! These are pure functions with no model/IO dependencies so they are unit
//! tested directly. They are config-driven from `[recognition]` in doorman.toml.

use image::DynamicImage;

/// Aggregate a slice of L2-normalized 512-d embeddings into a single
/// L2-normalized mean ("renormalized mean").
///
/// Averaging several normalized embeddings of the same identity suppresses
/// per-frame noise (pose/lighting/blur jitter) and raises the genuine cosine
/// against a stored template while leaving impostor scores essentially
/// unchanged. The mean is re-normalized so the result stays on the unit sphere
/// and cosine == dot product downstream.
///
/// Returns `None` for an empty input. A single-element input returns a
/// normalized copy of that element (no-op for already-normalized vectors).
pub fn aggregate_embeddings(embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = embeddings.first()?;
    let dim = first.len();
    if dim == 0 {
        return None;
    }

    let mut sum = vec![0.0f32; dim];
    let mut count = 0usize;
    for emb in embeddings {
        // Defensive: skip any mismatched-length vector rather than panic.
        if emb.len() != dim {
            continue;
        }
        for (s, v) in sum.iter_mut().zip(emb.iter()) {
            *s += *v;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }

    let inv = 1.0 / count as f32;
    for s in sum.iter_mut() {
        *s *= inv;
    }

    // Re-normalize the mean back onto the unit sphere.
    let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for s in sum.iter_mut() {
            *s /= norm;
        }
    }
    Some(sum)
}

/// Sharpness metric: variance of the Laplacian computed on a downsized 112x112
/// grayscale version of the image (or crop). Higher = sharper. A low value
/// indicates blur (out-of-focus / motion).
///
/// This is intentionally cheap (112x112 single pass). It is used as a
/// frame-quality gate before embedding extraction so blurry frames never reach
/// the recognizer or the aggregation window.
pub fn sharpness_score(image: &DynamicImage) -> f32 {
    const SIZE: u32 = 112;
    let gray = image
        .resize_exact(SIZE, SIZE, image::imageops::FilterType::Triangle)
        .to_luma8();
    let w = SIZE as i32;
    let h = SIZE as i32;
    let at = |x: i32, y: i32| -> f32 { gray.get_pixel(x as u32, y as u32)[0] as f32 };

    // 4-neighbour Laplacian on interior pixels.
    let mut values = Vec::with_capacity(((w - 2) * (h - 2)) as usize);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let lap = at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0 * at(x, y);
            values.push(lap);
        }
    }
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }

    fn normalize(mut v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() { *x /= n; }
        }
        v
    }

    #[test]
    fn aggregate_empty_is_none() {
        assert!(aggregate_embeddings(&[]).is_none());
    }

    #[test]
    fn aggregate_single_is_normalized_copy() {
        let v = normalize(vec![1.0, 2.0, 3.0, 4.0]);
        let agg = aggregate_embeddings(std::slice::from_ref(&v)).unwrap();
        let norm: f32 = agg.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "result must be unit-norm, got {norm}");
        for (a, b) in agg.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn aggregate_mean_is_unit_norm() {
        // Two near-identical normalized vectors -> mean stays ~unit-norm.
        let a = normalize(vec![1.0, 0.0, 0.1, 0.05]);
        let b = normalize(vec![0.95, 0.05, 0.12, 0.04]);
        let agg = aggregate_embeddings(&[a, b]).unwrap();
        let norm: f32 = agg.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "aggregated vector must be unit-norm, got {norm}");
    }

    #[test]
    fn aggregate_increases_cosine_to_reference() {
        // Reference (the stored template direction).
        let reference = normalize(vec![1.0, 0.0, 0.0, 0.0]);

        // Two genuine but noisy observations of the reference identity. Each is
        // off-axis (noisy) in different directions; averaging cancels the noise
        // and pulls the result closer to the reference.
        let noisy1 = normalize(vec![0.8, 0.5, 0.1, -0.2]);
        let noisy2 = normalize(vec![0.8, -0.5, -0.1, 0.2]);

        let single = cosine(&noisy1, &reference);
        let agg = aggregate_embeddings(&[noisy1.clone(), noisy2.clone()]).unwrap();
        let aggregated = cosine(&agg, &reference);

        assert!(
            aggregated > single,
            "aggregated cosine ({aggregated}) should exceed single-frame cosine ({single})"
        );
    }

    #[test]
    fn sharp_image_scores_higher_than_blurred() {
        use image::{DynamicImage, GrayImage, Luma};
        // Build a sharp high-frequency checkerboard.
        let mut sharp = GrayImage::new(112, 112);
        for (x, y, px) in sharp.enumerate_pixels_mut() {
            let v = if (x / 4 + y / 4) % 2 == 0 { 255u8 } else { 0u8 };
            *px = Luma([v]);
        }
        let sharp_img = DynamicImage::ImageLuma8(sharp);

        // Gaussian-blur it heavily to destroy the high frequencies.
        let blurred_img = sharp_img.blur(4.0);

        let s_sharp = sharpness_score(&sharp_img);
        let s_blur = sharpness_score(&blurred_img);
        assert!(
            s_sharp > s_blur,
            "sharp sharpness ({s_sharp}) must exceed blurred ({s_blur})"
        );
    }
}
