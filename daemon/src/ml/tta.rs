//! Flip test-time augmentation helpers (`[decision] flip_tta`): the embedding of
//! the aligned crop and of its horizontal mirror are averaged and renormalised.
//! Pure image/vector functions, so they are testable without a model.

// Only the ORT backend calls these; other feature sets compile them for tests.
#![cfg_attr(not(feature = "_ort"), allow(dead_code))]

use image::RgbImage;

/// Mirror `img` left-to-right.
pub fn flip_horizontal(img: &RgbImage) -> RgbImage {
    image::imageops::flip_horizontal(img)
}

/// Mean of two embeddings, L2-normalised. A zero mean is returned unchanged
/// (callers never feed zero embeddings; this avoids a division by zero).
pub fn average_normalized(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mean: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x + y) * 0.5).collect();
    let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        mean.iter().map(|x| x / norm).collect()
    } else {
        mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn flip_mirrors_columns_and_is_an_involution() {
        let mut img = RgbImage::new(3, 2);
        for y in 0..2 {
            for x in 0..3 {
                img.put_pixel(x, y, Rgb([x as u8, y as u8, 7]));
            }
        }
        let flipped = flip_horizontal(&img);
        assert_eq!(flipped.dimensions(), (3, 2));
        assert_eq!(*flipped.get_pixel(0, 1), Rgb([2, 1, 7]));
        assert_eq!(*flipped.get_pixel(2, 0), Rgb([0, 0, 7]));
        assert_eq!(flip_horizontal(&flipped), img);
    }

    #[test]
    fn average_is_unit_length_and_between_the_inputs() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let m = average_normalized(&a, &b);
        let norm = m.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((m[0] - m[1]).abs() < 1e-6);
        // Identical inputs come back unchanged.
        assert_eq!(average_normalized(&a, &a), vec![1.0, 0.0]);
        // Opposite inputs: zero mean passes through instead of dividing by zero.
        assert_eq!(
            average_normalized(&[1.0, 0.0], &[-1.0, 0.0]),
            vec![0.0, 0.0]
        );
    }
}
