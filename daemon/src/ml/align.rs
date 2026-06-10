//! Landmark-based face alignment for the recognizer.
//!
//! ArcFace-style recognizers are trained on faces aligned to a fixed 5-point
//! template in a 112x112 crop. We reproduce that alignment here:
//!
//! 1. Estimate a 2x3 **similarity transform** (uniform scale + rotation +
//!    translation, no shear) mapping the 5 detected landmarks onto the
//!    canonical ArcFace template via the Umeyama least-squares method.
//! 2. Inverse-warp the source frame through that transform into a 112x112
//!    RGB buffer using bilinear sampling.
//!
//! The result feeds straight into the recognizer preprocessing.

use image::{DynamicImage, RgbImage};

/// A 2x3 affine transform `[[a, b, tx], [c, d, ty]]` mapping source pixel
/// coordinates to destination (template) coordinates: `dst = M * [x, y, 1]^T`.
#[derive(Debug, Clone, Copy)]
pub struct Affine2x3 {
    pub m: [[f32; 3]; 2],
}

impl Affine2x3 {
    /// Invert the affine transform. Returns `None` if it is singular.
    fn inverse(&self) -> Option<Affine2x3> {
        let [[a, b, tx], [c, d, ty]] = self.m;
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv_det = 1.0 / det;
        // Inverse of the 2x2 linear part.
        let ia = d * inv_det;
        let ib = -b * inv_det;
        let ic = -c * inv_det;
        let id = a * inv_det;
        // Inverse translation: -A^{-1} * t.
        let itx = -(ia * tx + ib * ty);
        let ity = -(ic * tx + id * ty);
        Some(Affine2x3 {
            m: [[ia, ib, itx], [ic, id, ity]],
        })
    }

    #[inline]
    fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let [[a, b, tx], [c, d, ty]] = self.m;
        (a * x + b * y + tx, c * x + d * y + ty)
    }
}

/// Estimate a similarity transform (scale, rotation, translation) that best
/// maps `src` onto `dst` in the least-squares sense (Umeyama, 1991).
///
/// Both slices must have the same length (>= 2). Returns the 2x3 matrix mapping
/// `src -> dst`, or `None` if degenerate.
pub fn umeyama_similarity(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Option<Affine2x3> {
    let n = src.len();
    if n < 2 || n != dst.len() {
        return None;
    }
    let nf = n as f64;

    // Means.
    let (mut sx, mut sy, mut dx, mut dy) = (0.0f64, 0.0, 0.0, 0.0);
    for i in 0..n {
        sx += src[i].0 as f64;
        sy += src[i].1 as f64;
        dx += dst[i].0 as f64;
        dy += dst[i].1 as f64;
    }
    sx /= nf;
    sy /= nf;
    dx /= nf;
    dy /= nf;

    // Closed-form 2D similarity (rotation + uniform scale + translation) least
    // squares. With centered points, the optimal rotation angle theta and scale
    // come from the cross terms:
    //   a = sum(dx*sx + dy*sy)   (aligned component)
    //   b = sum(dy*sx - dx*sy)   (orthogonal / rotation component)
    //   src_var = sum(sx^2 + sy^2)
    //   scale = sqrt(a^2 + b^2) / src_var
    //   R = (1/sqrt(a^2+b^2)) * [[a, -b], [b, a]]
    // This is the standard "FitGeometricTransform"/cv2.estimateAffinePartial2D
    // solution and matches skimage's SimilarityTransform.estimate.
    let (mut a, mut b, mut src_var) = (0.0f64, 0.0, 0.0);
    for i in 0..n {
        let sxc = src[i].0 as f64 - sx;
        let syc = src[i].1 as f64 - sy;
        let dxc = dst[i].0 as f64 - dx;
        let dyc = dst[i].1 as f64 - dy;
        a += dxc * sxc + dyc * syc;
        b += dyc * sxc - dxc * syc;
        src_var += sxc * sxc + syc * syc;
    }

    if src_var < 1e-12 {
        return None;
    }
    let norm = (a * a + b * b).sqrt();
    if norm < 1e-12 {
        return None;
    }

    // scale * R, where R = [[cos, -sin], [sin, cos]] and cos = a/norm,
    // sin = b/norm; scale = norm / src_var. So scale*R = [[a, -b],[b, a]]/src_var.
    let sa = a / src_var; // scale * cos
    let sb = b / src_var; // scale * sin

    // Translation = dst_mean - (scale*R) * src_mean.
    let tx = dx - (sa * sx - sb * sy);
    let ty = dy - (sb * sx + sa * sy);

    Some(Affine2x3 {
        m: [
            [sa as f32, (-sb) as f32, tx as f32],
            [sb as f32, sa as f32, ty as f32],
        ],
    })
}

/// Align a detected face to the ArcFace 112x112 template.
///
/// `landmarks_px` are the 5 detector landmarks in **source pixel** coordinates
/// (right-eye, left-eye, nose, right-mouth, left-mouth). `template` is the
/// canonical 112x112 template in the same order. Returns a 112x112 RGB image,
/// inverse-warped with bilinear sampling. Falls back to `None` if the transform
/// is degenerate (caller should then use a plain crop+resize).
pub fn align_to_template(
    image: &DynamicImage,
    landmarks_px: &[(f32, f32); 5],
    template: &[(f32, f32); 5],
    out_size: u32,
) -> Option<RgbImage> {
    let m = umeyama_similarity(landmarks_px, template)?;
    let inv = m.inverse()?; // dst (template) -> src
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as i32, rgb.height() as i32);
    let mut out = RgbImage::new(out_size, out_size);

    for oy in 0..out_size {
        for ox in 0..out_size {
            // Map destination pixel center back into the source image.
            let (srcx, srcy) = inv.apply(ox as f32 + 0.5, oy as f32 + 0.5);
            let px = srcx - 0.5;
            let py = srcy - 0.5;
            let x0 = px.floor() as i32;
            let y0 = py.floor() as i32;
            let fx = px - x0 as f32;
            let fy = py - y0 as f32;

            let mut acc = [0.0f32; 3];
            for (dyi, wy) in [(0, 1.0 - fy), (1, fy)] {
                for (dxi, wx) in [(0, 1.0 - fx), (1, fx)] {
                    let sx = (x0 + dxi).clamp(0, w - 1);
                    let sy = (y0 + dyi).clamp(0, h - 1);
                    let p = rgb.get_pixel(sx as u32, sy as u32);
                    let wgt = wx * wy;
                    acc[0] += p[0] as f32 * wgt;
                    acc[1] += p[1] as f32 * wgt;
                    acc[2] += p[2] as f32 * wgt;
                }
            }
            out.put_pixel(
                ox,
                oy,
                image::Rgb([
                    acc[0].round().clamp(0.0, 255.0) as u8,
                    acc[1].round().clamp(0.0, 255.0) as u8,
                    acc[2].round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn umeyama_recovers_known_similarity() {
        // dst = scale*R*src + t with scale=2, 90deg rotation, t=(10,5).
        let src = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 3.0)];
        // 90deg rotation: (x,y) -> (-y, x); *2; + (10,5)
        let dst: Vec<(f32, f32)> = src
            .iter()
            .map(|&(x, y)| (2.0 * (-y) + 10.0, 2.0 * (x) + 5.0))
            .collect();
        let m = umeyama_similarity(&src, &dst).unwrap();
        for i in 0..src.len() {
            let (mx, my) = m.apply(src[i].0, src[i].1);
            assert!((mx - dst[i].0).abs() < 1e-3, "x {} vs {}", mx, dst[i].0);
            assert!((my - dst[i].1).abs() < 1e-3, "y {} vs {}", my, dst[i].1);
        }
    }

    #[test]
    fn inverse_roundtrip() {
        let m = Affine2x3 {
            m: [[1.5, -0.3, 4.0], [0.3, 1.5, -2.0]],
        };
        let inv = m.inverse().unwrap();
        let (x, y) = (12.0f32, 7.0);
        let (a, b) = m.apply(x, y);
        let (rx, ry) = inv.apply(a, b);
        assert!((rx - x).abs() < 1e-3 && (ry - y).abs() < 1e-3);
    }
}
