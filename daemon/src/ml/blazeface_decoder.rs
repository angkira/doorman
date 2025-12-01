use anyhow::{anyhow, Result};

use super::anchors::{generate_anchors, Anchor, AnchorConfig};

/// Decoded face detection
#[derive(Debug, Clone)]
pub struct Detection {
    pub ymin: f32,
    pub xmin: f32,
    pub ymax: f32,
    pub xmax: f32,
    pub confidence: f32,
    // Optional: 6 keypoints (eyes, nose, mouth, ears)
    pub keypoints: Option<Vec<(f32, f32)>>,
}

/// BlazeFace decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    pub x_scale: f32,
    pub y_scale: f32,
    pub w_scale: f32,
    pub h_scale: f32,
    pub score_clipping_thresh: f32,
    pub min_score_thresh: f32,
    pub min_suppression_threshold: f32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        // Default settings for BlazeFace front (128x128)
        Self {
            x_scale: 128.0,
            y_scale: 128.0,
            w_scale: 128.0,
            h_scale: 128.0,
            score_clipping_thresh: 100.0,
            min_score_thresh: 0.75,
            min_suppression_threshold: 0.3,
        }
    }
}

impl DecoderConfig {
    pub fn for_back_model() -> Self {
        Self {
            x_scale: 256.0,
            y_scale: 256.0,
            w_scale: 256.0,
            h_scale: 256.0,
            score_clipping_thresh: 100.0,
            min_score_thresh: 0.65,
            min_suppression_threshold: 0.3,
        }
    }
}

/// BlazeFace detection decoder
pub struct BlazeFaceDecoder {
    anchors: Vec<Anchor>,
    config: DecoderConfig,
}

impl BlazeFaceDecoder {
    pub fn new(anchor_config: AnchorConfig, decoder_config: DecoderConfig) -> Result<Self> {
        let anchors = generate_anchors(&anchor_config)?;
        Ok(Self {
            anchors,
            config: decoder_config,
        })
    }

    pub fn new_default() -> Result<Self> {
        Self::new(AnchorConfig::default(), DecoderConfig::default())
    }

    /// Decode raw model outputs into detections
    ///
    /// # Arguments
    /// * `raw_boxes` - Raw box predictions [num_anchors, 16] (4 coords + 12 keypoint coords)
    /// * `raw_scores` - Raw scores [num_anchors, 2] (background, face)
    ///
    /// # Returns
    /// Vector of detections sorted by confidence (highest first)
    pub fn decode(
        &self,
        raw_boxes: &[f32],
        raw_scores: &[f32],
        num_anchors: usize,
    ) -> Result<Vec<Detection>> {
        if raw_boxes.len() != num_anchors * 16 {
            return Err(anyhow!(
                "Expected {} box values, got {}",
                num_anchors * 16,
                raw_boxes.len()
            ));
        }
        if raw_scores.len() != num_anchors * 2 {
            return Err(anyhow!(
                "Expected {} score values, got {}",
                num_anchors * 2,
                raw_scores.len()
            ));
        }
        if self.anchors.len() != num_anchors {
            return Err(anyhow!(
                "Anchor count mismatch: decoder has {} anchors, model outputs {} anchors",
                self.anchors.len(),
                num_anchors
            ));
        }

        // Decode boxes
        let decoded_boxes = self.decode_boxes(raw_boxes)?;

        // Apply sigmoid to scores with clipping
        let mut detections = Vec::new();
        for i in 0..num_anchors {
            let face_score_raw = raw_scores[i * 2 + 1]; // index 1 is face class
            
            // Clip and apply sigmoid
            let clipped = face_score_raw.clamp(
                -self.config.score_clipping_thresh,
                self.config.score_clipping_thresh,
            );
            let face_score = 1.0 / (1.0 + (-clipped).exp()); // sigmoid

            // Filter by threshold
            if face_score >= self.config.min_score_thresh {
                let detection = Detection {
                    ymin: decoded_boxes[i].ymin,
                    xmin: decoded_boxes[i].xmin,
                    ymax: decoded_boxes[i].ymax,
                    xmax: decoded_boxes[i].xmax,
                    confidence: face_score,
                    keypoints: decoded_boxes[i].keypoints.clone(),
                };
                detections.push(detection);
            }
        }

        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // Apply NMS (weighted non-maximum suppression)
        let filtered = self.weighted_nms(detections);

        Ok(filtered)
    }

    /// Decode raw box coordinates using anchors
    fn decode_boxes(&self, raw_boxes: &[f32]) -> Result<Vec<Detection>> {
        let mut detections = Vec::new();

        for (i, anchor) in self.anchors.iter().enumerate() {
            let offset = i * 16; // 16 values per box: 4 coords + 6 keypoints (x,y each)

            // Decode center and size
            let x_center = raw_boxes[offset] / self.config.x_scale * anchor.w + anchor.x_center;
            let y_center =
                raw_boxes[offset + 1] / self.config.y_scale * anchor.h + anchor.y_center;
            let w = raw_boxes[offset + 2] / self.config.w_scale * anchor.w;
            let h = raw_boxes[offset + 3] / self.config.h_scale * anchor.h;

            // Convert to corner coordinates
            let xmin = x_center - w / 2.0;
            let ymin = y_center - h / 2.0;
            let xmax = x_center + w / 2.0;
            let ymax = y_center + h / 2.0;

            // Decode keypoints (6 points: left eye, right eye, nose, mouth, left ear, right ear)
            let mut keypoints = Vec::new();
            for k in 0..6 {
                let kp_offset = offset + 4 + k * 2;
                let kp_x =
                    raw_boxes[kp_offset] / self.config.x_scale * anchor.w + anchor.x_center;
                let kp_y =
                    raw_boxes[kp_offset + 1] / self.config.y_scale * anchor.h + anchor.y_center;
                keypoints.push((kp_x, kp_y));
            }

            detections.push(Detection {
                ymin,
                xmin,
                ymax,
                xmax,
                confidence: 0.0, // Will be set later
                keypoints: Some(keypoints),
            });
        }

        Ok(detections)
    }

    /// Weighted non-maximum suppression as described in BlazeFace paper
    fn weighted_nms(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        if detections.is_empty() {
            return detections;
        }

        let mut output = Vec::new();

        while !detections.is_empty() {
            let first = detections.remove(0);

            // Find overlapping detections
            let mut overlapping = vec![first.clone()];
            let mut remaining = Vec::new();

            for detection in detections {
                let iou = self.compute_iou(&first, &detection);
                if iou > self.config.min_suppression_threshold {
                    overlapping.push(detection);
                } else {
                    remaining.push(detection);
                }
            }

            // Compute weighted average
            let weighted = self.compute_weighted_detection(&overlapping);
            output.push(weighted);

            detections = remaining;
        }

        output
    }

    /// Compute IoU between two detections
    fn compute_iou(&self, a: &Detection, b: &Detection) -> f32 {
        let x1 = a.xmin.max(b.xmin);
        let y1 = a.ymin.max(b.ymin);
        let x2 = a.xmax.min(b.xmax);
        let y2 = a.ymax.min(b.ymax);

        if x2 < x1 || y2 < y1 {
            return 0.0;
        }

        let inter = (x2 - x1) * (y2 - y1);
        let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
        let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);
        let union = area_a + area_b - inter;

        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }

    /// Compute weighted average detection from overlapping detections
    fn compute_weighted_detection(&self, detections: &[Detection]) -> Detection {
        if detections.len() == 1 {
            return detections[0].clone();
        }

        let total_conf: f32 = detections.iter().map(|d| d.confidence).sum();

        let mut weighted = Detection {
            ymin: 0.0,
            xmin: 0.0,
            ymax: 0.0,
            xmax: 0.0,
            confidence: total_conf / detections.len() as f32,
            keypoints: None,
        };

        for det in detections {
            let weight = det.confidence / total_conf;
            weighted.ymin += det.ymin * weight;
            weighted.xmin += det.xmin * weight;
            weighted.ymax += det.ymax * weight;
            weighted.xmax += det.xmax * weight;
        }

        weighted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let decoder = BlazeFaceDecoder::new_default();
        assert!(decoder.is_ok());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.anchors.len(), 896); // 128x128 has 896 anchors
    }

    #[test]
    fn test_iou_computation() {
        let decoder = BlazeFaceDecoder::new_default().unwrap();

        // Perfect overlap
        let a = Detection {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            confidence: 1.0,
            keypoints: None,
        };
        let b = a.clone();
        assert!((decoder.compute_iou(&a, &b) - 1.0).abs() < 1e-6);

        // No overlap
        let c = Detection {
            xmin: 2.0,
            ymin: 2.0,
            xmax: 3.0,
            ymax: 3.0,
            confidence: 1.0,
            keypoints: None,
        };
        assert!((decoder.compute_iou(&a, &c) - 0.0).abs() < 1e-6);

        // Partial overlap
        let d = Detection {
            xmin: 0.5,
            ymin: 0.5,
            xmax: 1.5,
            ymax: 1.5,
            confidence: 1.0,
            keypoints: None,
        };
        let iou = decoder.compute_iou(&a, &d);
        assert!(iou > 0.0 && iou < 1.0);
    }
}
