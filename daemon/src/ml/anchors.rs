use anyhow::Result;

/// Anchor box for object detection
#[derive(Debug, Clone)]
pub struct Anchor {
    pub x_center: f32,
    pub y_center: f32,
    pub w: f32,
    pub h: f32,
}

/// Configuration for anchor generation
#[derive(Debug, Clone)]
pub struct AnchorConfig {
    pub num_layers: usize,
    pub min_scale: f32,
    pub max_scale: f32,
    pub input_size_height: usize,
    pub input_size_width: usize,
    pub anchor_offset_x: f32,
    pub anchor_offset_y: f32,
    pub strides: Vec<usize>,
    pub aspect_ratios: Vec<f32>,
    pub reduce_boxes_in_lowest_layer: bool,
    pub interpolated_scale_aspect_ratio: f32,
    pub fixed_anchor_size: bool,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        // Default BlazeFace configuration for 128x128
        Self {
            num_layers: 4,
            min_scale: 0.1484375,
            max_scale: 0.75,
            input_size_height: 128,
            input_size_width: 128,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![8, 16, 16, 16],
            aspect_ratios: vec![1.0],
            reduce_boxes_in_lowest_layer: false,
            interpolated_scale_aspect_ratio: 1.0,
            fixed_anchor_size: true,
        }
    }
}

impl AnchorConfig {
    /// Configuration for 240x320 input (custom)
    pub fn for_240x320() -> Self {
        // This is a guess - need to figure out the right config
        Self {
            num_layers: 4,
            min_scale: 0.1484375,
            max_scale: 0.75,
            input_size_height: 240,
            input_size_width: 320,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![8, 16, 32, 64],
            aspect_ratios: vec![1.0],
            reduce_boxes_in_lowest_layer: false,
            interpolated_scale_aspect_ratio: 1.0,
            fixed_anchor_size: true,
        }
    }

    /// Configuration for 256x256 back model
    pub fn for_256x256_back() -> Self {
        Self {
            num_layers: 4,
            min_scale: 0.15625,
            max_scale: 0.75,
            input_size_height: 256,
            input_size_width: 256,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![16, 32, 32, 32],
            aspect_ratios: vec![1.0],
            reduce_boxes_in_lowest_layer: false,
            interpolated_scale_aspect_ratio: 1.0,
            fixed_anchor_size: true,
        }
    }
}

fn calculate_scale(min_scale: f32, max_scale: f32, stride_index: usize, num_strides: usize) -> f32 {
    min_scale + (max_scale - min_scale) * (stride_index as f32) / (num_strides as f32 - 1.0)
}

/// Generate anchor boxes based on configuration
pub fn generate_anchors(config: &AnchorConfig) -> Result<Vec<Anchor>> {
    let strides_size = config.strides.len();
    assert_eq!(config.num_layers, strides_size, "num_layers must match strides length");

    let mut anchors = Vec::new();
    let mut layer_id = 0;

    while layer_id < strides_size {
        let mut anchor_height = Vec::new();
        let mut anchor_width = Vec::new();
        let mut aspect_ratios = Vec::new();
        let mut scales = Vec::new();

        // For same strides, we merge the anchors in the same order
        let mut last_same_stride_layer = layer_id;
        while last_same_stride_layer < strides_size
            && config.strides[last_same_stride_layer] == config.strides[layer_id]
        {
            let scale = calculate_scale(
                config.min_scale,
                config.max_scale,
                last_same_stride_layer,
                strides_size,
            );

            if last_same_stride_layer == 0 && config.reduce_boxes_in_lowest_layer {
                // For first layer, it can be specified to use predefined anchors
                aspect_ratios.extend_from_slice(&[1.0, 2.0, 0.5]);
                scales.extend_from_slice(&[0.1, scale, scale]);
            } else {
                for &aspect_ratio in &config.aspect_ratios {
                    aspect_ratios.push(aspect_ratio);
                    scales.push(scale);
                }

                if config.interpolated_scale_aspect_ratio > 0.0 {
                    let scale_next = if last_same_stride_layer == strides_size - 1 {
                        1.0
                    } else {
                        calculate_scale(
                            config.min_scale,
                            config.max_scale,
                            last_same_stride_layer + 1,
                            strides_size,
                        )
                    };
                    scales.push((scale * scale_next).sqrt());
                    aspect_ratios.push(config.interpolated_scale_aspect_ratio);
                }
            }

            last_same_stride_layer += 1;
        }

        for i in 0..aspect_ratios.len() {
            let ratio_sqrts = aspect_ratios[i].sqrt();
            anchor_height.push(scales[i] / ratio_sqrts);
            anchor_width.push(scales[i] * ratio_sqrts);
        }

        let stride = config.strides[layer_id];
        let feature_map_height = (config.input_size_height + stride - 1) / stride; // ceil division
        let feature_map_width = (config.input_size_width + stride - 1) / stride;

        for y in 0..feature_map_height {
            for x in 0..feature_map_width {
                for anchor_id in 0..anchor_height.len() {
                    let x_center =
                        (x as f32 + config.anchor_offset_x) / feature_map_width as f32;
                    let y_center =
                        (y as f32 + config.anchor_offset_y) / feature_map_height as f32;

                    let (w, h) = if config.fixed_anchor_size {
                        (1.0, 1.0)
                    } else {
                        (anchor_width[anchor_id], anchor_height[anchor_id])
                    };

                    anchors.push(Anchor {
                        x_center,
                        y_center,
                        w,
                        h,
                    });
                }
            }
        }

        layer_id = last_same_stride_layer;
    }

    Ok(anchors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_anchors_128x128() {
        let config = AnchorConfig::default();
        let anchors = generate_anchors(&config).unwrap();
        
        // BlazeFace 128x128 should generate 896 anchors
        assert_eq!(anchors.len(), 896);

        // Check first anchor
        assert!((anchors[0].x_center - 0.03125).abs() < 1e-5);
        assert!((anchors[0].y_center - 0.03125).abs() < 1e-5);
        assert_eq!(anchors[0].w, 1.0);
        assert_eq!(anchors[0].h, 1.0);
    }

    #[test]
    fn test_generate_anchors_256x256_back() {
        let config = AnchorConfig::for_256x256_back();
        let anchors = generate_anchors(&config).unwrap();
        
        // BlazeFace back 256x256 should also generate 896 anchors
        assert_eq!(anchors.len(), 896);
    }
}
