use anyhow::Result;
use ndarray::{Array, Array4};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;
use crate::DetectionResult;

pub struct FaceDetector {
    session: Session,
}

impl FaceDetector {
    pub fn new(models_dir: &str, device: &str) -> Result<Self> {
        let model_path = Path::new(models_dir).join("blazeface.onnx");
        
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        // Add GPU support if requested
        if device == "cuda" || device == "rocm" {
            #[cfg(feature = "rocm")]
            {
                use ort::execution_providers::ROCmExecutionProvider;
                builder = builder.with_execution_providers([
                    ROCmExecutionProvider::default().with_device_id(0).build()
                ])?;
            }
            
            #[cfg(not(feature = "rocm"))]
            {
                builder = builder.with_execution_providers([
                    CUDAExecutionProvider::default().with_device_id(0).build()
                ])?;
            }
        }

        let model_bytes = std::fs::read(&model_path)?;
        let session = builder.commit_from_memory(&model_bytes)?;

        Ok(Self { session })
    }

    pub fn detect(
        &mut self,
        image_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<DetectionResult>> {
        // Convert RGB bytes to CHW f32 tensor with letterboxing
        let (input_tensor, scale, offset_x, offset_y) = 
            self.preprocess_image(image_data, width, height)?;

        // Run inference and extract data
        let (scores_vec, boxes_vec) = {
            let outputs = self.session.run(ort::inputs![input_tensor])?;
            let (_, scores) = outputs[0].try_extract_tensor::<f32>()?;
            let (_, boxes) = outputs[1].try_extract_tensor::<f32>()?;
            (scores.to_vec(), boxes.to_vec())
        };

        self.parse_detections(&scores_vec, &boxes_vec, scale, offset_x, offset_y, width, height)
    }

    fn preprocess_image(
        &self,
        rgb_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(Value, f32, f32, f32)> {
        const TARGET_W: u32 = 320;
        const TARGET_H: u32 = 240;

        // Calculate scale for letterboxing
        let scale = (TARGET_W as f32 / width as f32).min(TARGET_H as f32 / height as f32);
        let resized_w = (width as f32 * scale) as u32;
        let resized_h = (height as f32 * scale) as u32;
        let offset_x = (TARGET_W - resized_w) as f32 / 2.0;
        let offset_y = (TARGET_H - resized_h) as f32 / 2.0;

        // Create tensor with letterboxing (simplified - no actual resizing, assume pre-resized)
        let mut tensor_data = vec![0.0f32; (3 * TARGET_W * TARGET_H) as usize];
        
        // Convert RGB to CHW format and normalize
        for c in 0..3 {
            for y in 0..height.min(TARGET_H) {
                for x in 0..width.min(TARGET_W) {
                    let src_idx = ((y * width + x) * 3 + c) as usize;
                    let dst_idx = (c * TARGET_H * TARGET_W + y * TARGET_W + x) as usize;
                    if src_idx < rgb_data.len() {
                        tensor_data[dst_idx] = rgb_data[src_idx] as f32 / 255.0;
                    }
                }
            }
        }

        let array: Array4<f32> = Array::from_shape_vec((1, 3, TARGET_H as usize, TARGET_W as usize), tensor_data)?;
        let value = Value::from_array(array)?;

        Ok((value.into(), scale, offset_x, offset_y))
    }

    fn parse_detections(
        &self,
        scores: &[f32],
        boxes: &[f32],
        scale: f32,
        offset_x: f32,
        offset_y: f32,
        orig_width: u32,
        orig_height: u32,
    ) -> Result<Vec<DetectionResult>> {
        let mut detections = Vec::new();

        let num_boxes = boxes.len() / 4;
        let num_classes = 2;

        for i in 0..num_boxes {
            let score_idx = i * num_classes + 1; // face class
            if score_idx >= scores.len() {
                break;
            }

            let confidence = scores[score_idx];
            if confidence < 0.5 {
                continue;
            }

            let box_idx = i * 4;
            if box_idx + 3 >= boxes.len() {
                break;
            }

            // Get box coordinates (model outputs normalized coordinates)
            let x1 = boxes[box_idx];
            let y1 = boxes[box_idx + 1];
            let x2 = boxes[box_idx + 2];
            let y2 = boxes[box_idx + 3];

            // Convert from letterboxed coordinates to original image coordinates
            let x1_orig = ((x1 * 320.0 - offset_x) / scale).max(0.0).min(orig_width as f32);
            let y1_orig = ((y1 * 240.0 - offset_y) / scale).max(0.0).min(orig_height as f32);
            let x2_orig = ((x2 * 320.0 - offset_x) / scale).max(0.0).min(orig_width as f32);
            let y2_orig = ((y2 * 240.0 - offset_y) / scale).max(0.0).min(orig_height as f32);

            detections.push(DetectionResult {
                bbox: (x1_orig, y1_orig, x2_orig, y2_orig),
                confidence,
                landmarks: vec![], // TODO: extract landmarks if needed
            });
        }

        // Sort by confidence
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(detections)
    }
}
