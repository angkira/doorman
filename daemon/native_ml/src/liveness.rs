use anyhow::Result;
use ndarray::{Array, Array4};
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;
use crate::LivenessResult;

pub struct LivenessChecker {
    session: Session,
}

impl LivenessChecker {
    pub fn new(models_dir: &str, device: &str) -> Result<Self> {
        let model_path = Path::new(models_dir).join("liveness.onnx");
        
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

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

    pub fn check(&mut self, face_crop: &[u8]) -> Result<LivenessResult> {
        // Face crop should be 112x112x3 RGB
        const SIZE: usize = 112;
        
        // Convert to CHW tensor
        let mut tensor_data = vec![0.0f32; 3 * SIZE * SIZE];
        for c in 0..3 {
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let src_idx = (y * SIZE + x) * 3 + c;
                    let dst_idx = c * SIZE * SIZE + y * SIZE + x;
                    if src_idx < face_crop.len() {
                        tensor_data[dst_idx] = face_crop[src_idx] as f32 / 255.0;
                    }
                }
            }
        }

        let array: Array4<f32> = Array::from_shape_vec((1, 3, SIZE, SIZE), tensor_data)?;
        let input_tensor = Value::from_array(array)?;

        // Run inference
        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let (_, scores) = outputs[0].try_extract_tensor::<f32>()?;

        // Liveness model outputs [fake_score, live_score]
        let live_score = if scores.len() >= 2 { scores[1] } else { 0.0 };
        let is_live = live_score > 0.5;

        Ok(LivenessResult {
            is_live,
            confidence: live_score,
        })
    }
}
