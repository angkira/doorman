#![cfg(all(feature = "_ort", feature = "backend-ncnn"))]

use anyhow::Result;
use async_trait::async_trait;
use doorman_shared::Config;
use image::DynamicImage;
use std::path::Path;
use std::sync::Mutex;
use tracing::{info, warn};

use super::backend::{Face, MLBackend};
use super::ncnn_backend::{DeviceSelector, NcnnNet};
use super::ort_backend::OrtBackend;

/// Hybrid ML backend that delegates face detection, embedding, and liveness to
/// the CPU-only `OrtBackend` while running the depth-relief PAD pass on the iGPU
/// via ncnn+Vulkan. Falls back to the CPU ORT depth path when the ncnn model is
/// unavailable or inference fails.
pub struct NcnnHybridBackend {
    inner: OrtBackend,
    depth_net: Option<Mutex<NcnnNet>>,
    relief_stat: doorman_shared::config::ReliefStat,
}

impl NcnnHybridBackend {
    pub fn new(models_dir: &Path, config: &Config) -> Result<Self> {
        let inner = OrtBackend::new(models_dir, config)?;

        let sel = match config.ml.vulkan_device.as_deref() {
            Some(s) if !s.is_empty() => DeviceSelector::NameSubstring(s.to_string()),
            _ => DeviceSelector::DefaultIntegrated,
        };

        let param = models_dir.join("depth.ncnn.param");
        let bin = models_dir.join("depth.ncnn.bin");

        let depth_net = match NcnnNet::new(&param, &bin, sel) {
            Ok(net) => {
                info!("ncnn hybrid: loaded depth model on iGPU");
                Some(Mutex::new(net))
            }
            Err(e) => {
                warn!("ncnn depth model not available, depth falls back to CPU ORT: {}", e);
                None
            }
        };

        Ok(Self {
            inner,
            depth_net,
            relief_stat: config.liveness.relief_stat,
        })
    }
}

#[async_trait]
impl MLBackend for NcnnHybridBackend {
    async fn detect_face(&self, image: &DynamicImage) -> Result<Option<Face>> {
        self.inner.detect_face(image).await
    }

    async fn check_liveness(&self, image: &DynamicImage, face: &Face) -> Result<bool> {
        self.inner.check_liveness(image, face).await
    }

    async fn depth_relief(&self, image: &DynamicImage, face: &Face) -> Result<Option<f32>> {
        const SIZE: u32 = 518;

        match &self.depth_net {
            Some(net) => {
                let data = super::ort_backend::depth_preprocess(image, SIZE);
                let result = {
                    let net = net.lock().unwrap();
                    net.infer("in0", &data, [3, SIZE as i32, SIZE as i32], "out0")
                };
                match result {
                    Ok(output) => {
                        let (vec, shape) = match output.get("out0") {
                            Some(v) => v,
                            None => {
                                warn!("ncnn depth: missing 'out0' in output, falling back to CPU ORT");
                                return self.inner.depth_relief(image, face).await;
                            }
                        };
                        let [_oc, oh, ow] = *shape;
                        let (d_h, d_w) = (oh as usize, ow as usize);
                        if d_h == 0 || d_w == 0 || vec.len() < d_h * d_w {
                            warn!("ncnn depth: unexpected output shape, falling back to CPU ORT");
                            return self.inner.depth_relief(image, face).await;
                        }
                        let relief = super::ort_backend::relief_from_depth_map(
                            self.relief_stat,
                            vec,
                            d_h,
                            d_w,
                            face,
                        );
                        Ok(Some(relief))
                    }
                    Err(e) => {
                        warn!("ncnn depth inference failed ({}), falling back to CPU ORT", e);
                        self.inner.depth_relief(image, face).await
                    }
                }
            }
            None => self.inner.depth_relief(image, face).await,
        }
    }

    async fn extract_embedding(&self, image: &DynamicImage, face: &Face) -> Result<Vec<f32>> {
        self.inner.extract_embedding(image, face).await
    }

    fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    fn name(&self) -> &'static str {
        "ncnn-hybrid (iGPU depth + CPU detect/recognize)"
    }
}
