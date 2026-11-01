/// Unit tests for ML backends
#[cfg(test)]
#[allow(clippy::module_inception)] // file is `ml::tests`; kept to avoid moving every test
mod tests {
    use super::super::*;

    #[test]
    fn test_blazeface_score_parsing() {
        // BlazeFace outputs scores in format: [bg0, face0, bg1, face1, ...]
        // Test the parsing logic extracts face scores correctly

        let scores = [
            0.9, 0.1, // anchor 0: high background, low face
            0.2, 0.8, // anchor 1: low background, high face
            0.7, 0.3, // anchor 2: high background, low face
        ];

        let num_classes = 2;
        let num_anchors = scores.len() / num_classes;

        let mut best_idx = 0;
        let mut best_score = 0.0f32;

        for anchor_idx in 0..num_anchors {
            let score_idx = anchor_idx * num_classes + 1;
            let face_score = scores[score_idx];

            if face_score > best_score {
                best_score = face_score;
                best_idx = anchor_idx;
            }
        }

        // Should find anchor 1 with score 0.8
        assert_eq!(best_idx, 1, "Should select anchor 1");
        assert_eq!(best_score, 0.8, "Should have score 0.8");
    }

    #[test]
    fn test_box_offset_calculation() {
        // BlazeFace actual tensor sizes from logs:
        // - boxes: 8840 elements = 2210 anchors * 4 coords
        // - scores: 17680 elements = 8840 scores (2 per anchor)

        let boxes_len = 8840;
        let scores_len = 17680;
        let num_classes = 2;

        // Number of anchors derived from scores
        let _num_anchors = scores_len / num_classes; // 8840

        // Number of boxes (each box has 4 coordinates)
        let num_boxes = boxes_len / 4; // 2210

        // The issue: scores has more anchors than boxes!
        // This suggests scores include both positive and negative anchors,
        // but boxes only include the valid detection anchors (2210)

        // Our parsing must use the box count, not score count
        for box_idx in 0..num_boxes {
            let box_offset = box_idx * 4;
            assert!(
                box_offset + 3 < boxes_len,
                "Box offset {} + 3 should be < {}",
                box_offset,
                boxes_len
            );
        }
    }

    #[test]
    fn test_score_to_box_mapping() {
        // BlazeFace format:
        // boxes: [x0,y0,w0,h0, ...] = 8840 elements = 2210 boxes
        // scores: [bg0,face0, bg1,face1, ...] = 17680 elements = 8840 class scores

        let boxes_len = 8840;
        let scores_len = 17680;
        let num_classes = 2;

        let num_score_anchors = scores_len / num_classes; // 8840
        let num_boxes = boxes_len / 4; // 2210

        // Key insight: Not all score anchors have corresponding boxes
        // We must only use indices < num_boxes when indexing into boxes
        assert!(
            num_boxes < num_score_anchors,
            "Boxes {} should be less than score anchors {}",
            num_boxes,
            num_score_anchors
        );
    }

    /// Backend whose detector occupies its thread, like a slow ORT inference.
    struct SlowBackend;

    const SLOW_INFERENCE: std::time::Duration = std::time::Duration::from_millis(500);
    /// Longest a timer may be late while an inference is in flight.
    const MAX_RUNTIME_STALL: std::time::Duration = std::time::Duration::from_millis(250);

    #[async_trait::async_trait]
    impl MLBackend for SlowBackend {
        async fn detect_face(&self, _image: &DynamicImage) -> Result<Option<Face>> {
            std::thread::sleep(SLOW_INFERENCE);
            Ok(None)
        }
        async fn check_liveness(&self, _image: &DynamicImage, _face: &Face) -> Result<bool> {
            Ok(true)
        }
        async fn extract_embedding(&self, _image: &DynamicImage, _face: &Face) -> Result<Vec<f32>> {
            Ok(Vec::new())
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn name(&self) -> &'static str {
            "slow-mock"
        }
    }

    /// A slow inference must not park the (single-threaded) runtime: timers,
    /// i.e. every other connection such as `Status`, keep being served.
    #[tokio::test(flavor = "current_thread")]
    async fn slow_inference_does_not_block_the_runtime() {
        let pipeline = Arc::new(MLPipeline::with_backend(
            Arc::new(SlowBackend),
            &Config::default(),
        ));
        let image = DynamicImage::new_rgb8(8, 8);
        let detect = {
            let pipeline = pipeline.clone();
            tokio::spawn(async move { pipeline.detect_face(&image).await })
        };
        // Let the detect task start its inference.
        tokio::task::yield_now().await;

        let started = std::time::Instant::now();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert!(
            started.elapsed() < MAX_RUNTIME_STALL,
            "runtime stalled for {:?} while inference was in flight",
            started.elapsed()
        );
        assert!(!detect.is_finished(), "inference is still in flight");
        assert!(detect.await.unwrap().unwrap().is_none());
    }
}
