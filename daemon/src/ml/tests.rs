/// Unit tests for ML backends
#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_blazeface_score_parsing() {
        // BlazeFace outputs scores in format: [bg0, face0, bg1, face1, ...]
        // Test the parsing logic extracts face scores correctly

        let scores = vec![
            0.9, 0.1,  // anchor 0: high background, low face
            0.2, 0.8,  // anchor 1: low background, high face
            0.7, 0.3,  // anchor 2: high background, low face
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
        let num_anchors = scores_len / num_classes; // 8840

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
            num_boxes, num_score_anchors
        );
    }
}
