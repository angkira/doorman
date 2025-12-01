/// Unit tests for coordinate transformations in the pipeline
/// Tests the flow: BlazeFace output -> pixel coords -> preview display

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    /// BlazeFace outputs coordinates in 512x512 space (for 128x128 input upscaled 4x)
    /// We need to convert these to camera resolution (e.g. 1024x720)
    #[test]
    fn test_blazeface_to_camera_coords() {
        // BlazeFace detects face at center of 512x512 space
        let blazeface_bbox = (200.0, 150.0, 100.0, 100.0); // (x, y, w, h) in 512x512
        let camera_res = (1024, 720);
        
        // Scale from 512x512 to camera resolution
        let scale_x = camera_res.0 as f32 / 512.0;
        let scale_y = camera_res.1 as f32 / 512.0;
        
        let camera_bbox = (
            blazeface_bbox.0 * scale_x,
            blazeface_bbox.1 * scale_y,
            blazeface_bbox.2 * scale_x,
            blazeface_bbox.3 * scale_y,
        );
        
        // Expected: (400, 210, 200, 140) approximately
        assert_relative_eq!(camera_bbox.0, 400.0, epsilon = 1.0);
        assert_relative_eq!(camera_bbox.1, 210.9375, epsilon = 1.0);
        assert_relative_eq!(camera_bbox.2, 200.0, epsilon = 1.0);
        assert_relative_eq!(camera_bbox.3, 140.625, epsilon = 1.0);
    }

    #[test]
    fn test_shrink_bbox_for_tight_fit() {
        let bbox = (400.0, 210.0, 200.0, 140.0);
        let shrink = 0.25;
        
        let tight_bbox = (
            bbox.0 + bbox.2 * shrink,      // x + w * 0.25
            bbox.1 + bbox.3 * shrink,      // y + h * 0.25
            bbox.2 * (1.0 - 2.0 * shrink), // w * 0.5
            bbox.3 * (1.0 - 2.0 * shrink), // h * 0.5
        );
        
        assert_relative_eq!(tight_bbox.0, 450.0, epsilon = 0.1);
        assert_relative_eq!(tight_bbox.1, 245.0, epsilon = 0.1);
        assert_relative_eq!(tight_bbox.2, 100.0, epsilon = 0.1);
        assert_relative_eq!(tight_bbox.3, 70.0, epsilon = 0.1);
    }

    #[test]
    fn test_full_coordinate_pipeline() {
        // Simulate full pipeline: BlazeFace -> Camera coords -> Preview
        let blazeface_bbox = (256.0, 256.0, 100.0, 100.0); // Center of 512x512
        let camera_res = (1024, 720);
        
        // Step 1: Scale from BlazeFace space (512x512) to camera resolution
        let scale_x = camera_res.0 as f32 / 512.0;
        let scale_y = camera_res.1 as f32 / 512.0;
        
        let camera_bbox = (
            blazeface_bbox.0 * scale_x,
            blazeface_bbox.1 * scale_y,
            blazeface_bbox.2 * scale_x,
            blazeface_bbox.3 * scale_y,
        );
        
        // Step 2: Shrink bbox for tight fit (no padding)
        let shrink = 0.25;
        let tight_bbox = (
            camera_bbox.0 + camera_bbox.2 * shrink,
            camera_bbox.1 + camera_bbox.3 * shrink,
            camera_bbox.2 * (1.0 - 2.0 * shrink),
            camera_bbox.3 * (1.0 - 2.0 * shrink),
        );
        
        // Result should be approximately center of 1024x720 with 100x70 size
        assert_relative_eq!(tight_bbox.0, 537.0, epsilon = 5.0); // ~512 + 25
        assert_relative_eq!(tight_bbox.1, 377.8, epsilon = 5.0); // ~360 + 18
        assert_relative_eq!(tight_bbox.2, 100.0, epsilon = 2.0);
        assert_relative_eq!(tight_bbox.3, 70.3, epsilon = 2.0);
    }
}
