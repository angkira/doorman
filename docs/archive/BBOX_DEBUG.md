# BBox Coordinate Debugging

## Problem
Preview shows huge bounding box (~657x695 pixels) that covers most of the frame.
Detection logs show bbox like `(125, 55, 666, 701)` which is too large.

## Current Flow
1. BlazeFace outputs: `[center_x, center_y, w, h]` in **normalized [0,1]** coordinates relative to letterboxed 128x128 input
2. tract_backend converts:
   - Center → Top-left corner
   - Removes letterbox padding
   - Scales to original camera resolution (1024x720)
   - Returns normalized coords `(x_norm, y_norm, w_norm, h_norm)` in [0,1]
3. detection_pipeline multiplies by camera dimensions to get pixels
4. Broadcasts to preview

## Issue
The width/height values are too large - seems like padding calculation is wrong or BlazeFace output interpretation is incorrect.

## Next Steps
1. Add unit test with known image to verify coordinate transformation
2. Check if BlazeFace output is actually center+size or corner+size
3. Verify letterbox padding math
4. Test with synthetic bbox values

## Test Case Needed
- Input: 1024x720 image with face roughly at center
- Expected bbox: ~150-250 pixels wide for typical face
- Actual bbox: 657x695 pixels (way too big)
