# Coordinate Flow Analysis

## Step-by-step coordinate transformations:

### 1. Camera Capture
- **Source**: GStreamer camera → 1024x720 RGB frame
- **Location**: `camera_producer.rs`

### 2. Detection Input Preprocessing
- **Input**: 1024x720 RGB frame
- **Process**: Resize to 128x128 for BlazeFace
- **Location**: `tract_backend.rs::detect_faces()`
- **Transform**: Scale down by ~8x (1024→128)

### 3. BlazeFace Output
- **Format**: Normalized coordinates [0.0, 1.0] relative to 128x128 input
- **Example**: (0.2, 0.1, 0.8, 0.5) means face at 20%-80% width, 10%-60% height of 128x128
- **Location**: BlazeFace model output

### 4. Convert to Original Frame Coordinates
- **Process**: Multiply normalized coords by original frame size (1024x720)
- **Location**: `tract_backend.rs` - NEED TO CHECK THIS
- **Expected**: Face bbox in 1024x720 space (e.g., 200, 100, 400, 300)

### 5. Send to Preview
- **Protocol**: Unix socket with bbox as (x, y, w, h) in pixels
- **Location**: `frame_fanout.rs::broadcast_detection()`

### 6. Preview Display
- **Input**: bbox (x, y, w, h) in original 1024x720 space
- **Display**: Draw on 1024x720 frame
- **Location**: `preview_ipc.py`

## Current Problem:
BBox arriving as (88, 54, 1004, 662) - width is 1004 pixels, almost entire frame!

This suggests the coordinates are being scaled incorrectly in step 4.
Need to check how BlazeFace normalized outputs are converted to pixel coordinates.
