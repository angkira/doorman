# Doorman Architecture - Non-Blocking Producer-Consumer Design

## Current Issues

The existing architecture has several blocking points that prevent optimal performance:

1. **Camera Lock Contention** (`background_detection.rs:47`)
   - Exclusive write lock on camera during capture
   - Blocks other operations from accessing camera

2. **Synchronous Capture**
   - `camera.capture_frame()` blocks async runtime
   - FFmpeg subprocess I/O causes thread stalls

3. **ML Pipeline Blocking** (`background_detection.rs:79, 130`)
   - CPU/GPU intensive operations in async context
   - No parallelism between capture and inference

4. **Sequential Processing**
   - Capture → Detect → Recognize all in single loop
   - If any stage is slow, entire pipeline stalls
   - Frame rate limited by slowest component

## Proposed Architecture: Staged Pipeline

### Design Principles

1. **Separation of Concerns**: Each stage is an independent task
2. **Non-blocking**: Use channels and `spawn_blocking` for CPU work
3. **Backpressure**: Bounded channels prevent memory exhaustion
4. **Graceful Degradation**: Skip frames rather than block on slow consumers

### Architecture Diagram

```
┌─────────────────────────┐
│   Camera Producer       │ (tokio task, owns camera)
│   - Captures at 30fps   │
│   - Never blocks on ML  │
│   - Uses spawn_blocking │
└───────────┬─────────────┘
            │ mpsc::channel(5) - Bounded
            │ RawFrame { image, timestamp, sequence }
            ↓
┌─────────────────────────┐
│   Frame Fanout          │ (tokio task)
│   - Receives frames     │
│   - Broadcasts to:      │
│     1. Preview clients  │
│     2. Detection        │
└───────┬──────────┬──────┘
        │          │
        │          └─→ mpsc::channel(2)
        │              → Detection Pipeline
        │
        └─→ broadcast::channel(30)
            → Preview Clients
            → JPEG encoding in spawn_blocking

┌─────────────────────────┐
│   Detection Pipeline    │ (tokio task)
│   - Receives @ 5fps     │
│   - Spawns ML work      │
│   - Face detection      │
└───────────┬─────────────┘
            │ mpsc::channel(10)
            │ DetectionResult { face, embedding?, timestamp }
            ↓
┌─────────────────────────┐
│  Recognition Pipeline   │ (tokio task)
│   - Matches embeddings  │
│   - Checks storage      │
│   - Unlocks system      │
│   - Broadcasts debug    │
└─────────────────────────┘
```

### Component Specifications

#### 1. Camera Producer Task

**Responsibilities:**
- Own camera exclusively (no lock contention)
- Capture frames at camera native FPS (30fps)
- Publish to channel without blocking on downstream
- Handle camera errors and reconnection

**Implementation:**
```rust
async fn run_camera_producer(
    camera: Camera,
    frame_tx: mpsc::Sender<RawFrame>,
    config: Arc<Config>,
) {
    let fps = config.camera.fps;
    let interval_ms = 1000 / fps as u64;
    let mut ticker = interval(Duration::from_millis(interval_ms));
    let mut sequence = 0u64;

    loop {
        ticker.tick().await;

        // Capture in blocking thread (FFmpeg I/O)
        let frame = match tokio::task::spawn_blocking(move || {
            camera.capture_frame()
        }).await {
            Ok(Ok(frame)) => frame,
            Ok(Err(e)) => {
                warn!("Capture error: {}", e);
                continue;
            }
            Err(e) => {
                error!("Task panic: {}", e);
                continue;
            }
        };

        sequence += 1;
        let raw_frame = RawFrame {
            image: frame,
            timestamp: Instant::now(),
            sequence,
        };

        // Non-blocking send (drop frame if channel full)
        if let Err(e) = frame_tx.try_send(raw_frame) {
            debug!("Dropped frame {} (channel full)", sequence);
        }
    }
}
```

**Channel:**
- Type: `mpsc::channel<RawFrame>(5)`
- Bounded to prevent memory buildup
- Producer uses `try_send()` to avoid blocking

#### 2. Frame Fanout Task

**Responsibilities:**
- Receive frames from camera
- Broadcast JPEG to preview clients
- Send to detection at target rate (5fps)
- Handle multiple consumers without blocking

**Implementation:**
```rust
async fn run_frame_fanout(
    mut frame_rx: mpsc::Receiver<RawFrame>,
    preview_tx: broadcast::Sender<FrameData>,
    detection_tx: mpsc::Sender<RawFrame>,
    target_detection_fps: u32,
) {
    let detection_interval = 1000 / target_detection_fps as u64;
    let mut last_detection = Instant::now();

    while let Some(raw_frame) = frame_rx.recv().await {
        // Always broadcast to preview (JPEG encoding in blocking thread)
        let preview_tx = preview_tx.clone();
        let image = raw_frame.image.clone();
        tokio::spawn(async move {
            let jpeg_bytes = tokio::task::spawn_blocking(move || {
                encode_jpeg(&image, 90)
            }).await.ok()?;

            let _ = preview_tx.send(jpeg_bytes);
        });

        // Send to detection at target FPS
        if last_detection.elapsed().as_millis() >= detection_interval as u128 {
            if let Err(_) = detection_tx.try_send(raw_frame.clone()) {
                debug!("Detection channel full, skipping frame");
            }
            last_detection = Instant::now();
        }
    }
}
```

**Channels:**
- Input: `mpsc::Receiver<RawFrame>` from camera
- Output 1: `broadcast::Sender<FrameData>` for preview (30 frame buffer)
- Output 2: `mpsc::Sender<RawFrame>` for detection (2 frame buffer)

#### 3. Detection Pipeline Task

**Responsibilities:**
- Receive frames at target detection rate
- Run ML inference in blocking thread pool
- Publish detection results
- Never block frame capture

**Implementation:**
```rust
async fn run_detection_pipeline(
    mut frame_rx: mpsc::Receiver<RawFrame>,
    result_tx: mpsc::Sender<DetectionResult>,
    ml_pipeline: Arc<MLPipeline>,
) {
    while let Some(raw_frame) = frame_rx.recv().await {
        let ml = ml_pipeline.clone();
        let tx = result_tx.clone();
        let frame = raw_frame.image.clone();
        let timestamp = raw_frame.timestamp;
        let sequence = raw_frame.sequence;

        // Spawn blocking ML work
        tokio::task::spawn_blocking(move || {
            let start = Instant::now();

            // Detect face (CPU/GPU intensive)
            let face = match ml.detect_face_sync(&frame) {
                Ok(Some(f)) => f,
                Ok(None) => {
                    // No face detected
                    let _ = tx.blocking_send(DetectionResult {
                        sequence,
                        timestamp,
                        face: None,
                        embedding: None,
                        processing_time: start.elapsed(),
                    });
                    return;
                }
                Err(e) => {
                    warn!("Detection error: {}", e);
                    return;
                }
            };

            // Extract embedding (CPU/GPU intensive)
            let embedding = match ml.extract_embedding_sync(&frame, &face) {
                Ok(emb) => Some(emb),
                Err(e) => {
                    warn!("Embedding error: {}", e);
                    None
                }
            };

            let _ = tx.blocking_send(DetectionResult {
                sequence,
                timestamp,
                face: Some(face),
                embedding,
                processing_time: start.elapsed(),
            });
        });
    }
}
```

**Key Changes:**
- All ML operations are `_sync` variants that run in blocking threads
- No `await` in ML code (runs on thread pool, not tokio runtime)
- Results published via channel, not broadcast (single consumer)

#### 4. Recognition Pipeline Task

**Responsibilities:**
- Receive detection results
- Match embeddings against storage
- Update system lock state
- Broadcast debug information
- Trigger system unlock

**Implementation:**
```rust
async fn run_recognition_pipeline(
    mut result_rx: mpsc::Receiver<DetectionResult>,
    storage: Arc<RwLock<Storage>>,
    debug_tx: broadcast::Sender<DebugStreamMessage>,
    system_locked: Arc<RwLock<bool>>,
) {
    while let Some(result) = result_rx.recv().await {
        let is_locked = *system_locked.read().await;

        // Check if we should process (locked or debug mode)
        if !is_locked && !DEBUG_MODE {
            continue;
        }

        // No face detected
        let (face, embedding) = match (result.face, result.embedding) {
            (Some(f), Some(e)) => (f, e),
            _ => {
                // Broadcast "no face" to debug stream
                broadcast_no_face(&debug_tx, result.timestamp, result.processing_time);
                continue;
            }
        };

        // Match against enrolled users
        let storage = storage.read().await;
        let mut best_match: Option<(String, f32)> = None;
        let mut best_similarity = 0.0f32;

        for user_info in storage.list_users() {
            if let Some(stored_emb) = storage.get_embedding(&user_info.username) {
                let similarity = cosine_similarity(&embedding, stored_emb);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    if similarity >= SIMILARITY_THRESHOLD {
                        best_match = Some((user_info.username.clone(), similarity));
                    }
                }
            }
        }
        drop(storage);

        // Handle recognition result
        if let Some((username, similarity)) = best_match {
            info!("✓ User recognized: {} (similarity: {:.2})", username, similarity);

            // Unlock system
            unlock_system(&system_locked).await;

            // Broadcast success to debug stream
            broadcast_recognition(&debug_tx, &face, Some(username), Some(similarity), result.processing_time);
        } else {
            // Unknown face
            broadcast_recognition(&debug_tx, &face, None, None, result.processing_time);
        }
    }
}
```

### Data Structures

```rust
/// Raw camera frame with metadata
#[derive(Clone)]
pub struct RawFrame {
    pub image: DynamicImage,
    pub timestamp: Instant,
    pub sequence: u64,
}

/// Detection result from ML pipeline
pub struct DetectionResult {
    pub sequence: u64,
    pub timestamp: Instant,
    pub face: Option<Face>,
    pub embedding: Option<Vec<f32>>,
    pub processing_time: Duration,
}

/// JPEG frame data for preview clients
pub type FrameData = Vec<u8>;
```

### Channel Configuration

| Channel | Type | Capacity | Behavior on Full |
|---------|------|----------|------------------|
| Camera → Fanout | `mpsc` | 5 | Drop frame (try_send) |
| Fanout → Preview | `broadcast` | 30 | Lag (clients skip frames) |
| Fanout → Detection | `mpsc` | 2 | Drop frame (try_send) |
| Detection → Recognition | `mpsc` | 10 | Backpressure (await send) |

### Performance Characteristics

**Frame Rates:**
- Camera capture: 30 fps (camera native)
- Preview stream: 30 fps (all frames)
- Detection processing: 5 fps (configurable)
- Recognition: On-demand (when face detected)

**Latency:**
- Camera to preview: ~10-20ms (JPEG encoding)
- Detection: ~50-200ms (ML inference)
- Recognition: ~10-50ms (embedding matching)
- End-to-end: ~100-300ms (capture to unlock)

**Throughput:**
- Camera never blocks (always captures at 30fps)
- Preview clients never block camera
- Detection runs in parallel with capture
- Recognition runs in parallel with detection

### Benefits Over Current Design

1. **No Blocking**: Camera captures continuously regardless of ML speed
2. **Parallelism**: Capture, preview, detection all run concurrently
3. **Scalability**: Add more detection pipelines for higher throughput
4. **Resilience**: Slow consumers don't affect producers
5. **Testability**: Each stage can be tested independently
6. **Observability**: Sequence numbers track frame processing

## Implementation Plan

### Phase 1: Core Refactoring
**Goal**: Replace monolithic background_detection with staged pipeline

**Tasks:**
1. Create `daemon/src/pipeline/mod.rs` module structure
2. Implement `CameraProducer` task
3. Implement `FrameFanout` task
4. Implement `DetectionPipeline` task
5. Implement `RecognitionPipeline` task
6. Update `main.rs` to spawn all pipeline tasks
7. Remove old `background_detection.rs`

**Success Criteria:**
- Preview works at 30fps without stuttering
- Detection runs at 5fps without blocking camera
- System unlock still works
- No performance regression

### Phase 2: Non-Blocking ML
**Goal**: Move all ML operations to blocking thread pool

**Tasks:**
1. Add `_sync` variants to ML pipeline methods
2. Update `DetectionPipeline` to use `spawn_blocking`
3. Configure thread pool size (via config)
4. Add metrics for thread pool utilization

**Success Criteria:**
- No blocking of async runtime
- Consistent latency under load
- Thread pool doesn't exhaust resources

### Phase 3: Performance Optimization
**Goal**: Tune performance and add observability

**Tasks:**
1. Add frame sequence tracking
2. Add metrics (fps, latency, drops)
3. Tune channel buffer sizes
4. Add configurable frame skipping
5. Add performance logging

**Success Criteria:**
- <100ms latency for detection
- 0% frame drops at 30fps capture
- <5% frame drops at 5fps detection
- Observable metrics in logs

### Phase 4: Advanced Features
**Goal**: Enable future enhancements

**Tasks:**
1. Support multiple detection pipelines
2. Add dynamic FPS adjustment
3. Support camera hot-swapping
4. Add background model updates

**Success Criteria:**
- Can run N detection pipelines in parallel
- FPS adjusts based on CPU usage
- Camera reconnects automatically
- Models update without restart

## Migration Strategy

### Backward Compatibility
- Keep existing IPC protocol unchanged
- Keep existing storage format unchanged
- Keep existing config schema (add new optional fields)

### Testing Plan
1. **Unit Tests**: Test each pipeline stage independently
2. **Integration Tests**: Test full pipeline end-to-end
3. **Performance Tests**: Verify frame rates and latency
4. **Regression Tests**: Ensure existing features still work

### Rollback Plan
- Keep old `background_detection.rs` as `background_detection_legacy.rs`
- Add feature flag to switch between old and new
- Monitor metrics for first week after deployment

## References

### Current Code Locations
- Background detection: `daemon/src/background_detection.rs`
- Camera module: `daemon/src/camera/mod.rs`
- ML pipeline: `daemon/src/ml/mod.rs`
- Frame streaming: `daemon/src/frame_stream.rs`
- Debug streaming: `daemon/src/debug_stream.rs`

### Configuration Parameters
```toml
[daemon]
processing_fps = 5  # Detection rate (existing)
camera_fps = 30     # Capture rate (new, defaults to camera native)
pipeline_threads = 4  # ML thread pool size (new, defaults to num_cpus)
frame_buffer_size = 5  # Camera channel buffer (new)
detection_buffer_size = 2  # Detection channel buffer (new)
```

### Metrics to Track
- `camera_capture_fps`: Actual camera frame rate
- `camera_drops`: Frames dropped due to full channel
- `preview_fps`: Preview stream frame rate
- `preview_clients`: Number of connected preview clients
- `detection_fps`: Detection processing rate
- `detection_latency_ms`: Time from capture to detection result
- `recognition_latency_ms`: Time from detection to recognition
- `ml_thread_utilization`: Thread pool usage percentage
