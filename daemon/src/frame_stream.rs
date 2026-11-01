use anyhow::Result;
use doormand::ipc_policy;
use doorman_shared::IpcConfig;
use image::DynamicImage;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

/// Frame data as JPEG bytes
pub type FrameData = Vec<u8>;

/// Broadcaster for streaming camera frames to preview clients
pub struct FrameStreamBroadcaster {
    sender: broadcast::Sender<FrameData>,
}

impl FrameStreamBroadcaster {
    /// Create a new broadcaster with given capacity
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Broadcast a frame to all connected preview clients
    /// Converts DynamicImage to JPEG and broadcasts
    pub fn broadcast_frame(&self, frame: &DynamicImage) -> Result<()> {
        use image::ImageEncoder;

        // Convert frame to JPEG bytes with quality setting
        let mut jpeg_bytes = Vec::new();

        // Convert to RGB8 first to ensure consistent format
        let rgb_frame = frame.to_rgb8();
        let (width, height) = rgb_frame.dimensions();

        // Encode as JPEG with quality 90
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_bytes, 90);
        encoder.write_image(
            rgb_frame.as_raw(),
            width,
            height,
            image::ExtendedColorType::Rgb8,
        )?;

        // Broadcast to all clients (ignore if no receivers)
        let _ = self.sender.send(jpeg_bytes);

        Ok(())
    }

    /// Subscribe to receive frames
    pub fn subscribe(&self) -> broadcast::Receiver<FrameData> {
        self.sender.subscribe()
    }

    /// Get current number of connected preview clients
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

/// Run the frame stream server that accepts preview client connections
pub async fn run_frame_server(
    socket_path: String,
    broadcaster: Arc<FrameStreamBroadcaster>,
    ipc: IpcConfig,
) -> Result<()> {
    // 0600 (or 0660 + group per [ipc] stream_socket_group): the stream carries face data.
    let listener = ipc_policy::bind_stream_socket(&socket_path, &ipc)?;
    info!("Frame stream server listening on {}", socket_path);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                // Defense in depth on top of the socket mode; fail closed.
                match ipc_policy::peer_from_stream(&stream) {
                    Ok(peer) if ipc_policy::stream_peer_allowed(&peer, &ipc) => {}
                    Ok(peer) => {
                        warn!(peer_uid = peer.uid, peer_pid = ?peer.pid, "dropped unauthorized frame stream client");
                        continue;
                    }
                    Err(e) => {
                        warn!("dropped frame stream client: {e:#}");
                        continue;
                    }
                }
                let broadcaster = broadcaster.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_frame_client(stream, broadcaster).await {
                        debug!("Frame client disconnected: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept frame connection: {}", e);
            }
        }
    }
}

/// Handle a single preview client connection for frame streaming
async fn handle_frame_client(
    mut stream: UnixStream,
    broadcaster: Arc<FrameStreamBroadcaster>,
) -> Result<()> {
    info!("Frame client connected");

    let mut receiver = broadcaster.subscribe();

    loop {
        match receiver.recv().await {
            Ok(jpeg_bytes) => {
                // Send frame size as 4-byte big-endian u32, then JPEG data
                let size = jpeg_bytes.len() as u32;
                let size_bytes = size.to_be_bytes();

                if let Err(e) = stream.write_all(&size_bytes).await {
                    debug!("Failed to send frame size to client: {}", e);
                    break;
                }

                if let Err(e) = stream.write_all(&jpeg_bytes).await {
                    debug!("Failed to send frame to client: {}", e);
                    break;
                }

                // Flush to ensure frame is sent immediately
                if let Err(e) = stream.flush().await {
                    debug!("Failed to flush frame to client: {}", e);
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                warn!("Frame client lagging, skipped {} frames", skipped);
                continue;
            }
            Err(broadcast::error::RecvError::Closed) => {
                info!("Frame stream broadcaster closed");
                break;
            }
        }
    }

    Ok(())
}
