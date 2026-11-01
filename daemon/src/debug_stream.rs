use anyhow::Result;
use doormand::ipc_policy;
use doorman_shared::{IpcConfig, StreamMessage};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};

/// Broadcaster for debug stream messages to preview clients
pub struct DebugStreamBroadcaster {
    sender: broadcast::Sender<StreamMessage>,
}

impl DebugStreamBroadcaster {
    /// Create a new broadcaster
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Send a message to all connected preview clients
    pub fn broadcast(&self, message: StreamMessage) {
        // Ignore error if no receivers (no preview connected)
        let _ = self.sender.send(message);
    }

    /// Subscribe to receive messages
    pub fn subscribe(&self) -> broadcast::Receiver<StreamMessage> {
        self.sender.subscribe()
    }

    /// Get current number of connected preview clients
    pub fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

/// Run the debug stream server that accepts preview client connections
pub async fn run_debug_server(
    socket_path: String,
    broadcaster: Arc<DebugStreamBroadcaster>,
    ipc: IpcConfig,
) -> Result<()> {
    // 0600 (or 0660 + group per [ipc] stream_socket_group): the stream carries face data.
    let listener = ipc_policy::bind_stream_socket(&socket_path, &ipc)?;
    info!("Debug stream server listening on {}", socket_path);

    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                // Defense in depth on top of the socket mode; fail closed.
                match ipc_policy::peer_from_stream(&stream) {
                    Ok(peer) if ipc_policy::stream_peer_allowed(&peer, &ipc) => {}
                    Ok(peer) => {
                        warn!(peer_uid = peer.uid, peer_pid = ?peer.pid, "dropped unauthorized debug stream client");
                        continue;
                    }
                    Err(e) => {
                        warn!("dropped debug stream client: {e:#}");
                        continue;
                    }
                }
                let broadcaster = broadcaster.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_debug_client(stream, broadcaster).await {
                        debug!("Preview client disconnected: {}", e);
                    }
                });
            }
            Err(e) => {
                error!("Failed to accept debug connection: {}", e);
            }
        }
    }
}

/// Handle a single preview client connection
async fn handle_debug_client(
    mut stream: UnixStream,
    broadcaster: Arc<DebugStreamBroadcaster>,
) -> Result<()> {
    info!("Preview client connected");

    let mut receiver = broadcaster.subscribe();

    loop {
        match receiver.recv().await {
            Ok(message) => {
                // Serialize message as JSON with newline delimiter
                let json = serde_json::to_string(&message)?;
                let data = format!("{}\n", json);

                // Send to client
                if let Err(e) = stream.write_all(data.as_bytes()).await {
                    debug!("Failed to send to preview client: {}", e);
                    break;
                }
            }
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                warn!("Preview client lagging, skipped {} messages", skipped);
                continue;
            }
            Err(broadcast::error::RecvError::Closed) => {
                info!("Debug stream broadcaster closed");
                break;
            }
        }
    }

    Ok(())
}
