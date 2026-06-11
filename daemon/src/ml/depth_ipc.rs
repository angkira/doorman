//! Wire protocol for the depth-liveness sidecar (`doorman-depthd`).
//!
//! WHY A SIDECAR: loading the Depth-Anything-V2 model on the ONNX Runtime **CPU
//! EP** inside the same process as ROCm/CUDA-EP sessions SEGVs — ORT's global
//! `OrtEnv` cannot mix a CPU session with GPU sessions. Confirmed root cause:
//! without the depth model the ROCm daemon runs fine on the iGPU. So when the
//! daemon runs on a GPU EP, the depth pass is moved OUT of process into this
//! sidecar (which only ever loads the CPU depth model). The detector + recognizer
//! stay on the iGPU in the main daemon.
//!
//! TRANSPORT: a Unix stream socket. One request → one response, length-prefixed,
//! little-endian. Kept deliberately tiny + dependency-free (no serde): the
//! payload is raw RGB8 pixel bytes plus a small fixed-size header, so we avoid
//! base64/JSON-encoding ~2 MB of image data per call.
//!
//! REQUEST  (header then body):
//!   magic:   u32  = `REQ_MAGIC`
//!   width:   u32  (image width, px)
//!   height:  u32  (image height, px)
//!   bbox:    4×f32 (normalized x, y, w, h in [0,1])
//!   then `width*height*3` bytes of RGB8 (row-major, R,G,B per pixel)
//!
//! RESPONSE (fixed 8 bytes):
//!   status:  u32  (`RESP_OK` | `RESP_NONE` | `RESP_ERR`)
//!   relief:  f32  (valid only when status == RESP_OK)
//!
//!   RESP_OK   → a relief score was computed.
//!   RESP_NONE → the sidecar has no depth model loaded (model missing). The
//!               daemon treats this exactly like the in-process `Ok(None)`:
//!               fail-safe reject when liveness is enabled.
//!   RESP_ERR  → an inference/IO error occurred sidecar-side; daemon rejects.

#![allow(dead_code)]

/// Default sidecar socket path (system mode). Lives under the runtime dir so the
/// sandboxed `doorman` user can bind/connect it (alongside `doorman.sock`).
pub const DEFAULT_DEPTH_SOCKET: &str = "/run/doorman/doorman-depth.sock";

/// Environment variable the daemon and sidecar both read to locate the socket,
/// overriding the compiled default. Set by `run_rocm.sh` / the systemd unit.
pub const DEPTH_SOCKET_ENV: &str = "DOORMAN_DEPTH_SOCKET";

pub const REQ_MAGIC: u32 = 0x44_50_54_48; // "DPTH"

pub const RESP_OK: u32 = 0;
pub const RESP_NONE: u32 = 1;
pub const RESP_ERR: u32 = 2;

/// Fixed request header size in bytes: magic(4) + w(4) + h(4) + bbox(4*4) = 28.
pub const REQ_HEADER_LEN: usize = 4 + 4 + 4 + 16;

/// Fixed response size in bytes: status(4) + relief(4) = 8.
pub const RESP_LEN: usize = 8;

/// Resolve the depth sidecar socket path: `DOORMAN_DEPTH_SOCKET` env if set,
/// else the supplied fallback (a runtime-dir default the caller computes).
pub fn resolve_socket_path(fallback: &str) -> String {
    std::env::var(DEPTH_SOCKET_ENV).unwrap_or_else(|_| fallback.to_string())
}

/// Encode a request header (does NOT include the pixel body).
pub fn encode_req_header(width: u32, height: u32, bbox: (f32, f32, f32, f32)) -> [u8; REQ_HEADER_LEN] {
    let mut buf = [0u8; REQ_HEADER_LEN];
    buf[0..4].copy_from_slice(&REQ_MAGIC.to_le_bytes());
    buf[4..8].copy_from_slice(&width.to_le_bytes());
    buf[8..12].copy_from_slice(&height.to_le_bytes());
    buf[12..16].copy_from_slice(&bbox.0.to_le_bytes());
    buf[16..20].copy_from_slice(&bbox.1.to_le_bytes());
    buf[20..24].copy_from_slice(&bbox.2.to_le_bytes());
    buf[24..28].copy_from_slice(&bbox.3.to_le_bytes());
    buf
}

/// Parsed request header.
pub struct ReqHeader {
    pub width: u32,
    pub height: u32,
    pub bbox: (f32, f32, f32, f32),
}

/// Decode a request header, validating the magic. Returns `None` on bad magic.
pub fn decode_req_header(buf: &[u8; REQ_HEADER_LEN]) -> Option<ReqHeader> {
    let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    if magic != REQ_MAGIC {
        return None;
    }
    let width = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let height = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    let bbox = (
        f32::from_le_bytes(buf[12..16].try_into().unwrap()),
        f32::from_le_bytes(buf[16..20].try_into().unwrap()),
        f32::from_le_bytes(buf[20..24].try_into().unwrap()),
        f32::from_le_bytes(buf[24..28].try_into().unwrap()),
    );
    Some(ReqHeader { width, height, bbox })
}

/// Encode a response (status + relief).
pub fn encode_resp(status: u32, relief: f32) -> [u8; RESP_LEN] {
    let mut buf = [0u8; RESP_LEN];
    buf[0..4].copy_from_slice(&status.to_le_bytes());
    buf[4..8].copy_from_slice(&relief.to_le_bytes());
    buf
}

/// Decode a response into `(status, relief)`.
pub fn decode_resp(buf: &[u8; RESP_LEN]) -> (u32, f32) {
    let status = u32::from_le_bytes(buf[0..4].try_into().unwrap());
    let relief = f32::from_le_bytes(buf[4..8].try_into().unwrap());
    (status, relief)
}
