#![no_main]
//! One IPC request line through the daemon's gate (size cap, UTF-8, JSON
//! envelope, protocol version, authorize) for a fixed unprivileged peer.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = doormand::fuzz_entry::ipc_request_line(data);
});
