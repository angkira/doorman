#![no_main]
//! `embeddings.bin` loader, version 1 (headerless) and version 2 (magic + version).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = doormand::fuzz_entry::embeddings_file(data);
});
