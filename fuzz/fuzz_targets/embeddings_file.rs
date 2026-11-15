#![no_main]
//! `embeddings.bin` loader: versions 1 (headerless), 2 (magic + version),
//! 3 (metadata) and 4 (sealed envelope; keyless input must refuse, not decode).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = doormand::fuzz_entry::embeddings_file(data);
});
