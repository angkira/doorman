#![no_main]
//! Config TOML: parse, `Config::validate`, and the `config check` unknown-key diff.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = doormand::fuzz_entry::config_toml(data);
});
