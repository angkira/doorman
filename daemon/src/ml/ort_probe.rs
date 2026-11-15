//! Find out which ONNX Runtime shared library the process actually mapped.
//!
//! With `ort/load-dynamic` the library is `dlopen`ed from `ORT_DYLIB_PATH` or, when
//! that is unset, by the bare name through the dynamic linker search path, so the
//! configured path alone does not prove what was loaded. `/proc/self/maps` does.
//! The parser is pure so it can be tested on a synthetic maps string.

// Only the ORT backend calls this outside tests. The module stays compiled
// without it (the bins re-declare `ml`) so its unit tests run in every build.
#![cfg_attr(not(feature = "_ort"), allow(dead_code))]

use std::path::Path;

/// File-name prefix of every ONNX Runtime shared object (core, providers).
const ORT_LIB_PREFIX: &str = "libonnxruntime";

/// Number of whitespace-separated columns before the path in `/proc/<pid>/maps`
/// (address, perms, offset, dev, inode).
const MAPS_COLUMNS_BEFORE_PATH: usize = 5;

/// Unique ONNX Runtime library paths mapped according to a `/proc/<pid>/maps`
/// dump, in order of first appearance.
pub fn onnxruntime_mappings(maps: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for line in maps.lines() {
        // Skip address, perms, offset, dev, inode; the next column is the path.
        let Some(path) = line.split_whitespace().nth(MAPS_COLUMNS_BEFORE_PATH) else {
            continue;
        };
        let is_ort = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(ORT_LIB_PREFIX));
        if is_ort && !found.iter().any(|p| p == path) {
            found.push(path.to_string());
        }
    }
    found
}

/// Human-readable summary of the ONNX Runtime libraries mapped into this process
/// and of `ORT_DYLIB_PATH`, for the startup log. Never fails: problems are
/// reported in the returned text.
pub fn describe_loaded_onnxruntime() -> String {
    let requested = match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) => format!("ORT_DYLIB_PATH={p}"),
        Err(std::env::VarError::NotPresent) => "ORT_DYLIB_PATH unset".to_string(),
        Err(e) => format!("ORT_DYLIB_PATH unreadable ({e})"),
    };
    let mapped = match std::fs::read_to_string("/proc/self/maps") {
        Ok(maps) => {
            let libs = onnxruntime_mappings(&maps);
            if libs.is_empty() {
                "no libonnxruntime mapping (runtime is statically linked)".to_string()
            } else {
                format!("mapped {}", libs.join(", "))
            }
        }
        Err(e) => format!("/proc/self/maps unreadable ({e})"),
    };
    format!("{requested}; {mapped}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAPS: &str = "\
7f0000000000-7f0000100000 r--p 00000000 00:2a 11 /usr/lib/doorman/libonnxruntime.so
7f0000100000-7f0000900000 r-xp 00100000 00:2a 11 /usr/lib/doorman/libonnxruntime.so
7f0001000000-7f0001100000 r--p 00000000 00:2a 12 /usr/lib/doorman/libonnxruntime_providers_shared.so
7f0002000000-7f0002100000 r--p 00000000 00:2a 13 /usr/lib/x86_64-linux-gnu/libc.so.6
7f0003000000-7f0003100000 rw-p 00000000 00:00 0
";

    #[test]
    fn lists_unique_ort_libraries_in_order() {
        assert_eq!(
            onnxruntime_mappings(MAPS),
            vec![
                "/usr/lib/doorman/libonnxruntime.so".to_string(),
                "/usr/lib/doorman/libonnxruntime_providers_shared.so".to_string(),
            ]
        );
    }

    #[test]
    fn anonymous_and_unrelated_mappings_are_ignored() {
        let maps = "7f00-7f01 rw-p 00000000 00:00 0\n7f02-7f03 r--p 0 00:2a 5 /lib/libc.so.6\n";
        assert!(onnxruntime_mappings(maps).is_empty());
    }
}
