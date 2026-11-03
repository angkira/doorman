//! Stable-toolchain stand-in for the cargo-fuzz targets in `fuzz/`: a fixed
//! seed, 10k random and mutated inputs per target, through the same entry
//! points (`doormand::fuzz_entry`). Asserts no panic (the entry points assert
//! their own invariants) and a generous wall-clock bound per target.

use doormand::fuzz_entry::{
    config_toml, embeddings_file, ipc_request_line, synthetic_user, IpcOutcome, FUZZ_USER,
};
use doormand::storage::{Storage, UserEmbedding};
use std::time::{Duration, Instant};

const SEED: u64 = 0x5eed_d00a_0001;
const ITERATIONS: usize = 10_000;
/// Upper bound for one target's whole run; a hang or blow-up fails the test.
const TARGET_BUDGET: Duration = Duration::from_secs(120);
/// Longest generated input (the daemon's own cap is 64 KiB per line).
const MAX_INPUT: usize = 4096;
/// Mutations applied to a seed to make one input.
const MAX_MUTATIONS: u64 = 8;

/// SplitMix64: tiny, deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// Mutate a seed: bit flips, byte replacement, truncation, insertion,
/// duplication, dictionary splice; or (1 in 8) pure random bytes.
fn mutate(rng: &mut Rng, seeds: &[Vec<u8>], dict: &[&[u8]]) -> Vec<u8> {
    if rng.below(8) == 0 {
        let len = rng.below(MAX_INPUT);
        return (0..len).map(|_| rng.next() as u8).collect();
    }
    let mut data = seeds[rng.below(seeds.len())].clone();
    for _ in 0..=rng.below(MAX_MUTATIONS as usize) {
        match rng.below(6) {
            0 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] ^= 1 << rng.below(8);
            }
            1 if !data.is_empty() => {
                let i = rng.below(data.len());
                data[i] = rng.next() as u8;
            }
            2 => data.truncate(rng.below(data.len() + 1)),
            3 => {
                let i = rng.below(data.len() + 1);
                data.insert(i, rng.next() as u8);
            }
            4 if !data.is_empty() => {
                let a = rng.below(data.len());
                let b = a + rng.below(data.len() - a);
                let chunk = data[a..b].to_vec();
                let at = rng.below(data.len() + 1);
                data.splice(at..at, chunk);
            }
            _ => {
                let token = dict[rng.below(dict.len())];
                let at = rng.below(data.len() + 1);
                data.splice(at..at, token.iter().copied());
            }
        }
        data.truncate(MAX_INPUT);
    }
    data
}

fn run_target(name: &str, seeds: &[Vec<u8>], dict: &[&[u8]], mut body: impl FnMut(&[u8])) {
    let mut rng = Rng(SEED);
    let started = Instant::now();
    for seed in seeds {
        body(seed);
    }
    for _ in 0..ITERATIONS {
        let input = mutate(&mut rng, seeds, dict);
        body(&input);
    }
    assert!(
        started.elapsed() < TARGET_BUDGET,
        "{name}: {ITERATIONS} inputs took {:?}",
        started.elapsed()
    );
}

fn lines(v: &[&str]) -> Vec<Vec<u8>> {
    v.iter().map(|s| format!("{s}\n").into_bytes()).collect()
}

#[test]
fn ipc_request_lines_never_panic_and_never_over_grant() {
    let seeds = lines(&[
        r#"{"type":"authenticate","username":"alice","protocol_version":1}"#,
        r#"{"type":"authenticate","username":"bob","service":"sudo","tty":"/dev/pts/3","protocol_version":1}"#,
        r#"{"type":"authenticate","username":"alice"}"#,
        r#"{"type":"wait_for_match","username":"alice","timeout_ms":5000,"protocol_version":1}"#,
        r#"{"type":"enroll","username":"alice","add":true,"protocol_version":1}"#,
        r#"{"type":"remove_user","username":"alice"}"#,
        r#"{"type":"status"}"#,
        r#"{"type":"list_users","protocol_version":2}"#,
        r#"{"type":"shutdown","protocol_version":null}"#,
        r#"{"type":"detect_and_recognize"}"#,
        r#"{"type":"get_latest_detection"}"#,
    ]);
    let dict: &[&[u8]] = &[
        b"\"type\"",
        b"\"username\"",
        b"\"protocol_version\"",
        b"\"authenticate\"",
        b"\"shutdown\"",
        b"4294967295",
        b"18446744073709551616",
        b"-1",
        b"null",
        b"\\u0000",
        b"\xff\xfe",
        b"\n",
        b"{",
        b"}",
    ];
    let mut allowed = 0usize;
    let mut rejected = 0usize;
    run_target("ipc", &seeds, dict, |data| match ipc_request_line(data) {
        IpcOutcome::Allowed(_) => allowed += 1,
        IpcOutcome::Rejected => rejected += 1,
        IpcOutcome::Unparsed => {}
    });
    // The corpus must actually reach the gate, otherwise the test proves nothing.
    assert!(
        allowed > 0 && rejected > 0,
        "{allowed} allowed, {rejected} rejected"
    );
}

#[test]
fn oversize_and_unterminated_lines_are_bounded() {
    // The cap is 64 KiB; one byte more must be refused without a panic.
    let big = vec![b'a'; 64 * 1024 + 1];
    assert_eq!(ipc_request_line(&big), IpcOutcome::Unparsed);
    let nested = "[".repeat(100_000);
    assert_eq!(ipc_request_line(nested.as_bytes()), IpcOutcome::Unparsed);
    let ok = format!(r#"{{"type":"status","username":"{FUZZ_USER}"}}"#);
    assert_eq!(
        ipc_request_line(ok.as_bytes()),
        IpcOutcome::Allowed("status")
    );
}

#[test]
fn config_toml_never_panics() {
    let shipped = include_str!("../../doorman.toml").as_bytes().to_vec();
    let seeds = vec![
        shipped,
        b"".to_vec(),
        b"schema_version = 1\n[auth]\nthreshold = 0.4\n".to_vec(),
        b"[ipc]\nadmin_uids = [0, 1000]\nmax_connections_per_uid = 8\n".to_vec(),
        b"[dataset]\nraw_face_frame_bundle_size = 3\n".to_vec(),
        b"[camera]\nfourcc = \"MJPG\"\nwidth = 640\n".to_vec(),
    ];
    let dict: &[&[u8]] = &[
        b"[auth]\n",
        b"[ipc]\n",
        b"[pam.sudo]\n",
        b"threshold = ",
        b"schema_version = 99\n",
        b"nan",
        b"inf",
        b"-1",
        b"18446744073709551615",
        b"\"\"",
        b"[[",
        b"= {",
        b"\"\"\"",
        b"\\u",
    ];
    let mut accepted = 0usize;
    run_target("config", &seeds, dict, |data| {
        if config_toml(data) {
            accepted += 1;
        }
    });
    assert!(accepted > 0, "no mutated config was ever accepted");
}

/// Version 1: headerless bincode of the user list.
fn legacy_file(users: &[UserEmbedding]) -> Vec<u8> {
    bincode::serialize(users).unwrap()
}

#[test]
fn embeddings_loader_never_panics() {
    let users = vec![synthetic_user("alice", 3, 8), synthetic_user("bob", 1, 4)];
    let seeds = vec![
        Storage::encode_embeddings(&users).unwrap(),
        Storage::encode_embeddings(&[]).unwrap(),
        legacy_file(&users),
        legacy_file(&[]),
        b"DRMNEMB\0".to_vec(),
        b"DRMNEMB\0\x02\x00\x00\x00".to_vec(),
    ];
    let dict: &[&[u8]] = &[
        b"DRMNEMB\0",
        b"\x01\x00\x00\x00",
        b"\x02\x00\x00\x00",
        b"\xff\xff\xff\xff\xff\xff\xff\xff",
        b"\x00\x00\x00\x00\x00\x00\x00\x00",
        b"\x00\x00\xc0\x7f",
        b"\x00\x00\x80\x7f",
    ];
    let mut decoded = 0usize;
    run_target("embeddings", &seeds, dict, |data| {
        if embeddings_file(data).is_some() {
            decoded += 1;
        }
    });
    assert!(decoded > 0, "no mutated file ever decoded");
}

#[test]
fn embeddings_header_cases_are_explicit_errors() {
    // Huge declared lengths must fail, not allocate or panic.
    let mut v1 = u64::MAX.to_le_bytes().to_vec();
    v1.extend_from_slice(&[0; 16]);
    assert_eq!(embeddings_file(&v1), None);
    let mut v2 = b"DRMNEMB\0\x02\x00\x00\x00".to_vec();
    v2.extend_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(embeddings_file(&v2), None);
    assert_eq!(embeddings_file(b"DRMNEMB\0\x63\x00\x00\x00"), None);
}

#[test]
fn deeply_nested_config_values_are_errors_not_stack_overflows() {
    let arrays = format!("x = {}{}\n", "[".repeat(100_000), "]".repeat(100_000));
    assert!(!config_toml(arrays.as_bytes()));
    let tables = format!("x = {}1{}\n", "{a=".repeat(50_000), "}".repeat(50_000));
    assert!(!config_toml(tables.as_bytes()));
    let dotted = format!("{}=1\n", "a.".repeat(50_000) + "b");
    assert!(!config_toml(dotted.as_bytes()));
}
