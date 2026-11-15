//! At-rest encryption key lifecycle (ROUND 10).
//!
//! The file envelope (v4) lives in [`crate::storage`]; this module owns the
//! 256-bit data key: generation, mlock'd in-RAM handling with zeroization on
//! drop, loading from `$CREDENTIALS_DIRECTORY` (systemd `LoadCredential*` /
//! `SetCredentialEncrypted`), AEAD seal/open of byte blobs, and the
//! password/PIN-wrapped recovery blob.
//!
//! Threat model: the sealed file and the recovery blob are unreadable off-host
//! and to unprivileged local users. A live root can still dump daemon RAM —
//! that is documented out of scope in `docs/STORAGE-ENCRYPTION.md`, not claimed
//! otherwise. Nothing here ever logs key material.

use anyhow::{Context, Result};
use argon2::{Argon2, Params};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305, Key, Nonce,
};
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

/// Length of the file data key and of a PIN-derived wrap key (bytes).
pub const DATA_KEY_LEN: usize = 32;
/// Name of the credential as seen under `$CREDENTIALS_DIRECTORY` and in the
/// unit drop-in (`SetCredentialEncrypted=<name>:...`).
pub const CREDENTIAL_NAME: &str = "doorman-storage-key";
/// Sidecar of `embeddings.bin` holding the PIN-wrapped data key.
pub const PIN_BLOB_NAME: &str = "embeddings.bin.pin";
/// ChaCha20-Poly1305 nonce length.
pub const NONCE_LEN: usize = 12;
/// Salt length for the PIN KDF.
pub const PIN_SALT_LEN: usize = 16;
/// AAD domain separators (never reused across contexts).
const AAD_FILE: &[u8] = b"doorman-embeddings-v4";
const AAD_PIN_WRAP: &[u8] = b"doorman-pin-wrap-v1";
const PIN_BLOB_VERSION: u32 = 1;
/// Argon2id work factors for the PIN wrap (interactive use by root only).
const PIN_M_COST_KIB: u32 = 19 * 1024;
const PIN_T_COST: u32 = 2;
const PIN_P_COST: u32 = 1;

/// OS randomness mapped into `anyhow` (`getrandom::Error` carries only an
/// errno, so `Context` does not apply to it).
fn rand_fill(buf: &mut [u8]) -> Result<()> {
    getrandom::fill(buf).map_err(|e| anyhow::anyhow!("OS RNG failed: {e}"))
}

/// A 256-bit data key. The bytes are `mlock`'d while alive when the platform
/// allows it (a failure only warns: containers often forbid it) and are
/// zeroized on drop. Never `Clone`, never logged, never serialized.
pub struct DataKey {
    key: Zeroizing<[u8; DATA_KEY_LEN]>,
    locked: bool,
}

impl DataKey {
    /// Wrap caller-provided bytes (e.g. from the credential directory or the
    /// unlock IPC). Copies them once into locked memory.
    pub fn new(bytes: [u8; DATA_KEY_LEN]) -> Self {
        let key = Zeroizing::new(bytes);
        let locked = lock_pages(key.as_ptr(), DATA_KEY_LEN);
        Self { key, locked }
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let arr: [u8; DATA_KEY_LEN] = bytes.try_into().map_err(|_| {
            anyhow::anyhow!("data key must be {DATA_KEY_LEN} bytes, got {}", bytes.len())
        })?;
        Ok(Self::new(arr))
    }

    /// Fresh random key from the OS RNG.
    pub fn generate() -> Result<Self> {
        let mut bytes = [0u8; DATA_KEY_LEN];
        rand_fill(&mut bytes)?;
        Ok(Self::new(bytes))
    }

    pub fn bytes(&self) -> &[u8; DATA_KEY_LEN] {
        &self.key
    }

    /// Whether `mlock` holds for the key pages (observability only).
    pub fn is_locked(&self) -> bool {
        self.locked
    }
}

impl Drop for DataKey {
    fn drop(&mut self) {
        if self.locked {
            unlock_pages(self.key.as_ptr(), DATA_KEY_LEN);
        }
        // `Zeroizing` zeroes the bytes after this runs.
    }
}

fn lock_pages(ptr: *const u8, len: usize) -> bool {
    // SAFETY: `ptr`/`len` cover the live `Zeroizing` array; `mlock` only pins
    // the pages, it never reads or writes through the pointer.
    let addr = std::ptr::NonNull::new(ptr as *mut std::os::raw::c_void)
        .expect("key array pointer is never null");
    let rc = unsafe { nix::sys::mman::mlock(addr, len) };
    match rc {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!("mlock of the data key failed ({e}); key stays pageable");
            false
        }
    }
}

fn unlock_pages(ptr: *const u8, len: usize) {
    // SAFETY: same range as the matching `lock_pages`.
    if let Some(addr) = std::ptr::NonNull::new(ptr as *mut std::os::raw::c_void) {
        let _ = unsafe { nix::sys::mman::munlock(addr, len) };
    }
}

/// Read the data key systemd decrypted into `$CREDENTIALS_DIRECTORY/<name>`.
/// `Ok(None)` = no credential configured (encryption not enrolled or TPM-only
/// mode not wired yet); `Err` = present but unusable (fail closed upstream).
pub fn load_from_credentials_dir() -> Result<Option<DataKey>> {
    let dir = match std::env::var("CREDENTIALS_DIRECTORY") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let path = Path::new(&dir).join(CREDENTIAL_NAME);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("cannot read {path:?}")),
    };
    if bytes.len() != DATA_KEY_LEN {
        anyhow::bail!(
            "credential {path:?} must hold {DATA_KEY_LEN} bytes, got {}",
            bytes.len()
        );
    }
    tracing::info!("storage data key loaded from the systemd credential");
    Ok(Some(DataKey::from_slice(&bytes)?))
}

/// AEAD-seal `plaintext` for the file envelope: `nonce || ciphertext+tag`.
/// The caller prefixes the version header; AAD binds this exact envelope.
pub fn seal_blob(plaintext: &[u8], key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand_fill(&mut nonce_bytes)?;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    let ct = cipher
        .encrypt(
            Nonce::from_slice(&nonce_bytes),
            Payload {
                msg: plaintext,
                aad: AAD_FILE,
            },
        )
        .map_err(|e| anyhow::anyhow!("AEAD seal failed: {e}"))?;
    let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    Ok(out)
}

/// Open a `seal_blob` envelope. Any error (wrong key, bit-flip, truncation)
/// is an authentication failure: the caller must fail closed.
pub fn open_blob(envelope: &[u8], key: &[u8; DATA_KEY_LEN]) -> Result<Vec<u8>> {
    let (nonce_bytes, ct) = envelope
        .split_at_checked(NONCE_LEN)
        .with_context(|| format!("sealed envelope is truncated ({} bytes)", envelope.len()))?;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    cipher
        .decrypt(
            Nonce::from_slice(nonce_bytes),
            Payload {
                msg: ct,
                aad: AAD_FILE,
            },
        )
        .map_err(|_| {
            anyhow::anyhow!("sealed envelope failed authentication (wrong key or tampered)")
        })
}

/// PIN-wrapped recovery blob (JSON sidecar next to `embeddings.bin`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct PinBlob {
    pub version: u32,
    pub salt: Vec<u8>,
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
}

fn pin_wrap_key(pin: &str, salt: &[u8]) -> Result<[u8; DATA_KEY_LEN]> {
    let params = Params::new(PIN_M_COST_KIB, PIN_T_COST, PIN_P_COST, Some(DATA_KEY_LEN))
        .map_err(|e| anyhow::anyhow!("PIN KDF params rejected: {e}"))?;
    let argon = Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut out = [0u8; DATA_KEY_LEN];
    argon
        .hash_password_into(pin.as_bytes(), salt, &mut out)
        .map_err(|e| anyhow::anyhow!("PIN KDF failed: {e}"))?;
    Ok(out)
}

/// Wrap `key` with `pin` into a storable blob. The PIN itself is never stored.
/// CLI-side only (the daemon binary never calls it); the dual lib/bin build
/// needs the explicit allow outside tests.
#[cfg_attr(not(test), allow(dead_code))]
pub fn wrap_with_pin(key: &[u8; DATA_KEY_LEN], pin: &str) -> Result<PinBlob> {
    if pin.is_empty() {
        anyhow::bail!("PIN must not be empty");
    }
    let mut salt = vec![0u8; PIN_SALT_LEN];
    rand_fill(&mut salt)?;
    let wrap_key = pin_wrap_key(pin, &salt)?;
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand_fill(&mut nonce_bytes)?;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&wrap_key));
    let ct = cipher
        .encrypt(
            Nonce::from_slice(&nonce_bytes),
            Payload {
                msg: key,
                aad: AAD_PIN_WRAP,
            },
        )
        .map_err(|e| anyhow::anyhow!("PIN wrap failed: {e}"))?;
    Ok(PinBlob {
        version: PIN_BLOB_VERSION,
        salt,
        nonce: nonce_bytes.to_vec(),
        ciphertext: ct,
    })
}

/// Unwrap a PIN blob. Wrong PIN and tampering are indistinguishable by design.
/// CLI-side only; see `wrap_with_pin`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn unwrap_with_pin(blob: &PinBlob, pin: &str) -> Result<[u8; DATA_KEY_LEN]> {
    if blob.version != PIN_BLOB_VERSION {
        anyhow::bail!("unsupported PIN blob version {}", blob.version);
    }
    if blob.salt.len() != PIN_SALT_LEN || blob.nonce.len() != NONCE_LEN {
        anyhow::bail!("malformed PIN blob");
    }
    let wrap_key = pin_wrap_key(pin, &blob.salt)?;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&wrap_key));
    let pt = cipher
        .decrypt(
            Nonce::from_slice(&blob.nonce),
            Payload {
                msg: &blob.ciphertext,
                aad: AAD_PIN_WRAP,
            },
        )
        .map_err(|_| anyhow::anyhow!("PIN unwrap failed (wrong PIN or tampered blob)"))?;
    if pt.len() != DATA_KEY_LEN {
        anyhow::bail!("PIN blob decrypted to {} bytes", pt.len());
    }
    let mut out = [0u8; DATA_KEY_LEN];
    out.copy_from_slice(&pt);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Enrollment wiring (pure builders; the CLI runs them as root).
// ---------------------------------------------------------------------------

/// Unit drop-in installed by `doorman key enroll`. CLI-side only.
#[cfg_attr(not(test), allow(dead_code))]
pub fn dropin_path() -> PathBuf {
    PathBuf::from("/etc/systemd/system/doormand.service.d/10-storage-key.conf")
}

/// `systemd-creds encrypt` argv sealing the key file with the host TPM.
/// `pcrs` is e.g. `"7"` (see docs/STORAGE-ENCRYPTION.md for the tradeoff).
/// `encrypt` always takes two positionals (`INPUT OUTPUT`); with `-p` the
/// sealed `SetCredentialEncrypted=` line goes to stdout and `-` (stdout) is
/// passed as OUTPUT. CLI-side only: pure builder, executed by `doorman key`.
#[cfg_attr(not(test), allow(dead_code))]
pub fn tpm_seal_argv(key_file: &Path, pcrs: &str) -> (String, Vec<String>) {
    (
        "systemd-creds".to_string(),
        vec![
            "encrypt".to_string(),
            "--with-key=tpm2".to_string(),
            "--tpm2-device=auto".to_string(),
            format!("--tpm2-pcrs={pcrs}"),
            format!("--name={CREDENTIAL_NAME}"),
            "-p".to_string(),
            key_file.as_os_str().to_string_lossy().into_owned(),
            "-".to_string(),
        ],
    )
}

/// Render the drop-in from one `systemd-creds encrypt -p` output line
/// (`SetCredentialEncrypted=<name>:<blob>`). CLI-side only.
#[cfg_attr(not(test), allow(dead_code))]
pub fn render_dropin(sealed_line: &str) -> Result<String> {
    let line = sealed_line.trim();
    let prefix = format!("SetCredentialEncrypted={CREDENTIAL_NAME}:");
    if !line.starts_with(&prefix) || line.len() <= prefix.len() {
        anyhow::bail!("systemd-creds did not return a {CREDENTIAL_NAME} line");
    }
    Ok(format!(
        "# Sealed storage key for doormand (see docs/STORAGE-ENCRYPTION.md).\n\
         # Written by `doorman key enroll`; root-owned. The blob below decrypts\n\
         # only on this machine in the enrolled boot state (TPM2).\n\
         [Service]\n{line}\n"
    ))
}

/// Path of the PIN sidecar for a data directory.
pub fn pin_blob_path(data_dir: &Path) -> PathBuf {
    data_dir.join(PIN_BLOB_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> [u8; DATA_KEY_LEN] {
        [7u8; DATA_KEY_LEN]
    }

    #[test]
    fn seal_open_round_trip() {
        let key = test_key();
        let pt = b"headerless-v1-body-bytes";
        let env = seal_blob(pt, &key).unwrap();
        assert!(env.len() > NONCE_LEN);
        assert_eq!(open_blob(&env, &key).unwrap(), pt);
    }

    #[test]
    fn open_rejects_wrong_key_tamper_and_truncation() {
        let env = seal_blob(b"secret", &test_key()).unwrap();
        let mut other = test_key();
        other[0] ^= 1;
        assert!(open_blob(&env, &other).is_err());
        let mut cut = env.clone();
        let n = cut.len();
        cut[n - 1] ^= 1;
        assert!(open_blob(&env[..], &test_key()).is_ok());
        assert!(open_blob(&cut, &test_key()).is_err());
        assert!(open_blob(&env[..4], &test_key()).is_err());
    }

    #[test]
    fn pin_wrap_round_trip_and_wrong_pin() {
        let blob = wrap_with_pin(&test_key(), "correct horse").unwrap();
        assert_eq!(blob.version, PIN_BLOB_VERSION);
        assert_eq!(unwrap_with_pin(&blob, "correct horse").unwrap(), test_key());
        assert!(unwrap_with_pin(&blob, "wrong pin").is_err());
        assert!(wrap_with_pin(&test_key(), "").is_err());
    }

    #[test]
    fn pin_blob_is_versioned_json() {
        let blob = wrap_with_pin(&test_key(), "pin").unwrap();
        let text = serde_json::to_string(&blob).unwrap();
        assert!(text.contains("\"version\":1"));
        let back: PinBlob = serde_json::from_str(&text).unwrap();
        assert_eq!(back, blob);
    }

    #[test]
    fn builders_produce_expected_wiring() {
        let (prog, args) = tpm_seal_argv(Path::new("/tmp/k"), "7");
        assert_eq!(prog, "systemd-creds");
        assert!(args.contains(&"--with-key=tpm2".to_string()));
        assert!(args.contains(&"--tpm2-pcrs=7".to_string()));
        assert!(args.contains(&"--name=doorman-storage-key".to_string()));
        assert_eq!(args.last().map(String::as_str), Some("-"));
        let dropin = render_dropin("SetCredentialEncrypted=doorman-storage-key: ABCDEF\n").unwrap();
        assert!(dropin.contains("[Service]"));
        assert!(dropin.contains("SetCredentialEncrypted=doorman-storage-key: ABCDEF"));
        assert!(render_dropin("garbage").is_err());
        assert_eq!(
            pin_blob_path(Path::new("/var/lib/doorman")),
            PathBuf::from("/var/lib/doorman/embeddings.bin.pin")
        );
        assert_eq!(
            dropin_path(),
            PathBuf::from("/etc/systemd/system/doormand.service.d/10-storage-key.conf")
        );
    }

    #[test]
    fn no_credential_dir_means_no_key() {
        let old = std::env::var("CREDENTIALS_DIRECTORY").ok();
        unsafe { std::env::remove_var("CREDENTIALS_DIRECTORY") };
        assert!(load_from_credentials_dir().unwrap().is_none());
        if let Some(v) = old {
            unsafe { std::env::set_var("CREDENTIALS_DIRECTORY", v) };
        }
    }

    #[test]
    fn data_key_holds_32_bytes() {
        let k = DataKey::from_slice(&[1u8; 32]).unwrap();
        assert_eq!(k.bytes(), &[1u8; 32]);
        assert!(DataKey::from_slice(&[1u8; 31]).is_err());
        let g = DataKey::generate().unwrap();
        assert_eq!(g.bytes().len(), DATA_KEY_LEN);
    }
}
