# Attempt Log M1/M2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the default-off attempt-log dataset (M1), then add the first ground-truth labeling pass for daemon-confirmed authentication success (M2).

**Architecture:** Keep the live gallery (`embeddings.bin`) separate from the R&D dataset. The daemon appends embedding-only `AttemptRecord` JSONL rows under `<data_dir>/dataset/attempts-YYYYMMDD.jsonl`; M2 upgrades labels by appending explicit label events instead of rewriting hot-path records in place. The auth and proactive-unlock paths call small dataset helpers best-effort, so logging failures never affect authentication.

**Tech Stack:** Rust, serde JSONL, chrono timestamps, existing doorman config, existing daemon IPC/auth pipeline tests.

---

## File Structure

- `daemon/src/dataset.rs`: Owns attempt-log schema, append-only JSONL writer, label-event schema, and read/update helpers needed by tests and future `doorman train`.
- `shared/src/config.rs`: Adds default-off dataset config and retention knobs.
- `doorman.toml`: Documents default-off dataset config.
- `daemon/src/pipeline/recognition_pipeline.rs`: Logs proactive-unlock scored attempts and rejected scored attempts when the screen-lock pipeline reaches scoring.
- `daemon/src/ipc.rs`: Logs on-demand `Authenticate` attempts and appends a confirmed-genuine label event when face authentication succeeds.
- `daemon/tests/config_tests.rs`: Covers dataset config defaults and parsing.

---

### Task 1: Finish M1 Schema

**Files:**
- Modify: `daemon/src/dataset.rs`
- Modify: `shared/src/config.rs`
- Test: `daemon/src/dataset.rs`
- Test: `daemon/tests/config_tests.rs`

- [x] **Step 1: Write failing schema tests**

Add tests in `daemon/src/dataset.rs` that expect `AttemptRecord` to include:

```rust
depth_proj_emb: None,
gallery_spread: GallerySpreadStats { nearest: 0.91, mean: 0.72, max: 0.91 },
gt_source: GtSource::None,
conditions: AttemptConditions { brightness: Some(0.5), face_area_frac: Some(0.08), sharpness: Some(12.0), hour_utc: Some(14) },
schema_version: 1,
```

Run: `cargo test -p doormand dataset::tests::enabled_logging_appends_jsonl_record_under_dataset_dir`

Expected: FAIL because these fields/types do not exist yet.

- [x] **Step 2: Implement minimal schema**

Add these types in `daemon/src/dataset.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GtSource {
    PamFaceAuth,
    ProactiveThenStayed,
    PasswordFallback,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GallerySpreadStats {
    pub nearest: f32,
    pub mean: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptConditions {
    pub brightness: Option<f32>,
    pub face_area_frac: Option<f32>,
    pub sharpness: Option<f32>,
    pub hour_utc: Option<u32>,
}
```

Extend `AttemptRecord` with `schema_version`, `depth_proj_emb`, `gallery_spread`, `gt_source`, and `conditions`.

- [x] **Step 3: Verify schema tests pass**

Run: `cargo test -p doormand dataset::tests`

Expected: PASS for dataset writer tests.

---

### Task 2: Finish M1 Logging Coverage

**Files:**
- Modify: `daemon/src/dataset.rs`
- Modify: `daemon/src/pipeline/recognition_pipeline.rs`
- Modify: `daemon/src/ipc.rs`
- Test: `daemon/src/pipeline/recognition_pipeline.rs`
- Test: unit tests in `daemon/src/ipc.rs` if needed for pure helpers

- [x] **Step 1: Write failing tests for helper calculations**

Add pure helper tests for:

```rust
gallery_spread_for(&[0.9, 0.7, 0.5]) == GallerySpreadStats { nearest: 0.9, mean: 0.7, max: 0.9 }
outcome_for_scored_attempt(best_match_present, liveness_pass) -> Outcome
```

Run: `cargo test -p doormand gallery_spread`

Expected: FAIL because helpers do not exist.

- [x] **Step 2: Implement helper calculations**

Add pure helpers in `daemon/src/dataset.rs`:

```rust
pub fn gallery_spread_from_similarities(similarities: &[f32]) -> GallerySpreadStats
pub fn attempt_conditions(brightness: Option<f32>, face_area_frac: Option<f32>, sharpness: Option<f32>) -> AttemptConditions
```

Add pipeline-local or dataset-level outcome helper:

```rust
pub fn outcome_from_match(matched: bool, liveness_pass: Option<bool>) -> Outcome
```

- [x] **Step 3: Log rejected scored attempts in recognition pipeline**

In `daemon/src/pipeline/recognition_pipeline.rs`, after scoring `best_similarity`, log `Outcome::Rejected` for no match when dataset logging is enabled. Use the same once-per-lock-generation throttle as proactive outcomes to avoid correlated-frame floods.

- [x] **Step 4: Log on-demand auth attempts**

In `daemon/src/ipc.rs`, after aggregated scoring in `handle_authenticate`, append an `AttemptRecord` with:

```rust
lock_generation: 0,
matched_user: Some(username.to_string()) only when best_similarity >= threshold,
outcome: Outcome::Unknown before final success/failure label,
gt_source: GtSource::None,
```

Then Task 3 will append the confirmation label event when success is known.

- [x] **Step 5: Verify M1 logging coverage**

Run: `cargo test -p doormand`

Expected: PASS.

---

### Task 3: M2 Label Events For Confirmed Face Auth Success

**Files:**
- Modify: `daemon/src/dataset.rs`
- Modify: `daemon/src/ipc.rs`
- Test: `daemon/src/dataset.rs`
- Test: `daemon/src/ipc.rs` pure helper tests if introduced

- [x] **Step 1: Write failing label-event tests**

In `daemon/src/dataset.rs`, add a test that writes a label event to `<data_dir>/dataset/labels-YYYYMMDD.jsonl`:

```rust
LabelEvent {
    schema_version: 1,
    attempt_id: "attempt-1".to_string(),
    timestamp_utc: "2026-06-23T12:00:01Z".to_string(),
    outcome: Outcome::ConfirmedGenuine,
    gt_source: GtSource::PamFaceAuth,
}
```

Run: `cargo test -p doormand dataset::tests::enabled_label_logging_appends_jsonl_event`

Expected: FAIL because `LabelEvent` and `log_label_event` do not exist.

- [x] **Step 2: Implement label-event writer**

Add:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEvent {
    pub schema_version: u32,
    pub attempt_id: String,
    pub timestamp_utc: String,
    pub outcome: Outcome,
    pub gt_source: GtSource,
}

pub fn log_label_event(data_dir: &str, enabled: bool, event: &LabelEvent)
```

Write to `<data_dir>/dataset/labels-YYYYMMDD.jsonl` with the same best-effort permissions as attempts.

- [x] **Step 3: Append confirmed label on auth success**

In `daemon/src/ipc.rs`, when `best_similarity >= threshold`, append a `LabelEvent` for the just-written auth attempt:

```rust
outcome: Outcome::ConfirmedGenuine,
gt_source: GtSource::PamFaceAuth,
```

Do not label failed face auth as confirmed genuine. Do not try to infer password fallback success; PAM password fallback happens after `pam_doorman` returns failure and is outside the daemon today.

- [x] **Step 4: Verify M2 tests pass**

Run: `cargo test -p doormand`

Expected: PASS.

---

### Task 4: Final Verification And R&D Notes

**Files:**
- Modify: `.plans/2026-06-23_learnable-projector-rnd.md` only if implementation semantics differ from the original text.

- [x] **Step 1: Verify whitespace**

Run: `git diff --check`

Expected: no output, exit code 0.

- [x] **Step 2: Verify daemon tests**

Run: `cargo test -p doormand`

Expected: all non-ignored tests pass. Existing warnings in ML tests are allowed if unchanged.

- [x] **Step 3: Record exact M1/M2 boundary**

If password fallback is not implemented, add a short note to `.plans/2026-06-23_learnable-projector-rnd.md` that current M2 labels daemon-confirmed face-auth success only, and password fallback labeling needs a separate trusted PAM-side signal.
