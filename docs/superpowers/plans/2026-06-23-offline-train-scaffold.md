# Offline Train Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe `doorman train` offline command that reads the attempt-log dataset and writes a versioned projector candidate artifact without promoting it.

**Architecture:** The training scaffold lives in the daemon library as pure filesystem code. It reads `<data_dir>/dataset/attempts-*.jsonl` and `labels-*.jsonl`, computes dataset counts, enforces a minimum confirmed-genuine label count, and writes `manifest.json` under `<models_dir>/projector/<version>/`. It never mutates `models/projector/current`.

**Tech Stack:** Rust, serde JSONL, chrono timestamps, clap CLI, tempfile-backed unit tests.

---

## File Structure

- `daemon/src/training.rs`: Pure offline dataset reader, validator, report/manifest writer, and tests.
- `daemon/src/lib.rs`: Exposes `dataset` and `training` modules to the `doorman` CLI.
- `daemon/src/bin/doorman.rs`: Adds `doorman train` command and prints the training report.
- `daemon/src/dataset.rs`: Reused schema types.

---

### Task 1: Library Training Reader

**Files:**
- Create: `daemon/src/training.rs`
- Modify: `daemon/src/lib.rs`

- [ ] Write a failing test that creates a temp `dataset/attempts-YYYYMMDD.jsonl` and `labels-YYYYMMDD.jsonl`, runs `run_offline_training`, and expects a report with `confirmed_genuine = 1`, `promoted = false`, and an artifact directory.
- [ ] Run: `cargo test -p doormand training::tests::writes_projector_candidate_without_promoting`
- [ ] Implement `run_offline_training`.
- [ ] Re-run the targeted test.

### Task 2: Safety Failure For Insufficient Labels

**Files:**
- Modify: `daemon/src/training.rs`

- [ ] Write a failing test that requires `min_confirmed = 2` with only one label and expects an error.
- [ ] Run: `cargo test -p doormand training::tests::refuses_training_without_enough_confirmed_labels`
- [ ] Implement the minimum confirmed-genuine gate.
- [ ] Re-run targeted tests.

### Task 3: CLI Command

**Files:**
- Modify: `daemon/src/bin/doorman.rs`

- [ ] Add `Command::Train { data_dir, models_dir, min_confirmed }`.
- [ ] Add `cmd_train` that calls `doormand::training::run_offline_training`.
- [ ] Print report text in normal mode and JSON in `--json` mode.
- [ ] Run: `cargo test -p doormand`.

### Task 4: Final Verification

- [ ] Run: `git diff --check`.
- [ ] Run: `cargo test -p doormand`.
- [ ] Commit the scaffold.
