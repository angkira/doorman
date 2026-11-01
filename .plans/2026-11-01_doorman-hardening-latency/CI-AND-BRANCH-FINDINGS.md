# CI and branch findings (owner host, 2026-11-01)

## 1. Why main's `deb` job is red (release run 41333094776)

- Failing step: job `deb`, step "Test doormand" (`cargo test -p doormand`, release.yml line 39), exit 101. Later steps (validate, build deb, upload) are skipped.
- Errors: `error[E0061]` at `daemon/src/ml/ort_backend.rs:424` and `:435`: `Session::builder()` is called with 0 arguments, but ort 2.0.0-rc.15 takes `&Environment`.
- Root cause:
  - `daemon/Cargo.toml:59` has `ort = { version = "2.0.0-rc.12", ... }`, a caret requirement, so CI resolves rc.15.
  - `Cargo.lock` is gitignored (`.gitignore` line 7) and untracked on main, so CI resolves fresh. The owner's local ignored lock pins rc.12, which is why local builds work.
  - The code was written against rc.12. Upgrading to rc.15 needs a code change; do NOT do it in the hotfix.

## 2. Minimal fix

1. `daemon/Cargo.toml:59`: `"2.0.0-rc.12"` -> `"=2.0.0-rc.12"`.
2. Remove `Cargo.lock` from `.gitignore` (line 7) and commit the lock.

The pin alone unblocks CI; the lock gives reproducible builds.
Both already exist on branch `claude/great-ride-i3oqn6` (pin in daemon/Cargo.toml:59, `.gitignore` hunk, lock pinning ort rc.12), but bundled with unrelated changes. Cherry-pick only those hunks into the hotfix PR.

## 3. Verification of `origin/claude/great-ride-i3oqn6` @ c9edb91

| Check | Result |
|---|---|
| `cargo fmt --all --check` | FAIL, ~40 files (style drift). CI runs this, so CI is red. |
| `cargo clippy --workspace --all-targets -- -D warnings` | FAIL. `shared/src/config.rs`: `derivable_impls`, `io_other_error` (both also present on main). With those allowed, `doormand` / `pam_doorman` fail with ~20 more: `too_many_arguments`, unused imports/variables (e.g. `num_anchors`), `needless_return`, `useless_vec`, `module_inception`. |
| `cargo test --workspace --locked` | PASS: 262 passed, 0 failed, 4 ignored |
| `pytest` | `python3 -m pytest tests` fails collection (no module `doorman`). Needs `PYTHONPATH=src`: 28 passed, 2 skipped. |
| `shellcheck` | Not installed on the owner host; not run. |

Consequence: the branch's own CI (`.github/workflows/ci.yml`) would be red on fmt and clippy until separate formatting and lint PRs land.
