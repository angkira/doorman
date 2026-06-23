# Debian Release Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a buildable Debian package and GitHub release workflow for doorman system installs.

**Architecture:** Keep packaging as a thin layer around the existing Rust binaries, PAM module, config, and systemd units. A repo-local shell builder stages the package under `target/debian`, while a validation script enforces the package contract without needing root.

**Tech Stack:** Rust/Cargo, POSIX shell, `dpkg-deb`, GitHub Actions.

---

### Task 1: Contract Validation

**Files:**
- Create: `scripts/check_deb_packaging.sh`

- [ ] **Step 1: Write the failing validation script**

The script must require the Debian metadata, maintainer scripts, package builder, and release workflow.

- [ ] **Step 2: Run test to verify it fails**

Run: `bash scripts/check_deb_packaging.sh`

Expected: failure mentioning at least one missing packaging file.

- [ ] **Step 3: Keep the script as the packaging regression check**

The script should run `bash -n` over shell files, reject maintainer scripts that touch `/etc/pam.d`, require `dpkg-deb --build`, and require the GitHub workflow to call both validation and package build.

### Task 2: Debian Metadata And Maintainer Scripts

**Files:**
- Create: `packaging/deb/control.in`
- Create: `packaging/deb/conffiles`
- Create: `packaging/deb/postinst`
- Create: `packaging/deb/prerm`
- Create: `packaging/deb/postrm`
- Create: `packaging/deb/README.Debian`

- [ ] **Step 1: Add metadata**

`control.in` must use substituted `@VERSION@`, `@ARCH@`, and `@INSTALLED_SIZE@`.

- [ ] **Step 2: Add safe maintainer scripts**

`postinst` creates the system user and state directories, then reloads systemd. `prerm` stops/disables the service on removal. `postrm purge` removes config, state, and user. No script edits PAM.

- [ ] **Step 3: Run validation**

Run: `bash scripts/check_deb_packaging.sh`

Expected: still fail until the builder and workflow exist.

### Task 3: Package Builder

**Files:**
- Create: `scripts/build_deb.sh`

- [ ] **Step 1: Build release binaries**

Run Cargo for `doormand`, `doorman`, and `pam_doorman`.

- [ ] **Step 2: Stage package files**

Copy binaries, PAM module, config, systemd units, docs, and Debian control files into `target/debian/doorman_<version>_<arch>`.

- [ ] **Step 3: Build `.deb`**

Run `dpkg-deb --build --root-owner-group` and write the artifact to `target/debian/dist`.

- [ ] **Step 4: Validate**

Run: `bash scripts/check_deb_packaging.sh`

Expected: pass once the workflow also exists.

### Task 4: GitHub Release Workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Add CI release workflow**

On pushes to `main` and tags `v*`, run tests, validation, and package build.

- [ ] **Step 2: Upload artifacts**

Upload `.deb` artifacts for all runs. For tags, publish them to the GitHub Release using `gh release upload`.

- [ ] **Step 3: Final verification**

Run:

```bash
bash scripts/check_deb_packaging.sh
cargo test -p doormand
git diff --check
```

Expected: all pass.
