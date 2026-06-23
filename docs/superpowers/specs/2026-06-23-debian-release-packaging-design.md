# Debian Release Packaging Design

## Goal

Make doorman easy to reinstall, update, and publish from GitHub by adding a first-class Debian package build and a GitHub release workflow.

## Scope

This first packaging milestone targets Debian/Ubuntu system installs only. It packages the daemon, CLI, PAM module, config template, and systemd units. It does not build Flatpak yet; Flatpak should later be limited to GUI/client surfaces because the daemon requires PAM, systemd, camera device access, a system user, `/etc/doorman`, and `/var/lib/doorman`.

## Package Layout

The package installs vendor-owned files, not local admin overrides:

- `/usr/bin/doormand`
- `/usr/bin/doorman`
- `/usr/lib/<multiarch>/security/libpam_doorman.so`
- `/usr/lib/<multiarch>/security/pam_doorman.so` as a compatibility symlink
- `/etc/doorman/doorman.toml` as a Debian conffile
- `/lib/systemd/system/doormand.service`
- `/usr/lib/systemd/user/doormand.service`
- `/usr/share/doc/doorman/README.Debian`

Runtime state remains outside the package payload:

- `/var/lib/doorman`
- `/var/lib/doorman/models`
- `/var/lib/doorman/miopen`
- `/var/lib/doorman/.cache`

`postinst` creates the `doorman` system user and state directories. It reloads systemd but does not enable or start the service automatically. PAM files under `/etc/pam.d` are never edited by maintainer scripts.

## Build Flow

`scripts/build_deb.sh` builds a lean CPU package using Rust release binaries:

- `doormand` and `doorman` from package `doormand`
- `libpam_doorman.so` from package `pam_doorman`

The script stages files under `target/debian/`, substitutes version and architecture into `packaging/deb/control.in`, then runs `dpkg-deb --build --root-owner-group`.

The default feature set is CPU ONNX Runtime plus V4L2 and ffmpeg cameras. GPU-specific packages are intentionally out of scope for this first milestone.

## Release Flow

GitHub Actions builds and tests on Ubuntu, runs the package validation script, builds the `.deb`, and uploads it as a release artifact when a tag matching `v*` is pushed. The workflow also uploads the package as a normal CI artifact for manual inspection.

## Safety

The package preserves enrolled data on normal removal. Purge removes `/etc/doorman`, `/var/lib/doorman`, and the system user. Password authentication is not changed by package install or upgrade because PAM integration remains an explicit user/admin step.
