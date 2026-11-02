# Install the PR #13 build on your host

Step by step, for the owner. The branch is `claude/great-ride-i3oqn6` (PR #13).
Every step that changes the system is a `sudo` command you run yourself; this
document changes nothing. Paths use `$HOME`; adjust the placeholders in angle
brackets.

> **Do NOT use `doorman install` for this path.** It is the from-source developer
> installer, not a way to upgrade a packaged install. It rebuilds from source,
> installs the PAM module, writes `/etc/pam.d/<service>`, copies the user's
> `~/.local/share/doorman/embeddings.bin` into the system store and restarts the
> services. This document installs the prebuilt `.deb` instead, which edits no
> PAM file and touches no enrolment.
>
> **Do NOT run `apt purge doorman` or `apt remove --purge doorman`**, now or
> during a rollback. See "Never purge" below.

## 0. Open a second root shell

In another terminal or TTY (Ctrl+Alt+F3), run `sudo -i` and keep that shell
open until the verification in step 9 is done (and, for step 12, until `sudo`
is tested from a third terminal). If the new daemon, the lock screen or PAM
misbehaves, a session that is already root needs no `sudo` and no face or
password prompt to repair it. Opening one afterwards may be impossible,
because `sudo` is exactly what can break. PAM and sudo come last (step 12),
never earlier.

## Never purge

`packaging/deb/postrm` runs on `purge` (`apt purge doorman`,
`apt remove --purge doorman`, `dpkg -P doorman`):

```sh
rm -rf /etc/doorman /var/lib/doorman
userdel doorman; groupdel doorman
```

That deletes, with no copy and no prompt:

* `/etc/doorman/` completely: `doorman.toml`, `doorman.toml.bak`,
  `doorman.toml.new`, anything else in it. The package declares no conffiles,
  so dpkg does not protect these files either.
* `/var/lib/doorman/` completely: `embeddings.bin` (all enrolments) and
  `embeddings.bin.v1.bak`, `models/` (including the depth model and any
  `depth.ncnn.*`), the `miopen` and `.cache` directories, and the dataset
  (attempt records and capture packs) if logging was enabled.
* The system user `doorman` and the group `doorman`.

A plain `apt remove doorman` (no `--purge`) leaves all of these in place.

## What is different from your installed build

* The installed daemon (about commit `5f78657`) has no `doorman config`
  subcommand. `doorman config check` exists only in this build.
* ONNX Runtime is statically linked in the installed build (`grep -ci onnx
  /proc/<pid>/maps` printed `0`) and stays so in the deb.
* The packaged unit grants no GPU access. Your host has a drop-in
  `10-drm.conf` that does; it must go (step 8).
* The first write of `embeddings.bin` converts it to a new format (version 2)
  that older daemons cannot read. A copy is made automatically
  (`embeddings.bin.v1.bak`), and step 1 makes your own tarball first.

## 1. Pre-flight backups

Do this before installing, in the root shell from step 0. The old `.deb` is not
available and the new package has the same version number, so `apt` cannot
downgrade: this tarball is the rollback (step 13). Nothing below is overwritten
without these copies.

```sh
# in the root shell
umask 077
systemctl stop doormand                  # a consistent embeddings.bin
R="/root/doorman-rollback-$(date +%Y%m%d-%H%M%S)"
install -d -m 0700 -o root -g root "$R"  # embeddings are biometric data

# 1. what the package owns right now (informational and the base of the list)
dpkg -L doorman | tee "$R/dpkg-L.txt"
dpkg -l doorman | tail -n 1 > "$R/installed-package.txt"

# 2. files to archive: package files (not directories), the PAM module wherever
#    it lives, unit files and drop-ins, config, enrolments
{
  while IFS= read -r f; do
    if [ -f "$f" ] || [ -L "$f" ]; then printf '%s\n' "$f"; fi
  done < "$R/dpkg-L.txt"
  find /lib/*/security /usr/lib/*/security -name '*pam_doorman.so' 2>/dev/null
  for p in /usr/bin/doorman* /usr/local/bin/doorman* \
           /etc/systemd/system/doormand.service /etc/systemd/system/doormand.service.d \
           /etc/doorman /var/lib/doorman/embeddings.bin*; do
    if [ -e "$p" ]; then printf '%s\n' "$p"; fi
  done
} | sort -u > "$R/files.txt"

# 3. archive them (tar strips the leading / and restores with -C /)
tar --numeric-owner -cpf "$R/doorman-installed.tar" -T "$R/files.txt"
tar -tvf "$R/doorman-installed.tar" | grep -E 'bin/doorman|pam_doorman|\.service|embeddings|doorman.toml'
ls -l "$R"
systemctl start doormand
```

The explicit set this covers: `/usr/bin/doorman*` (`doormand`, `doorman`), the
PAM module (`libpam_doorman.so` and the `pam_doorman.so` link, normally under
`/usr/lib/<multiarch>/security/`), `/lib/systemd/system/doormand.service`,
`/usr/lib/systemd/user/doormand.service`, `/etc/systemd/system/doormand.service`
and `doormand.service.d/` (your `10-drm.conf`), `/etc/doorman/`, and
`/var/lib/doorman/embeddings.bin` (with `.v1.bak` if present). Models under
`/var/lib/doorman/models` are not archived (large, restored by the package
`postinst`).

Check the tarball lists the binaries and `embeddings.bin` before going on.
Keep it until the new build has run for a while. Copy it off the machine if you
want it to survive a disk failure; it holds your face templates, so keep the
copy private.

Daemon-side, the first v2 write also copies a legacy `embeddings.bin` to
`/var/lib/doorman/embeddings.bin.v1.bak` (mode 0600, never overwritten).

## 2. Get the .deb

CI built it from the PR head. Either:

* download the artifact `doorman-deb` from the latest `release.yml` run for the
  PR #13 head (the `deb` job, step "Upload CI artifact"): web UI, Actions,
  `release`, the newest run on branch `claude/great-ride-i3oqn6`, "Artifacts"; or
  ```sh
  gh run list --repo angkira/doorman --workflow release.yml \
      --branch claude/great-ride-i3oqn6 --limit 5
  gh run download <release.yml-run-id> --repo angkira/doorman \
      --name doorman-deb --dir "$HOME/doorman-deb"
  ```
  Take the newest successful run whose head commit is the current branch tip
  (`git ls-remote https://github.com/angkira/doorman.git claude/great-ride-i3oqn6`).
  Do not reuse an old run id: older runs carry older builds, so this document
  does not hard-code one; or
* build it yourself from a checkout of the branch:
  ```sh
  git clone --branch claude/great-ride-i3oqn6 https://github.com/angkira/doorman.git "$HOME/doorman-pr13"
  cd "$HOME/doorman-pr13"
  scripts/build_deb.sh            # writes target/debian/dist/doorman_<version>_<arch>.deb
  # or only the binaries: cargo build --release --locked -p doormand --bin doormand --bin doorman
  ```
  Build dependencies: `INSTALL.md` section 1.

Check what you are about to install:

```sh
DEB="$(ls "$HOME"/doorman-deb/*.deb 2>/dev/null || ls target/debian/dist/*.deb)"
dpkg-deb -I "$DEB"
dpkg-deb -c "$DEB" | grep -E 'bin/|\.service|postinst|models/'
```

## 3. Stop the daemon

```sh
sudo systemctl stop doormand
systemctl is-active doormand    # prints "inactive"
```

## 4. Install

```sh
sudo apt install "$DEB"        # or: sudo dpkg -i "$DEB"
```

The package's `postinst` keeps your `/etc/doorman/doorman.toml` (it only
rewrites the old `/run/doorman.sock` socket path, keeping
`doorman.toml.bak`). It also scans `/etc/systemd/system/doormand.service` and
`doormand.service.d/*.conf` and prints a WARNING per file and line that grants
GPU access (`char-drm`, `/dev/dri`, `/dev/kfd`, a `render` group). It never edits
those files. It restarts the daemon at the end; the first start may log
warnings until the config is fixed (step 5).

If you used `doorman install` from a source checkout instead of the deb, an
existing `/etc/doorman/doorman.toml` is also left unchanged: the rendered config
is written to `/etc/doorman/doorman.toml.new` and a WARNING says so (see step 6).

## 5. Check the config

```sh
doorman config check --config /etc/doorman/doorman.toml; echo "exit=$?"
```

Exit codes: `0` ok, `1` invalid, `2` valid but unknown or deprecated keys.
Expected on your host: **exit 2** with

```
deprecated key: dataset.raw_face_frame_bundle_size (removed in 7c511d9 and ignored: ...)
```

The key has no effect. Remove it by hand (`sudo nano /etc/doorman/doorman.toml`,
delete the line, keep `raw_frame_bundle_size`), then run the check again until it
exits 0 (or lists only keys you understand). Your explicit values stay in force
(for example `keep_raw_frames = true`; set it to `false` yourself if you want the
privacy default). Defaults of keys your file omits can differ from the old build:
compare with `docs/CONFIG.md`.

## 6. Handle `doorman.toml.new` (only if it exists)

```sh
ls -l /etc/doorman/doorman.toml.new
diff -u /etc/doorman/doorman.toml /etc/doorman/doorman.toml.new
```

Merge what you want into `/etc/doorman/doorman.toml` by hand, then delete the
`.new` file. Do not `mv` it over your config without reading the diff.

## 7. Remove stale files (your decision)

Only after the diff and config check above. These are optional and
irreversible; look first.

```sh
ls -l /usr/local/bin/doormand /etc/doorman/*.bak* 2>/dev/null
sudo rm /usr/local/bin/doormand            # shadows /usr/bin/doormand if it is earlier on PATH
sudo rm /etc/doorman/*.bak*                # after you looked at them
```

`/var/lib/doorman/models` may hold legacy model files; `docs/MODELS.md` says which
ones this build uses. Do not delete anything there you are unsure about.

## 8. Remove the GPU drop-in

```sh
systemctl cat doormand | grep -nE 'char-drm|/dev/dri|/dev/kfd|render'
sudo rm /etc/systemd/system/doormand.service.d/10-drm.conf
sudo systemctl daemon-reload
systemctl cat doormand | grep -nE 'char-drm|/dev/dri|/dev/kfd|render' || echo "no GPU grants"
```

The first command lists what is granted now (the postinst warning shows the same).
If `systemctl cat` shows the grant in the unit file itself, edit or replace that
file by hand: the package unit is `/lib/systemd/system/doormand.service` and has none.

## 9. Restart and verify

```sh
sudo systemctl restart doormand
systemctl is-active doormand
journalctl -u doormand -b --no-pager | grep -E 'Using ML backend|ONNX Runtime|Dataset capture \(effective\)|Camera ready'
doorman status
ls -l /run/doorman
```

Expected:

* `Using ML backend: ...` (ONNX Runtime CPU backend; no ncnn or Vulkan).
* `ONNX Runtime: bundled (ort download-binaries); ... no libonnxruntime mapping
  (runtime is statically linked)`.
* `Dataset capture (effective): logging_enabled=... keep_raw_frames=...` matching
  your config.
* `Camera ready: backend=... resolution=... fourcc=... open_ms=... first_frame_ms=...`.
  This build requests MJPG (the camera offers MJPG 1280x720@30; YUYV only up to
  640x480, which is why the old V4L2 open failed within 0.1 ms). Note the
  `open_ms` and `first_frame_ms` values.
* `doorman status` shows models loaded, enrolled users, and the ONNX Runtime line.
* `/run/doorman` holds `doorman.sock` owned by `doorman` (the socket is 0666 by
  design; access is decided by peer credentials, see `docs/ipc-security.md`).

Then test before touching anything else:

```sh
sudo systemctl show doormand -p DeviceAllow -p SupplementaryGroups   # no GPU entries
doorman test "$USER"
```

If `doorman test` rejects a genuine face, do not change the liveness threshold
yet; collect timings first (step 11).

## 10. Lock-screen scripts (niri)

The niri scripts are not in the deb. They are installed **side by side** in
their own directory, so your current `~/.local/bin/doorman-lock` and
`doorman-lock-watch-poll` are never overwritten. From the checkout of the branch:

```sh
cd "$HOME/doorman-pr13"
EVENT_DIR="$HOME/.local/bin/doorman-event"
install -D -m 0755 packaging/adapters/niri/bin/doorman-lock "$EVENT_DIR/doorman-lock"
install -D -m 0755 packaging/adapters/niri/bin/doorman-lock-watch-poll "$EVENT_DIR/doorman-lock-watch-poll"
ls -l "$EVENT_DIR"
command -v doorman
```

`doorman-lock` finds `doorman-lock-watch-poll` next to itself, so keep the two in
the same directory. Point the niri bind at the new path in
`~/.config/niri/config.kdl` (your existing bind probably runs
`$HOME/.local/bin/doorman-lock`; change only that path):

```kdl
Mod+L allow-inhibiting=false hotkey-overlay-title="Lock" { spawn-sh "exec \"$HOME/.local/bin/doorman-event/doorman-lock\""; }
```

If you also lock from waybar (`waybar-niri-click`), change the path in that
script the same way. Apply with `niri msg action load-config-file` (or log out
and in), then lock once and read `$XDG_RUNTIME_DIR/doorman/lock.log`. When gtklock
exits with rc=1 while the session is locked, the log shows `attaching watcher` with
the PID of the live gtklock. Checklist: `docs/unlock-path.md`.

Roll back by pointing the bind (and the waybar script) at
`$HOME/.local/bin/doorman-lock` again; that file was never touched. To remove the
new scripts: `rm -r "$HOME/.local/bin/doorman-event"`.

(`scripts/install_local_niri_desktop.sh` is a different, from-source installer: it
overwrites `~/.local/bin/doorman-lock` and replaces your niri and waybar configs,
with backups. Do not use it for this path.)

## 11. Enable timing log (to collect numbers)

In `/etc/doorman/doorman.toml`:

```toml
[telemetry]
timing_log = true
timing_log_path = "/run/doorman/timing.jsonl"   # the default; rotated at 1 MiB
```

```sh
sudo systemctl restart doormand
for i in $(seq 20); do doorman test "$USER"; sleep 2; done
sudo wc -l /run/doorman/timing.jsonl
```

Summarise the latency (needs `jq`; the file is not world-readable, hence `sudo`).
`total_ms` is the whole attempt; `camera_open_ms` and `first_valid_frame_ms` are
the camera stages (`null` when a stage did not run, dropped by the `select`).
Any numeric field of `TimingRecord` in `daemon/src/timing.rs` works the same way:

```sh
# p50 and p95 of one field over the matched attempts
pct() {  # usage: pct FIELD
    sudo jq -s --arg f "$1" '
        [ .[] | select(.outcome == "matched") | .[$f] | select(. != null) ] | sort as $v
        | ($v | length) as $n
        | if $n == 0 then "no samples"
          else { field: $f, n: $n,
                 p50: $v[($n * 0.50 | floor)],
                 p95: $v[([($n * 0.95 | floor), $n - 1] | min)] }
          end' /run/doorman/timing.jsonl
}
pct total_ms
pct camera_open_ms
pct first_valid_frame_ms
# how the attempts ended
sudo jq -r .outcome /run/doorman/timing.jsonl | sort | uniq -c
```

The file holds timings and the relief value only, never frames or embeddings.
Send it back for the latency and threshold decisions (`docs/decision-and-liveness.md`).
Turn `timing_log` off again afterwards if you do not need it.

## 12. sudo and PAM, last

Not before steps 1 to 11 all pass. Follow `docs/SUDO.md` exactly, with a second
root shell open that stays open until you have tested `sudo` from a third
terminal. Nothing in the package edits PAM.

## 13. Rollback

The old `.deb` is not available and the new package has the same version, so
`apt install --allow-downgrades` does nothing useful. Restore from the step 1
tarball in `$R` instead, in the root shell. Never `apt purge` (see "Never purge").

```sh
# in the root shell; R is the directory from step 1
systemctl stop doormand
cp -a /var/lib/doorman/embeddings.bin "$R/embeddings.bin.v2-after-upgrade"   # enrolments made since
tar --numeric-owner -xpf "$R/doorman-installed.tar" -C /
ls -l /var/lib/doorman/embeddings.bin* /usr/bin/doorman*
systemctl daemon-reload
systemctl start doormand
systemctl is-active doormand
```

This restores the old binaries, PAM module, unit files and drop-in,
`/etc/doorman` and the v1 `embeddings.bin` over what the new package
installed. Files the new package added that the old one did not have stay.
dpkg still records the new package version, so `dpkg -V doorman` reports the
changed files; that is expected until you install a package again. Enrolments
made on the new build are not in the v1 file: re-enrol with the old build.
Restoring the old GPU drop-in is not recommended: the daemon does not need the
GPU. Restore the niri scripts with the commands in step 10.

Optional, **untested**: rebuild the old commit as a `.deb` and reinstall it
with `dpkg -i`, which accepts the same version. This is not needed if the
tarball restore works.

```sh
git clone https://github.com/angkira/doorman.git "$HOME/doorman-old"
cd "$HOME/doorman-old"
git checkout 5f78657
scripts/build_deb.sh          # may not exist or behave the same at that commit
sudo dpkg -i target/debian/dist/doorman_*.deb
```

Then put `embeddings.bin` and `/etc/doorman/doorman.toml` back from the tarball
as above.
