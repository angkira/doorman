# Install the PR #13 build on your host

Step by step, for the owner. The branch is `claude/great-ride-i3oqn6` (PR #13).
Every step that changes the system is a `sudo` command you run yourself; this
document changes nothing. Paths use `$HOME`; adjust the placeholders in angle
brackets.

Keep a second root shell open from step 1 until the end of step 9
(`sudo -i` in another terminal). PAM and sudo come last (step 12), never
earlier.

## What is different from your installed build

* The installed daemon (about commit `5f78657`) has no `doorman config`
  subcommand. `doorman config check` exists only in this build.
* ONNX Runtime is statically linked in the installed build (`grep -ci onnx
  /proc/<pid>/maps` printed `0`) and stays so in the deb.
* The packaged unit grants no GPU access. Your host has a drop-in
  `10-drm.conf` that does; it must go (step 8).
* The first write of `embeddings.bin` converts it to a new format (version 2)
  that older daemons cannot read. A backup is made automatically (step 1).

## 1. Pre-flight backups

Do this before installing. Nothing below is overwritten without these copies.

```sh
B="$HOME/doorman-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$B"
sudo cp -a /var/lib/doorman/embeddings.bin "$B/"
sudo cp -a /etc/doorman/doorman.toml "$B/"
sudo cp -a /etc/systemd/system/doormand.service "$B/" 2>/dev/null || true
sudo cp -a /etc/systemd/system/doormand.service.d "$B/" 2>/dev/null || true
sudo cp -a /usr/bin/doormand /usr/bin/doorman "$B/" 2>/dev/null || true
dpkg -l doorman 2>/dev/null | tail -n 1 > "$B/installed-package.txt" || true
sudo chown -R "$(id -u):$(id -g)" "$B"
ls -l "$B"
```

Also keep the old `.deb` file if you still have it; rollback (step 13) needs it.
Daemon-side, the first v2 write copies a legacy `embeddings.bin` to
`/var/lib/doorman/embeddings.bin.v1.bak` (mode 0600, never overwritten).

## 2. Get the .deb

CI built it from the PR head. Either:

* download the artifact `doorman-deb` from the `release.yml` run for PR #13 (the
  `deb` job, step "Upload CI artifact"): web UI, Actions, `release`, the run for
  the PR head, "Artifacts"; or
  ```sh
  gh run download <release.yml-run-id> --repo angkira/doorman \
      --name doorman-deb --dir "$HOME/doorman-deb"
  ```
  (the run id is the number at the end of the run URL, for the head
  `e50cd65` it is `41355386218`); or
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

The niri scripts are not in the deb. From the checkout of the branch:

```sh
cd "$HOME/doorman-pr13"
install -D -m 0755 packaging/adapters/niri/bin/doorman-lock "$HOME/.local/bin/doorman-lock"
install -D -m 0755 packaging/adapters/niri/bin/doorman-lock-watch-poll "$HOME/.local/bin/doorman-lock-watch-poll"
command -v doorman doorman-lock
```

(`scripts/install_local_niri_desktop.sh` also replaces your niri and waybar
configs, with backups; the two commands above do not.) Bind the lock key to
`doorman-lock`, lock once, and read `$XDG_RUNTIME_DIR/doorman/lock.log`. When gtklock
exits with rc=1 while the session is locked, the log shows `attaching watcher` with
the PID of the live gtklock. Checklist: `docs/unlock-path.md`.

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

The file holds timings and the relief value only, never frames or embeddings.
Send it back for the latency and threshold decisions (`docs/decision-and-liveness.md`).
Turn `timing_log` off again afterwards if you do not need it.

## 12. sudo and PAM, last

Not before steps 1 to 11 all pass. Follow `docs/SUDO.md` exactly, with a second
root shell open that stays open until you have tested `sudo` from a third
terminal. Nothing in the package edits PAM.

## 13. Rollback

Stop the daemon first. Use your step 1 copies in `$B`.

```sh
sudo systemctl stop doormand

# 1. old package (the .deb you kept) or the old binaries
sudo apt install --allow-downgrades "<path to the old doorman .deb>"
#    or: sudo install -m 0755 "$B/doormand" /usr/bin/doormand; sudo install -m 0755 "$B/doorman" /usr/bin/doorman

# 2. embeddings: the daemon wrote a v2 file; the old daemon needs the v1 file
sudo ls -l /var/lib/doorman/embeddings.bin*
sudo cp -a /var/lib/doorman/embeddings.bin "$B/embeddings.bin.v2-after-upgrade"
sudo install -o doorman -g doorman -m 0600 /var/lib/doorman/embeddings.bin.v1.bak /var/lib/doorman/embeddings.bin
#    if there is no .v1.bak, use the copy from step 1:
#    sudo install -o doorman -g doorman -m 0600 "$B/embeddings.bin" /var/lib/doorman/embeddings.bin

# 3. config and unit
sudo install -m 0644 "$B/doorman.toml" /etc/doorman/doorman.toml
#    only if you removed them and want the old unit back:
#    sudo cp -a "$B/doormand.service" /etc/systemd/system/ ; sudo cp -a "$B/doormand.service.d" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start doormand
systemctl is-active doormand
```

Enrolments made on the new build (after the backup) are not in the v1 file;
re-enrol with the old build. Restoring the old GPU drop-in is not recommended:
the daemon does not need the GPU.
