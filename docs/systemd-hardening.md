# systemd hardening review: `doormand.service`

Unit: `packaging/systemd/doormand.service` (system service, user `doorman`).
This is a static review; nothing here was run against a live systemd.

## Directive checklist

| Directive | State | Note |
|---|---|---|
| `NoNewPrivileges=true` | present | |
| `ProtectSystem=strict` | present | whole FS read-only except the paths below |
| `ProtectHome=true` | present | the daemon never needs `$HOME`; models live in `/var/lib/doorman/models` or `/usr/share/doorman/models` |
| `PrivateTmp=true` | present | |
| `MemoryMax=1G` | present (1G, not 2G) | see "Open points" |
| `StateDirectory=doorman` / `ReadWritePaths=/var/lib/doorman` | present | writable state, embeddings, dataset |
| `RuntimeDirectory=doorman` | present | `/run/doorman` (socket) is writable even under `ProtectSystem=strict` |
| `DevicePolicy=closed` + `DeviceAllow=char-video4linux rw` | present | camera only |
| `DeviceAllow=/dev/video0 rw` | present, redundant | subsumed by `char-video4linux`; left to keep the diff small |
| `DeviceAllow=/dev/dri rw` | **removed** | exposed GPU render nodes; the daemon runs on CPU (`--device cpu`). |
| `RestrictAddressFamilies=AF_UNIX` | present | no network |

## GPU policy

The unit grants no GPU device nodes (`/dev/dri/*`, `/dev/kfd`, `/dev/nvidia*`). With
`DevicePolicy=closed` they are inaccessible. If an owner deliberately configures a GPU
backend (`backend-ort-rocm`, `backend-ort-cuda`) they must add the needed nodes
themselves in a drop-in, for example `systemctl edit doormand` with

    [Service]
    DeviceAllow=/dev/kfd rw
    DeviceAllow=/dev/dri/renderD128 rw

Do not do this for a discrete GPU reserved for other work.

## Open points (documented, not changed)

- `DevicePolicy=closed` could block capture back-ends that open nodes other than V4L2
  character devices (the ffmpeg CLI fallback and GStreamer `v4l2src` both open
  `/dev/video*`, which `char-video4linux` covers; PipeWire capture does not work from this
  system unit at all and is for the user unit). Unverified on real hardware: after
  changes, check `journalctl -u doormand` for `Operation not permitted` on `/dev/video*`
  and confirm unlock still works.
- `MemoryMax=1G` is below the 2G suggested in the WS8 plan. ORT with the 518x518 depth model
  plus frame buffers may approach that on some hosts. Not changed without a measurement;
  watch `systemctl status doormand` (Memory:) and raise in a drop-in if OOM-killed.
- `MemoryDenyWriteExecute=false` is set explicitly; the unit gives no reason. ONNX Runtime
  may need executable mappings [unverified], so test before flipping it to `true`.
- `doormand-user.service` has only `NoNewPrivileges=true`; it is a dev unit and is out of
  this review.

## Owner commands

    systemd-analyze security doormand          # exposure score and per-directive table
    systemd-analyze verify /lib/systemd/system/doormand.service
    systemctl show doormand -p DeviceAllow -p DevicePolicy -p MemoryMax

Record the resulting score here after the first run on the host:
`systemd-analyze security doormand`: **[not yet measured]**.
