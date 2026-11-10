# Round 3: owner verification of PR #13 and follow-up tasks

Read after ROUND2.md. Continue on your branch. Do not push to main.

## Correction to earlier facts

The host models_dir `/var/lib/doorman/models` DOES exist. An earlier truncated listing wrongly said it did not. It holds depth_anything_v2_small_fp32.onnx and depth.ncnn.bin/.param among others (full listing: HOST-FACTS.md section 8). ORT linkage is still NOT confirmed: the owner will re-run `sudo grep -ci onnx /proc/<pid>/maps`.

## Context: verification of PR #13 (head 81a9686)

CI results:
- `check`, `ort-typecheck`, `deb`: pass.
- `shellcheck`: FAILS. The job selects files by a shebang regex and parses HOST-FACTS.md under `.plans/` as shell: 60 findings, all in that one file.
- `python`: FAILS. `tests/test_rnd_depth_fusion_experiment.py` imports numpy, which is not installed in CI; collection error.

Code claims verified OK:
- Unit generator: `render_unit` has no GPU access; `render_depthd_unit` keeps it for `--ep rocm`.
- M1 caps, M3 modes and fsync, M4 backup.
- The V4L2 YUYV-vs-MJPG diagnosis is supported.

Gaps:
- M2: in `mean` mode, mid-sequence frames skip depth, and there is no swap test for `mean`.
- No test asserts that the packaged .service and postinst are GPU-free.
- STATUS.md says branches pr/01..pr/12 were deleted, but they still exist on origin; it also says "CI not verified" (stale).

## Tasks

1. Fix shellcheck file selection: tracked `*.sh` plus files whose first line is a shebang, excluding `.plans/` and `docs/` (or an equivalent that cannot parse markdown).
2. Add numpy (and any other missing test deps) to the python job, or guard the experiment test with `pytest.importorskip`. Re-run so all five jobs are green on the PR.
3. Update STATUS.md accurately. Delete pr/* branches only if the PR chain is superseded; if unsure, ask the owner in the PR. Do not delete anything the owner may need.
4. Add a swap test for `mean` mode, or make the deciding frame plus at least N sampled frames depth-checked in `mean`. Measure the cost with doorman-bench. Defaults stay unchanged unless the owner approves.
5. Add a test asserting that packaging/systemd/doormand.service and packaging/deb/postinst contain none of: char-drm, /dev/dri, /dev/kfd, render.
6. Installed-build question. The host models dir has BOTH depth_anything_v2_small_fp32.onnx and depth.ncnn.bin/.param, and the installed unit header says "EP: ncnn" with Vulkan comments. From the code of the deployed commit (about 5f78657) and from current HEAD, determine which depth path is selected by `[ml] backend="ort", device="cpu"` and by `--device cpu`. Prove from code, with file:line references, that no configuration reachable from the shipped config can select a GPU or Vulkan device unless explicitly opted in.
7. docs/MODELS.md:
   - Licence table. Note that mobilefacenet.onnx and w600k_mbf.onnx come from insightface-style non-commercial weights if used. EdgeFace-S weights are likely CC BY-NC-SA; verify against the Idiap/EdgeFace-S-GAMMA model card.
   - The package must ship only models with a verified redistribution licence.
   - List which files in the host model dir are used by HEAD and which are legacy. Do not delete anything on the host; the owner decides.
8. Keep one PR per logical group where practical. No push to main.

## Owner-only items (do not attempt)

- `sudo grep -ci onnx /proc/$(pidof doormand)/maps`
- Cleanup of legacy model files and /etc/doorman/*.bak*
- Install and config check

## Update

ORT linkage is now confirmed static for the installed build (maps count 0). The installed CLI has no `config` subcommand, so `doorman config check` can only be validated after the PR build is installed. The owner will run the host runbook (install, config check, timing) when ready and report numbers.
