# Biometric self-assessment protocol

How the owner measures, on their own machine and camera, how well doorman
resists presentation attacks (PAD) and impostors (matching), and how to word the
result honestly. The method follows the practice of ISO/IEC 30107-3 (PAD
testing) and ISO/IEC 19795-1 (biometric performance testing). Both standards
are paywalled and were not available while writing this; the metrics below are
described generically and no clause text is quoted. This is a self-assessment,
NOT a certification, and nothing here is a statement that doorman conforms to
either standard.

Tools: `doorman-bench record`, `accuracy`, `pad-report`, `match-report`
(see [README.md](README.md) for build and the other subcommands).

## Metrics in one paragraph

* **APCER**: attack presentations wrongly accepted. Computed per attack
  species (one attack instrument type); the worst species is the headline, never
  a pooled number.
* **BPCER**: bona fide presentations wrongly rejected.
* **Non-response** (APNRR/BPNRR-style): presentations the system gave no answer
  to (no face found, timeout). Reported separately; excluded from the APCER and
  BPCER denominators.
* **FMR / FNMR**: impostor comparisons accepted / genuine comparisons rejected.
* **FTA**: failure to acquire (no usable face image).
* Every rate gets a one-sided 95% **Clopper-Pearson upper bound**. With `k`
  errors in `n` independent trials it is the exact binomial bound; with `k = 0`
  it is `1 - 0.05^(1/n)` (about `3/n`, the "rule of 3").

## How many presentations are enough (k = 0)

`n >= ln(0.05) / ln(1 - p)` presentations with zero accepted attacks show
`rate <= p` at 95% one-sided confidence. `pad-report` and `match-report` compute
the same numbers and say how many more you need.

| Target | n at k = 0 | Rule of 3 (3/p) |
|---|---|---|
| APCER <= 7% | 42 | 43 |
| APCER <= 1% | 299 | 300 |
| FMR <= 1e-4 | 29,956 | 30,000 |
| FMR <= 2e-5 | 149,786 | 150,000 |

One accepted attack raises the 1% requirement from 299 to 473 for that
species. Plan for the k = 0 number and treat it as a minimum, not a
prediction. The ISO/IEC 19795-1 "rule of 30" (about 30 errors for a usefully
narrow interval on a rate you expect to be nonzero) applies to rates you
measure, such as BPCER or FNMR, rather than to rates you hope are zero.

## Part 1: presentation attack detection

### What is measured

The PAD result describes the **liveness (depth relief) gate**: the daemon's
`liveness_relief` against `[authentication] liveness_depth_threshold`. An
attack that carries the owner's face also passes face matching, so the gate is
the part under test. Depth relief from an RGB camera is a deterrent, not a
security boundary (docs/decision-and-liveness.md); the protocol exists to put a
number on how weak it is.

### Attack species

One instrument per species; an attack is a fresh, complete unlock attempt (not
a frame of an ongoing replay: frames of one replay are correlated).

| species (CSV value) | Instrument |
|---|---|
| `print_matte` | Matte paper print of the owner's face, flat |
| `print_glossy` | Glossy photo print, flat |
| `print_curved` | Either print bent around a cylinder (bottle, mug) |
| `phone_photo` | Still photo shown on a phone screen |
| `phone_replay` | Video of the owner replayed on a phone |
| `screen_replay` | Video or photo shown on a laptop or monitor screen |
| `paper_mask` | (optional) Printed face with eye holes / paper mask worn |

Species that are absent from the CSV are listed as **untested** in the claim
line. Add your own species by using any other name in the CSV; it is reported
as its own row.

### Presentations per species

* To claim APCER <= 1%: at least **299** scored attacks per species, zero
  accepted (use 300).
* To claim APCER <= 7%: at least **42** per species (use 45).
* Seven species at 300 each is 2,100 attempts; choose which species you can
  afford and claim only those.
* Spread each species over **at least 3 sessions** (different days), **2
  lighting conditions** (daylight, evening lamp), and **2 distances** (about 40
  cm and 70 cm). Vary the instrument too (different print, different phone
  angle). Record the session in the CSV.
* Bona fide: at least **300** genuine attempts over the same sessions and
  lighting conditions, owner only. To claim BPCER <= 3% or better you need many
  more; BPCER is usually nonzero, so read the reported bound.

### Collecting scores

1. Enable the timing log (docs/bench/README.md, "Daemon timing log") and point
   it at one file per run so each file has one label, one species and one
   session. Each unlock attempt writes one JSON line with `liveness_relief`.
2. Run the attempts (`doorman test`, lock-screen unlock, or `sudo` as you
   normally use it).
3. Convert to the PAD CSV. Header: `label,species,score,response,session,subject`.
   `response` is `ok`, `no_face` or `timeout`; the score is empty for
   non-responses. The timing log has `liveness_relief = null` when no depth
   check ran; the command below calls that `no_face`. If you know an attempt
   was a timeout, edit its row.

```sh
# one run: label, species, session as shell variables; timing log copied to run.jsonl
add() { # add <jsonl> <bona_fide|attack> <species|none> <session>
  jq -r --arg l "$2" --arg sp "$3" --arg s "$4" \
    'if .liveness_relief != null
       then "\($l),\($sp),\(.liveness_relief),ok,\($s),owner"
       else "\($l),\($sp),,no_face,\($s),owner" end' "$1"
}
echo 'label,species,score,response,session,subject' > pad.csv
add genuine-day-s1.jsonl bona_fide none s1  >> pad.csv
add print_matte-s1.jsonl attack print_matte s1 >> pad.csv
add phone_replay-s2.jsonl attack phone_replay s2 >> pad.csv
```

`doorman-bench record` can additionally keep the raw frames of a session for
later offline work. Layout under the dataset root (outside any git repository;
`record` refuses to write inside one and refuses the mock camera):

```
~/doorman-data/<session>/
  me/<condition>/NNNN.jpg          bona fide frames (--user me --condition day)
  attacks/<species>/NNNN.jpg       attack frames (--attack print_matte)
```

Use one dataset root per session (`--out ~/doorman-data/2026-11-05`) and the
species names of the table as `--attack` values, so directory names and CSV
`species` values agree. The `record` command stores frames; the relief values
come from the daemon timing log, because the harness cannot yet score relief
from images (see the TODO list in README.md).

### Report

```sh
doorman-bench pad-report --scores pad.csv --threshold 0.05 --target-apcer 0.01 --out ~/doorman-reports/pad
```

`--threshold` is `[authentication] liveness_depth_threshold` (accept iff
`score >= threshold`). `--species-column NAME` reads another column name
instead of `species`; `--target-apcer` defaults to 0.01. Outputs: `pad_report.md`
(claim line, per-species table, BPCER, non-response rates, subject/session
counts, warnings) and `pad_report.csv` (one row per species plus BPCER:
`kind,species,presented,non_responses,n,errors,rate,upper_bound,required_n,additional_needed`).

## Part 2: matching (FMR / FNMR)

### What you need

* Genuine probes: the owner's frames in conditions that are **not** the
  enrolment images (a genuine probe identical to an enrolment image scores 1.0
  and flatters the result). Record enrolment under a separate user name:

```sh
doorman-bench record --out ~/doorman-data/2026-11-05 --user me_enrol --condition day
doorman-bench record --out ~/doorman-data/2026-11-05 --user me --condition evening_lamp
doorman-bench record --out ~/doorman-data/2026-11-05 --user me --condition backlit
```

* An impostor set: many different people. 29,956 comparisons are needed for
  FMR <= 1e-4 at zero false matches, but comparisons from few people are
  correlated; the report shows `n_subjects` and a conservative subject-level
  bound. Real impostor recordings (friends, with consent) are the most
  relevant and the smallest; a public set supplies the volume.

### Public impostor set: recommendation

Never download a dataset into this repository; keep it outside any git
repository and delete it when done.

| Dataset | Identities | Licence / status (checked 2026-11-03) | Use here |
|---|---|---|---|
| **DigiFace-1M** (Microsoft) | about 110,000 synthetic: 10,000 x 72 images and 100,000 x 5 images (about 1.22 M images) | Research Use of Data Agreement v1.0 (R-UDA): computational use for non-commercial research only, no commercial use of the data or results, redistribution only under the same terms. Source: <https://github.com/microsoft/DigiFace1M> and its LICENSE <https://raw.githubusercontent.com/microsoft/DigiFace1M/main/LICENSE> | **Recommended**: verified licence that allows a personal non-commercial evaluation, no real people involved, enough identities for 1e-4. Synthetic renders are a domain gap and may understate real-world FMR. |
| LFW (Labeled Faces in the Wild) | 5,749 real people, 13,233 images (secondary source: <https://docs.voxel51.com/dataset_zoo/datasets/lfw.html>) | Project page terms could not be fetched here; licence unverified. | Useful complement for realistic faces if you read the terms yourself. Too few images for 1e-4 (13,233 < 29,956). |
| FFHQ (NVIDIA) | 70,000 images, one per person | Dataset CC BY-NC-SA 4.0; each image has its own Flickr licence; NVIDIA states it should not be used for development or improvement of facial recognition. Source: <https://github.com/NVlabs/ffhq-dataset> | Avoid: the stated use restriction concerns face recognition. |
| VGGFace2 | about 9,000 identities (original release) | A 2023 paper (<https://arxiv.org/pdf/2311.11882>) says it is no longer publicly available because of image copyright; no official Oxford notice was found. | Do not use. |
| CelebA, CASIA-WebFace | n/a | Terms not verified here. | Check terms yourself before use. |

Use the 100,000 x 5 subset of DigiFace-1M: one image per identity folder from it
gives up to 100,000 distinct impostor subjects. Arrange the images as one
directory per identity (the harness names an impostor subject after the first
directory below `--impostor`, or after the file stem when images lie directly in
it). The dataset's own folder layout was not verified here; reorganise with
`mv`/symlinks if needed. Images without a face detected by the daemon's
detector are counted as failures to acquire (FTA), which also tells you how
usable a set is (the images may be small crops).

### Commands

```sh
# Enrolment from images (the images you would give `doorman enroll`)
doorman-bench accuracy \
  --enroll ~/doorman-data/2026-11-05/me_enrol \
  --genuine ~/doorman-data/2026-11-05/me \
  --impostor ~/datasets/digiface-impostors \
  --profile /etc/doorman/doorman.toml --out ~/doorman-reports/acc

# Or enrolment from the templates you actually enrolled. Work on a COPY: the
# loader creates the directory and sets its mode to 0700.
sudo cp -r /var/lib/doorman /tmp/doorman-copy && sudo chown -R "$USER" /tmp/doorman-copy
doorman-bench accuracy --enroll-store /tmp/doorman-copy --enroll-user "$USER" \
  --genuine ~/doorman-data/2026-11-05/me --impostor ~/datasets/digiface-impostors \
  --profile /etc/doorman/doorman.toml --out ~/doorman-reports/acc
rm -rf /tmp/doorman-copy

doorman-bench match-report --scores ~/doorman-reports/acc/match_scores.csv \
  --threshold 0.4 --target-fmr 1e-4 --out ~/doorman-reports/match
```

`accuracy` directory mode writes `scores.csv` (for the accuracy tables),
`match_scores.csv` (`label,score,subject,response`, the `match-report` input) and
`dataset_notes.md`. It needs the daemon ML backend (the default `ort` build) and
the model files. A probe is scored as the best cosine against the enrolled
templates, on single images; that matches `[decision] template_score = "max"` per
frame, not the mean mode's averaged embedding (docs/decision-and-liveness.md).
`--threshold` is `[authentication] similarity_threshold`; the accept rule is
`score > threshold`. `match-report` outputs `match_report.md` and
`match_report.csv` (`measure,attempts,fta,n,errors,rate,upper_bound`).

## Worded claims

Use the claim line the tools print; it states the bound, `n`, the species or
subject count, and what was not tested. Examples of honest wording:

* "On my own laptop camera, one subject, 3 sessions: APCER <= 1.0% (95%
  one-sided Clopper-Pearson) for each of print_matte, print_glossy and
  phone_replay at relief threshold 0.05; untested: print_curved, phone_photo,
  screen_replay, paper_mask."
* "FMR <= 0.010% (95% one-sided) over 29,956 comparisons against synthetic
  DigiFace-1M faces from 29,956 subjects, one enrolled owner."

Do not write: "passes iBeta", "meets ISO 30107-3", "spoof-proof", an APCER
without its species and bound, or a pooled APCER over mixed species. Always
state: self-assessment, single subject (the owner), same camera for enrolment
and testing, attacks made by the owner (not a skilled attacker), synthetic or
public impostor images rather than a population of live visitors. Results do
not transfer to other cameras, lighting or people.

## What formal programs require (context only)

**Thresholds from secondary sources, unverified.** Check the current official
documents before relying on any figure.

| Programme | Claimed requirement (unverified) |
|---|---|
| iBeta PAD (ISO/IEC 30107-3 based) Level 1 and Level 2 | Third-party lab; Level 1 uses simple artefacts (prints, screen replays), Level 2 adds masks and moulds by a more skilled tester; commonly cited pass criterion is no accepted attack in the test (0% attack accept rate) with a bounded bona fide rejection rate. |
| FIDO Alliance biometric component certification | Independent lab; commonly cited limits are an attack accept rate of at most 7% per attack species (the source of the 42-presentation figure above), plus FAR/FRR limits and many bona fide and attack samples. |
| Android "Class 3" (Strong) biometric | Compatibility definition requires spoof acceptance (SAR) of at most 7% for the attack set, a low FAR (commonly cited 1 in 50,000) and a bounded FRR. |
| Windows Hello face | Requires an infrared (IR) camera, plus a depth/IR liveness pipeline; an RGB-only webcam does not qualify (doorman's depth-from-RGB gate does not satisfy this). |

doorman is not certified under any of these programmes.
