# Dataset Reference for Doorman Face Pipeline Evaluation

## Face Recognition / Verification Datasets

| Dataset | Size | Pairs / Protocol | License | Openly downloadable tonight? | Notes |
|---|---|---|---|---|---|
| **LFW** (Labeled Faces in the Wild, deepfunneled) | ~13 k images, 5749 identities | 6000-pair verification protocol (pairs.txt) | Attribution (Gary B. Huang et al.) | **YES** — figshare mirror: https://ndownloader.figshare.com/files/5976015 (~233 MB) | Standard benchmark. Easy for modern models (EdgeFace-S expects ~0.99+). **Used for this evaluation.** |
| **CFP-FP** (Celebrities in Frontal-Profile) | 7000 images, 500 identities | 7000 pairs (frontal-vs-frontal + frontal-vs-profile) | Attribution (Sengupta et al.) | **YES** — http://cfpw.io/data/cfp-dataset.zip | Harder than LFW due to large pose variation. Good for pose-robustness testing. |
| **AgeDB-30** | 16 488 images, 568 identities | 6000 pairs across age gaps | Attribution (Moschoglou et al.) | **YES** — https://ibug.doc.ic.ac.uk/resources/agedb/ (may require email) | Harder than LFW due to large age gap. |
| **CALFW** (Cross-Age LFW) | 13 233 images (LFW images + augmented) | 6000 pairs | Attribution (Zheng et al.) | **YES** — http://whdeng.cn/CALFW/index.html | LFW pairs re-curated for cross-age evaluation. Moderate difficulty. |
| **CPLFW** (Cross-Pose LFW) | 13 233 images | 6000 pairs | Attribution (Zheng et al.) | **YES** — http://whdeng.cn/CPLFW/index.html | LFW pairs re-curated for cross-pose evaluation. |
| **IJB-B** (IARPA Janus Benchmark-B) | 21 798 images + 7011 video frames, 1845 subjects | Verification + identification | **NIST DUA required** — registration-walled | **NO** — requires NIST DUA agreement | Template-based (multi-image per identity). Critical for multi-frame aggregation testing. |
| **IJB-C** (IARPA Janus Benchmark-C) | ~31 k images + ~117 k video frames, 3531 subjects | Verification + identification | **NIST DUA required** — registration-walled | **NO** — requires NIST DUA agreement | Superset of IJB-B. The gold standard for template/aggregation evaluation. Apply at: https://nist.gov/programs-projects/face-challenges |
| **YouTube Faces (YTF)** | 3425 videos, 1595 subjects | 5000 video pairs | Attribution (Wolf et al., weizmann.ac.il) | **YES** (video download scripts available) — but large (~38 GB raw video) | Tests video-level (temporal) face recognition. Mirrors Phase-1 multi-frame scenario. |
| **MegaFace** | 1M+ images, 690k identities (distractors) | Identification @ scale | Attribution + registration | **Partial** — some mirrors exist but registration preferred | Large-scale 1:N identification. Not needed for current doorman use case (1:few). |
| **VGGFace2** | 3.31M images, 9131 identities | Test set: 500 identities × ~59 images | CC BY 4.0 | **YES** — download via script from https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ | Rich per-identity variation (age, pose, lighting). Excellent for multi-frame aggregation testing. No NIST DUA required. |

## Face Anti-Spoofing / Presentation-Attack Datasets

| Dataset | Size | Attacks | License | Openly downloadable? | Notes |
|---|---|---|---|---|---|
| **Replay-Attack** (Idiap) | 1200 videos, 50 subjects | Print, screen replay | **Idiap EULA** | **NO** — registration + EULA required. https://www.idiap.ch/en/dataset/replayattack | Well-known benchmark. 2D attacks only. |
| **CASIA-FASD** | 600 videos, 50 subjects | Print, warped, cut photo | **Academic license** | **NO** — registration required. http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofing.asp | Classic benchmark. 3 attack types. |
| **MSU-MFSD** | 280 videos, 35 subjects | Print, face replay | Attribution (Michigan State) | **Partial** — older mirrors exist; official site requires email | Small but widely used. |
| **OULU-NPU** | 5940 videos, 55 subjects | Print, display, 3 sessions | **Academic license** | **NO** — registration required. https://sites.google.com/site/oulunpudatabase/ | 4 sub-protocols; state-of-the-art benchmark as of 2017. |
| **SiW** (Spoof in the Wild) | 1641 videos, 165 subjects | Print, replay, 4 sessions | **Academic license** | **NO** — registration required. http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html | Better diversity than OULU-NPU. |
| **CelebA-Spoof** | 625k images, 10177 subjects | Print, replay, paper mask | CC BY-NC 4.0 | **YES** — https://github.com/ZhangYuanhan-AI/CelebA-Spoof | Largest 2D anti-spoofing dataset. MiniFASNetV2-SE (the doorman liveness model) reports 98.2% / 0.9984 AUC on this dataset. Good for liveness baseline evaluation. |
| **3DMADv2** | 255 videos, 17 subjects | 3D face mask | Attribution (Idiap) | **YES** — https://www.idiap.ch/en/dataset/3dmad | 3D mask attacks specifically. Small but relevant for hardware token bypass. |

## Summary: What Was Downloaded Tonight

| Dataset | Location | Images | How obtained |
|---|---|---|---|
| **LFW-funneled** | `/home/angkira/datasets/lfw/lfw_funneled/` | 13 233 JPEG images | figshare mirror `https://ndownloader.figshare.com/files/5976015` |
| **LFW pairs.txt** | `/home/angkira/datasets/lfw/pairs.txt` | 6000 pairs (10 splits × 300 genuine + 300 impostor) | facenet GitHub mirror: `https://raw.githubusercontent.com/davidsandberg/facenet/master/data/pairs.txt` |

Primary LFW mirror (vis-www.cs.umass.edu) was unreachable at time of download.
The figshare mirror provides the identical LFW-deepfunneled/funneled tarball.

## Recommended Next Steps

1. **IJB-C** — Apply for NIST DUA to get the registration-walled IJB-C dataset. This is the
   correct benchmark for Phase-1 multi-frame aggregation (template-vs-template evaluation),
   as LFW has very few identities with enough images.

2. **VGGFace2** test set — Freely available; ~500 identities × 59 images each provides a
   solid multi-frame aggregation evaluation without NIST registration.

3. **CelebA-Spoof** — For Phase-2 liveness evaluation; freely downloadable under CC BY-NC 4.0.
