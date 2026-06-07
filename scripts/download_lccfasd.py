#!/usr/bin/env python3
"""
Download a balanced subset of the LCC-FASD full-frame face anti-spoofing dataset
from Kainyyy/face-anti-spoof on HuggingFace.

This dataset contains full-frame 720P video captures from LCC FASD:
  - live: frontal webcam 720P captures of genuine faces
  - spoof: replay attacks (phone screens showing face video)

Downloads 300 live + 300 spoof images to ~/datasets/lccfasd_fullframe/
"""

import os
import sys
import random
from pathlib import Path

# Set HF token before imports
token_path = os.path.expanduser("~/.cache/huggingface/token")
if os.path.exists(token_path):
    with open(token_path) as f:
        os.environ["HF_TOKEN"] = f.read().strip()

from huggingface_hub import hf_hub_download, list_repo_files
import shutil

REPO_ID = "Kainyyy/face-anti-spoof"
OUT_DIR = os.path.expanduser("~/datasets/lccfasd_fullframe")
N_LIVE = 300
N_SPOOF = 300
SEED = 42

random.seed(SEED)

def main():
    print(f"Listing files in {REPO_ID}...")
    files = list(list_repo_files(repo_id=REPO_ID, repo_type="dataset"))

    live_files = [f for f in files if "/live/" in f and (f.endswith(".jpg") or f.endswith(".png"))]
    spoof_files = [f for f in files if "/spoof/" in f and (f.endswith(".jpg") or f.endswith(".png"))]

    print(f"  Live files available: {len(live_files)}")
    print(f"  Spoof files available: {len(spoof_files)}")

    # Prioritize FT720P (confirmed full-frame) over YOUTUBE for live-like resolution
    # For spoof: mix FT720P and YOUTUBE, skip tiny generic ones
    ft720p_spoof = [f for f in spoof_files if "FT720P" in f]
    youtube_spoof = [f for f in spoof_files if "YOUTUBE" in f]
    other_spoof   = [f for f in spoof_files if f not in ft720p_spoof and f not in youtube_spoof]

    print(f"  Spoof breakdown: FT720P={len(ft720p_spoof)}, YOUTUBE={len(youtube_spoof)}, other={len(other_spoof)}")

    # Select subset
    live_sel = random.sample(live_files, min(N_LIVE, len(live_files)))
    # Use FT720P first (confirmed 720P full frames), then YOUTUBE
    spoof_pool = ft720p_spoof + youtube_spoof + other_spoof
    spoof_sel = random.sample(spoof_pool, min(N_SPOOF, len(spoof_pool)))

    print(f"\nDownloading {len(live_sel)} live + {len(spoof_sel)} spoof images...")

    live_dir = Path(OUT_DIR) / "live"
    spoof_dir = Path(OUT_DIR) / "spoof"
    live_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing_live = set(f.name for f in live_dir.glob("*"))
    existing_spoof = set(f.name for f in spoof_dir.glob("*"))

    downloaded = 0
    skipped = 0
    errors = 0

    for i, (sel_files, out_dir, existing) in enumerate([
        (live_sel, live_dir, existing_live),
        (spoof_sel, spoof_dir, existing_spoof),
    ]):
        label = "live" if i == 0 else "spoof"
        for j, hf_path in enumerate(sel_files):
            fname = os.path.basename(hf_path)
            dest = out_dir / fname

            if fname in existing:
                skipped += 1
                continue

            if j % 50 == 0:
                print(f"  {label}: {j+1}/{len(sel_files)} (downloaded={downloaded}, skipped={skipped}, errors={errors})")

            try:
                local_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=hf_path,
                    repo_type="dataset",
                    local_dir="/tmp/hf_lccfasd_dl",
                )
                shutil.copy2(local_path, dest)
                downloaded += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error {hf_path}: {e}")

    n_live_final = len(list(live_dir.glob("*")))
    n_spoof_final = len(list(spoof_dir.glob("*")))

    print(f"\nDone. Live: {n_live_final}, Spoof: {n_spoof_final}")
    print(f"  Downloaded: {downloaded}, Skipped (existing): {skipped}, Errors: {errors}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
