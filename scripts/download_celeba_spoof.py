#!/usr/bin/env python3
"""
Download a balanced subset of CelebA-Spoof from HuggingFace (streaming mode).
Saves LIVE and SPOOF images to ~/datasets/celeba_spoof_subset/{live,spoof}/.

We pull at most --max-per-class images from each class (default 3000).
Streaming avoids pulling the full 5GB parquet corpus.
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="Max images per class (live/spoof) to download")
    parser.add_argument("--output-dir", default=os.path.expanduser(
        "~/datasets/celeba_spoof_subset"))
    args = parser.parse_args()

    from datasets import load_dataset

    out = Path(args.output_dir)
    live_dir = out / "live"
    spoof_dir = out / "spoof"
    live_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming CelebA-Spoof test split (max {args.max_per_class} per class)...")
    ds = load_dataset(
        "nguyenkhoa/celeba-spoof-for-face-antispoofing-test",
        split="test",
        streaming=True,
    )

    # Resume: count existing files to skip already-downloaded images
    n_live = len(list(live_dir.glob("live_*.jpg")))
    n_spoof = len(list(spoof_dir.glob("spoof_*.jpg")))
    n_skip = 0

    print(f"Resuming from: live={n_live} spoof={n_spoof}")

    bar = tqdm(total=args.max_per_class * 2 - n_live - n_spoof, desc="Downloading")

    for i, sample in enumerate(ds):
        label = int(sample["labels"])  # 0=live, 1=spoof
        img = sample["cropped_image"]  # PIL image

        if img is None:
            n_skip += 1
            continue

        if label == 0 and n_live < args.max_per_class:
            try:
                img.save(str(live_dir / f"live_{n_live:05d}.jpg"), quality=90)
                n_live += 1
                bar.update(1)
            except Exception:
                n_skip += 1
        elif label == 1 and n_spoof < args.max_per_class:
            try:
                img.save(str(spoof_dir / f"spoof_{n_spoof:05d}.jpg"), quality=90)
                n_spoof += 1
                bar.update(1)
            except Exception:
                n_skip += 1
        else:
            n_skip += 1

        if n_live >= args.max_per_class and n_spoof >= args.max_per_class:
            break

    bar.close()
    print(f"\nDone. Live: {n_live}  Spoof: {n_spoof}  Skipped/passed: {n_skip}")
    print(f"Live images  -> {live_dir}")
    print(f"Spoof images -> {spoof_dir}")


if __name__ == "__main__":
    main()
