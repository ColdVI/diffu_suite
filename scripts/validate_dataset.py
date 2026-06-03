#!/usr/bin/env python3
"""Audit an extracted CIFAR-10 image-folder dataset."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image

EXPECTED_SPLIT_COUNTS = {"train": 50_000, "test": 10_000}
EXPECTED_CLASS_COUNTS = {"train": 5_000, "test": 1_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10_dataset"))
    parser.add_argument(
        "--hashes",
        action="store_true",
        help="Also hash every PNG and report byte-identical duplicates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_files: list[Path] = []
    valid = True
    for split in ("train", "test"):
        split_total = 0
        for class_id in range(10):
            class_root = args.data_root / split / str(class_id)
            files = sorted(class_root.glob("*.png"))
            split_total += len(files)
            all_files.extend(files)
            expected = EXPECTED_CLASS_COUNTS[split]
            print(f"{split}/{class_id}: {len(files)} PNG files (expected {expected})")
            valid &= len(files) == expected
        print(f"{split}: {split_total} PNG files (expected {EXPECTED_SPLIT_COUNTS[split]})")
        valid &= split_total == EXPECTED_SPLIT_COUNTS[split]

    size_counts: Counter[tuple[int, int]] = Counter()
    mode_counts: Counter[str] = Counter()
    digests: Counter[str] = Counter()
    for path in all_files:
        with Image.open(path) as image:
            size_counts[image.size] += 1
            mode_counts[image.mode] += 1
        if args.hashes:
            digests[hashlib.sha256(path.read_bytes()).hexdigest()] += 1

    print(f"sizes: {dict(size_counts)}")
    print(f"modes: {dict(mode_counts)}")
    valid &= size_counts == {(32, 32): 60_000}
    valid &= mode_counts == {"RGB": 60_000}
    if args.hashes:
        duplicate_files = sum(count - 1 for count in digests.values() if count > 1)
        print(f"byte-identical duplicate files beyond first occurrence: {duplicate_files}")
        valid &= duplicate_files == 0

    if not valid:
        raise SystemExit("dataset audit failed")
    print("dataset audit passed")


if __name__ == "__main__":
    main()

