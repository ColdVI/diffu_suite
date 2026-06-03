#!/usr/bin/env python3
"""Rebuild an exact extracted CIFAR-10 ImageFolder dataset with torchvision.

Use this when a copied Drive dataset has extra or missing PNG files. By default
the script refuses to overwrite an existing output directory. Pass
``--backup-existing`` to move the current folder aside first.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from torchvision.datasets import CIFAR10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cifar10_dataset"),
        help="Destination ImageFolder root.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw_cifar10"),
        help="Where torchvision stores/downloads the raw CIFAR-10 archive.",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Move an existing output-root to a timestamped backup before rebuilding.",
    )
    parser.add_argument(
        "--force-delete-existing",
        action="store_true",
        help="Delete an existing output-root instead of backing it up.",
    )
    return parser.parse_args()


def prepare_output_root(args: argparse.Namespace) -> None:
    """Create a clean output directory without silently overwriting data."""

    if args.backup_existing and args.force_delete_existing:
        raise ValueError("choose only one of --backup-existing or --force-delete-existing")
    if not args.output_root.exists():
        args.output_root.mkdir(parents=True, exist_ok=True)
        return
    if args.backup_existing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = args.output_root.with_name(f"{args.output_root.name}_backup_{timestamp}")
        shutil.move(str(args.output_root), backup_path)
        print(f"moved existing dataset to backup: {backup_path}")
        args.output_root.mkdir(parents=True, exist_ok=True)
        return
    if args.force_delete_existing:
        shutil.rmtree(args.output_root)
        args.output_root.mkdir(parents=True, exist_ok=True)
        return
    raise FileExistsError(
        f"{args.output_root} already exists. Use --backup-existing or "
        "--force-delete-existing to rebuild it."
    )


def export_split(args: argparse.Namespace, *, split: str, train: bool) -> int:
    """Export one CIFAR-10 split into class-id folders."""

    dataset = CIFAR10(root=str(args.raw_root), train=train, download=True)
    total = 0
    for index, (image, label) in enumerate(dataset):
        output_dir = args.output_root / split / str(label)
        output_dir.mkdir(parents=True, exist_ok=True)
        image.save(output_dir / f"{index}.png")
        total += 1
    return total


def main() -> None:
    args = parse_args()
    prepare_output_root(args)
    train_total = export_split(args, split="train", train=True)
    test_total = export_split(args, split="test", train=False)
    print(f"rebuilt CIFAR-10 ImageFolder at: {args.output_root}")
    print(f"train: {train_total}")
    print(f"test: {test_total}")
    print("next: python3 scripts/validate_dataset.py --data-root", args.output_root, "--hashes")


if __name__ == "__main__":
    main()

