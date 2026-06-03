"""Dataset loader for the extracted CIFAR-10 image-folder layout."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utils.images import pil_to_tensor

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Cifar10ImageFolder(Dataset[tuple[Tensor, Tensor]]):
    """Read ``root/{train|test}/{0..9}/*.png`` as normalized RGB tensors."""

    def __init__(
        self,
        root: str | Path = "data/cifar10_dataset",
        *,
        split: str = "train",
        augment: bool | None = None,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.augment = split == "train" if augment is None else augment
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")

        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"CIFAR-10 split directory does not exist: {split_root}")

        samples = []
        for class_id in range(len(CIFAR10_CLASSES)):
            class_root = split_root / str(class_id)
            if not class_root.is_dir():
                raise FileNotFoundError(f"missing CIFAR-10 class directory: {class_root}")
            samples.extend((path, class_id) for path in sorted(class_root.glob("*.png")))
        if not samples:
            raise ValueError(f"no PNG files found below {split_root}")
        self.samples = samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, class_id = self.samples[index]
        with Image.open(path) as image:
            image_tensor = pil_to_tensor(image)[0]
        if self.augment and torch.rand(()) < 0.5:
            image_tensor = image_tensor.flip(-1)
        return image_tensor, torch.tensor(class_id, dtype=torch.long)


def build_dataloader(
    root: str | Path,
    *,
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool | None = None,
    limit: int | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Construct a CIFAR-10 loader with practical training defaults."""

    dataset = Cifar10ImageFolder(root, split=split, limit=limit)
    should_shuffle = split == "train" if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=split == "train" and len(dataset) >= batch_size,
    )

