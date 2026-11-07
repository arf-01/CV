"""Create stratified train/validation/test folders from TrashNet."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.20, "test": 0.10}


def split_counts(total: int) -> dict[str, int]:
    """Return counts whose sum is total and that are closest to the target ratios."""
    counts = {name: int(total * ratio) for name, ratio in RATIOS.items()}
    remaining = total - sum(counts.values())

    fractions = sorted(
        SPLITS,
        key=lambda name: total * RATIOS[name] - counts[name],
        reverse=True,
    )
    for name in fractions[:remaining]:
        counts[name] += 1
    return counts


def split_dataset(source: Path, destination: Path, seed: int, overwrite: bool) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Destination is not empty: {destination}. "
            "Use --overwrite to replace it."
        )

    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class folders found in {source}")

    rng = random.Random(seed)
    totals = {split: 0 for split in SPLITS}

    for class_dir in classes:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            print(f"Skipping empty class folder: {class_dir.name}")
            continue

        rng.shuffle(images)
        counts = split_counts(len(images))
        start = 0

        for split in SPLITS:
            split_images = images[start : start + counts[split]]
            target_dir = destination / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image in split_images:
                shutil.copy2(image, target_dir / image.name)
            totals[split] += len(split_images)
            start += counts[split]

        print(
            f"{class_dir.name}: "
            f"{counts['train']} train, "
            f"{counts['validation']} validation, "
            f"{counts['test']} test"
        )

    print(
        f"\nCreated {destination}\n"
        f"Total: {totals['train']} train, "
        f"{totals['validation']} validation, "
        f"{totals['test']} test"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a class-folder image dataset into train/validation/test."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data") / "trashnet",
        help="Original dataset folder (default: data/trashnet)",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data") / "trashnet_split",
        help="Output folder (default: data/trashnet_split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for a reproducible split (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the destination if it already contains files",
    )
    args = parser.parse_args()

    split_dataset(args.source, args.destination, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
