"""
Build train/val/test lists for the Weizmann action clips used in this repo.

Writes TSV lines: absolute_video_path, tab, class_name.
Stratification is per-class so each split keeps at least one clip per label when
counts allow. Adjust DATASET_ROOT if your tree differs.
"""
from pathlib import Path
import random


random.seed(42)

DATASET_ROOT = Path.home() / "Desktop" / "weizmann_video_data" / "Weizmann Dataset"

SELECTED_CLASSES = [
    "walk",
    "bend",
    "jump in place",
]

SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def collect_videos(dataset_root: Path, class_names: list[str]) -> list[tuple[str, str]]:
    """Walk class folders; return sorted (path, label) pairs for *.avi only."""
    samples: list[tuple[str, str]] = []

    for class_name in class_names:
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            print(f"[Warning] Class directory not found: {class_dir}")
            continue

        video_paths = sorted(class_dir.glob("*.avi"))
        for video_path in video_paths:
            samples.append((str(video_path), class_name))

    return samples


def split_by_class(
    samples: list[tuple[str, str]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Per-label shuffle then slice; n_test forced >= 1 when mathematically 0."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    grouped_samples: dict[str, list[tuple[str, str]]] = {}
    for video_path, class_name in samples:
        grouped_samples.setdefault(class_name, []).append((video_path, class_name))

    train_samples: list[tuple[str, str]] = []
    val_samples: list[tuple[str, str]] = []
    test_samples: list[tuple[str, str]] = []

    for class_name, class_samples in grouped_samples.items():
        random.shuffle(class_samples)
        n_samples = len(class_samples)

        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(1, int(n_samples * val_ratio))
        n_test = n_samples - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                raise ValueError(
                    f"Not enough samples in class '{class_name}' to create all splits."
                )

        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train : n_train + n_val])
        test_samples.extend(class_samples[n_train + n_val :])

    return train_samples, val_samples, test_samples


def save_split(samples: list[tuple[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for video_path, class_name in samples:
            f.write(f"{video_path}\t{class_name}\n")


def main() -> None:
    samples = collect_videos(DATASET_ROOT, SELECTED_CLASSES)
    print(f"Total videos collected: {len(samples)}")

    train_samples, val_samples, test_samples = split_by_class(samples)

    save_split(train_samples, SPLIT_DIR / "train.txt")
    save_split(val_samples, SPLIT_DIR / "val.txt")
    save_split(test_samples, SPLIT_DIR / "test.txt")

    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")
    print(f"Test:  {len(test_samples)}")

    print("\nSample entries from train split:")
    for item in train_samples[:5]:
        print(item)


if __name__ == "__main__":
    main()
