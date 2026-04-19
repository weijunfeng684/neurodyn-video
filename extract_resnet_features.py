"""
Fixed-length ResNet18 frame embeddings (ImageNet weights) per clip.

Each video becomes (16, 512): sixteen frames picked by linspace over the decoded
frame list, then torchvision's preprocessing + truncated ResNet (fc removed).
Index file format matches extract_features.py for drop-in use with the GRU trainer.
"""
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


SPLIT_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/processed/resnet_features")
FRAMES_PER_VIDEO = 16


def read_split_file(split_path: Path) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            video_path, class_name = line.split("\t")
            samples.append((video_path, class_name))
    return samples


def read_all_frames(video_path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def pick_fixed_frames(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Evenly spaced indices; repeats when clip shorter than n (linspace rounding)."""
    m = len(frames)
    if m == 0:
        return []
    idxs = np.linspace(0, m - 1, n).round().astype(int)
    return [frames[i] for i in idxs]


def build_output_filename(video_path: str, class_name: str) -> str:
    video_stem = Path(video_path).stem
    safe_class_name = class_name.replace(" ", "_")
    return f"{safe_class_name}__{video_stem}.npy"


def build_backbone(device: torch.device) -> tuple[nn.Module, Callable[[Image.Image], torch.Tensor]]:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device)
    backbone = nn.Sequential(*list(model.children())[:-1]).to(device)
    backbone.eval()
    return backbone, weights.transforms()


def frames_to_tensor(
    frames: list[np.ndarray],
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(preprocess(pil))
    return torch.stack(tensors, dim=0).to(device)


def extract_sequence(
    frames: list[np.ndarray],
    backbone: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    x = frames_to_tensor(frames, preprocess, device)
    with torch.no_grad():
        feats = backbone(x)
    # (T, 512, 1, 1) after avgpool
    feats = feats.flatten(1).cpu().numpy().astype(np.float32)
    return feats


def process_split(
    split_name: str,
    backbone: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> None:
    split_file = SPLIT_DIR / f"{split_name}.txt"
    split_output_dir = OUTPUT_DIR / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_split_file(split_file)
    index_lines: list[str] = []

    print(f"\nProcessing split: {split_name}")
    print(f"Number of videos: {len(samples)}")

    kept = 0
    skipped = 0

    for video_path, class_name in samples:
        try:
            frames = read_all_frames(video_path)
            picked = pick_fixed_frames(frames, FRAMES_PER_VIDEO)
            if not picked:
                print(f"[Skip] No frames: {video_path}")
                skipped += 1
                continue
            feats = extract_sequence(picked, backbone, preprocess, device)
        except Exception as e:
            print(f"[Error] {video_path}: {e}")
            skipped += 1
            continue

        out_name = build_output_filename(video_path, class_name)
        out_path = split_output_dir / out_name
        np.save(out_path, feats)
        index_lines.append(f"{out_path}\t{class_name}\t{feats.shape[0]}")
        kept += 1

    index_path = OUTPUT_DIR / f"{split_name}_index.txt"
    with index_path.open("w", encoding="utf-8") as f:
        for line in index_lines:
            f.write(line + "\n")

    print(f"Saved index: {index_path}")
    print(f"Kept: {kept}, Skipped: {skipped}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, preprocess = build_backbone(device)
    print(f"Device: {device}")

    for name in ("train", "val", "test"):
        process_split(name, backbone, preprocess, device)

    print("\nResNet feature extraction finished.")


if __name__ == "__main__":
    main()
