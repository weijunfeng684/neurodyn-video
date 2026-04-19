"""
Handcrafted temporal features per video (histogram + simple motion stats).

Downstream expects:
  data/processed/features/{split}/*.npy  with shape (T, 20)
  data/processed/features/{split}_index.txt — each line: path, label, seq length (tab-separated)

Sequence length varies with SAMPLE_EVERY_N_FRAMES and clip length; MIN_SEQUENCE_LENGTH
drops unusably short clips.
"""
from pathlib import Path

import cv2
import numpy as np


SPLIT_DIR = Path("data/splits")

OUTPUT_DIR = Path("data/processed/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_EVERY_N_FRAMES = 3

MIN_SEQUENCE_LENGTH = 5


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


def sample_video_frames(video_path: str, sample_every_n_frames: int = 3) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            frames.append(frame)

        frame_idx += 1

    cap.release()
    return frames


def extract_frame_feature(
    frame: np.ndarray,
    prev_gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    20-D vector: 4 scalars (brightness stats + frame diff + mean flow magnitude)
    concatenated with a 16-bin grayscale histogram (L1-normalized).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))

    hist = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)

    if prev_gray is None:
        frame_diff_mean = 0.0
        optical_flow_mean = 0.0
    else:
        diff = cv2.absdiff(gray, prev_gray)
        frame_diff_mean = float(np.mean(diff))

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flow_mean = float(np.mean(mag))

    feature = np.concatenate(
        [
            np.array(
                [
                    mean_brightness,
                    std_brightness,
                    frame_diff_mean,
                    optical_flow_mean,
                ],
                dtype=np.float32,
            ),
            hist.astype(np.float32),
        ]
    )

    return feature, gray


def extract_video_features(video_path: str, sample_every_n_frames: int = 3) -> np.ndarray:
    frames = sample_video_frames(video_path, sample_every_n_frames=sample_every_n_frames)

    features: list[np.ndarray] = []
    prev_gray = None

    for frame in frames:
        feature, prev_gray = extract_frame_feature(frame, prev_gray)
        features.append(feature)

    if not features:
        return np.empty((0, 20), dtype=np.float32)

    return np.stack(features).astype(np.float32)


def build_output_filename(video_path: str, class_name: str) -> str:
    video_stem = Path(video_path).stem
    safe_class_name = class_name.replace(" ", "_")
    return f"{safe_class_name}__{video_stem}.npy"


def process_split(split_name: str) -> None:
    split_file = SPLIT_DIR / f"{split_name}.txt"
    split_output_dir = OUTPUT_DIR / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_split_file(split_file)
    index_lines: list[str] = []

    print(f"\nProcessing split: {split_name}")
    print(f"Number of videos: {len(samples)}")

    kept_count = 0
    skipped_count = 0

    for video_path, class_name in samples:
        try:
            features = extract_video_features(
                video_path,
                sample_every_n_frames=SAMPLE_EVERY_N_FRAMES,
            )
        except Exception as e:
            print(f"[Error] Failed to process {video_path}: {e}")
            skipped_count += 1
            continue

        if len(features) < MIN_SEQUENCE_LENGTH:
            print(f"[Skip] Sequence too short: {video_path}")
            skipped_count += 1
            continue

        output_name = build_output_filename(video_path, class_name)
        output_path = split_output_dir / output_name
        np.save(output_path, features)

        index_lines.append(f"{output_path}\t{class_name}\t{len(features)}")
        kept_count += 1

    index_path = OUTPUT_DIR / f"{split_name}_index.txt"
    with index_path.open("w", encoding="utf-8") as f:
        for line in index_lines:
            f.write(line + "\n")

    print(f"Saved index file: {index_path}")
    print(f"Kept: {kept_count}")
    print(f"Skipped: {skipped_count}")


def main() -> None:
    process_split("train")
    process_split("val")
    process_split("test")
    print("\nFeature extraction finished.")


if __name__ == "__main__":
    main()
