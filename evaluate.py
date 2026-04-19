"""
Load best handcrafted-feature GRU, run test loader, print per-sample labels + acc.

Confusion matrix PNG + a small npz cache (for refresh_output_figures.py) are
written under outputs/ and data/plot_cache/ respectively.
"""
from pathlib import Path

import numpy as np
import torch

from model import GRUClassifier
from plot_style import plot_confusion_matrix_fancy

from train_gru import (
    CLASS_NAMES,
    DEVICE,
    DROPOUT,
    FEATURE_ROOT,
    HIDDEN_DIM,
    IDX_TO_CLASS,
    NUM_LAYERS,
    OUTPUT_DIR,
    build_dataloader,
)


def confusion_matrix_counts(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    plot_confusion_matrix_fancy(cm, class_names, output_path, title="Confusion matrix (handcrafted features)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_loader = build_dataloader(FEATURE_ROOT / "test_index.txt", shuffle=False)
    sample_x, _ = next(iter(test_loader))
    input_dim = sample_x.shape[-1]
    num_classes = len(CLASS_NAMES)

    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(OUTPUT_DIR / "best_gru_classifier.pt", map_location=DEVICE)
    )
    model.eval()

    y_true_idx: list[int] = []
    y_pred_idx: list[int] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true_idx.extend(y.cpu().numpy().tolist())
            y_pred_idx.extend(preds.cpu().numpy().tolist())

    y_true_arr = np.array(y_true_idx, dtype=int)
    y_pred_arr = np.array(y_pred_idx, dtype=int)

    true_labels = [IDX_TO_CLASS[i] for i in y_true_idx]
    pred_labels = [IDX_TO_CLASS[i] for i in y_pred_idx]
    test_acc = float((y_true_arr == y_pred_arr).mean())

    print("True labels:")
    print(true_labels)
    print("Predicted labels:")
    print(pred_labels)
    print(f"Test accuracy: {test_acc:.4f}")

    cm = confusion_matrix_counts(y_true_arr, y_pred_arr, num_classes)
    cache_dir = Path("data/plot_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / "confusion_baseline.npz", cm=cm.astype(np.float64))
    plot_confusion_matrix(cm, CLASS_NAMES, OUTPUT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
