"""
2D PCA of per-clip GRU final hidden states (handcrafted features, test split).

One point per video: hidden_seq[:, -1, :]. PCA is mean-centered SVD; order of
printed labels matches dataloader iteration order.
"""
import numpy as np
import torch

from model import GRUClassifier
from plot_style import plot_hidden_embedding_scatter

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


def pca_to_2d(X: np.ndarray) -> np.ndarray:
    """Rows = observations; returns first two score columns (pad if rank < 2)."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    X = X - X.mean(axis=0, keepdims=True)
    if n <= 1:
        return np.zeros((n, 2), dtype=np.float64)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = min(2, U.shape[1])
    Z = U[:, :k] * S[:k]
    if k < 2:
        Z = np.concatenate([Z, np.zeros((n, 2 - k), dtype=np.float64)], axis=1)
    return Z


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

    representations: list[np.ndarray] = []
    label_indices: list[int] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            _, hidden_seq = model(x)
            last_hidden = hidden_seq[:, -1, :].cpu().numpy()
            for i in range(last_hidden.shape[0]):
                representations.append(last_hidden[i])
                label_indices.append(int(y[i].item()))

    X = np.stack(representations, axis=0)
    Z = pca_to_2d(X)

    video_labels = [IDX_TO_CLASS[i] for i in label_indices]
    print("Video-level labels (plot order):")
    print(video_labels)

    plot_hidden_embedding_scatter(
        Z,
        np.array(label_indices, dtype=int),
        CLASS_NAMES,
        OUTPUT_DIR / "hidden_pca.png",
        title="Last GRU hidden state (PCA)\nHandcrafted temporal features · test split",
    )


if __name__ == "__main__":
    main()
