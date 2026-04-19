"""
PCA trajectories of full GRU hidden sequences (ResNet pipeline, test split).

Fit: one PCA on all timesteps concatenated across clips (same basis for every
curve). Draw: B-spline in the projected plane; markers sit on raw first/last
timestep projections.
"""
import numpy as np
import torch

from model import GRUClassifier
from plot_style import plot_hidden_trajectories_smoothed

from train_gru_resnet import (
    BEST_MODEL_PATH,
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


def pca_project_rows(X: np.ndarray) -> np.ndarray:
    """Same subspace as usual row-PCA; Vt from SVD on centered X."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    if n <= 1:
        return np.zeros((n, 2), dtype=np.float64)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(2, Vt.shape[0])
    Z = Xc @ Vt[:k, :].T
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
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()

    trajectories: list[np.ndarray] = []
    label_indices: list[int] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            _, hidden_seq = model(x)
            h = hidden_seq.cpu().numpy()
            for i in range(h.shape[0]):
                trajectories.append(h[i])
                label_indices.append(int(y[i].item()))

    X_all = np.concatenate(trajectories, axis=0)
    Z_all = pca_project_rows(X_all)

    lengths = [t.shape[0] for t in trajectories]
    traj_2d: list[np.ndarray] = []
    offset = 0
    for L in lengths:
        traj_2d.append(Z_all[offset : offset + L])
        offset += L

    video_labels = [IDX_TO_CLASS[i] for i in label_indices]
    print("Video-level labels (plot order):")
    print(video_labels)

    plot_hidden_trajectories_smoothed(
        traj_2d,
        label_indices,
        CLASS_NAMES,
        OUTPUT_DIR / "hidden_trajectories_resnet.png",
        title="GRU hidden trajectories in PCA space (parametric B-spline, C²)",
    )


if __name__ == "__main__":
    main()
