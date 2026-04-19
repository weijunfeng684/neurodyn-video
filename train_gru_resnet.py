"""
Same training recipe as train_gru.py, but reads ResNet18 frame embeddings.

FEATURE_ROOT points at extract_resnet_features.py output; checkpoint and curve
filenames are suffixed _resnet so baseline runs are not overwritten.
"""
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import GRUClassifier
from plot_style import plot_training_curves


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

FEATURE_ROOT = Path("data/processed/resnet_features")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "best_gru_classifier_resnet.pt"
LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve_resnet.png"
ACC_CURVE_PATH = OUTPUT_DIR / "accuracy_curve_resnet.png"

CLASS_NAMES = [
    "walk",
    "bend",
    "jump in place",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_index_file(index_path):
    samples = []

    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            feature_path, class_name, seq_len = line.split("\t")
            samples.append((feature_path, class_name, int(seq_len)))

    return samples


class VideoFeatureDataset(Dataset):
    def __init__(self, index_path):
        self.samples = read_index_file(index_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, class_name, _ = self.samples[idx]
        features = np.load(feature_path).astype(np.float32)
        label = CLASS_TO_IDX[class_name]
        return features, label


def pad_collate_fn(batch):
    """Right-pad with zeros to max T in batch; GRU sees full padded length."""
    feature_list, label_list = zip(*batch)

    max_len = max(seq.shape[0] for seq in feature_list)
    feat_dim = feature_list[0].shape[1]

    batch_size = len(feature_list)
    padded = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)

    for i, seq in enumerate(feature_list):
        padded[i, : seq.shape[0], :] = seq

    x = torch.tensor(padded, dtype=torch.float32)
    y = torch.tensor(label_list, dtype=torch.long)

    return x, y


def build_dataloader(index_path, shuffle):
    dataset = VideoFeatureDataset(index_path)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=pad_collate_fn,
    )
    return loader


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    cache_dir = Path("data/plot_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        LOSS_CURVE_PATH,
        ACC_CURVE_PATH,
        metrics_cache=cache_dir / "resnet_training.npz",
    )


def main():
    train_loader = build_dataloader(FEATURE_ROOT / "train_index.txt", shuffle=True)
    val_loader = build_dataloader(FEATURE_ROOT / "val_index.txt", shuffle=False)
    test_loader = build_dataloader(FEATURE_ROOT / "test_index.txt", shuffle=False)

    sample_batch = next(iter(train_loader))
    sample_x, _ = sample_batch
    input_dim = sample_x.shape[-1]

    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = -1.0

    print(f"Using device: {DEVICE}")
    print(f"Input dim: {input_dim}")

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits, _ = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            running_samples += x.size(0)

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    plot_curves(train_losses, val_losses, train_accs, val_accs)

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print("\nBest model evaluation on test split:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")


if __name__ == "__main__":
    main()
