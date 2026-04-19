#!/usr/bin/env python3
"""
Regenerate PNGs under outputs/ from cached tensors in data/plot_cache/.

Use case: tweak plot_style.py, then rerun this instead of full training to
refresh loss/accuracy/confusion figures. PCA and trajectory PNGs are rebuilt by
spawning the visualize_* scripts (needs checkpoints + feature files on disk).

Does not introduce new filenames under outputs/ — existing paths are overwritten.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _refresh_training(name: str, loss_rel: str, acc_rel: str) -> None:
    from plot_style import plot_training_curves

    p = ROOT / "data" / "plot_cache" / f"{name}_training.npz"
    if not p.exists():
        print(f"[skip] {p.name} not found — run train_gru.py or train_gru_resnet.py once.")
        return
    d = np.load(p)
    plot_training_curves(
        d["train_losses"],
        d["val_losses"],
        d["train_accs"],
        d["val_accs"],
        ROOT / loss_rel,
        ROOT / acc_rel,
    )
    print(f"[ok] {loss_rel} + {acc_rel}")


def _refresh_confusion(npz_name: str, out_rel: str, title: str, class_names: list[str]) -> None:
    from plot_style import plot_confusion_matrix_fancy

    p = ROOT / "data" / "plot_cache" / npz_name
    if not p.exists():
        print(f"[skip] {npz_name} not found — run evaluate.py / evaluate_resnet.py once.")
        return
    cm = np.load(p)["cm"]
    plot_confusion_matrix_fancy(cm, class_names, ROOT / out_rel, title=title)
    print(f"[ok] {out_rel}")


def _run_viz(script: str) -> None:
    path = ROOT / script
    if not path.is_file():
        return
    r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[warn] {script} exited with code {r.returncode}")


def main() -> None:
    sys.path.insert(0, str(ROOT))

    _refresh_training("baseline", "outputs/loss_curve.png", "outputs/accuracy_curve.png")
    _refresh_training("resnet", "outputs/loss_curve_resnet.png", "outputs/accuracy_curve_resnet.png")

    from train_gru import CLASS_NAMES as BL_NAMES
    from train_gru_resnet import CLASS_NAMES as RN_NAMES

    _refresh_confusion(
        "confusion_baseline.npz",
        "outputs/confusion_matrix.png",
        "Confusion matrix (handcrafted features)",
        list(BL_NAMES),
    )
    _refresh_confusion(
        "confusion_resnet.npz",
        "outputs/confusion_matrix_resnet.png",
        "Confusion matrix (ResNet + GRU)",
        list(RN_NAMES),
    )

    for script in (
        "visualize_hidden.py",
        "visualize_hidden_resnet.py",
        "visualize_hidden_trajectories_resnet.py",
    ):
        _run_viz(script)

    print("Done. Same filenames under outputs/ were overwritten where caches existed.")


if __name__ == "__main__":
    main()
