"""
Figure helpers shared by train_*, evaluate_*, and visualize_* entry points.

Design notes:
  - configure_matplotlib() is intentionally side-effectful (global rc + seaborn theme).
  - Training curves stash an optional npz so refresh_output_figures.py can redraw
    without replaying epochs.
  - Confusion matrix heatmap uses row-normalized counts for color; cell text is raw n.
  - Trajectories: splprep in (x,y); duplicate PCA vertices removed before fit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, PercentFormatter
from scipy.spatial import ConvexHull

FIG_DPI = 180

CLASS_COLORS = ("#4f46e5", "#db2777", "#059669", "#a855f7", "#ea580c")

TRAIN_COLOR = "#312e81"
VAL_COLOR = "#9d174d"

CM_CMAP = LinearSegmentedColormap.from_list(
    "ink_teal",
    ["#f8fafc", "#e0f2fe", "#38bdf8", "#0369a1", "#0c4a6e"],
)


def _save_figure(fig: mpl.figure.Figure, path: Path) -> None:
    # bbox_inches='tight' + explicit facecolor avoids a flat white crop killing the grid tone.
    fig.savefig(
        path,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.16,
        dpi=FIG_DPI,
    )


def configure_matplotlib() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font="sans-serif",
        rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    mpl.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": "tight",
            "savefig.facecolor": "#f1f5f9",
            "figure.facecolor": "#f1f5f9",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#94a3b8",
            "axes.linewidth": 1.05,
            "axes.grid": True,
            "grid.alpha": 0.55,
            "grid.linestyle": "--",
            "grid.linewidth": 0.9,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "axes.titlepad": 12,
            "legend.frameon": True,
            "legend.framealpha": 0.98,
            "legend.edgecolor": "#cbd5e1",
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0f172a",
        }
    )


def _line_with_glow(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    markevery: int,
    label: str,
    marker: str,
) -> None:
    # Wide translucent stroke underneath reads as a glow on projectors / slides.
    rgb = np.array(mcolors.to_rgb(color))
    ax.plot(x, y, color=(*rgb, 0.22), lw=7.0, solid_capstyle="round", zorder=1)
    ax.plot(
        x,
        y,
        color=color,
        lw=2.4,
        marker=marker,
        ms=5.0,
        markevery=markevery,
        label=label,
        solid_capstyle="round",
        zorder=3,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )


def plot_training_curves(
    train_losses: list[float] | np.ndarray,
    val_losses: list[float] | np.ndarray,
    train_accs: list[float] | np.ndarray,
    val_accs: list[float] | np.ndarray,
    loss_path: Path,
    acc_path: Path,
    metrics_cache: Path | None = None,
) -> None:
    configure_matplotlib()
    n = len(train_losses)
    epochs = np.arange(1, n + 1, dtype=float)
    markevery = max(1, n // 8)

    def _style_axis(ax: plt.Axes, ylabel: str, title: str) -> None:
        ax.set_xlabel("Epoch", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.set_title(title)
        ax.set_xlim(0.85, n + 0.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(12, n + 1)))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="minor", length=3, color="#94a3b8")

    # —— Loss ——
    fig, ax = plt.subplots(figsize=(10.0, 4.85))
    tl, vl = np.asarray(train_losses, float), np.asarray(val_losses, float)
    ymin = max(0.0, float(min(tl.min(), vl.min()) * 0.92))
    ymax = float(max(tl.max(), vl.max()) * 1.08)
    ax.fill_between(epochs, tl, ymin, alpha=0.12, color=TRAIN_COLOR, linewidth=0, zorder=0)
    ax.fill_between(epochs, vl, ymin, alpha=0.12, color=VAL_COLOR, linewidth=0, zorder=0)
    _line_with_glow(ax, epochs, tl, TRAIN_COLOR, markevery, "Train", "o")
    _line_with_glow(ax, epochs, vl, VAL_COLOR, markevery, "Validation", "s")
    _style_axis(ax, "Loss (cross-entropy)", "Model convergence · loss (train vs validation)")
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper right", title="Split", alignment="left")
    fig.text(
        0.99,
        0.02,
        f"Final  ·  train {tl[-1]:.4f}  ·  val {vl[-1]:.4f}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748b",
        family="monospace",
    )
    _save_figure(fig, loss_path)
    plt.close(fig)

    # —— Accuracy ——
    fig, ax = plt.subplots(figsize=(10.0, 4.85))
    ta, va = np.asarray(train_accs, float), np.asarray(val_accs, float)
    ax.axhline(1.0, color="#cbd5e1", lw=1.0, ls=(0, (4, 4)), zorder=0)
    ax.fill_between(epochs, ta, alpha=0.1, color=TRAIN_COLOR, linewidth=0, zorder=0)
    ax.fill_between(epochs, va, alpha=0.1, color=VAL_COLOR, linewidth=0, zorder=0)
    _line_with_glow(ax, epochs, ta, TRAIN_COLOR, markevery, "Train", "o")
    _line_with_glow(ax, epochs, va, VAL_COLOR, markevery, "Validation", "s")
    _style_axis(ax, "Accuracy", "Model convergence · accuracy (train vs validation)")
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right", title="Split", alignment="left")
    fig.text(
        0.99,
        0.02,
        f"Final  ·  train {ta[-1]:.1%}  ·  val {va[-1]:.1%}",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#64748b",
        family="monospace",
    )
    _save_figure(fig, acc_path)
    plt.close(fig)

    if metrics_cache is not None:
        # Small on-disk replay buffer for refresh_output_figures.py
        metrics_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            metrics_cache,
            train_losses=np.asarray(train_losses, dtype=np.float64),
            val_losses=np.asarray(val_losses, dtype=np.float64),
            train_accs=np.asarray(train_accs, dtype=np.float64),
            val_accs=np.asarray(val_accs, dtype=np.float64),
        )


def plot_confusion_matrix_fancy(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Confusion matrix",
) -> None:
    configure_matplotlib()
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    norm = cm / row_sum
    overall_acc = 100.0 * float(np.trace(cm)) / max(float(cm.sum()), 1e-9)

    annot = np.array(
        [
            [f"{int(cm[i, j])}\n{100.0 * cm[i, j] / row_sum[i, 0]:.0f}%" for j in range(n)]
            for i in range(n)
        ],
        dtype=object,
    )

    fig, ax = plt.subplots(figsize=(8.4, 6.85))
    sns.heatmap(
        norm,
        annot=annot,
        fmt="",
        cmap=CM_CMAP,
        vmin=0.0,
        vmax=1.0,
        square=True,
        linewidths=2.2,
        linecolor="white",
        cbar_kws={
            "label": "Fraction of true-class samples",
            "shrink": 0.82,
            "aspect": 26,
            "pad": 0.02,
        },
        ax=ax,
        annot_kws={
            "fontsize": 10.5,
            "fontweight": "semibold",
            "color": "#0f172a",
            "ha": "center",
            "va": "center",
        },
        xticklabels=class_names,
        yticklabels=class_names,
    )
    for t in ax.texts:
        t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white", alpha=0.95)])

    ax.set_xticklabels(ax.get_xticklabels(), rotation=32, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel("Predicted label", labelpad=10, fontweight="medium")
    ax.set_ylabel("True label", labelpad=10, fontweight="medium")
    ax.set_title(f"{title}\nOverall accuracy: {overall_acc:.1f}%", fontsize=13.5, pad=14, fontweight="semibold")

    _save_figure(fig, output_path)
    plt.close(fig)


def _convex_hull_patch(ax: plt.Axes, pts: np.ndarray, facecolor: str, alpha: float = 0.09) -> None:
    if pts.shape[0] < 3:
        return
    try:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=facecolor,
                edgecolor=facecolor,
                lw=0,
                alpha=alpha,
                zorder=0,
            )
        )
    except Exception:
        return


def plot_hidden_embedding_scatter(
    Z: np.ndarray,
    label_indices: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.4, 6.35))
    ax.set_facecolor("#f8fafc")
    y_idx = np.asarray(label_indices, dtype=int)

    for class_idx, name in enumerate(class_names):
        mask = y_idx == class_idx
        if not np.any(mask):
            continue
        c = CLASS_COLORS[class_idx % len(CLASS_COLORS)]
        pts = Z[mask]
        _convex_hull_patch(ax, pts.astype(float), c, alpha=0.11)
        ax.scatter(
            pts[:, 0] + 0.04,
            pts[:, 1] - 0.04,
            s=220,
            c="#0f172a",
            alpha=0.08,
            zorder=1,
            linewidths=0,
        )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            label=name,
            s=185,
            alpha=0.92,
            c=c,
            edgecolors="#f8fafc",
            linewidths=1.35,
            zorder=3,
        )

    ax.axhline(0.0, color="#e2e8f0", lw=0.9, zorder=0)
    ax.axvline(0.0, color="#e2e8f0", lw=0.9, zorder=0)
    ax.set_xlabel("Principal component 1", labelpad=8, fontweight="medium")
    ax.set_ylabel("Principal component 2", labelpad=8, fontweight="medium")
    ax.set_title(title, pad=14, fontweight="semibold")
    ax.legend(loc="best", scatterpoints=1, title="Class", alignment="left", framealpha=0.98)
    ax.margins(0.12)
    fig.subplots_adjust(top=0.9)
    _save_figure(fig, output_path)
    plt.close(fig)


def _dedupe_vertices(Z: np.ndarray, min_step: float = 1e-10) -> np.ndarray:
    # splprep chokes on repeated knots in the (x,y) plane; keep temporal order.
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape[0] <= 1:
        return Z
    out = [Z[0]]
    for i in range(1, Z.shape[0]):
        if np.linalg.norm(Z[i] - out[-1]) > min_step:
            out.append(Z[i])
    P = np.stack(out, axis=0)
    if P.shape[0] == 1:
        P = np.vstack([P, P[0] + 1e-6 * np.array([1.0, 1.0])])
    return P


def smooth_trajectory_2d(
    Z: np.ndarray,
    n_out: int = 512,
    s: float = 0.0,
) -> np.ndarray:
    # 2D parametric spline in index order; s=0 interpolates control points, s>0 relaxes.
    from scipy.interpolate import splprep, splev

    Z = np.asarray(Z, dtype=np.float64)
    m = Z.shape[0]
    if m == 0:
        return Z
    if m == 1:
        return np.repeat(Z, n_out, axis=0)

    P = _dedupe_vertices(Z)
    m = P.shape[0]
    x, y = P[:, 0], P[:, 1]
    k = min(3, m - 1)
    u_fine = np.linspace(0.0, 1.0, n_out, dtype=np.float64)

    def _dense_linear() -> np.ndarray:
        t_old = np.linspace(0.0, 1.0, m)
        xs = np.interp(u_fine, t_old, x)
        ys = np.interp(u_fine, t_old, y)
        return np.column_stack([xs, ys])

    sm = float(s)
    for _ in range(8):
        try:
            tck, _u = splprep([x, y], k=k, s=sm, nest=-1)
            out = splev(u_fine, tck)
            return np.column_stack([np.asarray(out[0]).ravel(), np.asarray(out[1]).ravel()])
        except Exception:
            sm = sm * 5.0 + 1e-6 * m
    return _dense_linear()


def plot_hidden_trajectories_smoothed(
    traj_2d: list[np.ndarray],
    label_indices: list[int],
    class_names: list[str],
    output_path: Path,
    title: str,
    n_curve_points: int = 512,
    spline_s: float = 0.0,
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.85, 6.65))
    ax.set_facecolor("#f8fafc")

    for Z, lab in zip(traj_2d, label_indices):
        c = CLASS_COLORS[lab % len(CLASS_COLORS)]
        Zs = smooth_trajectory_2d(Z, n_out=n_curve_points, s=spline_s)
        rgb = np.array(mcolors.to_rgb(c))
        ax.plot(
            Zs[:, 0],
            Zs[:, 1],
            "-",
            color=(*rgb, 0.2),
            lw=7.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )
        ax.plot(
            Zs[:, 0],
            Zs[:, 1],
            "-",
            color=c,
            lw=2.55,
            alpha=0.95,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
            zorder=2,
        )
        ax.scatter(
            Z[0, 0],
            Z[0, 1],
            c=c,
            marker="o",
            s=95,
            zorder=5,
            edgecolors="#f8fafc",
            linewidths=1.1,
        )
        ax.scatter(
            Z[-1, 0],
            Z[-1, 1],
            c=c,
            marker="s",
            s=88,
            zorder=5,
            edgecolors="#f8fafc",
            linewidths=1.1,
        )

    ax.axhline(0.0, color="#e2e8f0", lw=0.85, zorder=0)
    ax.axvline(0.0, color="#e2e8f0", lw=0.85, zorder=0)
    ax.set_xlabel("Principal component 1", labelpad=8, fontweight="medium")
    ax.set_ylabel("Principal component 2", labelpad=8, fontweight="medium")
    ax.set_title(title, pad=12, fontweight="semibold")

    legend_handles = [
        mlines.Line2D([], [], color=CLASS_COLORS[i % len(CLASS_COLORS)], lw=3.0, label=class_names[i])
        for i in range(len(class_names))
    ]
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color="#64748b",
            marker="o",
            linestyle="None",
            markersize=10,
            label="Start",
        )
    )
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color="#64748b",
            marker="s",
            linestyle="None",
            markersize=10,
            label="End",
        )
    )
    ax.legend(handles=legend_handles, loc="best", fontsize=10, title="Class & markers", framealpha=0.98)
    ax.margins(0.1)

    _save_figure(fig, output_path)
    plt.close(fig)
