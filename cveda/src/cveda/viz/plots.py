"""
Plotting utilities built with matplotlib.

Each plotting function returns a matplotlib Figure object.
All functions handle empty input gracefully and return a figure with
a readable message when no data is available.
"""

from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


def plot_class_distribution(counts: Dict[str, int]) -> plt.Figure:
    """
    Plot a horizontal bar chart of class counts.

    Parameters
    counts dict class to integer count

    Returns matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if not counts:
        ax.text(0.5, 0.5, "No class counts available", ha="center", va="center")
        return fig
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [str(k) for k, _ in items]
    vals = [v for _, v in items]
    y = np.arange(len(labels))
    ax.barh(y, vals)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Annotation count")
    ax.set_title("Class distribution")
    fig.tight_layout()
    return fig


def plot_bbox_statistics(stats: Dict[str, Any]) -> plt.Figure:
    """
    Plot a simple visualization of overall bbox area distribution using a histogram.

    stats expected to be the overall summary returned by compute_bbox_statistics
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if not stats:
        ax.text(0.5, 0.5, "No bbox statistics available", ha="center", va="center")
        return fig
    # stats may include mean median min max
    mean = stats.get("mean", 0.0)
    median = stats.get("median", 0.0)
    ax.hist([mean], bins=1)
    ax.text(0.05, 0.95, f"Mean area {mean:.2f}", transform=ax.transAxes)
    ax.text(0.05, 0.90, f"Median area {median:.2f}", transform=ax.transAxes)
    ax.set_title("BBox overall summary")
    fig.tight_layout()
    return fig


def plot_heatmap(heatmap: Optional[np.ndarray], title: str = "Heatmap") -> plt.Figure:
    """
    Plot a 2D heatmap array.

    If heatmap is None produce a placeholder figure that informs the user.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    if heatmap is None:
        ax.text(0.5, 0.5, "Not enough data for heatmap", ha="center", va="center")
        return fig
    im = ax.imshow(heatmap, interpolation="nearest", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def make_sample_collage(image_paths: List[str], grid: Tuple[int, int] = (3, 4), thumb_size: Tuple[int, int] = (128, 128)) -> plt.Figure:
    """
    Create a simple collage figure of images given by absolute paths.

    Parameters
    image_paths list of str absolute paths
    grid tuple rows cols default 3 4
    thumb_size tuple w h thumbnail size in pixels

    Returns a matplotlib Figure object with the collage.
    """
    rows, cols = grid
    total = rows * cols
    chosen = image_paths[:total]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() if total > 1 else [axes]
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(chosen):
            try:
                img = plt.imread(chosen[i])
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image not readable", ha="center", va="center")
        else:
            ax.set_facecolor((0.95, 0.95, 0.95))
    fig.suptitle("Sample images")
    fig.tight_layout()
    return fig
