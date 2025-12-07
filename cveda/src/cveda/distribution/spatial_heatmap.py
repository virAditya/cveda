"""
Spatial heatmap computation.

Compute normalized center heatmaps for each class and for the whole dataset.
The output includes numpy arrays suitable for plotting in viz.plot functions.
"""

from typing import Dict, Any, Tuple
import numpy as np


def compute_spatial_heatmaps(index: Dict[str, Any], cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute per class and overall normalized center heatmaps.

    cfg supports:
    - bins tuple default (64 64)
    - min_samples int default 10 skip heatmap generation for rare classes

    Returns:
    {
        "overall": heatmap_array,
        "per_class": {"class_name": heatmap_array, ...}
    }
    """
    cfg = cfg or {}
    bins = cfg.get("bins", (64, 64))
    min_samples = cfg.get("min_samples", 10)

    overall = np.zeros(bins, dtype=float)
    class_maps = {}
    counts = {}

    for fname, rec in index.items():
        w = rec.get("width")
        h = rec.get("height")
        if not w or not h:
            continue
        for ann in rec.get("annotations", []):
            cls = str(ann.get("class"))
            xmin, ymin, xmax, ymax = ann["bbox"]
            cx = (xmin + xmax) / 2.0 / float(w)
            cy = (ymin + ymax) / 2.0 / float(h)
            # clamp normalized coords
            cx = min(max(0.0, cx), 1.0)
            cy = min(max(0.0, cy), 1.0)
            ix = int(cx * (bins[1] - 1))
            iy = int(cy * (bins[0] - 1))
            overall[iy, ix] += 1.0
            cm = class_maps.setdefault(cls, np.zeros(bins, dtype=float))
            cm[iy, ix] += 1.0
            counts[cls] = counts.get(cls, 0) + 1

    # normalize heatmaps to density
    overall_norm = overall / (overall.sum() + 1e-12)
    per_class_norm = {}
    for cls, mat in class_maps.items():
        if counts.get(cls, 0) >= min_samples:
            per_class_norm[cls] = mat / (mat.sum() + 1e-12)
        else:
            per_class_norm[cls] = None

    return {"overall": overall_norm, "per_class": per_class_norm, "counts": counts}
