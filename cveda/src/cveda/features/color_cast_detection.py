"""
color_cast_detection

Detect images with a global color cast (dominant tint) using per-channel mean differences.
Process a sample of images from index that have abs_path.

Returns:
- sample mean channel deltas
- list of images whose channel distribution deviates strongly from dataset median
"""

from typing import Dict, Any, List
from PIL import Image
import numpy as np
import os
import statistics

def _mean_rgb(img_path, downscale=0.25):
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w,h = im.size
            small = im.resize((max(1,int(w*downscale)), max(1,int(h*downscale))))
            arr = np.asarray(small).astype(np.float32)/255.0
            m = arr.mean(axis=(0,1))  # R,G,B means
            return float(m[0]), float(m[1]), float(m[2])
    except Exception:
        return None

def run_color_cast_detection(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 200))
    downscale = float(cfg.get("downscale", 0.25))
    samples = []
    for fname, rec in (index or {}).items():
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        rgb = _mean_rgb(path, downscale=downscale)
        if rgb is None:
            continue
        samples.append({"file": fname, "r": rgb[0], "g": rgb[1], "b": rgb[2]})
        if len(samples) >= sample_limit:
            break

    if not samples:
        return {"feature": "color_cast_detection", "status": "no_images_processed"}

    # compute median per-channel
    rvals = [s["r"] for s in samples]
    gvals = [s["g"] for s in samples]
    bvals = [s["b"] for s in samples]
    med_r = statistics.median(rvals)
    med_g = statistics.median(gvals)
    med_b = statistics.median(bvals)

    # compute per-image color cast score: average absolute deviation from medians
    outliers = []
    for s in samples:
        dev = (abs(s["r"]-med_r) + abs(s["g"]-med_g) + abs(s["b"]-med_b))/3.0
        if dev > cfg.get("outlier_threshold", 0.06):  # empirical threshold
            outliers.append({"file": s["file"], "dev": dev, "r": s["r"], "g": s["g"], "b": s["b"]})

    return {"feature": "color_cast_detection", "median_rgb": [med_r, med_g, med_b], "outliers": outliers[:cfg.get("sample_limit", 20)], "status": "ok"}
