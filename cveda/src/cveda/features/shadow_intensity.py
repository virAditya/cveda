"""
shadow_intensity

Estimate shadow coverage fraction per image by using a simple local contrast heuristic:
- convert image to grayscale, compute local minimum filter via downsample smoothing,
- pixels significantly darker than local neighborhood are likely shadow.
Requires images to be readable; if index entries lack abs_path, module returns note.
"""

from typing import Dict, Any
import numpy as np
from PIL import Image
from collections import defaultdict
import os

def _estimate_shadow_fraction(img_path, downscale=0.25, threshold=0.6):
    try:
        with Image.open(img_path) as im:
            im = im.convert("L")
            w,h = im.size
            # downsample for speed
            small = im.resize((max(1,int(w*downscale)), max(1,int(h*downscale))))
            arr = np.asarray(small).astype(np.float32)/255.0
            # local mean via simple uniform filter using convolution (fast via mean of small blocks)
            kernel = 7
            # pad and compute local mean using simple sliding window by convolution via integral image
            integral = arr.cumsum(axis=0).cumsum(axis=1)
            # approximate local mean by block sampling, simpler: use uniform filter via numpy's pad and mean of neighbors
            padded = np.pad(arr, kernel//2, mode='reflect')
            local_mean = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    block = padded[i:i+kernel, j:j+kernel]
                    local_mean[i,j] = block.mean()
            shadow_mask = arr < (local_mean * threshold)
            frac = float(shadow_mask.sum())/shadow_mask.size
            return frac
    except Exception:
        return None

def run_shadow_intensity(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 20))
    downscale = float(cfg.get("downscale", 0.25))
    threshold = float(cfg.get("threshold", 0.6))

    results = []
    n_processed = 0
    for fname, rec in (index or {}).items():
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        frac = _estimate_shadow_fraction(path, downscale=downscale, threshold=threshold)
        if frac is None:
            continue
        results.append({"file": fname, "shadow_fraction": frac})
        n_processed += 1
        if n_processed >= sample_limit:
            break

    if not results:
        return {"feature": "shadow_intensity", "status": "no_images_processed"}

    mean_frac = sum(r["shadow_fraction"] for r in results)/len(results)
    return {"feature": "shadow_intensity", "sampled_images": len(results), "mean_shadow_fraction": mean_frac, "examples": results, "status": "ok"}
