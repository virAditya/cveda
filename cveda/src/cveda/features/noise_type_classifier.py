"""
noise_type_classifier

Lightweight heuristics to classify noise type using local patch statistics.
It returns counts for three categories: gaussian_like, saltpepper_like, unknown.

Heuristics
---------
- gaussian_like: global variance high but no extreme outliers
- saltpepper_like: presence of many extreme pixel values indicative of impulse noise

This is indicative only. For high accuracy a trained classifier or statistical tests are needed.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
import os

def run_noise_type_classifier(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 200))
    downscale = float(cfg.get("downscale", 0.25))
    results = {"gaussian_like": 0, "saltpepper_like": 0, "unknown": 0}
    examples = []

    processed = 0
    for fname, rec in (index or {}).items():
        if processed >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        processed += 1
        try:
            with Image.open(path) as im:
                im = im.convert("L")
                w,h = im.size
                small = im.resize((max(1,int(w*downscale)), max(1,int(h*downscale))))
                arr = np.asarray(small).astype(np.float32)
                var = float(np.var(arr))
                p01 = np.percentile(arr, 1)
                p99 = np.percentile(arr, 99)
                # salt pepper heuristic: many pixels at extremes, e.g., many zeros or 255s
                extreme_ratio = float(((arr<=1).sum() + (arr>=254).sum())) / arr.size
                if extreme_ratio > 0.01:
                    results["saltpepper_like"] += 1
                    examples.append({"file": fname, "type": "saltpepper", "extreme_ratio": extreme_ratio, "var": var})
                elif var > cfg.get("var_threshold", 500.0):
                    results["gaussian_like"] += 1
                    examples.append({"file": fname, "type": "gaussian_like", "extreme_ratio": extreme_ratio, "var": var})
                else:
                    results["unknown"] += 1
        except Exception:
            results["unknown"] += 1

    return {"feature": "noise_type_classifier", "processed": processed, "counts": results, "examples": examples[:cfg.get("sample_limit",20)], "status": "ok"}
