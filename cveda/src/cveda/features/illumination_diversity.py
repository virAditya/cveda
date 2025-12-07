"""
illumination_diversity

Estimate per-image mean luminance and categorize images into buckets: dark, normal, bright.

Return proportions and sample examples from each bucket.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
import os
from collections import defaultdict

def run_illumination_diversity(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 500))
    downscale = float(cfg.get("downscale", 0.25))
    buckets = {"dark": [], "normal": [], "bright": []}
    scanned = 0

    for fname, rec in (index or {}).items():
        if scanned >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        scanned += 1
        try:
            with Image.open(path) as im:
                im = im.convert("L")
                w,h = im.size
                small = im.resize((max(1,int(w*downscale)), max(1,int(h*downscale))))
                arr = np.asarray(small).astype(float)/255.0
                mean = float(arr.mean())
                if mean < cfg.get("dark_threshold", 0.35):
                    buckets["dark"].append({"file": fname, "mean": mean})
                elif mean > cfg.get("bright_threshold", 0.75):
                    buckets["bright"].append({"file": fname, "mean": mean})
                else:
                    buckets["normal"].append({"file": fname, "mean": mean})
        except Exception:
            continue

    total = sum(len(v) for v in buckets.values())
    proportions = {k: (len(v)/total if total>0 else 0.0) for k,v in buckets.items()}
    samples = {k: v[:cfg.get("sample_limit", 10)] for k,v in buckets.items()}

    return {"feature": "illumination_diversity", "scanned": scanned, "proportions": proportions, "samples": samples, "status": "ok"}
