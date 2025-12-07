"""
environment_diversity

A simple environment classifier using heuristics:
- indoor: presence of small texture, uniform lighting, low green channel
- outdoor: larger green content, sky like blue region

This is coarse and heuristic driven. For production use a classifier model.

Return counts of predicted categories and example files.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
import os
from collections import defaultdict

def run_environment_diversity(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 500))
    downscale = float(cfg.get("downscale", 0.25))
    counts = defaultdict(int)
    examples = defaultdict(list)
    scanned = 0

    for fname, rec in (index or {}).items():
        if scanned >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w,h = im.size
                small = im.resize((max(1,int(w*downscale)), max(1,int(h*downscale))))
                arr = np.asarray(small).astype(float)/255.0
                mean_r = arr[:,:,0].mean()
                mean_g = arr[:,:,1].mean()
                mean_b = arr[:,:,2].mean()
                # heuristic rules
                if mean_g > 0.35 and mean_b < 0.6:
                    label = "outdoor_green"
                elif mean_r > 0.45 and mean_g > 0.4:
                    label = "indoor_warm"
                elif mean_b > 0.5:
                    label = "outdoor_sky"
                else:
                    label = "unknown"
                counts[label] += 1
                if len(examples[label]) < 10:
                    examples[label].append({"file": fname, "r": mean_r, "g": mean_g, "b": mean_b})
                scanned += 1
        except Exception:
            continue

    return {"feature": "environment_diversity", "scanned": scanned, "counts": dict(counts), "examples": dict(examples), "status": "ok"}
