"""
multiresolution_similarity

Check whether important annotations vanish under progressive downsampling.
We compute, per class or globally, the fraction of annotations that would
have bbox area under a visibility threshold at each downscale.

Heuristic
---------
- For each sampled image, for each annotation compute normalized bbox area
- simulate downscales and check if area falls below threshold

Config
------
- sample_limit int default 200 images
- downscale_steps list default [1.0, 0.75, 0.5, 0.25]
- visibility_area_frac default 0.001 area fraction considered visible

Return
------
{
  "feature":"multiresolution_similarity",
  "scales": {"0.5": {"visible_fraction": 0.9}, ...},
  "status":"ok"
}
"""

from typing import Dict, Any
import os
from collections import defaultdict

def run_multiresolution_similarity(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 200))
    steps = cfg.get("downscale_steps", [1.0, 0.75, 0.5, 0.25])
    vis_frac = float(cfg.get("visibility_area_frac", 0.001))

    scale_stats = {s: {"total": 0, "visible": 0} for s in steps}
    sampled = 0

    for fname, rec in (index or {}).items():
        if sampled >= sample_limit:
            break
        sampled += 1
        w = rec.get("width", 0) or 1
        h = rec.get("height", 0) or 1
        anns = rec.get("annotations", []) or []
        for ann in anns:
            try:
                x0,y0,x1,y1 = map(float, ann.get("bbox", [0,0,0,0]))
            except Exception:
                continue
            area = max(0.0, (x1-x0)*(y1-y0)) / (w*h)
            for s in steps:
                # downscale reduces both image dims so area scales by s^2
                scaled_area = area * (s*s)
                scale_stats[s]["total"] += 1
                if scaled_area >= vis_frac:
                    scale_stats[s]["visible"] += 1

    result = {}
    for s, vals in scale_stats.items():
        total = vals["total"]
        visible = vals["visible"]
        result[str(s)] = {"total_annotations": total, "visible_annotations": visible, "visible_fraction": (visible/total if total>0 else None)}

    return {"feature": "multiresolution_similarity", "scales": result, "status": "ok"}
