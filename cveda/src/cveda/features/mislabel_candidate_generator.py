"""
mislabel_candidate_generator

Simple mislabel candidate generator using per-class bounding box area and aspect ratio
outliers. Without embeddings a cheap heuristic is to find annotations that are far from
class mean statistics.

Algorithm
---------
- compute per-class mean and std of area fraction and aspect ratio
- mark annotations with z-score > config.threshold as candidate mislabels
Return top candidates per class.

This is a heuristic scaffold. For production use use embeddings clustering.
"""

from typing import Dict, Any, List
from collections import defaultdict
import math

def run_mislabel_candidate_generator(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    z_thresh = float(cfg.get("z_threshold", 3.0))
    sample_limit = int(cfg.get("sample_limit", 50))

    per_class_metrics = defaultdict(list)
    # first pass compute area fraction and aspect ratio per class
    for fname, rec in (index or {}).items():
        w = rec.get("width") or 1
        h = rec.get("height") or 1
        for ann in (rec.get("annotations", []) or []):
            cls = str(ann.get("class",""))
            try:
                x0,y0,x1,y1 = map(float, ann.get("bbox", [0,0,0,0]))
            except Exception:
                continue
            area_frac = max(0.0, (x1-x0)*(y1-y0) / max(1.0, w*h))
            ar = (x1-x0) / max(1.0, (y1-y0))
            per_class_metrics[cls].append({"file": fname, "area_frac": area_frac, "aspect_ratio": ar})

    candidates = []
    for cls, vals in per_class_metrics.items():
        areas = [v["area_frac"] for v in vals]
        ars = [v["aspect_ratio"] for v in vals]
        if not areas:
            continue
        mean_area = sum(areas)/len(areas)
        var_area = sum((a-mean_area)**2 for a in areas)/len(areas)
        std_area = math.sqrt(var_area)
        mean_ar = sum(ars)/len(ars)
        var_ar = sum((a-mean_ar)**2 for a in ars)/len(ars)
        std_ar = math.sqrt(var_ar)
        for v in vals:
            z_area = (v["area_frac"] - mean_area) / (std_area if std_area>0 else 1e-6)
            z_ar = (v["aspect_ratio"] - mean_ar) / (std_ar if std_ar>0 else 1e-6)
            score = max(abs(z_area), abs(z_ar))
            if score >= z_thresh:
                candidates.append({"class": cls, "file": v["file"], "z_area": z_area, "z_ar": z_ar, "score": score})
    # sort by score descending and return top samples
    candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)[:sample_limit]
    return {"feature": "mislabel_candidate_generator", "candidates": candidates_sorted, "status": "ok"}
