"""
train_ready_scoring

Produce a lightweight 'train ready' score per image combining:
- annotation coverage fraction
- presence of corrupted or missing annotations
- number of annotations
- image size

Score is 0..1 higher is better.

Return top and bottom image examples and overall distribution summary.

Config
------
- weights: dict with keys coverage, size, annotations, default weights used
- sample_limit: how many images included in examples
"""

from typing import Dict, Any, List
from collections import defaultdict
import math
import os

def run_train_ready_scoring(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    weights = cfg.get("weights", {"coverage": 0.4, "size": 0.2, "annotations": 0.3, "corrupt": 0.1})
    sample_limit = int(cfg.get("sample_limit", 20))

    scores = []
    for fname, rec in (index or {}).items():
        # skip corrupted if indicated in meta
        corrupted = bool(rec.get("meta", {}).get("corrupt", False))
        width = rec.get("width") or rec.get("meta", {}).get("width") or 0
        height = rec.get("height") or rec.get("meta", {}).get("height") or 0
        anns = rec.get("annotations", []) or []
        # coverage
        img_area = float(max(1.0, width*height))
        ann_area = 0.0
        for ann in anns:
            try:
                x0,y0,x1,y1 = map(float, ann.get("bbox", [0,0,0,0]))
                ann_area += max(0.0, (x1-x0)*(y1-y0))
            except Exception:
                continue
        coverage = min(1.0, ann_area / img_area)
        # size score: prefer larger images, normalized log scale
        size_score = math.tanh(math.log(max(1.0, width*height))/10.0)
        # annotation count score: sigmoid-like
        n_ann = len(anns)
        ann_score = 1.0 - math.exp(-n_ann/5.0)
        corrupt_penalty = 0.0 if not corrupted else 1.0
        score = (weights["coverage"] * coverage +
                 weights["size"] * size_score +
                 weights["annotations"] * ann_score -
                 weights["corrupt"] * corrupt_penalty)
        score = max(0.0, min(1.0, score))
        scores.append({"file": fname, "score": score, "coverage": coverage, "n_annotations": n_ann, "size": width*height, "corrupt": corrupted})

    if not scores:
        return {"feature": "train_ready_scoring", "status": "no_images"}

    scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
    top = scores_sorted[:sample_limit]
    bottom = scores_sorted[-sample_limit:]
    mean_score = sum(s["score"] for s in scores)/len(scores)
    return {"feature": "train_ready_scoring", "mean_score": mean_score, "top_examples": top, "bottom_examples": bottom, "status": "ok"}
