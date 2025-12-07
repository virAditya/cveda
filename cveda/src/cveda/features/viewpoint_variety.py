"""
viewpoint_variety

Estimate coarse viewpoint using bounding box aspect ratio and object centroid vertical position.
This is heuristic. For robust viewpoint detection use a trained model.

Returns counts for buckets: top_view, side_view, front_view, unknown
"""

from typing import Dict, Any
from collections import defaultdict

def run_viewpoint_variety(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    counts = defaultdict(int)
    examples = defaultdict(list)
    for fname, rec in (index or {}).items():
        w = rec.get("width") or 0; h = rec.get("height") or 0
        for ann in (rec.get("annotations", []) or []):
            try:
                x0,y0,x1,y1 = map(float, ann.get("bbox", [0,0,0,0]))
            except Exception:
                continue
            ar = (x1-x0) / max(1.0, (y1-y0))
            cy = (y0+y1)/2.0 / max(1.0, h)
            # heuristics
            if ar < 0.5:
                vp = "vertical_view"  # tall object maybe side or top depending on class
            elif ar > 2.0:
                vp = "long_horizontal_view"
            elif cy < 0.2:
                vp = "top_view"
            elif cy > 0.8:
                vp = "bottom_view"
            else:
                vp = "frontal"
            counts[vp] += 1
            if len(examples[vp]) < cfg.get("sample_limit", 5):
                examples[vp].append({"file": fname, "class": str(ann.get("class","")), "bbox": [x0,y0,x1,y1]})
    return {"feature": "viewpoint_variety", "counts": dict(counts), "examples": dict(examples), "status": "ok"}
