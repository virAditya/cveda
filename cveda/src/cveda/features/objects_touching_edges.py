"""
objects_touching_edges

Report per-image and per-class counts of objects that touch image edges.
Useful to surface crops, occlusions, or truncated objects that may need special handling.
"""

from typing import Dict, Any, List
from collections import defaultdict

def run_objects_touching_edges(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - eps: float pixels threshold default 1.0
      - sample_limit: int default 20

    Returns:
      {
        "feature": "objects_touching_edges",
        "touch_fraction_global": float,
        "per_class_touch_fraction": {class: fraction},
        "examples": [...]
      }
    """
    cfg = config or {}
    eps = float(cfg.get("eps", 1.0))
    sample_limit = int(cfg.get("sample_limit", 20))

    total_boxes = 0
    total_touch = 0
    per_class_counts = defaultdict(int)
    per_class_touch = defaultdict(int)
    examples = []

    for fname, rec in (index or {}).items():
        w = rec.get("width", 0) or rec.get("meta", {}).get("width", 0) or 0
        h = rec.get("height", 0) or rec.get("meta", {}).get("height", 0) or 0
        for ann in (rec.get("annotations", []) or []):
            bbox = ann.get("bbox", [0,0,0,0])
            try:
                x0,y0,x1,y1 = map(float, bbox)
            except Exception:
                continue
            total_boxes += 1
            cls = str(ann.get("class",""))
            per_class_counts[cls] += 1
            touching = (x0 <= eps) or (y0 <= eps) or (abs(x1 - (w)) <= eps) or (abs(y1 - (h)) <= eps)
            if touching:
                total_touch += 1
                per_class_touch[cls] += 1
                if len(examples) < sample_limit:
                    examples.append({"file": fname, "class": cls, "bbox": [x0,y0,x1,y1]})

    global_frac = total_touch / max(1, total_boxes)
    per_class_frac = {cls: per_class_touch[cls] / per_class_counts[cls] for cls in per_class_counts}

    return {
        "feature": "objects_touching_edges",
        "n_boxes": total_boxes,
        "touching_boxes": total_touch,
        "touch_fraction_global": global_frac,
        "per_class_touch_fraction": per_class_frac,
        "examples": examples,
        "status": "ok"
    }
