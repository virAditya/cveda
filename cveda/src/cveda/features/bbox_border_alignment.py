"""
bbox_border_alignment

Detect bounding boxes that align with image borders (touch or sit very close).
This often signals cropping or careless annotation.

Outputs:
- global fraction of border-touching boxes
- per-class border-touch fraction
- small sample of offending annotations
"""

from typing import Dict, Any, List
from collections import defaultdict

def run_bbox_border_alignment(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - eps: float (pixels) threshold to consider "touching" (default 1.0)
      - sample_limit: int (default 20)

    Returns:
      {
        "feature": "bbox_border_alignment",
        "n_boxes": int,
        "border_touch_fraction": float,
        "per_class": {class: fraction},
        "examples": [...]
      }
    """
    cfg = config or {}
    eps = float(cfg.get("eps", 1.0))
    sample_limit = int(cfg.get("sample_limit", 20))

    n_boxes = 0
    border_count = 0
    per_class_counts = defaultdict(int)
    per_class_border = defaultdict(int)
    examples = []

    for fname, rec in (index or {}).items():
        width = rec.get("width", 0) or rec.get("meta", {}).get("width", 0) or 0
        height = rec.get("height", 0) or rec.get("meta", {}).get("height", 0) or 0
        for ann in (rec.get("annotations", []) or []):
            bbox = ann.get("bbox", [0,0,0,0])
            try:
                x0,y0,x1,y1 = map(float, bbox)
            except Exception:
                continue
            n_boxes += 1
            cls = str(ann.get("class",""))
            per_class_counts[cls] += 1
            touches = (abs(x0 - 0.0) <= eps) or (abs(y0 - 0.0) <= eps) or (abs(x1 - width) <= eps) or (abs(y1 - height) <= eps)
            if touches:
                per_class_border[cls] += 1
                border_count += 1
                if len(examples) < sample_limit:
                    examples.append({"file": fname, "class": cls, "bbox": [x0,y0,x1,y1]})

    overall_frac = border_count / max(1, n_boxes)
    per_class_frac = {cls: per_class_border[cls] / per_class_counts[cls] for cls in per_class_counts}

    return {
        "feature": "bbox_border_alignment",
        "n_boxes": n_boxes,
        "border_touch_fraction": overall_frac,
        "per_class_border_fraction": per_class_frac,
        "examples": examples,
        "status": "ok"
    }
