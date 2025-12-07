"""
occlusion_estimator

Estimate likely occlusion by computing pairwise IoU between boxes in the same image.
If a box overlaps significantly with one or more boxes of different classes, it is
considered possibly occluded.

Output:
- per-class occlusion rate
- sample occluded boxes
"""

from typing import Dict, Any, List
from collections import defaultdict
import math

def _iou(b1, b2):
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    inter = w*h
    a1 = max(0.0, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(0.0, (b2[2]-b2[0])*(b2[3]-b2[1]))
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0

def run_occlusion_estimator(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - iou_threshold: float default 0.5
      - sample_limit: int default 20

    Returns:
      {
        "feature": "occlusion_estimator",
        "per_class_occlusion_fraction": {class: frac},
        "occluded_examples": [...]
      }
    """
    cfg = config or {}
    iou_thresh = float(cfg.get("iou_threshold", 0.5))
    sample_limit = int(cfg.get("sample_limit", 20))

    class_counts = defaultdict(int)
    class_occluded = defaultdict(int)
    examples = []

    for fname, rec in (index or {}).items():
        anns = rec.get("annotations", []) or []
        # precompute
        boxes = []
        for ann in anns:
            try:
                bbox = list(map(float, ann.get("bbox", [0,0,0,0])))
            except Exception:
                continue
            cls = str(ann.get("class",""))
            boxes.append((bbox, cls, ann))
        # for each box compute max IoU with boxes of other objects
        for i in range(len(boxes)):
            b1, cls1, ann1 = boxes[i]
            class_counts[cls1] += 1
            max_iou = 0.0
            for j in range(len(boxes)):
                if i == j:
                    continue
                b2, cls2, ann2 = boxes[j]
                # consider occlusion with ANY other box (even same class)
                try:
                    iou = _iou(b1, b2)
                except Exception:
                    iou = 0.0
                if iou > max_iou:
                    max_iou = iou
            if max_iou >= iou_thresh:
                class_occluded[cls1] += 1
                if len(examples) < sample_limit:
                    examples.append({"file": fname, "class": cls1, "bbox": b1, "max_iou": max_iou})

    per_class = {cls: (class_occluded[cls] / class_counts[cls]) if class_counts[cls] > 0 else 0.0 for cls in class_counts}
    return {
        "feature": "occlusion_estimator",
        "per_class_occlusion_fraction": per_class,
        "occluded_examples": examples,
        "status": "ok"
    }
