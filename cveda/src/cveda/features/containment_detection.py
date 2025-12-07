"""
containment_detection

Detect frequent containment relationships where one bbox is (nearly) entirely inside another.
Useful for part->whole relationships like wheel inside car or plate inside vehicle.
"""

from typing import Dict, Any, List
from collections import defaultdict

def _contains(b_outer, b_inner, eps=1e-6):
    return (b_inner[0] >= b_outer[0]-eps and b_inner[1] >= b_outer[1]-eps and b_inner[2] <= b_outer[2]+eps and b_inner[3] <= b_outer[3]+eps)

def run_containment_detection(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - min_support: minimum occurrences to report a containment pair (default 5)
      - sample_limit: how many example pairs to return

    Returns:
      {
        "feature": "containment_detection",
        "top_pairs": [ {"container": A, "contained": B, "count": n, "examples": [...] } ],
        "status": "ok"
      }
    """
    cfg = config or {}
    min_support = int(cfg.get("min_support", 5))
    sample_limit = int(cfg.get("sample_limit", 10))

    pair_counts = defaultdict(int)
    pair_examples = defaultdict(list)

    for fname, rec in (index or {}).items():
        anns = rec.get("annotations", []) or []
        boxes = []
        for ann in anns:
            try:
                bbox = list(map(float, ann.get("bbox", [0,0,0,0])))
            except Exception:
                continue
            boxes.append((bbox, str(ann.get("class",""))))
        # check containment for each ordered pair
        for i in range(len(boxes)):
            b1, cls1 = boxes[i]
            for j in range(len(boxes)):
                if i==j: continue
                b2, cls2 = boxes[j]
                if _contains(b1, b2):
                    pair_counts[(cls1, cls2)] += 1
                    if len(pair_examples[(cls1, cls2)]) < sample_limit:
                        pair_examples[(cls1, cls2)].append({"file": fname, "container_bbox": b1, "contained_bbox": b2})

    # prepare top pairs above support threshold
    top_pairs = []
    for (c1,c2), cnt in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        if cnt >= min_support:
            top_pairs.append({"container": c1, "contained": c2, "count": cnt, "examples": pair_examples[(c1,c2)][:sample_limit]})

    return {"feature": "containment_detection", "top_pairs": top_pairs, "status": "ok"}
