"""
absolute_coordinate_patterns

Detect repeated absolute bbox coordinates across images that hint at copy-paste
or templated annotation mistakes.

Approach:
- round coordinates to nearest integer and count identical bbox tuples
- return top repeated bbox entries and a small list of example filenames for each
"""

from typing import Dict, Any, Tuple, List
from collections import Counter, defaultdict
import math

def run_absolute_coordinate_patterns(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - top_k: int default 30
      - min_count: int default 3

    Returns:
      {
        "feature": "absolute_coordinate_patterns",
        "top_repeats": [ {"bbox": (x0,y0,x1,y1), "count": n, "examples": [file1,...]} ],
        "status": "ok"
      }
    """
    cfg = config or {}
    top_k = int(cfg.get("top_k", 30))
    min_count = int(cfg.get("min_count", 3))

    bbox_counter = Counter()
    bbox_examples = defaultdict(list)

    for fname, rec in (index or {}).items():
        for ann in (rec.get("annotations", []) or []):
            bbox = ann.get("bbox", [0,0,0,0])
            try:
                rounded = tuple(int(round(float(x))) for x in bbox)
            except Exception:
                continue
            bbox_counter[rounded] += 1
            if len(bbox_examples[rounded]) < top_k:
                bbox_examples[rounded].append(fname)

    repeated = []
    for bbox, count in bbox_counter.most_common(top_k):
        if count >= min_count:
            repeated.append({"bbox": bbox, "count": count, "examples": bbox_examples.get(bbox, [])[:5]})

    return {
        "feature": "absolute_coordinate_patterns",
        "top_repeats": repeated,
        "status": "ok"
    }
