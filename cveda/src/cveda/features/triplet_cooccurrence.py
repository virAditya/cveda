"""
triplet_cooccurrence

Count triplets of classes that appear together in images, report top triplets.
This helps discover recurring scene structures.
"""

from typing import Dict, Any
from collections import Counter
import itertools

def run_triplet_cooccurrence(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - top_k: int default 50

    Returns:
      {
        "feature": "triplet_cooccurrence",
        "top_triplets": [ {"triplet": (A,B,C), "count": n} ],
        "status": "ok"
      }
    """
    cfg = config or {}
    top_k = int(cfg.get("top_k", 50))
    counter = Counter()
    for fname, rec in (index or {}).items():
        classes = sorted({str(ann.get("class","")) for ann in (rec.get("annotations", []) or [])})
        if len(classes) < 3:
            continue
        for t in itertools.combinations(classes, 3):
            counter[t] += 1

    top = [{"triplet": t, "count": c} for t,c in counter.most_common(top_k)]
    return {"feature": "triplet_cooccurrence", "top_triplets": top, "status": "ok"}
