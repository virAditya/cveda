"""
mutual_exclusion

Find class pairs that never or rarely co-occur. Useful to detect label inconsistencies
or to discover classes that split the dataset.
"""

from typing import Dict, Any
from collections import defaultdict
import itertools

def run_mutual_exclusion(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - min_support: ignore pairs where total support is below this (default 5)
      - sample_limit

    Returns:
      {
        "feature": "mutual_exclusion",
        "rare_pairs": [ {"pair": (A,B), "cooccurrence": 0, "support": x } ],
        "status": "ok"
      }
    """
    cfg = config or {}
    min_support = int(cfg.get("min_support", 5))

    class_images = defaultdict(set)
    for fname, rec in (index or {}).items():
        classes = set()
        for ann in (rec.get("annotations", []) or []):
            classes.add(str(ann.get("class","")))
        for c in classes:
            class_images[c].add(fname)

    classes = list(class_images.keys())
    rare_pairs = []
    for a,b in itertools.combinations(classes, 2):
        imgs_a = class_images.get(a, set())
        imgs_b = class_images.get(b, set())
        support = len(imgs_a.union(imgs_b))
        co = len(imgs_a.intersection(imgs_b))
        if support >= min_support and co == 0:
            rare_pairs.append({"pair": (a,b), "cooccurrence": co, "support": support})

    return {"feature": "mutual_exclusion", "rare_pairs": rare_pairs, "status": "ok"}
