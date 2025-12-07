"""
annotation_rarity_buckets

Bucket classes into rarity buckets to prioritize collection/augmentation.

Buckets by default:
- very_rare: < 10
- rare: 10-99
- common: 100-999
- abundant: >= 1000
"""

from typing import Dict, Any
from collections import Counter, defaultdict

def run_annotation_rarity_buckets(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - buckets: list of (name, min_inclusive, max_inclusive) optional

    Returns:
      {
        "feature": "annotation_rarity_buckets",
        "counts": {class: count},
        "buckets": {"very_rare": [...], ...}
      }
    """
    cfg = config or {}
    # default bucket thresholds
    buckets = cfg.get("buckets", [
        ("very_rare", 0, 9),
        ("rare", 10, 99),
        ("common", 100, 999),
        ("abundant", 1000, 10**12)
    ])

    counter = Counter()
    for fname, rec in (index or {}).items():
        for ann in (rec.get("annotations", []) or []):
            counter[str(ann.get("class",""))] += 1

    bucket_map = defaultdict(list)
    for cls, cnt in counter.items():
        for name, lo, hi in buckets:
            if lo <= cnt <= hi:
                bucket_map[name].append({"class": cls, "count": cnt})
                break

    return {
        "feature": "annotation_rarity_buckets",
        "counts": dict(counter),
        "buckets": dict(bucket_map),
        "status": "ok"
    }
