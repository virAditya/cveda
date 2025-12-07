"""
relative_size_ratios

Compute area ratios between pairs of classes within the same image.
Return per-pair mean and std of ratios and examples where ratios are extreme.
Useful to detect annotation scale inconsistencies.
"""

from typing import Dict, Any, List
from collections import defaultdict
import math

def run_relative_size_ratios(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - sample_limit: int default 20
      - min_support: only consider pairs with >= min_support co-occurrences

    Returns:
      {
        "feature": "relative_size_ratios",
        "pairs": { (A,B): {mean, std, support, examples} },
        "status": "ok"
      }
    """
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 20))
    min_support = int(cfg.get("min_support", 5))

    pair_values = defaultdict(list)
    pair_examples = defaultdict(list)

    for fname, rec in (index or {}).items():
        anns = rec.get("annotations", []) or []
        # compute areas per annotation
        entries = []
        for ann in anns:
            try:
                x0,y0,x1,y1 = map(float, ann.get("bbox", [0,0,0,0]))
            except Exception:
                continue
            area = max(0.0, (x1-x0)*(y1-y0))
            entries.append((str(ann.get("class","")), area, ann))
        # for each pair A,B compute areaA/areaB for co-occurring pairs
        for i in range(len(entries)):
            for j in range(i+1, len(entries)):
                a_cls, a_area, _ = entries[i]
                b_cls, b_area, _ = entries[j]
                if b_area > 0:
                    ratio = a_area / b_area
                    pair_values[(a_cls, b_cls)].append(ratio)
                    if len(pair_examples[(a_cls,b_cls)]) < sample_limit:
                        pair_examples[(a_cls,b_cls)].append({"file": fname, "ratio": ratio})
                if a_area > 0:
                    ratio2 = b_area / a_area
                    pair_values[(b_cls, a_cls)].append(ratio2)
                    if len(pair_examples[(b_cls,a_cls)]) < sample_limit:
                        pair_examples[(b_cls,a_cls)].append({"file": fname, "ratio": ratio2})

    result_pairs = {}
    for pair, vals in pair_values.items():
        if len(vals) < min_support:
            continue
        mean = sum(vals)/len(vals)
        var = sum((v-mean)**2 for v in vals)/len(vals)
        std = math.sqrt(var)
        result_pairs[pair] = {"mean": mean, "std": std, "support": len(vals), "examples": pair_examples.get(pair, [])[:sample_limit]}

    return {"feature": "relative_size_ratios", "pairs": result_pairs, "status": "ok"}
