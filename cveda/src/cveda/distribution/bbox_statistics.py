"""
Bounding box statistics module.

Compute distribution of bbox sizes per class and overall aggregate metrics.
"""

from typing import Dict, Any, List
import math


def compute_bbox_statistics(index: Dict[str, Any], cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute stats per class:
    - count
    - mean_area
    - median_area
    - min_area
    - max_area
    - relative_area_mean compared to image area

    Returns a dict with per_class statistics and overall summary.
    """
    cfg = cfg or {}
    per_class = {}
    all_areas = []
    for fname, rec in index.items():
        w = rec.get("width")
        h = rec.get("height")
        img_area = (w * h) if w and h else None
        for ann in rec.get("annotations", []):
            cls = str(ann.get("class"))
            xmin, ymin, xmax, ymax = ann["bbox"]
            area = max(0.0, (xmax - xmin) * (ymax - ymin))
            rel = (area / img_area) if img_area else None
            entry = per_class.setdefault(cls, {"areas": [], "relative": []})
            entry["areas"].append(area)
            if rel is not None:
                entry["relative"].append(rel)
            all_areas.append(area)

    def summarize(lst: List[float]) -> Dict[str, float]:
        if not lst:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        n = len(lst)
        mean = sum(lst) / n
        sorted_lst = sorted(lst)
        median = sorted_lst[n // 2] if n % 2 == 1 else 0.5 * (sorted_lst[n // 2 - 1] + sorted_lst[n // 2])
        variance = sum((x - mean) ** 2 for x in lst) / n
        std = math.sqrt(variance)
        return {"count": n, "mean": mean, "median": median, "min": sorted_lst[0], "max": sorted_lst[-1], "std": std}

    per_class_summary = {}
    for cls, data in per_class.items():
        per_class_summary[cls] = {
            "area_stats": summarize(data["areas"]),
            "relative_stats": summarize(data["relative"]) if data["relative"] else {}
        }

    overall = summarize(all_areas)
    return {"per_class": per_class_summary, "overall": overall}
