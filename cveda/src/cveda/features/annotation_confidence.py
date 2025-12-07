"""
annotation_confidence

Heuristic proxies for how "confident" annotations look, based on geometry
consistency. This is not model confidence. Use as a quick first-pass to find
classes or images with unusual annotation geometry.

Exports
-------
run_annotation_confidence(index: dict, config: dict=None) -> dict
"""

from typing import Dict, Any, List, Tuple
import math
from collections import defaultdict

def _bbox_area(bbox: List[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, (x1 - x0) * (y1 - y0))

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def run_annotation_confidence(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute geometry-based annotation "confidence" proxies.

    Heuristics used:
    - per-class bbox area variance normalized by mean (high variance may indicate mixed annotation granularity)
    - per-image annotation count extremes (images with extremely many boxes may be noisy)
    - fraction of boxes with very extreme aspect ratios (e.g., > 10 or < 0.1)
    - returns short samples for manual inspection

    Parameters
    ----------
    index : dict
        canonical index mapping filenames -> record.
        record example:
        {
          "file.jpg": {
              "file_name": "file.jpg",
              "abs_path": "/abs/path/file.jpg",
              "width": 1024,
              "height": 768,
              "annotations": [
                  {"class": "car", "bbox": [x0,y0,x1,y1], "raw": {...}}
              ],
              "meta": {...}
          }, ...
        }

    config : dict, optional
        - aspect_ratio_threshold: float default 8.0 (boxes with aspect ratio > this are extreme)
        - small_area_threshold: float default 1e-4 (fraction of image area considered very small)
        - sample_limit: int default 10 (number of sample filenames to return for issues)

    Returns
    -------
    dict
        {
          "feature": "annotation_confidence",
          "n_images": int,
          "per_class_stats": { class: {"count": int, "mean_area_frac": float, "area_std_frac": float} },
          "global": { "extreme_aspect_fraction": float, "high_box_count_images": int },
          "samples": { "extreme_aspect_examples": [...], "high_box_images": [...] },
          "status": "ok"
        }

    Complexity
    ----------
    O(N + A) where N = number of images and A = total annotations.
    Memory O(C) for class stats.

    Notes
    -----
    This function provides quick heuristics. For real "confidence" you'd combine
    annotator metadata, duplicate-image agreement, and model predictions.
    """
    cfg = config or {}
    ar_thresh = float(cfg.get("aspect_ratio_threshold", 8.0))
    small_area_frac = float(cfg.get("small_area_threshold", 1e-4))
    sample_limit = int(cfg.get("sample_limit", 10))

    n_images = 0
    class_areas = defaultdict(list)
    extreme_aspect_examples = []
    high_box_images = []

    for fname, rec in (index or {}).items():
        n_images += 1
        anns = rec.get("annotations", []) or []
        width = _safe_float(rec.get("width", 0))
        height = _safe_float(rec.get("height", 0))
        img_area = max(1.0, width * height)
        if len(anns) > cfg.get("high_box_threshold", 200):
            high_box_images.append({"file": fname, "n_boxes": len(anns)})
        for ann in anns:
            bbox = ann.get("bbox", [0,0,0,0])
            try:
                x0, y0, x1, y1 = map(float, bbox)
            except Exception:
                continue
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            if w <= 0 or h <= 0:
                continue
            area = w * h
            cls = str(ann.get("class", ""))
            class_areas[cls].append(area / img_area)
            # aspect ratio max(width/height, height/width)
            ar = (w / h) if h > 0 else float('inf')
            ar = ar if ar >= 1 else 1.0 / ar
            if ar >= ar_thresh:
                if len(extreme_aspect_examples) < sample_limit:
                    extreme_aspect_examples.append({"file": fname, "class": cls, "bbox": [x0,y0,x1,y1], "aspect_ratio": ar})

    per_class_stats = {}
    for cls, areas in class_areas.items():
        if not areas:
            continue
        mean = sum(areas) / len(areas)
        variance = sum((a - mean) ** 2 for a in areas) / len(areas)
        std = math.sqrt(variance)
        per_class_stats[cls] = {
            "count": len(areas),
            "mean_area_frac": mean,
            "area_std_frac": std
        }

    total_anns = sum(v["count"] for v in per_class_stats.values()) if per_class_stats else 0
    extreme_aspect_fraction = len(extreme_aspect_examples) / max(1, total_anns)

    result = {
        "feature": "annotation_confidence",
        "n_images": n_images,
        "per_class_stats": per_class_stats,
        "global": {
            "extreme_aspect_fraction": extreme_aspect_fraction,
            "high_box_image_count": len(high_box_images)
        },
        "samples": {
            "extreme_aspect_examples": extreme_aspect_examples[:sample_limit],
            "high_box_images": high_box_images[:sample_limit]
        },
        "status": "ok"
    }
    return result
