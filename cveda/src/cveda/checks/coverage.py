"""
Annotation coverage audit.

Compute fraction of image area covered by annotations and flag images with very low coverage.
"""

from typing import Dict, Any, List
from ..annotations import bbox_area


def annotation_coverage(index: Dict[str, Any], low_threshold: float = 0.01) -> Dict[str, Any]:
    """
    Returns dict with:
    - per_image coverage fraction
    - low_coverage list of images below threshold
    - stats aggregates
    """
    per = {}
    low = []
    areas = []
    for fname, rec in index.items():
        w = rec.get("width")
        h = rec.get("height")
        if not w or not h:
            per[fname] = {"coverage": None}
            continue
        img_area = float(w) * float(h)
        total = 0.0
        for ann in rec.get("annotations", []):
            total += bbox_area(ann["bbox"])
        frac = total / img_area if img_area > 0 else 0.0
        per[fname] = {"coverage": frac, "n_ann": len(rec.get("annotations", []))}
        areas.append(frac)
        if frac <= low_threshold:
            low.append({"file_name": fname, "coverage": frac})
    # basic aggregates
    agg = {"mean": float(sum(areas) / len(areas)) if areas else 0.0, "min": float(min(areas)) if areas else 0.0, "max": float(max(areas)) if areas else 0.0}
    return {"per_image": per, "low_coverage": low, "aggregates": agg}
