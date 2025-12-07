"""
background_relevance

Compute per-image annotated area fraction and identify low-information images
where foreground annotation area is tiny compared to image area.

Outputs:
- per-image annotation_area_fraction (aggregated stats)
- list of low-coverage images for manual inspection
"""

from typing import Dict, Any, List
from collections import defaultdict

def run_background_relevance(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Parameters
    ----------
    config:
      - low_coverage_threshold: float between 0 and 1 (default 0.01) images below this are low-coverage
      - sample_limit: int default 20

    Returns
    -------
    dict with:
      - feature
      - n_images
      - mean_annotation_fraction
      - median_annotation_fraction
      - low_coverage_images: list of samples with their coverage
    """
    cfg = config or {}
    low_thresh = float(cfg.get("low_coverage_threshold", 0.01))
    sample_limit = int(cfg.get("sample_limit", 20))

    fractions = []
    low_imgs = []
    n_images = 0

    for fname, rec in (index or {}).items():
        n_images += 1
        width = rec.get("width") or rec.get("meta", {}).get("width", 0) or 0
        height = rec.get("height") or rec.get("meta", {}).get("height", 0) or 0
        try:
            img_area = float(width) * float(height)
            if img_area <= 0:
                img_area = 1.0
        except Exception:
            img_area = 1.0
        anns = rec.get("annotations", []) or []
        total_ann_area = 0.0
        for ann in anns:
            bbox = ann.get("bbox", [0,0,0,0])
            try:
                x0,y0,x1,y1 = map(float, bbox)
                w = max(0.0, x1-x0)
                h = max(0.0, y1-y0)
                total_ann_area += w*h
            except Exception:
                continue
        frac = total_ann_area / img_area
        fractions.append(frac)
        if frac <= low_thresh:
            if len(low_imgs) < sample_limit:
                low_imgs.append({"file": fname, "coverage": frac})

    # basic stats
    fractions_sorted = sorted(fractions)
    mean = sum(fractions)/max(1, len(fractions))
    median = fractions_sorted[len(fractions_sorted)//2] if fractions_sorted else 0.0

    return {
        "feature": "background_relevance",
        "n_images": n_images,
        "mean_annotation_fraction": mean,
        "median_annotation_fraction": median,
        "low_coverage_count": sum(1 for f in fractions if f <= low_thresh),
        "low_coverage_examples": low_imgs,
        "status": "ok"
    }
