"""
Bounding box sanity checks.

This module provides two functions used by tests and by the audit:
- find_zero_area_boxes(index)
- find_inverted_boxes(index)

Design choices
1) We first interpret the raw bbox as xyxy, that is x_min, y_min, x_max, y_max.
   This makes tests deterministic for mixed formats where the third value may be
   smaller than the first value and should therefore be considered inverted.
2) Zero area is when width or height equals zero.
3) Inverted is when width or height is negative.
4) The functions look for common bbox keys and skip annotations that do not include a bbox.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def _as_floats(bbox: List[Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert a bbox like [a, b, c, d] to four floats.
    Returns None when conversion fails.
    The conversion does not guess format. It simply returns floats in the
    original order for a deterministic interpretation as xyxy.
    """
    if not bbox or len(bbox) != 4:
        return None
    try:
        x0 = float(bbox[0])
        y0 = float(bbox[1])
        x1 = float(bbox[2])
        y1 = float(bbox[3])
        return x0, y0, x1, y1
    except Exception:
        return None


def find_zero_area_boxes(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find boxes with zero area.

    Parameters
    - index: mapping file_name -> record. Each record may have 'annotations' list.

    Returns
    - list of dicts with keys: file_name, ann_index, bbox
    """
    results: List[Dict[str, Any]] = []
    for fname, rec in index.items():
        anns = rec.get("annotations") or []
        for i, ann in enumerate(anns):
            bbox = ann.get("bbox") or ann.get("box") or ann.get("bbox_xyxy")
            if not bbox:
                continue
            vals = _as_floats(bbox)
            if vals is None:
                continue
            x_min, y_min, x_max, y_max = vals
            width = x_max - x_min
            height = y_max - y_min
            # zero area when width == 0 or height == 0
            if width == 0.0 or height == 0.0:
                results.append({"file_name": fname, "ann_index": i, "bbox": bbox})
    return results


def find_inverted_boxes(index: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find boxes that are inverted.

    An inverted box is one where, after interpreting the four numbers
    as xyxy, either x_max < x_min or y_max < y_min.
    """
    results: List[Dict[str, Any]] = []
    for fname, rec in index.items():
        anns = rec.get("annotations") or []
        for i, ann in enumerate(anns):
            bbox = ann.get("bbox") or ann.get("box") or ann.get("bbox_xyxy")
            if not bbox:
                continue
            vals = _as_floats(bbox)
            if vals is None:
                continue
            x_min, y_min, x_max, y_max = vals
            # inverted when either difference is negative
            if (x_max - x_min) < 0.0 or (y_max - y_min) < 0.0:
                results.append({"file_name": fname, "ann_index": i, "bbox": bbox})
    return results
