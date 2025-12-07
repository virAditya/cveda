"""
Polygon and mask based annotation helpers.

Compute mask area and bounding box from COCO segmentation polygons or RLE when present.

This helper is conservative it computes tight bbox from polygon coordinates and
gives a mask area estimate. It does not require heavy dependencies.
"""

from typing import Dict, Any, List, Optional
import math


def polygon_bbox_and_area(polygon: List[float]) -> Optional[Dict[str, float]]:
    """
    Given a flat list [x1 y1 x2 y2 ...] compute bbox and polygon area via shoelace.
    Returns {"bbox": [xmin ymin xmax ymax], "area": area} or None on bad input.
    """
    if not polygon or len(polygon) < 6:
        return None
    xs = polygon[0::2]
    ys = polygon[1::2]
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)
    # shoelace formula
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    area = abs(area) / 2.0
    return {"bbox": [xmin, ymin, xmax, ymax], "area": area}
