"""
Shared metrics utilities such as iou computation.

Functions are designed to be small deterministic building blocks that other modules reuse.
"""

from typing import List


def iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.

    Each box is expected as [xmin ymin xmax ymax]. Coordinates can be floats.
    Returns IoU float in [0 1] or 0 for degenerate boxes.

    Behavior notes
    - If union is zero returns 0.0
    - Negative widths or heights are clipped to zero
    """
    ax0, ay0, ax1, ay1 = boxA
    bx0, by0, bx1, by1 = boxB
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    areaA = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    areaB = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = areaA + areaB - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union
