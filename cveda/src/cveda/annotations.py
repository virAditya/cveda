"""
Annotation helpers.

This module contains utilities to convert arbitrary annotation payloads
into the canonical schema and to perform small transformation
operations like clamping and swapping inverted coordinates.
"""

from typing import List, Dict, Any, Tuple


def clamp_bbox_to_image(bbox: List[float], image_w: int, image_h: int) -> List[float]:
    """
    Clamp a bbox to image boundaries.

    Parameters
    bbox list of float
        Expected format xmin ymin xmax ymax
    image_w int image width in pixels
    image_h int image height in pixels

    Returns
    A new bbox list with coordinates clamped to [0 image_w] and [0 image_h]
    """
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0.0, min(xmin, float(image_w)))
    ymin = max(0.0, min(ymin, float(image_h)))
    xmax = max(0.0, min(xmax, float(image_w)))
    ymax = max(0.0, min(ymax, float(image_h)))
    return [xmin, ymin, xmax, ymax]


def is_bbox_inverted(bbox: List[float]) -> bool:
    """
    Check whether the bbox has inverted coordinates.

    Inverted means xmin > xmax or ymin > ymax which indicates a labeling error.

    Returns True when inverted.
    """
    xmin, ymin, xmax, ymax = bbox
    return xmin > xmax or ymin > ymax


def swap_inverted_bbox(bbox: List[float]) -> List[float]:
    """
    If bbox is inverted swap coordinates to make it valid.

    Behavior
    - If xmin > xmax swap them
    - If ymin > ymax swap them

    Returns corrected bbox
    """
    xmin, ymin, xmax, ymax = bbox
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    return [xmin, ymin, xmax, ymax]


def bbox_area(bbox: List[float]) -> float:
    """
    Compute area of a bbox.

    If bbox coordinates are invalid negative area is returned as zero.
    """
    xmin, ymin, xmax, ymax = bbox
    w = max(0.0, xmax - xmin)
    h = max(0.0, ymax - ymin)
    return w * h
