"""
Canonical data model and sanitization helpers for CVEDA.

This module defines:
- canonical image record structure used by all features
- validators and converters for annotation types
- sanitizer that ensures JSON friendly outputs
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np

# Canonical annotation types allowed
ANN_TYPES = ("bbox", "polygon", "mask", "keypoints", "seg_class", "rle")

def make_annotation_bbox(class_id: int, class_name: str, x_min: int, y_min: int, x_max: int, y_max: int, ann_id: Optional[str]=None, score: Optional[float]=None, attributes: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    """
    Create and return a canonical bbox annotation dict.

    Coordinates are expected in pixel space, integer values.
    x_min, y_min are top left corners.
    x_max, y_max are bottom right corners.
    This function will sanitize coordinates to ensure x_min <= x_max and y_min <= y_max
    and compute area.

    Parameters
    ----------
    class_id
        numeric class id
    class_name
        human readable class name
    x_min, y_min, x_max, y_max
        pixel coordinates
    ann_id
        optional unique annotation id
    score
        optional confidence score
    attributes
        optional dict for arbitrary attributes

    Returns
    -------
    dict
        canonical annotation dictionary
    """
    # defensive rounding and ordering
    x_min_i = int(round(x_min))
    y_min_i = int(round(y_min))
    x_max_i = int(round(x_max))
    y_max_i = int(round(y_max))
    if x_max_i < x_min_i:
        x_min_i, x_max_i = x_max_i, x_min_i
    if y_max_i < y_min_i:
        y_min_i, y_max_i = y_max_i, y_min_i
    w = max(0, x_max_i - x_min_i)
    h = max(0, y_max_i - y_min_i)
    area = int(w * h)
    return {
        "ann_id": ann_id or f"bbox_{x_min_i}_{y_min_i}_{x_max_i}_{y_max_i}",
        "type": "bbox",
        "class_id": int(class_id) if class_id is not None else None,
        "class_name": class_name,
        "bbox": [x_min_i, y_min_i, x_max_i, y_max_i],
        "area": area,
        "score": float(score) if score is not None else None,
        "attributes": attributes or {}
    }

def make_annotation_polygon(class_id: int, class_name: str, polygon: List[Tuple[float,float]], ann_id: Optional[str]=None, attributes: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    """
    Create canonical polygon annotation.

    polygon is a sequence of (x, y) points in pixel coordinates.
    This will flatten to [x1, y1, x2, y2, ...] for storage.
    Area is approximated using the shoelace formula.
    """
    pts = [(float(x), float(y)) for x, y in polygon]
    flat = [float(v) for p in pts for v in p]
    # compute polygon area via shoelace
    area = 0.0
    if len(pts) >= 3:
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i+1) % len(pts)]
            area += x1*y2 - x2*y1
        area = abs(area) / 2.0
    return {
        "ann_id": ann_id or f"poly_{len(flat)}",
        "type": "polygon",
        "class_id": int(class_id) if class_id is not None else None,
        "class_name": class_name,
        "polygon": flat,
        "area": float(area),
        "attributes": attributes or {}
    }

def make_annotation_mask(class_id: int, class_name: str, mask_path: str, ann_id: Optional[str]=None, attributes: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    """
    Create canonical mask annotation. We store the path to the mask image or RLE string.
    Features may decode mask lazily.
    """
    return {
        "ann_id": ann_id or f"mask_{Path(mask_path).name}",
        "type": "mask",
        "class_id": int(class_id) if class_id is not None else None,
        "class_name": class_name,
        "mask": mask_path,
        "area": None,
        "attributes": attributes or {}
    }

def make_annotation_keypoints(class_id: int, class_name: str, keypoints: List[Tuple[float,float,int]], ann_id: Optional[str]=None, attributes: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    """
    keypoints is a list of tuples (x, y, v) where v is visibility flag often 0,1,2.
    """
    flat = [float(v) for k in keypoints for v in k]
    return {
        "ann_id": ann_id or f"kp_{len(flat)}",
        "type": "keypoints",
        "class_id": int(class_id) if class_id is not None else None,
        "class_name": class_name,
        "keypoints": flat,
        "attributes": attributes or {}
    }

def make_image_record(image_id: str, abs_path: str, width: Optional[int]=None, height: Optional[int]=None, thumbnail: Optional[str]=None, mean_rgb: Optional[List[float]]=None, annotations: Optional[List[Dict[str,Any]]]=None, metadata: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    """
    Create a canonical image record. All features expect this shape.

    Required minimal keys are id and abs_path. Missing optional keys set to None or empty.
    """
    rec = {
        "id": image_id,
        "abs_path": str(abs_path),
        "file_name": str(Path(abs_path).name),
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "thumbnail": str(thumbnail) if thumbnail else None,
        "mean_rgb": [float(x) for x in mean_rgb] if mean_rgb else None,
        "annotations": annotations or [],
        "n_annotations": len(annotations) if annotations else 0,
        "annotation_area_frac": None,
        "metadata": metadata or {}
    }
    return rec

def sanitize_for_json(obj: Any) -> Any:
    """
    Convert numpy types, Path, and other non json friendly items into JSON serializable counterparts.

    - numpy scalars become Python scalars
    - numpy arrays become lists
    - Path become strings
    - bytes are described by length
    - fallback uses str(obj)
    """
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [sanitize_for_json(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (np.ndarray, )):
            try:
                return obj.tolist()
            except Exception:
                return [sanitize_for_json(x) for x in obj]
        if isinstance(obj, bytes):
            return {"__bytes_len": len(obj)}
        # last resort
        return str(obj)
    except Exception:
        return str(obj)
