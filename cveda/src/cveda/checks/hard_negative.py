"""
Hard negative suggestions.

Find images that likely serve as hard negatives such as images with many small boxes
or many boxes that are extremely small relative to image area.

This is a heuristic helper intended as guidance for building a hard negative mining set.
It is defensive and returns an empty list on error.
"""

from typing import Dict, Any, List


def find_hard_negative_candidates(index: Dict[str, Any], min_boxes: int = 10, small_box_relative: float = 1e-4) -> List[Dict[str, Any]]:
    """
    Return images that may be good hard negative mining candidates.

    Heuristics:
    1. Images with at least min_boxes annotations
    2. Among those, images where the majority of boxes are very small relative to image area

    Parameters
    index dict
        Canonical index mapping relative filenames to records produced by ImageCollectionLoader.
    min_boxes int
        Minimum number of annotation boxes in an image to be considered.
    small_box_relative float
        A box is considered small when its area is <= small_box_relative times the image area.

    Returns
    A list of dictionaries for candidate images, each containing:
      {
        "file_name": relative path in index,
        "n_boxes": number of boxes,
        "small_fraction": fraction of boxes that are small,
        "example_small_boxes": [first up to 5 bbox entries]
      }

    Notes
    The function is intentionally simple, it avoids heavy computation and skips images
    without width or height metadata.
    """
    out: List[Dict[str, Any]] = []
    try:
        for fname, rec in index.items():
            anns = rec.get("annotations", []) or []
            w = rec.get("width")
            h = rec.get("height")
            if not w or not h:
                # cannot reason about relative sizes without image dimensions
                continue
            img_area = float(w) * float(h)
            if len(anns) < min_boxes:
                continue
            small_count = 0
            small_examples = []
            for ann in anns:
                bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
                try:
                    x0, y0, x1, y1 = map(float, bbox)
                except Exception:
                    continue
                area = max(0.0, (x1 - x0) * (y1 - y0))
                if img_area > 0 and (area / img_area) <= small_box_relative:
                    small_count += 1
                    if len(small_examples) < 5:
                        small_examples.append({"bbox": [x0, y0, x1, y1], "area": area})
            if len(anns) > 0:
                fraction = small_count / float(len(anns))
            else:
                fraction = 0.0
            # require that a majority are small to flag it as a candidate
            if fraction >= 0.6:
                out.append({
                    "file_name": fname,
                    "n_boxes": len(anns),
                    "small_fraction": fraction,
                    "example_small_boxes": small_examples
                })
    except Exception:
        # fail safe, return what we gathered so far
        return out
    return out
