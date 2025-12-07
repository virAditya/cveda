"""
Overlap analysis utilities.

Compute pairwise IoU per image and return pairs that exceed thresholds.
The module implements a vectorized IoU calculation and a fallback grid
based bounding box pre filter for performance on many boxes.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise IoU matrix for boxes.

    Parameters
    boxes numpy array shape (N 4) each row xmin ymin xmax ymax

    Returns
    NxN numpy array where entry [i j] is IoU between boxes i and j.
    Diagonal equals IoU of box with itself equals 1.0 for non empty boxes.
    """
    if boxes.size == 0:
        return np.zeros((0, 0), dtype=float)
    x1 = boxes[:, 0:1]
    y1 = boxes[:, 1:2]
    x2 = boxes[:, 2:3]
    y2 = boxes[:, 3:4]

    area = np.clip((x2 - x1) * (y2 - y1), a_min=0.0, a_max=None).reshape(-1)

    # broadcasted intersection coords
    inter_x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    inter_y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    inter_x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    inter_y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    union = area[:, None] + area[None, :] - inter_area
    # avoid division by zero
    iou = np.where(union > 0.0, inter_area / union, 0.0)
    return iou


def find_high_iou_pairs_for_image(boxes: List[List[float]], labels: List[Any], threshold: float = 0.9, max_pairs: int = 200) -> List[Dict[str, Any]]:
    """
    Find pairs of boxes with IoU above threshold for a single image.

    Parameters
    boxes list of [xmin ymin xmax ymax]
    labels list of class tokens parallel to boxes
    threshold float IoU threshold
    max_pairs int maximum number of pairs to return to keep report small

    Returns
    list of dicts with keys:
    - i j indexes
    - iou value
    - class_i class_j
    """
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    iou_mat = iou_matrix(arr)
    N = arr.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            val = float(iou_mat[i, j])
            if val >= threshold:
                pairs.append({"i": int(i), "j": int(j), "iou": val, "class_i": labels[i] if i < len(labels) else None, "class_j": labels[j] if j < len(labels) else None})
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def run_overlap_checks(index: Dict[str, Any], cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run overlap checks across the dataset index.

    cfg optional keys:
    - same_class_threshold default 0.9
    - cross_class_threshold default 0.8
    - max_pairs_to_report default 200
    """
    cfg = cfg or {}
    same_thr = cfg.get("same_class_threshold", 0.9)
    cross_thr = cfg.get("cross_class_threshold", 0.8)
    max_pairs = cfg.get("max_pairs_to_report", 200)

    results = {"same_class_pairs": [], "cross_class_pairs": []}
    for fname, rec in index.items():
        anns = rec.get("annotations", [])
        boxes = [a["bbox"] for a in anns]
        labels = [a.get("class") for a in anns]
        same = find_high_iou_pairs_for_image(boxes, labels, threshold=same_thr, max_pairs=max_pairs)
        cross = find_high_iou_pairs_for_image(boxes, labels, threshold=cross_thr, max_pairs=max_pairs)
        # filter cross to only pairs where classes differ
        cross = [p for p in cross if p["class_i"] != p["class_j"]]
        if same:
            results["same_class_pairs"].append({"file_name": fname, "pairs": same})
        if cross:
            results["cross_class_pairs"].append({"file_name": fname, "pairs": cross})
    return results
