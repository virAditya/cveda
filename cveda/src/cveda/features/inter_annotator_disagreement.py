"""
inter_annotator_disagreement

Compute basic inter-annotator disagreement metrics if annotator identifiers
are available in annotation metadata.

This module expects annotator information in either:
- annotation['raw']['annotator_id']
or
- record['meta']['annotators'] keyed per annotation
If absent, the module returns a note explaining it's unavailable.
"""

from typing import Dict, Any, List
from collections import defaultdict
import math

def _iou(b1, b2):
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    inter = w * h
    a1 = max(0.0, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(0.0, (b2[2]-b2[0])*(b2[3]-b2[1]))
    union = a1 + a2 - inter
    return inter/union if union > 0 else 0.0

def run_inter_annotator_disagreement(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute simple inter-annotator disagreement metrics.

    Behavior
    --------
    - If no annotator ids are found the function returns a note explaining that computation is unavailable.
    - If annotator ids present, for each image we group annotations by annotator and compute pairwise
      IoU distributions between annotators for matched classes (greedy match by highest IoU).
    - Returns per-image average pairwise IoU, and a small set of images with low agreement.

    Parameters
    ----------
    index : dict
    config : dict optional
      - iou_threshold : float, threshold to consider a match (default 0.3)
      - sample_limit : int, how many low-agreement images to return (default 10)

    Returns
    -------
    dict:
      {
        "feature": "inter_annotator_disagreement",
        "available": bool,
        "avg_pairwise_iou": float or None,
        "low_agreement_examples": [ {file, avg_iou} ],
        "status": "ok" or "no_annotators"
      }
    """
    cfg = config or {}
    iou_thresh = float(cfg.get("iou_threshold", 0.3))
    sample_limit = int(cfg.get("sample_limit", 10))

    found_any = False
    per_image_scores = {}
    low_examples = []

    for fname, rec in (index or {}).items():
        anns = rec.get("annotations", []) or []
        # group annotations by annotator id found in ann['raw']['annotator_id']
        groups = defaultdict(list)
        for ann in anns:
            raw = ann.get("raw", {}) or {}
            annotator = raw.get("annotator_id") or raw.get("annotator") or ann.get("annotator")
            if annotator is None:
                annotator = "__no_annotator__"
            groups[annotator].append(ann)
        if len(groups) <= 1:
            continue
        found_any = True
        annotators = list(groups.keys())
        pairwise_iou = []
        # compute pairwise IoU between annotators by greedy matching of boxes with same class
        for i in range(len(annotators)):
            for j in range(i+1, len(annotators)):
                a = groups[annotators[i]]
                b = groups[annotators[j]]
                # for each ann in a, find best matching ann in b of same class
                ious = []
                for ann_a in a:
                    cls_a = str(ann_a.get("class", ""))
                    best = 0.0
                    ba = ann_a.get("bbox", [0,0,0,0])
                    for ann_b in b:
                        if str(ann_b.get("class","")) != cls_a:
                            continue
                        bb = ann_b.get("bbox", [0,0,0,0])
                        try:
                            score = _iou(ba, bb)
                        except Exception:
                            score = 0.0
                        if score > best:
                            best = score
                    ious.append(best)
                # also symmetric: match b->a
                for ann_b in b:
                    cls_b = str(ann_b.get("class",""))
                    best = 0.0
                    bb = ann_b.get("bbox", [0,0,0,0])
                    for ann_a in a:
                        if str(ann_a.get("class","")) != cls_b:
                            continue
                        ba = ann_a.get("bbox", [0,0,0,0])
                        try:
                            score = _iou(ba, bb)
                        except Exception:
                            score = 0.0
                        if score > best:
                            best = score
                    ious.append(best)
                if ious:
                    pairwise_iou.append(sum(ious)/len(ious))
        if pairwise_iou:
            avg = sum(pairwise_iou)/len(pairwise_iou)
        else:
            avg = 0.0
        per_image_scores[fname] = avg
        if avg < cfg.get("low_agreement_threshold", 0.5):
            low_examples.append({"file": fname, "avg_iou": avg})

    if not found_any:
        return {
            "feature": "inter_annotator_disagreement",
            "available": False,
            "message": "No annotator identifiers found in annotations; cannot compute disagreement",
            "status": "no_annotators"
        }
    # aggregate
    avg_pairwise_iou = sum(per_image_scores.values()) / max(1, len(per_image_scores))
    low_examples_sorted = sorted(low_examples, key=lambda x: x["avg_iou"])[:sample_limit]
    return {
        "feature": "inter_annotator_disagreement",
        "available": True,
        "avg_pairwise_iou": avg_pairwise_iou,
        "per_image": {"count": len(per_image_scores), "scores": None},  # omit large data
        "low_agreement_examples": low_examples_sorted,
        "status": "ok"
    }
