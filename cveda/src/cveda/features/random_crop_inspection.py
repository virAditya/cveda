"""
random_crop_inspection

Simulate random crops and compute crop survival statistics for annotations.
This helps evaluate robustness to random cropping augmentations.

Heuristic
---------
For each sampled image, sample several random crops of given size fraction.
For each crop, count annotations that remain with IoU >= threshold with box intersection.
Return fraction of crops that remove all annotations and per-class survival rates.

Config
------
- sample_images int default 200
- crops_per_image int default 5
- crop_frac float default 0.6 portion of shorter side
- iou_threshold float default 0.3
- sample_limit int default 20 returned examples

Return
------
{
  "feature": "random_crop_inspection",
  "n_images_sampled": int,
  "mean_crop_kills_fraction": float,
  "per_class_survival": {class: rate},
  "examples": [...]
}
"""

from typing import Dict, Any, List
import random
import math
import os
from PIL import Image
from collections import defaultdict

def _iou(b1, b2):
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    w = max(0.0, x1-x0); h = max(0.0, y1-y0)
    inter = w*h
    a1 = max(0.0, (b1[2]-b1[0])*(b1[3]-b1[1]))
    a2 = max(0.0, (b2[2]-b2[0])*(b2[3]-b2[1]))
    union = a1 + a2 - inter
    return inter/union if union>0 else 0.0

def run_random_crop_inspection(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_images = int(cfg.get("sample_images", 200))
    crops_per_image = int(cfg.get("crops_per_image", 5))
    crop_frac = float(cfg.get("crop_frac", 0.6))
    iou_threshold = float(cfg.get("iou_threshold", 0.3))
    sample_limit = int(cfg.get("sample_limit", 20))

    images_done = 0
    kill_counts = 0
    per_class_survive = defaultdict(int)
    per_class_total = defaultdict(int)
    examples = []

    for fname, rec in (index or {}).items():
        if images_done >= sample_images:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                w,h = im.size
        except Exception:
            continue
        anns = rec.get("annotations", []) or []
        if not anns:
            images_done += 1
            continue
        images_done += 1
        kills_for_image = 0
        for _ in range(crops_per_image):
            # sample crop size
            crop_w = int(min(w, max(1, crop_frac * min(w,h))))
            crop_h = int(min(h, max(1, crop_frac * min(w,h))))
            x0 = random.randint(0, max(0, w - crop_w))
            y0 = random.randint(0, max(0, h - crop_h))
            crop_box = [x0, y0, x0 + crop_w, y0 + crop_h]
            surviving = 0
            for ann in anns:
                try:
                    bx = list(map(float, ann.get("bbox", [0,0,0,0])))
                except Exception:
                    continue
                # compute intersection with crop then IoU of cropped box vs original
                inter_x0 = max(bx[0], crop_box[0])
                inter_y0 = max(bx[1], crop_box[1])
                inter_x1 = min(bx[2], crop_box[2])
                inter_y1 = min(bx[3], crop_box[3])
                if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
                    continue
                inter_box = [inter_x0, inter_y0, inter_x1, inter_y1]
                iou = _iou(bx, inter_box)
                if iou >= iou_threshold:
                    surviving += 1
            if surviving == 0:
                kills_for_image += 1
        kill_fraction_for_image = kills_for_image / max(1, crops_per_image)
        kill_counts += kill_fraction_for_image
        # update per-class survival approximations
        for ann in anns:
            cls = str(ann.get("class",""))
            per_class_total[cls] += 1
            # naive increment if any crop survives at least once
            if kill_fraction_for_image < 1.0:
                per_class_survive[cls] += 1
        if len(examples) < sample_limit:
            examples.append({"file": fname, "kill_fraction": kill_fraction_for_image})

    mean_kill_fraction = kill_counts / max(1, images_done) if images_done else 0.0
    per_class_survival_rate = {c: (per_class_survive[c] / per_class_total[c]) if per_class_total[c]>0 else 0.0 for c in per_class_total}

    return {
        "feature": "random_crop_inspection",
        "n_images_sampled": images_done,
        "mean_crop_kills_fraction": mean_kill_fraction,
        "per_class_survival": per_class_survival_rate,
        "examples": examples,
        "status": "ok"
    }
