"""
pose_diversity

Coarse pose estimation heuristic using elongated axis of bounding box crop.
This is not a keypoint model. It returns whether object appears horizontal vertical or square.

Steps
-----
- crop bbox region
- compute edge orientation via Sobel gradient energy dominating axis

Return
------
{
 "feature":"pose_diversity",
 "counts": {"horizontal": n, "vertical": n, "square": n},
 "examples": { ... }
}
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
import os
from collections import defaultdict

def run_pose_diversity(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 500))
    counts = defaultdict(int)
    examples = defaultdict(list)
    processed = 0

    for fname, rec in (index or {}).items():
        if processed >= sample_limit:
            break
        for ann in (rec.get("annotations", []) or []):
            bbox = ann.get("bbox", None)
            if not bbox:
                continue
            path = rec.get("abs_path")
            if not path or not os.path.exists(path):
                continue
            try:
                with Image.open(path) as im:
                    im = im.convert("L")
                    w,h = im.size
                    x0,y0,x1,y1 = map(int, bbox)
                    x0 = max(0, min(x0, w-1)); x1 = max(1, min(x1, w))
                    y0 = max(0, min(y0, h-1)); y1 = max(1, min(y1, h))
                    crop = im.crop((x0,y0,x1,y1)).resize((64,64))
                    arr = np.asarray(crop).astype(float)
                    gx = np.abs(np.diff(arr, axis=1)).mean()
                    gy = np.abs(np.diff(arr, axis=0)).mean()
                    if gx > 1.2 * gy:
                        orient = "vertical_edges_dom"  # vertical edges imply horizontal features orientation
                    elif gy > 1.2 * gx:
                        orient = "horizontal_edges_dom"
                    else:
                        orient = "square_like"
                    counts[orient] += 1
                    cls = str(ann.get("class",""))
                    if len(examples[orient]) < 5:
                        examples[orient].append({"file": fname, "class": cls})
                processed += 1
            except Exception:
                continue

    return {"feature": "pose_diversity", "processed": processed, "counts": dict(counts), "examples": dict(examples), "status": "ok"}
