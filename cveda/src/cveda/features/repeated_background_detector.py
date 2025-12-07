"""
repeated_background_detector

Detect similar backgrounds by hashing blurred, foreground-masked image regions.
Simpler approach here uses global downsampled average color of background region
estimated by excluding union of bounding boxes.

Steps
-----
- compute bounding box union per image
- mask union and compute mean color of remaining pixels
- cluster exact matches by rounding mean color to small bins
- report clusters with size above threshold

Config
------
- sample_limit default 200 images
- binsize default 8 rounding step per channel
- min_cluster_size default 3
"""

from typing import Dict, Any
from collections import defaultdict
from PIL import Image
import numpy as np
import os

def run_repeated_background_detector(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 200))
    binsize = int(cfg.get("binsize", 8))
    min_cluster_size = int(cfg.get("min_cluster_size", 3))

    clusters = defaultdict(list)
    processed = 0

    for fname, rec in (index or {}).items():
        if processed >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w,h = im.size
                arr = np.asarray(im.resize((max(1,int(w*0.25)), max(1,int(h*0.25)))), dtype=np.uint8)
                mask = np.ones((arr.shape[0], arr.shape[1]), dtype=bool)
                # mask out boxes scaled to downsampled coordinate space
                for ann in (rec.get("annotations", []) or []):
                    try:
                        bx = ann.get("bbox", [0,0,0,0])
                        x0,y0,x1,y1 = map(float, bx)
                        sx = arr.shape[1]/w; sy = arr.shape[0]/h
                        rx0 = int(max(0, round(x0*sx))); ry0 = int(max(0, round(y0*sy)))
                        rx1 = int(min(arr.shape[1], round(x1*sx))); ry1 = int(min(arr.shape[0], round(y1*sy)))
                        mask[ry0:ry1, rx0:rx1] = False
                    except Exception:
                        continue
                bg_pixels = arr[mask]
                if bg_pixels.size == 0:
                    continue
                mean_rgb = tuple(int(bg_pixels[:,i].mean()) for i in range(3))
                # quantize by binsize
                q = tuple(int((v//binsize)*binsize) for v in mean_rgb)
                clusters[q].append({"file": fname, "mean_rgb": mean_rgb})
                processed += 1
        except Exception:
            continue

    repeated = []
    for key, members in clusters.items():
        if len(members) >= min_cluster_size:
            repeated.append({"cluster_key": key, "count": len(members), "examples": members[:10]})

    return {"feature": "repeated_background_detector", "clusters_found": repeated, "status": "ok"}
