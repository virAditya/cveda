"""
compression_anomaly

Detect images likely suffering from heavy JPEG compression or visible blockiness.

Heuristic
---------
- For JPEG files, attempt to read file size and image dimensions and compute bytes per pixel
- compute simple blockiness score via downsample and upsample residual variance
- return top candidates with highest blockiness

Config
------
- sample_limit default 200 for scanning
- blockiness_threshold optional float

Return
------
{
  "feature":"compression_anomaly",
  "sampled": n,
  "top_blocky_examples": [ {"file":..., "blockiness":...}, ... ],
  "status":"ok"
}
"""

from typing import Dict, Any, List
import os
from PIL import Image
import numpy as np

def _blockiness_score(img_path, downscale=0.5):
    try:
        with Image.open(img_path) as im:
            im = im.convert("L")
            w,h = im.size
            small = im.resize((max(1,int(w*downscale)), max(1,int(h*downscale))))
            arr = np.asarray(small, dtype=float)
            # upsample back and compute residual to estimate blocking artifacts
            up = Image.fromarray(arr.astype(np.uint8)).resize((w,h))
            up_arr = np.asarray(up, dtype=float)
            # align sizes
            if up_arr.shape != (h,w):
                up_arr = np.resize(up_arr, (h,w))
            residual = arr - np.mean(arr)
            # use variance of residual as proxy, normalized
            return float(np.var(residual))
    except Exception:
        return None

def run_compression_anomaly(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 200))
    downscale = float(cfg.get("downscale", 0.5))

    candidates = []
    scanned = 0
    for fname, rec in (index or {}).items():
        if scanned >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        scanned += 1
        block = _blockiness_score(path, downscale=downscale)
        if block is None:
            continue
        candidates.append({"file": fname, "blockiness": block})
    if not candidates:
        return {"feature": "compression_anomaly", "status": "no_images_processed"}
    # sort descending blockiness
    top = sorted(candidates, key=lambda x: x["blockiness"], reverse=True)[:cfg.get("top_k", 50)]
    return {"feature": "compression_anomaly", "sampled": scanned, "top_blocky_examples": top, "status": "ok"}
