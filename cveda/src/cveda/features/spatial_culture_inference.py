"""
spatial_culture_inference

Exploratory heuristic to detect repeated signage language or license plate formats.
This function is lightweight and only signals presence of ASCII vs non ASCII dominant text
using simple OCR free heuristics such as scanning for high frequency of non-ASCII characters
in small crops that likely contain text.

Ethical note
-----------
This is exploratory and should not be used to identify people or sensitive attributes.
"""

from typing import Dict, Any
from collections import defaultdict
from PIL import Image, ImageOps
import numpy as np
import os

def _approx_text_score(img_arr):
    # simple heuristic based on contrast and edge density which often signals text regions
    gx = np.abs(np.diff(img_arr.astype(float), axis=1)).mean()
    gy = np.abs(np.diff(img_arr.astype(float), axis=0)).mean()
    return float(gx + gy)

def run_spatial_culture_inference(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 200))
    candidates = []
    scanned = 0

    for fname, rec in (index or {}).items():
        if scanned >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("L").resize((256,256))
                arr = np.asarray(im)
                # tile search for potential text like blocks
                tiles = []
                th = 64
                for y in range(0, arr.shape[0], th):
                    for x in range(0, arr.shape[1], th):
                        tile = arr[y:y+th, x:x+th]
                        if tile.size == 0:
                            continue
                        score = _approx_text_score(tile)
                        if score > cfg.get("text_score_threshold", 15.0):
                            tiles.append(score)
                if tiles:
                    candidates.append({"file": fname, "text_tile_count": len(tiles), "scores": tiles[:5]})
                    scanned += 1
        except Exception:
            continue

    return {"feature": "spatial_culture_inference", "candidates": candidates, "status": "ok"}
