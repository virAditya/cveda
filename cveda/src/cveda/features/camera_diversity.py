"""
camera_diversity

Extract basic camera model diversity using PIL EXIF tags when available.
If EXIF Model not present the module returns counts based on available metadata.

Notes
-----
PIL EXIF tag access varies by image format and camera. This module uses common keys
and normalizes strings for grouping.
"""

from typing import Dict, Any
from collections import Counter
from PIL import Image
import os

def run_camera_diversity(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 500))
    counts = Counter()
    examples = {}
    scanned = 0

    for fname, rec in (index or {}).items():
        if scanned >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                exif = im._getexif() or {}
                # EXIF tag 272 is Model in many images; fallback to raw meta
                model = None
                # safe access common keys
                if isinstance(exif, dict):
                    for k,v in exif.items():
                        # some PIL builds return tag ids others names
                        try:
                            if str(k).lower() == 'model' or k==272:
                                model = str(v)
                                break
                        except Exception:
                            continue
                if not model:
                    model = rec.get("meta", {}).get("camera_model") or rec.get("meta", {}).get("model")
                model = (str(model).strip() if model else "unknown")
                counts[model] += 1
                if model not in examples:
                    examples[model] = fname
                scanned += 1
        except Exception:
            continue

    return {"feature": "camera_diversity", "scanned": scanned, "camera_counts": dict(counts), "examples": examples, "status": "ok"}
