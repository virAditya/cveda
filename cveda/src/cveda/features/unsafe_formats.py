"""
unsafe_formats

Detect images using uncommon or problematic formats for training pipelines
like CMYK mode or progressive JPEG.

Approach
--------
- open image via PIL and inspect mode
- for JPEGs attempt to detect progressive attribute via info dict

Return
------
{
 "feature":"unsafe_formats",
 "problems": [ {"file":..., "issue":"CMYK" }, ... ]
}
"""

from typing import Dict, Any, List
from PIL import Image
import os

def run_unsafe_formats(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 1000))
    problems = []
    scanned = 0
    for fname, rec in (index or {}).items():
        if scanned >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                scanned += 1
                mode = im.mode
                info = im.info or {}
                if mode.upper() == "CMYK":
                    problems.append({"file": fname, "issue": "CMYK mode"})
                if info.get("progression") or info.get("progressive"):
                    problems.append({"file": fname, "issue": "progressive_jpeg"})
        except Exception:
            problems.append({"file": fname, "issue": "cannot_open"})
    return {"feature": "unsafe_formats", "scanned": scanned, "problems": problems, "status": "ok"}
