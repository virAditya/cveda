"""
Dataset capability inspector.

Provides inspect_dataset which returns which dataset capabilities are present so we
can avoid running irrelevant features.
"""

from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json

def inspect_dataset(root: str) -> Dict[str, Any]:
    """
    Quickly inspect the dataset root and return a capability dictionary.

    Keys include:
    - has_boxes bool
    - has_masks bool
    - has_polygons bool
    - has_keypoints bool
    - has_multiclass_masks bool
    - has_annotations bool
    - format str detected format name
    - sample_image str path for sample
    """
    p = Path(root)
    res = {
        "has_boxes": False,
        "has_masks": False,
        "has_polygons": False,
        "has_keypoints": False,
        "has_multiclass_masks": False,
        "has_annotations": False,
        "format": "unknown",
        "sample_image": None
    }
    if not p.exists():
        return res
    # format detection simple
    files = list(p.glob("*"))
    for f in files:
        if f.name.lower().endswith(".json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                if isinstance(d, dict) and "annotations" in d and "images" in d:
                    res["format"] = "coco"
                    res["has_annotations"] = True
                    # scan annotations for types
                    anns = d.get("annotations", [])
                    for a in anns:
                        if "bbox" in a:
                            res["has_boxes"] = True
                        if "segmentation" in a:
                            res["has_polygons"] = True
                            seg = a.get("segmentation")
                            if isinstance(seg, dict):
                                res["has_masks"] = True
                        if "keypoints" in a:
                            res["has_keypoints"] = True
                    break
            except Exception:
                continue
    # check voc xml
    for f in p.rglob("*.xml"):
        res["format"] = "voc"
        res["has_annotations"] = True
        res["has_boxes"] = True
        break
    # check yolo labels folder
    if (p / "labels").is_dir():
        txts = list((p / "labels").glob("*.txt"))
        if txts:
            res["format"] = "yolo"
            res["has_annotations"] = True
            res["has_boxes"] = True
    # check masks folder
    for cand in ("masks", "masks_png", "segmentation_masks"):
        mp = p / cand
        if mp.exists():
            res["format"] = "mask_folder"
            res["has_masks"] = True
            res["has_annotations"] = True
            # peek a mask to see if multiclass
            for m in mp.glob("*"):
                try:
                    from PIL import Image
                    import numpy as np
                    with Image.open(m) as im:
                        arr = np.asarray(im)
                        if arr.ndim == 2:
                            unique = set(arr.flatten().tolist())
                            if not unique.issubset({0,1,255}):
                                res["has_multiclass_masks"] = True
                                break
                except Exception:
                    continue
            break
    # sample image
    for f in p.rglob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            res["sample_image"] = str(f)
            break
    return res
