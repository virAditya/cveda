"""
Dataset format detection and ingestion.

Provides robust loaders for:
- COCO JSON annotations with polygons, boxes and RLE
- YOLO TXT label folders
- Mask folder patterns where one mask image corresponds to each image
- Mixed fallback for per image JSON or CSV

All loaders convert to the canonical schema defined in src/cveda/schema.py
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import logging
import math

from PIL import Image
import numpy as np

from .schema import make_image_record, make_annotation_bbox, make_annotation_polygon, make_annotation_mask, make_annotation_keypoints, sanitize_for_json

# try optional import for RLE decoding
try:
    from pycocotools import mask as coco_mask
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False

logger = logging.getLogger(__name__)

def detect_dataset_format(root: str) -> str:
    """
    Heuristic detection of dataset format.

    Returns one of: "coco", "voc", "yolo", "mask_folder", "mixed", "unknown"

    Behavior
    - look for coco style json files
    - look for voc xml files
    - look for yolo label txt files inside labels folder
    - look for separate masks folder
    - otherwise mixed or unknown
    """
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"{root} does not exist")
    # COCO style check
    for cand in ("annotations.json", "instances.json", "coco.json", "annotations/instances_train.json"):
        if (p / cand).exists():
            return "coco"
    # VOC
    xmls = list(p.rglob("*.xml"))
    if xmls:
        return "voc"
    # YOLO style: labels folder with *.txt
    if (p / "labels").is_dir():
        txts = list((p / "labels").glob("*.txt"))
        if txts:
            return "yolo"
    # mask folder
    for m in ("masks", "masks_png", "segmentation_masks"):
        if (p / m).exists():
            return "mask_folder"
    # mixed heuristics
    jsons = list(p.glob("*.json"))
    if jsons:
        # if contains images array maybe coco
        for j in jsons:
            try:
                text = j.read_text(encoding="utf-8")
                data = json.loads(text)
                if isinstance(data, dict) and ("images" in data and "annotations" in data):
                    return "coco"
            except Exception:
                continue
    return "unknown"

# -----------------------------
# COCO loader
# -----------------------------
def _safe_image_size(image_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Return width and height for an image path or None None on failure.
    """
    try:
        with Image.open(image_path) as im:
            w, h = im.size
            return int(w), int(h)
    except Exception:
        return None, None

def parse_coco(coco_json: str, images_root: Optional[str]=None) -> Dict[str,Any]:
    """
    Parse COCO JSON and return canonical index mapping image file name to record.

    Parameters
    ----------
    coco_json
        path to coco style JSON
    images_root
        optional images root used to resolve image relative paths. When None the JSON image path is used directly.

    Returns
    -------
    dict
        mapping image file_name to canonical image record
    """
    path = Path(coco_json)
    if not path.exists():
        raise FileNotFoundError(coco_json)
    raw = json.loads(path.read_text(encoding="utf-8"))
    images = raw.get("images", [])
    annotations = raw.get("annotations", [])
    categories = raw.get("categories", [])
    # build category id to name map
    cat_map = {int(cat.get("id")): cat.get("name") for cat in categories}
    # build image id map
    img_map = {}
    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name") or img.get("file")
        if not file_name:
            continue
        full = Path(images_root) / file_name if images_root else Path(file_name)
        w = img.get("width")
        h = img.get("height")
        if w is None or h is None:
            try:
                w, h = _safe_image_size(full)
            except Exception:
                w, h = None, None
        img_map[int(img_id)] = {
            "file_name": str(file_name),
            "abs_path": str(full),
            "width": int(w) if w else None,
            "height": int(h) if h else None,
            "annotations": []
        }
    # attach annotations
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id not in img_map:
            # orphan annotation record, skip but log
            logger.debug("Orphan annotation for image id %s", img_id)
            continue
        rec = img_map[img_id]
        ann_type = None
        if "bbox" in ann:
            x, y, w, h = ann.get("bbox", [0,0,0,0])
            # convert to pixel x_min y_min x_max y_max
            xyxy = [x, y, x + w, y + h]
            cat = ann.get("category_id")
            class_name = cat_map.get(int(cat)) if cat is not None else ann.get("category_name") or "unknown"
            a = make_annotation_bbox(class_id=cat, class_name=class_name, x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3], ann_id=str(ann.get("id")), score=ann.get("score"))
            rec["annotations"].append(a)
        if "segmentation" in ann and ann.get("segmentation"):
            seg = ann.get("segmentation")
            cat = ann.get("category_id")
            class_name = cat_map.get(int(cat)) if cat is not None else "unknown"
            # segmentation may be polygon list or RLE dict
            if isinstance(seg, list):
                # list of polygons possibly multiple. take each polygon separate
                for poly in seg:
                    if not poly:
                        continue
                    # poly is [x1, y1, x2, y2 ...]
                    try:
                        # pair into tuples
                        pairs = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                        p = make_annotation_polygon(class_id=cat, class_name=class_name, polygon=pairs, ann_id=str(ann.get("id")))
                        rec["annotations"].append(p)
                    except Exception:
                        logger.debug("Bad polygon in coco ann", exc_info=True)
            elif isinstance(seg, dict):
                # RLE. store as rle string placeholder. features may decode later
                if HAS_PYCOCO:
                    try:
                        mask = coco_mask.decode(seg)
                        # compute bbox from mask
                        ys, xs = np.where(mask)
                        if ys.size and xs.size:
                            y_min = int(ys.min()); y_max = int(ys.max())
                            x_min = int(xs.min()); x_max = int(xs.max())
                            b = make_annotation_bbox(class_id=cat, class_name=class_name, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, ann_id=str(ann.get("id")))
                            rec["annotations"].append(b)
                        # store RLE string as mask pointer
                        rec["annotations"].append(make_annotation_mask(class_id=cat, class_name=class_name, mask_path=json.dumps(seg), ann_id=str(ann.get("id"))))
                    except Exception:
                        logger.debug("Failed to decode RLE", exc_info=True)
                else:
                    # store RLE metadata for later decode by features
                    rec["annotations"].append(make_annotation_mask(class_id=cat, class_name=class_name, mask_path=json.dumps(seg), ann_id=str(ann.get("id"))))
        if "keypoints" in ann and ann.get("keypoints"):
            kps = ann.get("keypoints", [])
            # COCO keypoints are flat [x1,y1,v1,...]
            pts = []
            for i in range(0, len(kps), 3):
                pts.append((kps[i], kps[i+1], kps[i+2]))
            kp = make_annotation_keypoints(class_id=ann.get("category_id"), class_name=cat_map.get(ann.get("category_id"), "unknown"), keypoints=pts, ann_id=str(ann.get("id")))
            rec["annotations"].append(kp)

    # finalize image records
    index = {}
    for iid, r in img_map.items():
        anns = r.pop("annotations", [])
        rec = make_image_record(image_id=r.get("file_name"), abs_path=r.get("abs_path"), width=r.get("width"), height=r.get("height"), thumbnail=None, mean_rgb=None, annotations=anns, metadata={})
        index[rec["id"]] = sanitize_for_json(rec)
    return index

# -----------------------------
# YOLO loader
# -----------------------------
def _load_image_size_if_needed(image_path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return None, None

def parse_yolo(labels_root: str, images_root: Optional[str]=None, class_map: Optional[List[str]]=None) -> Dict[str,Any]:
    """
    Parse a directory of YOLO label files. YOLO text files typically have lines:
      class_id x_center y_center width height
    where coordinates are normalized in [0,1].

    Parameters
    ----------
    labels_root
        path containing .txt files matching images by name
    images_root
        optional location of images to peek size
    class_map
        optional list mapping class id to class name

    Returns mapping image file_name to canonical image record.
    """
    p = Path(labels_root)
    if not p.exists():
        raise FileNotFoundError(labels_root)
    index = {}
    for txt in p.glob("*.txt"):
        stem = txt.stem
        # try find image by common extensions
        image_path = None
        if images_root:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif"):
                cand = Path(images_root) / (stem + ext)
                if cand.exists():
                    image_path = cand
                    break
        else:
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif"):
                cand = txt.with_suffix(ext)
                if cand.exists():
                    image_path = cand
                    break
        w, h = None, None
        if image_path:
            w, h = _load_image_size_if_needed(image_path)
        anns = []
        try:
            text = txt.read_text(encoding="utf-8").strip()
            if not text:
                # empty label file means no annotations
                rec = make_image_record(image_id=stem, abs_path=str(image_path) if image_path else str(txt.with_suffix(".jpg")), width=w, height=h, annotations=[])
                index[rec["id"]] = sanitize_for_json(rec)
                continue
            for line in text.splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                cx = float(parts[1]); cy = float(parts[2]); bw = float(parts[3]); bh = float(parts[4])
                # if image size known convert to absolute
                if w and h:
                    x_min = (cx - bw/2.0) * w
                    y_min = (cy - bh/2.0) * h
                    x_max = (cx + bw/2.0) * w
                    y_max = (cy + bh/2.0) * h
                else:
                    # store normalized box as 0..1; some features may not accept this. We attempt to scale when possible later.
                    x_min = cx - bw/2.0
                    y_min = cy - bh/2.0
                    x_max = cx + bw/2.0
                    y_max = cy + bh/2.0
                name = class_map[cls] if (class_map and cls < len(class_map)) else str(cls)
                a = make_annotation_bbox(class_id=cls, class_name=name, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
                anns.append(a)
        except Exception:
            logger.debug("Failed parsing YOLO file %s", txt, exc_info=True)
        rec = make_image_record(image_id=stem, abs_path=str(image_path) if image_path else str(txt.with_suffix(".jpg")), width=w, height=h, annotations=anns, metadata={})
        index[rec["id"]] = sanitize_for_json(rec)
    return index

# -----------------------------
# Mask folder loader
# -----------------------------
def parse_mask_folder(images_dir: str, masks_dir: str, mapping: Optional[Dict[int,str]]=None, multi_instance: bool=False) -> Dict[str,Any]:
    """
    Parse datasets where masks are stored in a folder that mirrors image file names.

    Parameters
    ----------
    images_dir
        folder with image files
    masks_dir
        folder with mask images. Mask filenames match image filenames by stem.
    mapping
        optional mapping from pixel value to class name. e.g. {0: "bg", 1: "car", 2: "person"}
    multi_instance
        if True multiple instance masks per image may exist with naming pattern image__inst1.png

    Behavior
    - For each image file, try to find mask file with same stem
    - If mask is binary create a single mask annotation
    - If mask has multiple class values create one annotation per class referencing the mask path and class id
    - When mapping not provided and mask has values other than 0 and 255 we use value as class id and class_name as str(value)
    """
    imgp = Path(images_dir)
    maskp = Path(masks_dir)
    if not imgp.exists():
        raise FileNotFoundError(images_dir)
    if not maskp.exists():
        raise FileNotFoundError(masks_dir)
    index = {}
    for img in imgp.iterdir():
        if not img.is_file():
            continue
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            continue
        stem = img.stem
        # candidate mask file
        candidate = maskp / (stem + ".png")
        anns = []
        if not candidate.exists():
            # check multi instance patterns
            if multi_instance:
                for f in maskp.glob(f"{stem}__*.png"):
                    candidate = f
                    # treat each instance as its own mask with instance id
                    anns.append(make_annotation_mask(class_id=None, class_name="instance", mask_path=str(candidate), ann_id=stem + "__" + f.name))
            # else leave annotations empty
        else:
            # examine the mask
            try:
                with Image.open(candidate) as m:
                    arr = np.asarray(m)
                    if arr.ndim == 3:
                        # palette or rgb. try take first channel
                        arrc = arr[..., 0]
                    else:
                        arrc = arr
                    unique = np.unique(arrc)
                    # binary mask common case
                    if set(unique.tolist()) <= {0, 255} or set(unique.tolist()) <= {0,1}:
                        # single binary mask
                        class_id = 1
                        class_name = mapping.get(1) if mapping and 1 in mapping else "object"
                        anns.append(make_annotation_mask(class_id=class_id, class_name=class_name, mask_path=str(candidate)))
                    else:
                        # multi class mask, create annotation per unique non zero value
                        for v in unique:
                            if v == 0:
                                continue
                            cid = int(v)
                            cname = mapping.get(cid) if mapping and cid in mapping else str(cid)
                            anns.append(make_annotation_mask(class_id=cid, class_name=cname, mask_path=str(candidate)))
            except Exception:
                logger.debug("Failed reading mask %s", candidate, exc_info=True)
        rec = make_image_record(image_id=stem, abs_path=str(img), width=None, height=None, annotations=anns, metadata={})
        index[rec["id"]] = sanitize_for_json(rec)
    return index

# -----------------------------
# Generic detect and parse entry
# -----------------------------
def parse_dataset(root: str, prefer: Optional[str]=None, **kwargs) -> Dict[str,Any]:
    """
    Detect dataset type and parse into canonical index.

    prefer may be "coco", "yolo", "mask_folder" to force parser selection.

    kwargs passed to parser functions.

    Returns canonical index mapping image id to record.
    """
    fmt = prefer or detect_dataset_format(root)
    if fmt == "coco":
        # try common coco json names
        path = Path(root)
        candidates = [path / "annotations.json", path / "instances.json", path / "coco.json"]
        chosen = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is None:
            # try parent annotations folder
            for c in path.glob("**/*.json"):
                try:
                    d = json.loads(c.read_text(encoding="utf-8"))
                    if isinstance(d, dict) and "images" in d and "annotations" in d:
                        chosen = c
                        break
                except Exception:
                    continue
        if chosen is None:
            raise FileNotFoundError("COCO JSON not found under root")
        return parse_coco(str(chosen), images_root=kwargs.get("images_root"))
    if fmt == "yolo":
        # default labels folder
        return parse_yolo(str(Path(root) / "labels"), images_root=kwargs.get("images_root"), class_map=kwargs.get("class_map"))
    if fmt == "mask_folder":
        images_dir = str(root)
        masks_dir = str(Path(root) / "masks")
        # try multiple mask folder names
        if not Path(masks_dir).exists():
            for cand in ("masks_png", "segmentation_masks"):
                if (Path(root) / cand).exists():
                    masks_dir = str(Path(root) / cand)
                    break
        return parse_mask_folder(images_dir=images_dir, masks_dir=masks_dir, mapping=kwargs.get("mapping"), multi_instance=kwargs.get("multi_instance", False))
    # fallback naive parse: treat images folder and optional labels
    # scan for images and create empty annotation records
    out = {}
    p = Path(root)
    for f in p.rglob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            rec = make_image_record(image_id=f.stem, abs_path=str(f), width=None, height=None, annotations=[], metadata={})
            out[rec["id"]] = sanitize_for_json(rec)
    return out
