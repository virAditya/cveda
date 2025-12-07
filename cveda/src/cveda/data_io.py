"""
Data IO and annotation loader module.

This module provides robust utilities to discover images and annotations
in a dataset root, to normalize common annotation formats into a canonical
schema, and to handle conventional train val test split layouts.

Key classes and functions
- ImageCollectionLoader
    Discover images and annotations under a provided root path and build a
    canonical index mapping relative file names to records. The loader
    understands common nested folder layouts used by many datasets.

- discover_splits
    Convenience function to detect common split folders such as train val test.

- build_index_for_split
    Convenience wrapper to build an index for a specific split path.

Canonical per image record format returned by ImageCollectionLoader.build_index:

{
    "file_name": "relative/path/to/img.jpg",  # relative to loader.root
    "abs_path": "/absolute/path/to/img.jpg",
    "width": 1024,                             # or None when unavailable
    "height": 768,                             # or None when unavailable
    "annotations": [
        {"class": "person", "bbox": [xmin, ymin, xmax, ymax], "raw": {...}},
        ...
    ],
    "meta": {...}                              # optional metadata such as parse errors
}

Design goals and behaviors
- Be resilient to common real world problems in datasets
  such as missing width height in COCO entries normalized YOLO coords
  XML parse errors mixed annotation formats and nested folder layouts.
- Never raise on a single corrupted file. Instead record per file meta errors
  inside the index so calling code can inspect and decide how to proceed.
- Normalize all bounding boxes to absolute pixel coordinates in the form
  xmin ymin xmax ymax represented as floats.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import xml.etree.ElementTree as ET
import os
import logging

from PIL import Image, UnidentifiedImageError

from .utils.io_helpers import get_image_size_safe

logger = logging.getLogger(__name__)


def discover_splits(root: str, split_names: Optional[List[str]] = None) -> Dict[str, Path]:
    """
    Discover conventional split subfolders under a dataset root.

    Parameters
    root str
        Path to the dataset root folder to scan for splits.
    split_names List[str] optional
        Iterable of split folder names to look for. By default the function
        checks: ['train', 'val', 'test', 'validation'] in that order.

    Returns
    Dict[str, Path] mapping split name to absolute Path for splits that exist.
    """
    if split_names is None:
        split_names = ["train", "val", "test", "validation"]
    root_p = Path(root)
    found: Dict[str, Path] = {}
    for s in split_names:
        p = root_p / s
        if p.exists() and p.is_dir():
            found[s] = p.resolve()
    return found


def build_index_for_split(split_path: str, recursive: bool = True) -> Dict[str, Any]:
    """
    Convenience wrapper that builds a canonical index for a single split folder.

    Parameters
    split_path str
        Path to the split folder, for example dataset/train
    recursive bool
        Whether to search recursively for images inside the split folder.

    Returns
    Dict[str, Any] canonical index as produced by ImageCollectionLoader.build_index
    """
    loader = ImageCollectionLoader(str(split_path), recursive=recursive)
    return loader.build_index()


class ImageCollectionLoader:
    """
    Discover images and annotation files and build a canonical dataset index.

    Usage examples

    1) Single folder with images and annotations side by side:
       loader = ImageCollectionLoader("/path/to/dataset", recursive=True)
       index = loader.build_index()

    2) Split folder usage where images are under train/images and annotations under train/labels:
       loader = ImageCollectionLoader("/path/to/dataset/train", recursive=True)
       index = loader.build_index()

    Behavior summary
    - Detects common image subfolders such as images image imgs img JPEGImages
    - Detects common annotation subfolders such as labels label annotations annotation
    - Supports COCO json with images and annotations arrays
    - Supports VOC xml single file per image
    - Supports YOLO txt single file per image or label folder with .txt files
    - When multiple annotation sources are available they are merged in order
      COCO global json first VOC xml next YOLO txt last
    - The loader records parsing problems inside record["meta"]["errors"]
      rather than raising, enabling robust batch processing
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    COMMON_IMAGE_DIRS = ("images", "image", "imgs", "img", "JPEGImages", "Images")
    COMMON_ANNOT_DIRS = ("labels", "label", "annotations", "annotation", "ann", "xmls", "jsons")

    def __init__(self, root: str, recursive: bool = True):
        """
        Initialize the loader.

        Parameters
        root str
            Root folder to scan. This can be a dataset root containing train val test
            subfolders or a single split folder containing images and labels.
        recursive bool
            When true the loader will search recursively in directories where appropriate.
        """
        self.root = Path(root)
        self.recursive = bool(recursive)
        if not self.root.exists():
            raise FileNotFoundError(f"Root path not found: {root}")

    # ---------- low level discovery helpers ----------

    def _list_image_files(self, base: Optional[Path] = None) -> List[Path]:
        """
        Return a sorted list of image files under base.

        Strategy:
        1) If base contains known image subfolders use them
        2) If not, scan base recursively when recursive is True
        3) Return an empty list if no images were found
        """
        base = base or self.root

        # try common image subdirectories first
        candidate_dirs: List[Path] = []
        for dname in self.COMMON_IMAGE_DIRS:
            p = base / dname
            if p.exists() and p.is_dir():
                candidate_dirs.append(p)

        # also search one level deeper: some datasets use split/<something>/images
        if not candidate_dirs:
            for child in base.iterdir():
                if child.is_dir():
                    for dname in self.COMMON_IMAGE_DIRS:
                        p = child / dname
                        if p.exists() and p.is_dir():
                            candidate_dirs.append(p)

        files: List[Path] = []
        if candidate_dirs:
            for d in candidate_dirs:
                if self.recursive:
                    files.extend([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS])
                else:
                    files.extend([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS])
            return sorted(files)

        # fallback: treat base itself as containing images
        if base.is_dir():
            if self.recursive:
                files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS]
            else:
                files = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS]
        return sorted(files)

    def _locate_annotation_folder(self, img_path: Path) -> Optional[Path]:
        """
        Heuristic to locate a sibling annotation folder for a given image.

        Looks for labels label annotations annotation and similar folders in the
        parent and grandparent directories. Returns the first folder found or None.
        """
        # candidate search starts at the image's parent then climbs up
        p = img_path.parent
        checked = set()
        depth = 0
        while p and p not in checked and depth < 5:
            checked.add(p)
            for a_name in self.COMMON_ANNOT_DIRS:
                cand = p / a_name
                if cand.exists() and cand.is_dir():
                    return cand
            # also check siblings of the parent
            parent_parent = p.parent
            if parent_parent and parent_parent != p:
                for a_name in self.COMMON_ANNOT_DIRS:
                    cand = parent_parent / a_name
                    if cand.exists() and cand.is_dir():
                        return cand
            if p == self.root:
                break
            p = p.parent
            depth += 1
        return None

    def _find_annotation_candidates(self, img_path: Path) -> List[Path]:
        """
        Return a prioritized list of annotation file paths or folders that may contain annotations
        relevant to the provided image path.

        Order of candidates:
        - COCO style global json files near the image or at the loader root
        - annotation file next to the image with same stem (.txt .xml .json)
        - sibling annotation folder containing a matching stem.txt or many label files
        - returned candidate list is deduplicated and preserves order
        """
        candidates: List[Path] = []

        # look for common global JSON files at loader.root
        for name in ("annotations.json", "instances.json", "coco.json"):
            g = self.root / name
            if g.exists():
                candidates.append(g)

        # look for COCO style files in ancestor folders up to 3 levels
        cur = img_path.parent
        for _ in range(3):
            for name in ("annotations.json", "instances.json", "coco.json"):
                g = cur / name
                if g.exists():
                    candidates.append(g)
            if cur == cur.parent:
                break
            cur = cur.parent

        # check for annotation file next to image
        stem = img_path.stem
        for ext in (".txt", ".xml", ".json"):
            p = img_path.with_suffix(ext)
            if p.exists():
                candidates.append(p)

        # look for sibling annotation folder
        annot_folder = self._locate_annotation_folder(img_path)
        if annot_folder:
            # include the whole folder as a candidate
            candidates.append(annot_folder)
            # include same stem file inside folder if present
            for ext in (".txt", ".xml", ".json"):
                p = annot_folder / f"{stem}{ext}"
                if p.exists():
                    candidates.append(p)

        # deduplicate preserving order
        seen = set()
        out: List[Path] = []
        for c in candidates:
            try:
                rp = str(c.resolve())
            except Exception:
                rp = str(c)
            if rp not in seen:
                seen.add(rp)
                out.append(c)
        return out

    # ---------- annotation format parsers ----------

    def _load_coco_json(self, json_path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[int, str]]:
        """
        Parse a COCO style JSON file while tolerating minor inconsistencies.

        Returns:
        - labels_map: mapping from file_name to a list of annotation dicts
          each annotation contains keys category_id bbox raw where bbox is COCO x y w h
        - category_map: mapping category id to category name when available
        """
        labels_map: Dict[str, List[Dict[str, Any]]] = {}
        category_map: Dict[int, str] = {}
        try:
            with json_path.open("r", encoding="utf8") as f:
                data = json.load(f)
        except Exception as e:
            logger.debug("Failed to parse COCO json %s error %s", json_path, e)
            return labels_map, category_map

        images = {img.get("id"): img for img in data.get("images", []) if isinstance(img, dict)}
        for c in data.get("categories", []) or []:
            if isinstance(c, dict):
                cid = c.get("id")
                name = c.get("name") or str(cid)
                if cid is not None:
                    category_map[cid] = name

        for ann in data.get("annotations", []) or []:
            if not isinstance(ann, dict):
                continue
            img_id = ann.get("image_id")
            img_info = images.get(img_id) if img_id is not None else None
            if not img_info:
                continue
            fname = img_info.get("file_name") or img_info.get("file", img_info.get("id"))
            if not fname:
                continue
            labels_map.setdefault(fname, []).append({
                "category_id": ann.get("category_id"),
                "bbox": ann.get("bbox"),
                "raw": ann
            })
        return labels_map, category_map

    def _load_voc_xml_file(self, xml_path: Path) -> Optional[Dict[str, Any]]:
        """
        Parse a single VOC XML annotation file.

        Returns a dict with keys file_name width height annotations or None on failure.
        Each annotation is a dict with name bbox raw
        """
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            filename = root.findtext("filename") or xml_path.stem
            size = root.find("size")
            width = int(size.findtext("width")) if size is not None and size.findtext("width") else None
            height = int(size.findtext("height")) if size is not None and size.findtext("height") else None
            objs: List[Dict[str, Any]] = []
            for obj in root.findall("object"):
                name = obj.findtext("name")
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                try:
                    xmin = float(bnd.findtext("xmin"))
                    ymin = float(bnd.findtext("ymin"))
                    xmax = float(bnd.findtext("xmax"))
                    ymax = float(bnd.findtext("ymax"))
                except Exception:
                    continue
                objs.append({"name": name, "bbox": [xmin, ymin, xmax, ymax], "raw": None})
            return {"file_name": filename, "width": width, "height": height, "annotations": objs}
        except ET.ParseError:
            logger.debug("XML parse error for %s", xml_path)
            return None
        except Exception:
            logger.debug("Unexpected error parsing VOC xml %s", xml_path)
            return None

    def _load_yolo_txt_file(self, txt_path: Path, image_size: Tuple[int, int]) -> Optional[List[Dict[str, Any]]]:
        """
        Parse a YOLO style txt file with normalized center coordinates cx cy w h.

        Parameters
        txt_path Path
            Path to the .txt file
        image_size Tuple[int, int]
            (width, height) of the corresponding image. If None the function cannot convert normalized coords

        Returns list of annotation dicts with class bbox raw or None on error
        """
        if image_size is None:
            return None
        w, h = image_size
        anns: List[Dict[str, Any]] = []
        try:
            with txt_path.open("r", encoding="utf8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_token = parts[0]
                    try:
                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                    except Exception:
                        continue
                    # normalized to absolute
                    xmin = (cx - bw / 2.0) * w
                    ymin = (cy - bh / 2.0) * h
                    xmax = (cx + bw / 2.0) * w
                    ymax = (cy + bh / 2.0) * h
                    anns.append({"class": cls_token, "bbox": [xmin, ymin, xmax, ymax], "raw": None})
            return anns
        except Exception:
            logger.debug("Failed to parse YOLO txt %s", txt_path)
            return None

    def _normalize_coco_bbox(self, cocobox: List[float]) -> List[float]:
        """
        Convert COCO bbox x y w h to xmin ymin xmax ymax
        """
        if not isinstance(cocobox, (list, tuple)) or len(cocobox) < 4:
            return [0.0, 0.0, 0.0, 0.0]
        x, y, w, h = map(float, cocobox[:4])
        return [x, y, x + w, y + h]

    # ---------- canonicalization pipeline for a single image ----------

    def _canonicalize_annotations_for_image(self, img_path: Path, candidates: List[Path]) -> Dict[str, Any]:
        """
        Build the canonical per image record by trying annotation candidates in order.

        The function always returns a record dict. If parsing problems occur details are stored
        in record["meta"]["errors"] so the caller can inspect them.
        """
        rel_path = img_path.relative_to(self.root).as_posix()
        abs_path = str(img_path.resolve())
        width, height = get_image_size_safe(abs_path)
        record: Dict[str, Any] = {
            "file_name": rel_path,
            "abs_path": abs_path,
            "width": width,
            "height": height,
            "annotations": [],
            "meta": {},
        }
        errors: List[str] = []

        # process candidates in priority order
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.suffix.lower() == ".json":
                    # try COCO style parsing
                    labels_map, cat_map = self._load_coco_json(candidate)
                    # try matching by basename first then relative path
                    anns_for_file = labels_map.get(img_path.name) or labels_map.get(rel_path) or []
                    for a in anns_for_file:
                        raw_bbox = a.get("bbox")
                        if raw_bbox:
                            bbox = self._normalize_coco_bbox(raw_bbox)
                            record["annotations"].append({
                                "class": a.get("category_id"),
                                "bbox": bbox,
                                "raw": a.get("raw", a)
                            })
                elif candidate.is_file() and candidate.suffix.lower() == ".xml":
                    parsed = self._load_voc_xml_file(candidate)
                    if parsed and parsed.get("annotations"):
                        for a in parsed.get("annotations", []):
                            bbox = a.get("bbox")
                            if bbox and len(bbox) == 4:
                                record["annotations"].append({
                                    "class": a.get("name"),
                                    "bbox": bbox,
                                    "raw": a.get("raw")
                                })
                elif candidate.is_file() and candidate.suffix.lower() == ".txt":
                    anns = self._load_yolo_txt_file(candidate, (width, height))
                    if anns:
                        record["annotations"].extend(anns)
                elif candidate.is_dir():
                    # treat dir as YOLO style labels folder containing <image_stem>.txt
                    txt = candidate / f"{img_path.stem}.txt"
                    if txt.exists():
                        anns = self._load_yolo_txt_file(txt, (width, height))
                        if anns:
                            record["annotations"].extend(anns)
            except Exception as e:
                logger.debug("Error processing candidate %s for image %s: %s", candidate, img_path, e)
                errors.append(f"{candidate}: {e}")

        # final normalization step: ensure bbox arrays are floats and in xmin ymin xmax ymax order
        normalized: List[Dict[str, Any]] = []
        for ann in record.get("annotations", []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = map(float, bbox)
            except Exception:
                continue
            normalized.append({"class": ann.get("class"), "bbox": [x0, y0, x1, y1], "raw": ann.get("raw")})
        record["annotations"] = normalized
        if errors:
            record["meta"]["errors"] = errors
        return record

    # ---------- public API ----------

    def build_index(self) -> Dict[str, Any]:
        """
        Build and return the canonical index mapping relative file names to records.

        The method is intentionally tolerant. Files that cause unexpected exceptions
        will have an entry with meta.error describing the problem and an empty
        annotations list.
        """
        files = self._list_image_files(base=self.root)
        index: Dict[str, Any] = {}
        for img in files:
            try:
                candidates = self._find_annotation_candidates(img)
                rec = self._canonicalize_annotations_for_image(img, candidates)
                index[rec["file_name"]] = rec
            except Exception as e:
                try:
                    rel = img.relative_to(self.root).as_posix()
                except Exception:
                    rel = str(img)
                index[rel] = {
                    "file_name": rel,
                    "abs_path": str(img.resolve()),
                    "width": None,
                    "height": None,
                    "annotations": [],
                    "meta": {"error": str(e)}
                }
        return index
