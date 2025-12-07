"""
repetitive_layouts

Heuristic detection of repetitive layouts using bbox normalized positions.
If many annotation centroids fall on a grid-like pattern, flag the image.

Approach:
- form centroid coordinates normalized to image size
- quantize spatial coordinates and look for repeated quantized cells
"""

from typing import Dict, Any, List
from collections import Counter
import math

def run_repetitive_layouts(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - grid_size: int default 10 quantization bins per axis
      - repeat_threshold: int default 8 occurrences in same cell to consider repeated
      - sample_limit: int

    Returns:
      {
        "feature": "repetitive_layouts",
        "repeated_cells": [ {cell: (i,j), count: n, examples: [...] } ],
        "status": "ok"
      }
    """
    cfg = config or {}
    grid_size = int(cfg.get("grid_size", 10))
    repeat_threshold = int(cfg.get("repeat_threshold", 8))
    sample_limit = int(cfg.get("sample_limit", 10))

    cell_counter = Counter()
    cell_examples = {}

    for fname, rec in (index or {}).items():
        w = rec.get("width") or 0; h = rec.get("height") or 0
        if w <= 0 or h <= 0:
            continue
        for ann in (rec.get("annotations", []) or []):
            try:
                x0,y0,x1,y1 = map(float, ann.get("bbox", [0,0,0,0]))
            except Exception:
                continue
            cx = (x0+x1)/(2.0*w)
            cy = (y0+y1)/(2.0*h)
            ix = min(grid_size-1, int(cx * grid_size))
            iy = min(grid_size-1, int(cy * grid_size))
            cell_counter[(ix,iy)] += 1
            if (ix,iy) not in cell_examples:
                cell_examples[(ix,iy)] = []
            if len(cell_examples[(ix,iy)]) < sample_limit:
                cell_examples[(ix,iy)].append(fname)

    repeated = []
    for cell, cnt in cell_counter.most_common():
        if cnt >= repeat_threshold:
            repeated.append({"cell": cell, "count": cnt, "examples": cell_examples.get(cell, [])[:sample_limit]})

    return {"feature": "repetitive_layouts", "repeated_cells": repeated, "status": "ok"}
