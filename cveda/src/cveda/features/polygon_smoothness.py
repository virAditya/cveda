"""
polygon_smoothness

For segmentation polygons (COCO-style lists in ann['raw']['segmentation']), compute
a simple boundary complexity metric: ratio of perimeter^2 to area (isoperimetric ratio)
and mean edge length variance.

This detects jagged hand-drawn masks vs smoothed polygons.
"""

from typing import Dict, Any, List
import math

def _poly_metrics(points: List[float]):
    # points is flat list [x0,y0,x1,y1,...]
    if not points or len(points) < 6:
        return None
    coords = [(float(points[i]), float(points[i+1])) for i in range(0, len(points), 2)]
    # area via shoelace
    area = 0.0
    perim = 0.0
    n = len(coords)
    for i in range(n):
        x1,y1 = coords[i]
        x2,y2 = coords[(i+1)%n]
        area += x1*y2 - x2*y1
        dx = x2-x1; dy=y2-y1
        perim += math.hypot(dx, dy)
    area = abs(area)/2.0
    # edge length variance
    edges = []
    for i in range(n):
        x1,y1 = coords[i]; x2,y2 = coords[(i+1)%n]
        edges.append(math.hypot(x2-x1, y2-y1))
    mean_edge = sum(edges)/len(edges) if edges else 0.0
    var_edge = sum((e-mean_edge)**2 for e in edges)/len(edges) if edges else 0.0
    iso = (perim*perim)/(4*math.pi*area) if area>0 else float('inf')
    return {"area": area, "perimeter": perim, "isoperimetric_ratio": iso, "edge_mean": mean_edge, "edge_var": var_edge}

def run_polygon_smoothness(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - sample_limit: how many polygon examples to return

    Returns:
      {
        "feature": "polygon_smoothness",
        "n_polygons": int,
        "avg_isoperimetric_ratio": float,
        "examples": [ {file, class, metrics}, ... ]
      }
    """
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 20))
    metrics_list = []
    examples = []

    for fname, rec in (index or {}).items():
        for ann in (rec.get("annotations", []) or []):
            raw = ann.get("raw") or {}
            seg = raw.get("segmentation")
            if not seg:
                continue
            # seg may be list of polygons; iterate
            for poly in seg if isinstance(seg, list) else [seg]:
                if not poly:
                    continue
                m = _poly_metrics(poly)
                if m:
                    metrics_list.append(m)
                    if len(examples) < sample_limit:
                        examples.append({"file": fname, "class": str(ann.get("class","")), "metrics": m})

    if not metrics_list:
        return {"feature": "polygon_smoothness", "n_polygons": 0, "status": "no_polygons"}

    avg_iso = sum(m["isoperimetric_ratio"] for m in metrics_list)/len(metrics_list)
    return {
        "feature": "polygon_smoothness",
        "n_polygons": len(metrics_list),
        "avg_isoperimetric_ratio": avg_iso,
        "examples": examples,
        "status": "ok"
    }
