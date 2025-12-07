"""
object_distance_distribution

Compute distributions of nearest-neighbor centroid distances per image and per class.
Useful to identify clustered vs sparse object distributions.
"""

from typing import Dict, Any, List
import math
from collections import defaultdict

def run_object_distance_distribution(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - sample_limit: int default 20 to return images with tight clustering

    Returns:
      {
        "feature": "object_distance_distribution",
        "global_stats": {"mean_nn_distance": float, "median_nn_distance": float},
        "tight_clusters_examples": [...]
      }
    """
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 20))
    nn_distances = []
    tight_examples = []

    for fname, rec in (index or {}).items():
        anns = (rec.get("annotations", []) or [])
        centers = []
        for ann in anns:
            bbox = ann.get("bbox", [0,0,0,0])
            try:
                x0,y0,x1,y1 = map(float, bbox)
            except Exception:
                continue
            cx = 0.5*(x0+x1); cy = 0.5*(y0+y1)
            centers.append((cx,cy))
        if len(centers) < 2:
            continue
        # compute nearest neighbor distance for each center
        for i,c in enumerate(centers):
            min_d = float('inf')
            for j,other in enumerate(centers):
                if i==j: continue
                d = math.hypot(c[0]-other[0], c[1]-other[1])
                if d < min_d: min_d = d
            nn_distances.append(min_d)
        # detect tight cluster: mean nn distance small relative to image diag
        mean_nn = sum(nn_distances[-len(centers):])/len(centers)
        w = rec.get("width") or 0; h = rec.get("height") or 0
        diag = math.hypot(w,h) if w and h else 1.0
        if mean_nn < cfg.get("tight_cluster_frac", 0.05)*diag:
            if len(tight_examples) < sample_limit:
                tight_examples.append({"file": fname, "mean_nn": mean_nn})

    if not nn_distances:
        return {"feature": "object_distance_distribution", "status": "no_data"}

    nn_sorted = sorted(nn_distances)
    mean = sum(nn_distances)/len(nn_distances)
    median = nn_sorted[len(nn_sorted)//2]
    return {
        "feature": "object_distance_distribution",
        "global_stats": {"mean_nn_distance": mean, "median_nn_distance": median},
        "tight_cluster_examples": tight_examples,
        "status": "ok"
    }
