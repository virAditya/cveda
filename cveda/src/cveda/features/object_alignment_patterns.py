"""
object_alignment_patterns

Detect linear alignments of object centroids using a simple RANSAC-like approach:
- pick random pairs of centroids and compute inlier counts for collinearity within tolerance.
- report images with strong alignment evidence.
"""

from typing import Dict, Any, List
import random
import math

def _collinearity_score(centroids, a, b, tol):
    # line through a,b compute perpendicular distance for each point
    (x1,y1) = a; (x2,y2) = b
    if x1==x2 and y1==y2:
        return 0, []
    inliers = []
    denom = math.hypot(x2-x1, y2-y1)
    for c in centroids:
        x0,y0 = c
        num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        dist = num / denom if denom>0 else float('inf')
        if dist <= tol:
            inliers.append(c)
    return len(inliers), inliers

def run_object_alignment_patterns(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - trials: int default 100
      - tol: float default 5.0 (pixels)
      - sample_limit: int default 10

    Returns:
      {
        "feature": "object_alignment_patterns",
        "aligned_images": [ {"file": fname, "score": best_inlier_count, "inliers": n } ],
        "status": "ok"
      }
    """
    cfg = config or {}
    trials = int(cfg.get("trials", 100))
    tol = float(cfg.get("tol", 5.0))
    sample_limit = int(cfg.get("sample_limit", 10))

    aligned_images = []

    for fname, rec in (index or {}).items():
        anns = rec.get("annotations", []) or []
        centroids = []
        for ann in anns:
            try:
                x0,y0,x1,y1 = map(float, ann.get("bbox", [0,0,0,0]))
                centroids.append(((x0+x1)/2.0, (y0+y1)/2.0))
            except Exception:
                continue
        if len(centroids) < 4:  # need some points to detect alignment
            continue
        best = 0
        best_inliers = []
        for _ in range(trials):
            a,b = random.sample(centroids, 2)
            count, inliers = _collinearity_score(centroids, a, b, tol)
            if count > best:
                best = count
                best_inliers = inliers
        if best >= cfg.get("min_inliers", max(3, int(0.4*len(centroids)))):
            aligned_images.append({"file": fname, "inlier_count": best, "n_points": len(centroids)})
            if len(aligned_images) >= sample_limit:
                break

    return {"feature": "object_alignment_patterns", "aligned_images": aligned_images, "status": "ok"}
