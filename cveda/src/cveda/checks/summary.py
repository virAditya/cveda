"""
Dataset sanity checklist.

Provides a single well named function dataset_health_summary that the API imports.
The function is defensive and returns a simple health score and badges even when parts
of the audit are missing or malformed.
"""

from typing import Dict, Any, List


def dataset_health_summary(audit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a short health summary of the audit.

    The function uses a simple heuristic combining counts of critical issues normalized
    by image count. It never raises, it returns a dictionary containing:
    {
        "score": int 0 to 100,
        "badges": { ... boolean badges ... },
        "notes": [ ... human readable notes ... ]
    }

    Parameters
    audit dict
        The full audit result produced by CVEDA.run_audit. The function reads a few
        well known keys and tolerates absent keys gracefully.
    """
    try:
        n_images = int(audit.get("checks", {}).get("completeness", {}).get("n_images", 0) or 0)
    except Exception:
        n_images = 0

    if n_images == 0:
        return {"score": 0, "badges": {"images_found": False}, "notes": ["No images scanned"]}

    zero = len(audit.get("checks", {}).get("bbox_sanity", {}).get("zero_area", []) or [])
    inverted = len(audit.get("checks", {}).get("bbox_sanity", {}).get("inverted", []) or [])
    outside = len(audit.get("checks", {}).get("bbox_sanity", {}).get("outside_bounds", []) or [])
    corrupted = len(audit.get("checks", {}).get("corrupt", {}).get("files", []) or [])
    missing_ann = len(audit.get("checks", {}).get("completeness", {}).get("images_without_annotations", []) or [])

    # weighted penalties
    penalty = zero * 3 + inverted * 4 + outside * 1 + corrupted * 6 + missing_ann * 1
    base = max(1.0, n_images / 10.0)
    raw_score = max(0.0, 100.0 - (penalty / base))
    score = int(round(raw_score))

    badges = {
        "images_found": n_images > 0,
        "low_corruption": corrupted == 0,
        "no_critical_bbox_errors": (zero + inverted) == 0,
        "enough_annotations": missing_ann < (n_images * 0.1)
    }

    notes: List[str] = []
    if corrupted:
        notes.append(f"{corrupted} corrupted images detected")
    if missing_ann:
        notes.append(f"{missing_ann} images have no annotations")
    if zero:
        notes.append(f"{zero} zero area bounding boxes detected")
    if inverted:
        notes.append(f"{inverted} inverted bounding boxes detected")

    return {"score": score, "badges": badges, "notes": notes}
