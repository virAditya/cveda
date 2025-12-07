"""
Class distribution helpers.

Compute per class counts number of images per class and top classes.
"""

from typing import Dict, Any
from collections import Counter


def compute_class_distribution(index: Dict[str, Any], cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute class counts aggregated by annotations and by images.

    Returns:
    {
        "annotation_counts": {class: count},
        "image_counts": {class: count_of_images_where_class_appears},
        "top_classes": [(class, count) ...]
    }
    """
    cfg = cfg or {}
    counter = Counter()
    imageset = {}
    for fname, rec in index.items():
        classes_in_image = set()
        for ann in rec.get("annotations", []):
            cls = str(ann.get("class"))
            counter[cls] += 1
            classes_in_image.add(cls)
        for c in classes_in_image:
            imageset.setdefault(c, 0)
            imageset[c] += 1

    top_n = cfg.get("top_n", 30)
    top = counter.most_common(top_n)
    return {
        "annotation_counts": dict(counter),
        "image_counts": imageset,
        "top_classes": top
    }
