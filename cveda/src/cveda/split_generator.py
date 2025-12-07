"""
Simple stratified split generator.

Generates train val splits preserving class frequencies roughly.
This is a helper for dataset preparation and for cross validation folds.
It writes manifest dictionaries that map relative image names to split names.

Note this is a simple greedy stratified splitter not a full balanced fold generator.
"""

from typing import Dict, Any, List, Tuple
import random
from collections import defaultdict, Counter


def stratified_split(index: Dict[str, Any], val_fraction: float = 0.2, seed: int = 42) -> Dict[str, List[str]]:
    """
    Create a train val split mapping.

    Returns {"train": [filenames], "val": [filenames]}
    """
    random.seed(seed)
    # build per class lists of images
    class_to_images = defaultdict(set)
    for fname, rec in index.items():
        for ann in rec.get("annotations", []):
            cls = str(ann.get("class"))
            class_to_images[cls].add(fname)
    # flatten image list and allocate greedily
    image_pool = list(index.keys())
    random.shuffle(image_pool)
    val_target = int(len(image_pool) * val_fraction)
    val_set = set()
    # choose images that cover many rare classes first
    class_counts = {c: len(s) for c, s in class_to_images.items()}
    rare_classes = sorted(class_counts.items(), key=lambda x: x[1])
    for cls, _ in rare_classes:
        for im in class_to_images.get(cls, []):
            if len(val_set) >= val_target:
                break
            if im not in val_set:
                val_set.add(im)
    # fill up
    i = 0
    while len(val_set) < val_target and i < len(image_pool):
        val_set.add(image_pool[i])
        i += 1
    train = [i for i in image_pool if i not in val_set]
    val = [i for i in image_pool if i in val_set]
    return {"train": train, "val": val}
