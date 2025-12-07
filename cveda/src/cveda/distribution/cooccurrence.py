"""
Class co occurrence matrix and utilities.

Compute per image co occurrence counts and a global co occurrence matrix.
"""

from typing import Dict, Any, Tuple
from collections import Counter, defaultdict
import numpy as np


def compute_cooccurrence(index: Dict[str, Any], top_n: int = 50) -> Dict[str, Any]:
    """
    Build a co occurrence matrix for classes in the index.

    Returns:
    {
        "classes": [class1 class2 ...],
        "matrix": 2D list matrix counts,
        "top_pairs": [((c1 c2) count) ...]
    }
    """
    # count classes
    class_counts = Counter()
    for _, rec in index.items():
        classes = set(str(a.get("class")) for a in rec.get("annotations", []))
        for c in classes:
            class_counts[c] += 1

    classes = [c for c, _ in class_counts.most_common(top_n)]
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    pair_counts = Counter()
    for _, rec in index.items():
        classes_in_image = sorted(set(str(a.get("class")) for a in rec.get("annotations", [])))
        # count pairwise
        for i in range(len(classes_in_image)):
            for j in range(i + 1, len(classes_in_image)):
                a = classes_in_image[i]
                b = classes_in_image[j]
                if a in idx and b in idx:
                    M[idx[a], idx[b]] += 1
                    M[idx[b], idx[a]] += 1
                    pair_counts[(a, b)] += 1
    top_pairs = pair_counts.most_common(50)
    return {"classes": classes, "matrix": M.tolist(), "top_pairs": top_pairs}
