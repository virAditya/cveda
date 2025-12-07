"""
Duplicate detection utilities.

Wraps phash based duplicate detection and provides a lightweight cluster helper.

This module is used by checks/splits and can be used elsewhere for leakage
detection or for building a duplicate free sample.
"""

from typing import Dict, List, Tuple, Optional
from .hashing import phash_image
import os
import logging

logger = logging.getLogger(__name__)


def find_duplicates(paths: List[str], threshold: int = 8) -> List[Tuple[str, str, int]]:
    """
    Naive O n squared duplicate detector using phash.

    Returns list of tuples (path_a path_b hamming)
    """
    hashes = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        h = phash_image(p)
        if h:
            hashes[p] = h
    items = list(hashes.items())
    out = []
    for i in range(len(items)):
        a, ha = items[i]
        for j in range(i + 1, len(items)):
            b, hb = items[j]
            try:
                d = int(hamming_distance_hex(ha, hb))
            except Exception:
                continue
            if d <= threshold:
                out.append((a, b, d))
    return out


def hamming_distance_hex(h1: str, h2: str) -> int:
    try:
        v1 = int(h1, 16)
        v2 = int(h2, 16)
        return (v1 ^ v2).bit_count()
    except Exception:
        return 9999
