"""
Split aware checks, leakage detection and distribution comparison.

This module focuses on:
- exact filename overlap detection
- perceptual hash based near duplicate detection across splits
- class distribution pairwise comparison
"""

from typing import Dict, Any, List
from collections import Counter
from ..utils.hashing import phash_image
import os
import logging

logger = logging.getLogger(__name__)


def check_filename_leakage(indices_by_split: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find exact filename overlap across splits.

    Returns dict with overlaps list and per_pair counts.
    """
    splits = list(indices_by_split.keys())
    overlaps = []
    per_pair_counts = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a = splits[i]
            b = splits[j]
            set_a = set(indices_by_split[a].keys())
            set_b = set(indices_by_split[b].keys())
            common = sorted(list(set_a.intersection(set_b)))
            per_pair_counts[f"{a}|{b}"] = len(common)
            if common:
                overlaps.append({"a": a, "b": b, "common_files": common[:50], "n_common": len(common)})
    return {"overlaps": overlaps, "per_pair_counts": per_pair_counts}


def check_phash_leakage(indices_by_split: Dict[str, Dict[str, Any]], threshold: int = 10, max_compare_pairs: int = 200000) -> Dict[str, Any]:
    """
    Compute pHash for each image in each split and detect near duplicates across splits.

    threshold Hamming distance threshold for a match
    max_compare_pairs to limit O n squared explosion
    """
    # compute hashes
    hash_map = {}
    total_files = 0
    for s, idx in indices_by_split.items():
        hash_map[s] = {}
        for fname, rec in idx.items():
            p = rec.get("abs_path")
            if not p or not os.path.exists(p):
                continue
            h = phash_image(p)
            if h:
                hash_map[s][fname] = h
            total_files += 1

    # create comparisons across split pairs with cap
    matches = []
    splits = list(hash_map.keys())
    comparisons = 0
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            items_a = list(hash_map[a].items())
            items_b = list(hash_map[b].items())
            # cap approximate
            if len(items_a) * len(items_b) > max_compare_pairs:
                # sample
                import random
                k = int(max_compare_pairs ** 0.5)
                if k < 1:
                    continue
                items_a = random.sample(items_a, min(len(items_a), k))
                items_b = random.sample(items_b, min(len(items_b), k))
            for fa, ha in items_a:
                for fb, hb in items_b:
                    comparisons += 1
                    try:
                        d = hamming_distance_hex(ha, hb)
                    except Exception:
                        continue
                    if d <= threshold:
                        matches.append({"split_a": a, "split_b": b, "file_a": fa, "file_b": fb, "hamming": d})
    stats = {"comparisons": comparisons, "matches": len(matches), "total_files": total_files}
    return {"matches": matches, "stats": stats}


def hamming_distance_hex(h1: str, h2: str) -> int:
    """
    Hamming distance between two hex string hashes.
    """
    if not h1 or not h2:
        return 9999
    v1 = int(h1, 16)
    v2 = int(h2, 16)
    x = v1 ^ v2
    return x.bit_count()


def compare_class_distributions(indices_by_split: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute simple class counts per split and return pairwise relative differences.
    """
    counts = {}
    for s, idx in indices_by_split.items():
        c = Counter()
        for fname, rec in idx.items():
            for ann in rec.get("annotations", []):
                cls = str(ann.get("class"))
                c[cls] += 1
        counts[s] = dict(c)
    # pairwise differences
    splits = list(counts.keys())
    pairwise = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            classes = set(counts[a].keys()).union(set(counts[b].keys()))
            diffs = []
            for cls in classes:
                va = counts[a].get(cls, 0)
                vb = counts[b].get(cls, 0)
                mean = (va + vb) / 2 if (va + vb) > 0 else 1
                rel = abs(va - vb) / mean if mean > 0 else 0.0
                diffs.append((cls, rel, va, vb))
            pairwise[f"{a}|{b}"] = {"top_diffs": sorted(diffs, key=lambda x: x[1], reverse=True)[:20], "n_classes": len(diffs)}
    return {"counts": counts, "pairwise": pairwise}
