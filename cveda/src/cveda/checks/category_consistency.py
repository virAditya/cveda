"""
Category consistency checks.

Compare declared categories with used categories in annotations and produce useful hints.
"""

from typing import Dict, Any, List, Set


def run_category_consistency(index: Dict[str, Any], cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Build declared and used class sets and return differences.

    Since many datasets do not include a declared category list
    this function will attempt to recover a declared set from
    common COCO style metadata when present in records.

    Returns dictionary with keys:
    - declared_classes list
    - used_classes list
    - declared_but_unused list
    - used_but_undeclared list
    - suggestions dict with possible next steps
    """
    declared: Set[str] = set()
    used: Set[str] = set()

    # find declared classes from index meta if present
    for fname, rec in index.items():
        meta = rec.get("meta", {})
        categories = meta.get("categories") or meta.get("category_map")
        if categories and isinstance(categories, dict):
            for v in categories.values():
                declared.add(str(v))

        for ann in rec.get("annotations", []):
            cls = ann.get("class")
            if cls is not None:
                used.add(str(cls))

    declared_list = sorted(list(declared))
    used_list = sorted(list(used))
    declared_but_unused = [c for c in declared_list if c not in used]
    used_but_undeclared = [c for c in used_list if c not in declared]
    suggestions = {
        "provide_class_map": bool(used and not declared),
        "auto_reindex_option": True
    }
    return {
        "declared_classes": declared_list,
        "used_classes": used_list,
        "declared_but_unused": declared_but_unused,
        "used_but_undeclared": used_but_undeclared,
        "suggestions": suggestions
    }
