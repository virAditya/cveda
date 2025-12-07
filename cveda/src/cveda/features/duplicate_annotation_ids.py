"""
duplicate_annotation_ids

Search for duplicate annotation ids inside annotation raw fields.
Common keys: 'id', 'annotation_id', 'ann_id'.

Return collisions and sample filenames.
"""

from typing import Dict, Any
from collections import defaultdict

def run_duplicate_annotation_ids(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    id_fields = cfg.get("id_fields", ["id", "annotation_id", "ann_id"])
    seen = {}
    duplicates = defaultdict(list)
    for fname, rec in (index or {}).items():
        for ann in (rec.get("annotations", []) or []):
            raw = ann.get("raw", {}) or {}
            for f in id_fields:
                if f in raw:
                    aid = raw[f]
                    if aid in seen:
                        duplicates[aid].append({"file": fname, "existing_file": seen[aid]})
                    else:
                        seen[aid] = fname
    dup_list = [{"id": k, "occurrences": v} for k,v in duplicates.items()]
    return {"feature": "duplicate_annotation_ids", "duplicates": dup_list, "status": "ok"}
