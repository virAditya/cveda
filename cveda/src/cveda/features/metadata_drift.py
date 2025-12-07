"""
metadata_drift

Compute basic statistics of metadata fields like width, height, camera_model and
report divergence between first half and second half of dataset as a proxy for drift.

Return per-field changes and example values.
"""

from typing import Dict, Any
from collections import defaultdict
import statistics

def run_metadata_drift(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    # convert index to list deterministic order
    items = list((index or {}).items())
    if not items:
        return {"feature": "metadata_drift", "status": "no_data"}
    half = len(items)//2
    first = items[:half]
    second = items[half:]
    fields = cfg.get("fields", ["width", "height", "meta.camera_model"])
    results = {}
    def _get_field(rec, field):
        if field == "width":
            return rec.get("width")
        if field == "height":
            return rec.get("height")
        if field.startswith("meta."):
            k = field.split(".",1)[1]
            return rec.get("meta", {}).get(k)
        return None

    for field in fields:
        vals1 = [ _get_field(rec, field) for _,rec in first if _get_field(rec, field) is not None ]
        vals2 = [ _get_field(rec, field) for _,rec in second if _get_field(rec, field) is not None ]
        if not vals1 or not vals2:
            results[field] = {"status": "insufficient_data"}
            continue
        # numeric fields summary
        try:
            fmean1 = statistics.mean([float(v) for v in vals1])
            fmean2 = statistics.mean([float(v) for v in vals2])
            # relative change
            pct_change = (fmean2 - fmean1) / (abs(fmean1) if fmean1!=0 else 1.0)
            results[field] = {"mean_first": fmean1, "mean_second": fmean2, "pct_change": pct_change}
        except Exception:
            # categorical field, compare top values
            from collections import Counter
            c1 = Counter(vals1).most_common(3)
            c2 = Counter(vals2).most_common(3)
            results[field] = {"top_first": c1, "top_second": c2}
    return {"feature": "metadata_drift", "results": results, "status": "ok"}
