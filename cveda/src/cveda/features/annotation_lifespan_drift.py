"""
annotation_lifespan_drift

If annotation timestamps exist in ann['raw']['timestamp'] or record['meta'],
compute count per time bucket and highlight major changes. This is a lightweight
temporal drift detector using month buckets.

Notes:
- timestamps should be ISO8601 strings or unix epoch numbers.
"""

from typing import Dict, Any
from collections import defaultdict
from datetime import datetime
import math

def _parse_ts(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return datetime.utcfromtimestamp(int(x))
        except Exception:
            return None
    s = str(x)
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def run_annotation_lifespan_drift(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    config:
      - bucket: 'month' or 'year' default 'month'
      - sample_limit: int default 10

    Returns:
      {
        "feature": "annotation_lifespan_drift",
        "time_histogram": {"YYYY-MM": count, ...},
        "top_changes": [ {"period": p, "count": c, "pct_change": x} ],
        "status": "ok" or note if no timestamps
      }
    """
    cfg = config or {}
    bucket = cfg.get("bucket", "month")
    counter = defaultdict(int)
    any_ts = False

    for fname, rec in (index or {}).items():
        for ann in (rec.get("annotations", []) or []):
            raw = ann.get("raw", {}) or {}
            ts = raw.get("timestamp") or raw.get("time") or ann.get("time")
            dt = _parse_ts(ts)
            if not dt:
                # fallback to record meta
                meta_ts = rec.get("meta", {}).get("timestamp")
                dt = _parse_ts(meta_ts)
            if dt:
                any_ts = True
                if bucket == "month":
                    key = f"{dt.year:04d}-{dt.month:02d}"
                else:
                    key = f"{dt.year:04d}"
                counter[key] += 1

    if not any_ts:
        return {"feature": "annotation_lifespan_drift", "status": "no_timestamps", "message": "No timestamp metadata found"}

    # compute sorted histogram and simple pct change between adjacent buckets
    items = sorted(counter.items())
    changes = []
    last = None
    for k, v in items:
        if last is not None:
            prev_v = last
            pct = (v - prev_v) / prev_v if prev_v != 0 else float("inf")
            changes.append({"period": k, "count": v, "pct_change": pct})
        last = v

    # top changes by absolute pct
    top_changes = sorted(changes, key=lambda x: abs(x["pct_change"]), reverse=True)[:cfg.get("sample_limit", 10)]

    return {
        "feature": "annotation_lifespan_drift",
        "time_histogram": dict(items),
        "top_changes": top_changes,
        "status": "ok"
    }
