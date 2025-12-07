"""
time_of_day_mapping

Bucket images into time of day categories using timestamp metadata in record['meta']
or annotation raw fields. Buckets are: night, morning, afternoon, evening.

If no timestamps found the function returns a no_timestamps status.
"""

from typing import Dict, Any
from datetime import datetime
from collections import defaultdict

def _parse_ts(ts):
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(int(ts))
        s = str(ts)
        return datetime.fromisoformat(s)
    except Exception:
        # try common format
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
    return None

def _hour_bucket(hour):
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"

def run_time_of_day_mapping(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    counts = defaultdict(int)
    examples = defaultdict(list)
    found = False
    for fname, rec in (index or {}).items():
        ts = rec.get("meta", {}).get("timestamp") or rec.get("meta", {}).get("datetime")
        if ts is None:
            # try annotation raw timestamps
            anns = rec.get("annotations", []) or []
            for ann in anns:
                raw = ann.get("raw", {}) or {}
                ts = raw.get("timestamp") or raw.get("time")
                if ts:
                    break
        dt = _parse_ts(ts)
        if dt:
            found = True
            bucket = _hour_bucket(dt.hour)
            counts[bucket] += 1
            if len(examples[bucket]) < cfg.get("sample_limit", 10):
                examples[bucket].append({"file": fname, "timestamp": dt.isoformat()})
    if not found:
        return {"feature": "time_of_day_mapping", "status": "no_timestamps"}
    total = sum(counts.values())
    proportions = {k: counts[k]/total for k in counts}
    return {"feature": "time_of_day_mapping", "counts": dict(counts), "proportions": proportions, "examples": dict(examples), "status": "ok"}
