"""
seasonal_coverage

Map image timestamps to seasons by month. Seasons assumed northern hemisphere:
- winter: Dec Feb
- spring: Mar May
- summer: Jun Aug
- autumn: Sep Nov

If timestamps missing returns no_timestamps.
"""

from typing import Dict, Any
from collections import defaultdict
from datetime import datetime

def _parse_ts(ts):
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(int(ts))
        return datetime.fromisoformat(str(ts))
    except Exception:
        try:
            return datetime.strptime(str(ts), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

def _month_to_season(m):
    if m in (12,1,2):
        return "winter"
    if m in (3,4,5):
        return "spring"
    if m in (6,7,8):
        return "summer"
    return "autumn"

def run_seasonal_coverage(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    counts = defaultdict(int)
    examples = defaultdict(list)
    found = False
    for fname, rec in (index or {}).items():
        ts = rec.get("meta", {}).get("timestamp") or rec.get("meta", {}).get("datetime")
        if not ts:
            anns = rec.get("annotations", []) or []
            for ann in anns:
                ts = (ann.get("raw") or {}).get("timestamp")
                if ts:
                    break
        dt = _parse_ts(ts)
        if dt:
            found = True
            season = _month_to_season(dt.month)
            counts[season] += 1
            if len(examples[season]) < cfg.get("sample_limit", 10):
                examples[season].append({"file": fname, "month": dt.month})
    if not found:
        return {"feature": "seasonal_coverage", "status": "no_timestamps"}
    total = sum(counts.values())
    proportions = {k: counts[k]/total for k in counts}
    return {"feature": "seasonal_coverage", "counts": dict(counts), "proportions": proportions, "examples": dict(examples), "status": "ok"}
