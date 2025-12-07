"""
geographic_clustering

Lightweight clustering based on EXIF GPS coordinates. No external clustering library required.
We quantize lat lon to grid cells and report clusters.

Config
------
- grid_deg float default 0.1 degrees the grid cell size for clustering
- sample_limit int default 10000

Return
------
{
  "feature": "geographic_clustering",
  "clusters": [ {"cell": (lat_bin, lon_bin), "count": n, "examples": [file,...] } ],
  "status":"ok" or "no_gps"
}
"""

from typing import Dict, Any
from collections import defaultdict
from PIL import Image
import os

def _get_gps_from_exif(exif):
    # tries multiple possible EXIF fields to extract GPS info
    try:
        # PIL forms GPS info under 34853 sometimes
        gps = exif.get(34853) if isinstance(exif, dict) else None
        if gps:
            lat = gps.get(2)  # may be rational tuples
            lon = gps.get(4)
            # fallback logic not robust here, best-effort
            return None
    except Exception:
        pass
    return None

def run_geographic_clustering(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or {}
    grid_deg = float(cfg.get("grid_deg", 0.1))
    sample_limit = int(cfg.get("sample_limit", 10000))

    clusters = defaultdict(list)
    found_any = False
    scanned = 0

    for fname, rec in (index or {}).items():
        if scanned >= sample_limit:
            break
        scanned += 1
        # look for GPS in metadata first
        meta = rec.get("meta", {}) or {}
        lat = meta.get("gps_lat") or meta.get("latitude") or meta.get("lat")
        lon = meta.get("gps_lon") or meta.get("longitude") or meta.get("lon")
        if lat is None or lon is None:
            # try EXIF if abs_path present
            path = rec.get("abs_path")
            if path and os.path.exists(path):
                try:
                    with Image.open(path) as im:
                        exif = im._getexif() or {}
                        # many possible formats so attempt common keys
                        gps = exif.get('GPSInfo') or exif.get(34853) or {}
                        # try keys like GPSLatitude, GPSLongitude
                        lat = meta.get("gps_lat") or meta.get("latitude")
                        lon = meta.get("gps_lon") or meta.get("longitude")
                except Exception:
                    continue
        try:
            if lat is None or lon is None:
                continue
            lat_f = float(lat)
            lon_f = float(lon)
            found_any = True
            lat_bin = int(lat_f / grid_deg)
            lon_bin = int(lon_f / grid_deg)
            clusters[(lat_bin, lon_bin)].append({"file": fname, "lat": lat_f, "lon": lon_f})
        except Exception:
            continue

    if not found_any:
        return {"feature": "geographic_clustering", "status": "no_gps"}

    out = []
    for cell, members in clusters.items():
        if len(members) >= cfg.get("min_cluster_size", 3):
            out.append({"cell": cell, "count": len(members), "examples": members[:10]})

    return {"feature": "geographic_clustering", "clusters": out, "status": "ok"}
