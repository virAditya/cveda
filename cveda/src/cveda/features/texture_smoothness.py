"""
texture_smoothness

Purpose
-------
Estimate high frequency content and smoothness of images to detect over-smoothing,
excessive denoising, synthetic images or low-detail photos.

Heuristic
---------
Compute Laplacian variance on a downsampled grayscale image per sample.
Low Laplacian variance suggests strong smoothing.

Exports
-------
run_texture_smoothness(index, config=None) -> dict

Config options
------------
- sample_limit int default 100 number of images to sample
- downscale float default 0.25 fraction to downsample image for speed
- laplacian_kernel_size int default 3 not used directly uses convolution via numpy
- smooth_threshold float optional threshold to mark image as overly smooth

Return format
------------
{
  "feature": "texture_smoothness",
  "n_sampled": int,
  "mean_laplacian_variance": float,
  "low_detail_examples": [ {"file": fname, "lap_var": value}, ... ],
  "status": "ok"
}

Complexity
----------
O(S * p) where S is sample_limit and p is pixels per sampled image.
Memory O(p) per image.

Failure modes
-------------
- images without abs_path or unreadable images are skipped
- very small images may produce noisy laplacian values

Next steps
----------
- use local entropy or wavelet high frequency energy
- compare to pretrained feature distributions for GAN detection
"""

from typing import Dict, Any, List
import numpy as np
from PIL import Image
import os
import statistics

def _laplacian_variance(arr: np.ndarray) -> float:
    """
    Compute a simple Laplacian via discrete kernel and return variance of result.
    """
    # kernel [[0,1,0],[1,-4,1],[0,1,0]]
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=float)
    # pad arr
    padded = np.pad(arr, ((1,1),(1,1)), mode='reflect')
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            patch = padded[i:i+3, j:j+3]
            out[i,j] = (patch * kernel).sum()
    return float(np.var(out))

def run_texture_smoothness(index: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute per-image Laplacian variance on a sample and return summary statistics.

    Parameters
    ----------
    index dict: canonical index
    config dict:
      - sample_limit int default 100
      - downscale float default 0.25
      - smooth_threshold float optional, default None. If set images below threshold are flagged.

    Returns
    -------
    dict as described above
    """
    cfg = config or {}
    sample_limit = int(cfg.get("sample_limit", 100))
    downscale = float(cfg.get("downscale", 0.25))
    smooth_threshold = cfg.get("smooth_threshold")  # may be None

    lap_vars = []
    examples = []

    for fname, rec in (index or {}).items():
        if len(lap_vars) >= sample_limit:
            break
        path = rec.get("abs_path")
        if not path or not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("L")
                w,h = im.size
                small = im.resize((max(1,int(w*downscale)), max(1,int(h*downscale))))
                arr = np.asarray(small, dtype=float)/255.0
                lv = _laplacian_variance(arr)
                lap_vars.append(lv)
                examples.append({"file": fname, "laplacian_variance": lv})
        except Exception:
            continue

    if not lap_vars:
        return {"feature": "texture_smoothness", "status": "no_images_processed"}

    mean_lv = float(statistics.mean(lap_vars))
    low_detail_examples = []
    if smooth_threshold is not None:
        low_detail_examples = [e for e in examples if e["laplacian_variance"] <= smooth_threshold][:cfg.get("sample_limit", 20)]
    else:
        # heuristically mark bottom 5 percent as low detail
        cutoff = np.percentile(lap_vars, 5)
        low_detail_examples = [e for e in examples if e["laplacian_variance"] <= cutoff][:cfg.get("sample_limit", 20)]

    return {
        "feature": "texture_smoothness",
        "n_sampled": len(lap_vars),
        "mean_laplacian_variance": mean_lv,
        "low_detail_examples": low_detail_examples,
        "status": "ok"
    }
