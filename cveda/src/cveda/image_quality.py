"""
Image quality metrics.

Functions to compute brightness contrast blur and simple noise estimates
and to produce a small per image summary plus aggregated distributions.

Dependencies
- pillow
- numpy

All functions are defensive and return None or empty structures on error.
"""

from typing import Tuple, Optional, Dict, Any, List
from PIL import Image, ImageStat, ImageFilter, ImageOps, UnidentifiedImageError
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


def brightness_mean(path: str) -> Optional[float]:
    """
    Compute mean brightness of an image as the average of L channel in L mode.

    Returns float in [0 255] or None on error.
    """
    try:
        with Image.open(path) as im:
            l = im.convert("L")
            stat = ImageStat.Stat(l)
            return float(stat.mean[0])
    except Exception as e:
        logger.debug("brightness_mean error %s %s", path, e)
        return None


def contrast_rms(path: str) -> Optional[float]:
    """
    Estimate contrast using RMS of pixel luminance.

    Returns RMS float or None on error.
    """
    try:
        with Image.open(path) as im:
            l = im.convert("L")
            stat = ImageStat.Stat(l)
            return float(stat.rms[0])
    except Exception as e:
        logger.debug("contrast_rms error %s %s", path, e)
        return None


def laplacian_variance(path: str) -> Optional[float]:
    """
    Approximate Laplacian variance for blur estimation using a fast kernel.

    Pillow does not expose Laplacian directly so this uses a small kernel via filter
    and computes variance of the result, which correlates with sharpness.

    Returns float or None on error.
    """
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            # 3x3 Laplacian kernel approximation via filter
            kernel = ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1)
            lap = gray.filter(kernel)
            arr = np.asarray(lap, dtype=np.float32)
            var = float(np.var(arr))
            return var
    except Exception as e:
        logger.debug("laplacian_variance error %s %s", path, e)
        return None


def noise_estimate(path: str, sample_size: int = 10000) -> Optional[float]:
    """
    Rough noise estimate using local variance on a random sample of pixels.

    Returns estimated noise standard deviation or None on error.
    """
    try:
        with Image.open(path) as im:
            gray = im.convert("L")
            arr = np.asarray(gray, dtype=np.float32).ravel()
            if arr.size == 0:
                return None
            if arr.size > sample_size:
                # sample uniformly
                idx = np.random.choice(arr.size, sample_size, replace=False)
                s = arr[idx]
            else:
                s = arr
            # use median absolute deviation scaled to std
            mad = np.median(np.abs(s - np.median(s)))
            # convert MAD to approximate std
            std = mad * 1.4826
            return float(std)
    except Exception as e:
        logger.debug("noise_estimate error %s %s", path, e)
        return None


def image_quality_summary(paths: List[str]) -> Dict[str, Any]:
    """
    Compute quality metrics for a list of image absolute paths.

    Returns a dict with per_image metrics and aggregates:
    {
        "per_image": {path: {"brightness": .., "contrast": .., "blur": .., "noise": ..}},
        "aggregates": {"brightness": {"mean": .., "std": ..}, ...}
    }
    """
    per = {}
    brightness = []
    contrast = []
    blur = []
    noise = []
    for p in paths:
        try:
            b = brightness_mean(p)
            c = contrast_rms(p)
            l = laplacian_variance(p)
            n = noise_estimate(p)
            per[p] = {"brightness": b, "contrast": c, "blur": l, "noise": n}
            if b is not None:
                brightness.append(b)
            if c is not None:
                contrast.append(c)
            if l is not None:
                blur.append(l)
            if n is not None:
                noise.append(n)
        except Exception as e:
            logger.debug("image_quality_summary skipping %s %s", p, e)
            per[p] = {"error": str(e)}
    def stats(lst):
        if not lst:
            return {"count": 0}
        a = float(np.mean(lst))
        return {"count": len(lst), "mean": a, "std": float(np.std(lst)), "min": float(np.min(lst)), "max": float(np.max(lst))}
    return {"per_image": per, "aggregates": {"brightness": stats(brightness), "contrast": stats(contrast), "blur": stats(blur), "noise": stats(noise)}}
