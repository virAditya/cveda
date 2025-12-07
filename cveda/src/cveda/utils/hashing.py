"""
Perceptual hashing helpers.

This module uses the imagehash library when available and falls back to
a simple average hash if not present.

The functions are designed to compute a compact hash that can be compared
with hamming distance for fast duplicate detection.
"""

from typing import Optional
try:
    import imagehash
    from PIL import Image
    _IMAGEHASH_AVAILABLE = True
except Exception:
    _IMAGEHASH_AVAILABLE = False
    from PIL import Image


def phash_image(path: str, hash_size: int = 8) -> Optional[str]:
    """
    Compute a perceptual hash string for the image at path.

    Parameters
    path str absolute path to image
    hash_size int size used by imagehash library default 8

    Returns hex string representation of hash or None on error.
    """
    try:
        img = Image.open(path).convert("RGB")
        if _IMAGEHASH_AVAILABLE:
            h = imagehash.phash(img, hash_size=hash_size)
            return h.__str__()
        else:
            # fallback quick average hash
            small = img.resize((hash_size, hash_size), Image.BILINEAR).convert("L")
            pixels = list(small.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if p > avg else "0" for p in pixels)
            hexstr = "%0*x" % ((len(bits) + 3) // 4, int(bits, 2))
            return hexstr
    except Exception:
        return None
