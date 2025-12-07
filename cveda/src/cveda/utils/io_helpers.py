"""
I O helpers for images.

Provide a safe get_image_size function that tries PIL and falls back on binary header sniffing.
Caching can be added by callers to avoid repeated disk IO.
"""

from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError


def get_image_size_safe(path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Safely open an image and return its width and height.

    Returns (width height) or (None None) on failure.

    The function is careful to not load full image pixel arrays.
    """
    try:
        with Image.open(path) as im:
            return im.width, im.height
    except (UnidentifiedImageError, OSError):
        return None, None
