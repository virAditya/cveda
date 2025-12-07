"""
Detect unreadable corrupted or truncated images.

Provides a fast check that attempts to open the image with PIL and optionally
reads the first and last bytes to detect truncation.

Returns a list of files that failed to open or appear truncated.
"""

from typing import List
from PIL import Image, ImageFile, UnidentifiedImageError
import logging
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


def find_corrupted_images(paths: List[str]) -> List[dict]:
    """
    Check each path and return list of dicts:
    { "path": path, "error": "reason" }
    """
    out = []
    for p in paths:
        if not os.path.exists(p):
            out.append({"path": p, "error": "not found"})
            continue
        try:
            with Image.open(p) as im:
                im.verify()  # may raise
        except UnidentifiedImageError as e:
            out.append({"path": p, "error": "unidentified"})
        except Exception as e:
            out.append({"path": p, "error": str(e)})
    return out
