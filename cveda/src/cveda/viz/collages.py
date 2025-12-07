"""
Image collage helpers.

This module contains simpler wrappers around viz.plots.make_sample_collage
but is left here for future more advanced collage needs.
"""

from typing import List, Tuple
from .plots import make_sample_collage


def make_collage(image_paths: List[str], grid: Tuple[int, int] = (3, 4), thumb_size: Tuple[int, int] = (128, 128)):
    """
    Return a matplotlib Figure containing a collage for the provided images.

    This is a convenience wrapper that keeps the API consistent.
    """
    return make_sample_collage(image_paths, grid=grid, thumb_size=thumb_size)
