import tempfile
from pathlib import Path
from PIL import Image
from cveda.data_io import ImageCollectionLoader
from cveda.annotations import clamp_bbox_to_image, swap_inverted_bbox

def create_image(path, size=(64,64), color=(128,128,128)):
    Image.new("RGB", size, color).save(path)

def test_image_loader_and_annotation_norm(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    img = d / "img1.jpg"
    create_image(img)
    loader = ImageCollectionLoader(str(d), recursive=False)
    index = loader.build_index()
    assert "img1.jpg" in index
    rec = index["img1.jpg"]
    assert rec["width"] == 64 and rec["height"] == 64

def test_clamp_and_swap():
    bbox = [-10, -5, 200, 300]
    clamped = clamp_bbox_to_image(bbox, 100, 100)
    assert clamped[0] >= 0 and clamped[1] >= 0 and clamped[2] <= 100 and clamped[3] <= 100
    inverted = [50, 60, 10, 20]
    fixed = swap_inverted_bbox(inverted)
    assert fixed[0] <= fixed[2] and fixed[1] <= fixed[3]
