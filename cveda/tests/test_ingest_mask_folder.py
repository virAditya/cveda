from pathlib import Path
from PIL import Image
import numpy as np
from cveda.ingest import parse_mask_folder

def _create_mask(path: Path, arr):
    img = Image.fromarray(arr.astype('uint8'))
    img.save(path)

def test_parse_mask_folder(tmp_path):
    imgs = tmp_path / "images"
    masks = tmp_path / "masks"
    imgs.mkdir()
    masks.mkdir()
    # create image
    img_path = imgs / "a.jpg"
    img_path.write_bytes(b"\xFF\xD8\xFF")
    # binary mask
    mask = masks / "a.png"
    arr = np.zeros((10,10), dtype=np.uint8)
    arr[2:6, 3:7] = 255
    _create_mask(mask, arr)
    idx = parse_mask_folder(str(imgs), str(masks))
    assert "a" in idx
    rec = idx["a"]
    assert rec["n_annotations"] >= 1
    ann = rec["annotations"][0]
    assert ann["type"] == "mask"
