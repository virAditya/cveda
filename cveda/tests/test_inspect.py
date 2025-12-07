from pathlib import Path
from cveda.inspect import inspect_dataset
from PIL import Image
import numpy as np

def test_inspect_mask_folder(tmp_path):
    p = tmp_path
    masks = p / "masks"
    masks.mkdir()
    # create a multiclass mask
    arr = np.zeros((5,5), dtype=np.uint8)
    arr[0:2, 0:2] = 2
    arr[2:4, 2:4] = 3
    from PIL import Image
    Image.fromarray(arr).save(masks / "dummy.png")
    res = inspect_dataset(str(p))
    assert res["has_masks"] is True
    assert res["has_multiclass_masks"] is True
