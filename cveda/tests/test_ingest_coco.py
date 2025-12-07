import json
import tempfile
from pathlib import Path
from cveda.ingest import parse_coco
from cveda.schema import sanitize_for_json

def _make_fake_coco(tmpdir: Path):
    images = [
        {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 80},
        {"id": 2, "file_name": "img2.jpg", "width": 200, "height": 120}
    ]
    annotations = [
        {"id": 10, "image_id": 1, "category_id": 3, "bbox": [10, 5, 20, 20]},
        {"id": 11, "image_id": 1, "category_id": 3, "segmentation": [[10,5, 30,5, 30,25, 10,25]]},
        # orphan annotation should be ignored by loader
        {"id": 12, "image_id": 999, "category_id": 4, "bbox": [0,0,1,1]}
    ]
    categories = [{"id": 3, "name": "car"}, {"id": 4, "name": "person"}]
    data = {"images": images, "annotations": annotations, "categories": categories}
    p = tmpdir / "coco.json"
    p.write_text(json.dumps(data))
    # create dummy image files
    for img in images:
        (tmpdir / img["file_name"]).write_bytes(b"\x89PNG\r\n")
    return str(p)

def test_parse_coco(tmp_path):
    coco = _make_fake_coco(tmp_path)
    idx = parse_coco(coco, images_root=str(tmp_path))
    assert "img1.jpg" in idx
    rec = idx["img1.jpg"]
    # annotations should include bbox and polygon converted entries
    assert rec["n_annotations"] >= 2
