import tempfile
from pathlib import Path
from cveda.ingest import parse_yolo

def test_parse_yolo(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    # create dummy label for image named dog.txt
    txt = labels / "dog.txt"
    # class 0, cx cy w h normalized
    txt.write_text("0 0.5 0.5 0.4 0.4\n")
    # create a dummy image file to allow proper conversion
    (tmp_path / "dog.jpg").write_bytes(b"\xFF\xD8\xFF")
    idx = parse_yolo(str(labels), images_root=str(tmp_path))
    assert "dog" in idx
    rec = idx["dog"]
    # should have annotation
    assert rec["n_annotations"] >= 1
    ann = rec["annotations"][0]
    assert ann["type"] == "bbox"
