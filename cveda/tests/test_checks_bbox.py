from cveda.checks.bbox_sanity import find_zero_area_boxes, find_inverted_boxes
def test_bbox_zero_and_inverted():
    index = {
        "a.jpg": {"width":100, "height":100, "annotations":[{"class":"c", "bbox":[10,10,10,10]}]},
        "b.jpg": {"width":100, "height":100, "annotations":[{"class":"c", "bbox":[50,60,10,20]}]},
    }
    zero = find_zero_area_boxes(index)
    inv = find_inverted_boxes(index)
    assert len(zero) == 1
    assert len(inv) == 1
