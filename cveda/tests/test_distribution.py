from cveda.distribution.class_distribution import compute_class_distribution
def test_class_distribution_basic():
    index = {
        "a.jpg": {"annotations":[{"class":"cat"},{"class":"dog"}]},
        "b.jpg": {"annotations":[{"class":"cat"}]},
    }
    out = compute_class_distribution(index)
    assert out["annotation_counts"]["cat"] == 2
    assert out["annotation_counts"]["dog"] == 1
