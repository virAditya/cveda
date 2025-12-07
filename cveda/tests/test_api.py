from cveda.api import CVEDA

class FakeLoader:
    def __init__(self):
        self.root = "."
    def build_index(self):
        return {
            "img1.jpg": {
                "file_name": "img1.jpg",
                "abs_path": None,
                "width": 128,
                "height": 128,
                "annotations": [],
                "meta": {}
            }
        }

def test_cveda_runs_minimal():
    loader = FakeLoader()
    c = CVEDA(loader)
    res = c.run_audit(out_pdf=None, config={})
    assert isinstance(res, dict)
    assert "index" in res
    assert "features" in res
    assert "checks" in res
