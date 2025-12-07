import os
from cveda.report.pdf_report import generate_pdf_report

def test_generate_pdf_minimal(tmp_path):
    audit = {
        "checks": {
            "completeness": {"n_images": 1, "images_without_annotations": []},
            "bbox_sanity": {"zero_area": [], "inverted": []},
            "corrupt": {"files": []}
        },
        "distributions": {"class_distribution": {"annotation_counts": {}}},
        "index": {}
    }
    out = tmp_path / "report.pdf"
    p = generate_pdf_report(audit, str(out), cfg={})
    assert os.path.exists(p)
    assert os.path.getsize(p) > 0
