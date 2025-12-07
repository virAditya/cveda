import tempfile
from cveda.report.pdf_report import generate_pdf_report

def test_pdf_generation_minimal(tmp_path):
    fake = {
        "index": {},
        "checks": {"completeness":{"n_images":0},"bbox_sanity":{}},
        "distributions": {"class_distribution":{"annotation_counts":{}}, "bbox_statistics":{"overall":{}}, "spatial_heatmaps":{"overall":None}}
    }
    out = tmp_path / "report.pdf"
    path = generate_pdf_report(fake, str(out))
    assert path and out.exists()
