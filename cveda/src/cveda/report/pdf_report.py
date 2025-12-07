"""
Simple PDF report generator for CVEDA.

The function generate_pdf_report(audit_result, out_path, config=None, cfg=None)
is intentionally small and defensive. It accepts either `config` or `cfg`
so older callers remain compatible.

The resulting PDF is intentionally concise and human friendly.
"""

from typing import Dict, Any, Optional
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os
import datetime

def _safe_text(obj: Any) -> str:
    """
    Convert basic Python structures to short human readable strings.
    Keep output compact so the PDF stays small.
    """
    if obj is None:
        return "None"
    if isinstance(obj, str):
        return obj
    try:
        return str(obj)
    except Exception:
        return "<unrepresentable>"

def _add_section(story, styles, title: str, body_lines):
    """
    Helper to add a titled section to the document.
    body_lines is an iterable of strings.
    """
    story.append(Paragraph(title, styles.get("Heading2", styles.get("Heading1"))))
    story.append(Spacer(1, 0.08 * inch))
    for line in body_lines:
        story.append(Paragraph(line, styles.get("BodyText")))
        story.append(Spacer(1, 0.02 * inch))
    story.append(Spacer(1, 0.12 * inch))


def generate_pdf_report(audit_result: Dict[str, Any], out_path: str, config: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """
    Generate a basic PDF report.

    Parameters
    - audit_result: dictionary produced by CVEDA run_audit
    - out_path: target filename for the PDF
    - config: optional configuration mapping
    - cfg: legacy alias for config, accepted for compatibility

    Returns
    - the out_path string when PDF was written successfully

    Notes
    - This function intentionally avoids creating new named styles in the
      global stylesheet, this prevents collisions when the report module is
      imported repeatedly during tests or interactive sessions.
    """
    if config is None and cfg is not None:
        config = cfg
    config = config or {}

    # Prepare document
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    doc = SimpleDocTemplate(out_path, pagesize=letter, title="CVEDA Report")
    styles = getSampleStyleSheet()

    story = []
    # Title block
    title = "CVEDA Dataset Audit Report"
    story.append(Paragraph(title, styles.get("Title", styles.get("Heading1"))))
    story.append(Spacer(1, 0.08 * inch))

    meta_line = f"Generated: {datetime.datetime.utcnow().isoformat()} UTC"
    story.append(Paragraph(meta_line, styles.get("BodyText")))
    story.append(Spacer(1, 0.12 * inch))

    # Summary section
    checks = audit_result.get("checks", {})
    summary_lines = []
    # attempt to create a concise human friendly summary
    if isinstance(checks, dict):
        n_images = checks.get("summary", {}).get("n_images", None) if checks.get("summary") else None
        if n_images is not None:
            summary_lines.append(f"Images discovered: {_safe_text(n_images)}")
        # report counts for common checks when present
        completeness = checks.get("completeness", {})
        if isinstance(completeness, dict):
            summary_lines.append(f"Images without annotations: {_safe_text(completeness.get('images_without_annotations', []))}")
            summary_lines.append(f"Total annotations: {_safe_text(completeness.get('n_annotations', 'unknown'))}")
        bbox_sanity = checks.get("bbox_sanity", {})
        if isinstance(bbox_sanity, dict):
            zc = bbox_sanity.get("zero_area", [])
            inv = bbox_sanity.get("inverted", [])
            summary_lines.append(f"Zero area boxes: {len(zc) if isinstance(zc, (list, tuple)) else _safe_text(zc)}")
            summary_lines.append(f"Inverted boxes: {len(inv) if isinstance(inv, (list, tuple)) else _safe_text(inv)}")
    else:
        summary_lines.append("No checks found in audit result")

    _add_section(story, styles, "Summary", summary_lines)

    # Distributions section
    distributions = audit_result.get("distributions", {})
    dist_lines = []
    if isinstance(distributions, dict) and distributions:
        # class distribution is common
        class_dist = distributions.get("class_distribution", {})
        if isinstance(class_dist, dict):
            counts = class_dist.get("annotation_counts") or class_dist.get("counts") or {}
            if isinstance(counts, dict):
                # show top categories
                top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
                if top:
                    dist_lines.append("Top categories and counts")
                    for name, cnt in top:
                        dist_lines.append(f"{_safe_text(name)}: {_safe_text(cnt)}")
                else:
                    dist_lines.append("No category counts available")
        # bbox statistics summary
        bbox_stats = distributions.get("bbox_statistics", {})
        if isinstance(bbox_stats, dict):
            dist_lines.append(f"Median bbox area: {_safe_text(bbox_stats.get('median_area', 'n/a'))}")
            dist_lines.append(f"Median bbox width: {_safe_text(bbox_stats.get('median_width', 'n/a'))}")
    else:
        dist_lines.append("No distributions computed")

    _add_section(story, styles, "Distributions", dist_lines)

    # Add short diagnostics section with highlights from feature outputs when present
    features = audit_result.get("features", {})
    feat_lines = []
    if isinstance(features, dict) and features:
        for fname, output in features.items():
            if not isinstance(output, dict):
                continue
            status = output.get("status", "ok")
            line = f"{fname}: {_safe_text(status)}"
            # if there are small metrics include them
            metrics = output.get("metrics")
            if isinstance(metrics, dict) and metrics:
                # include up to three metric summaries
                items = list(metrics.items())[:3]
                metrics_str = ", ".join(f"{k}={_safe_text(v)}" for k, v in items)
                line = f"{line}, {metrics_str}"
            feat_lines.append(line)
    else:
        feat_lines.append("No feature outputs found")

    _add_section(story, styles, "Feature diagnostics", feat_lines)

    # Footer note
    story.append(Paragraph("For details run the audit with detailed config or inspect the JSON output keys.", styles.get("Italic", styles.get("BodyText"))))
    story.append(Spacer(1, 0.08 * inch))

    # Build document
    try:
        doc.build(story)
    except Exception as exc:
        # If PDF writing fails we raise so callers can capture the issue
        raise

    return out_path
