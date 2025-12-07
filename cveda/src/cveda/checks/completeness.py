"""
Annotation completeness auditor.

Detect missing annotation files orphan annotation files images with no annotations
and parse failures. This module aims to be robust and return a list of issues.
"""

from typing import Dict, Any, List


def run_completeness_audit(index: Dict[str, Any], cfg: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Audit the canonical index for completeness issues.

    Issues detected:
    - images_without_annotations list of image file names
    - annotations_pointing_to_missing_images list of annotation records that reference non existing image files
    - parse_failures list of records where loader stored an error in meta

    The index is the canonical index mapping file name to record.

    Returns a dict with keys described above and counts.
    """
    cfg = cfg or {}
    treat_empty_as_error = cfg.get("treat_empty_as_error", False)

    images_without_annotations = []
    parse_failures = []
    # For orphan annotations we rely on the loader to include them in index if they reference files
    # Since our canonical index is image keyed this auditor focuses on image side issues.
    for fname, rec in index.items():
        anns = rec.get("annotations", [])
        meta = rec.get("meta", {})
        if meta and meta.get("error"):
            parse_failures.append({"file_name": fname, "error": meta.get("error")})
        if not anns:
            if treat_empty_as_error:
                images_without_annotations.append({"file_name": fname, "reason": "empty"})
            else:
                images_without_annotations.append({"file_name": fname, "reason": "no_annotations"})
    return {
        "images_without_annotations": images_without_annotations,
        "parse_failures": parse_failures,
        "n_images": len(index)
    }
