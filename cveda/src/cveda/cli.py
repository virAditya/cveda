"""
Command line interface.

Simple CLI that accepts a dataset path and writes a JSON or PDF report.
This version includes a robust serializer to make the audit result JSON safe.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from .data_io import ImageCollectionLoader
from .api import CVEDA

# try to import numpy for specialized handling
try:
    import numpy as _np  # private alias
except Exception:
    _np = None


def parse_args():
    parser = argparse.ArgumentParser(prog="cveda_cli", description="Run CV dataset audit")
    parser.add_argument("root", help="Dataset root path containing images and optional annotations")
    parser.add_argument("--pdf", help="Write a PDF report to this path", default=None)
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--max-sample", type=int, default=None, help="Limit number of images scanned")
    return parser.parse_args()


def _make_json_safe(obj: Any) -> Any:
    """
    Recursively convert objects into JSON serializable types.

    Handles:
    - numpy arrays and scalars converted to lists and Python numbers
    - pathlib.Path converted to string
    - bytes decoded to utf-8 string when possible
    - nested dicts and lists processed recursively

    For unknown objects we fall back to str(obj)
    """
    # primitives that json already supports
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # numpy arrays and scalars
    if _np is not None:
        if isinstance(obj, _np.ndarray):
            # convert to nested lists, cast to Python scalar types
            try:
                lst = obj.tolist()
                return _make_json_safe(lst)
            except Exception:
                # fallback to flattened list
                try:
                    return _make_json_safe(obj.reshape(-1).tolist())
                except Exception:
                    return str(obj)
        if isinstance(obj, (_np.integer, _np.floating, _np.bool_)):
            return obj.item()

    # pathlib Path
    if isinstance(obj, Path):
        return str(obj)

    # bytes
    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except Exception:
            return str(obj)

    # dict
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            # ensure keys are strings
            key = k if isinstance(k, str) else str(k)
            new[key] = _make_json_safe(v)
        return new

    # list or tuple
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]

    # fallback for other iterables
    try:
        iter(obj)
        return [_make_json_safe(x) for x in obj]
    except Exception:
        return str(obj)


def main():
    args = parse_args()
    root = args.root
    loader = ImageCollectionLoader(root, recursive=args.recursive)
    cveda = CVEDA(loader)
    result = cveda.run_audit(out_pdf=args.pdf)

    # optionally shrink the index for display if max sample requested
    if args.max_sample and isinstance(result.get("index"), dict):
        idx_keys = list(result["index"].keys())[: args.max_sample]
        result["index"] = {k: result["index"][k] for k in idx_keys}

    safe = _make_json_safe(result)
    print(json.dumps(safe, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
