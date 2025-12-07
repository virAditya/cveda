"""
Top level orchestration API for CVEDA.

This module implements the CVEDA class which orchestrates a dataset audit:
- builds a canonical index using the project's ImageCollectionLoader
- runs core checks and distribution computations
- dynamically discovers and runs feature modules placed in src/cveda/features
- optionally emits a human friendly PDF report via report.pdf_report.generate_pdf_report

Key runtime features
- Feature modules discovered at runtime under cveda.features
- Feature execution runs in a process pool by default
- Per feature timeout and optional disk cache
- Defensive execution so one broken feature does not stop the whole audit
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Callable, Tuple
import logging
import importlib
import pkgutil
import json
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import cpu_count
from pathlib import Path
from collections import OrderedDict

from .data_io import ImageCollectionLoader, discover_splits, build_index_for_split

# core checks and distribution modules. If any import fails we will capture the error during runtime
try:
    from .checks.completeness import run_completeness_audit
except Exception:  # pragma: no cover
    run_completeness_audit = None

try:
    from .checks.bbox_sanity import run_bbox_sanity_checks
except Exception:  # pragma: no cover
    run_bbox_sanity_checks = None

try:
    from .distribution.class_distribution import compute_class_distribution
except Exception:  # pragma: no cover
    compute_class_distribution = None

try:
    from .distribution.bbox_statistics import compute_bbox_statistics
except Exception:  # pragma: no cover
    compute_bbox_statistics = None

try:
    from .distribution.spatial_heatmap import compute_spatial_heatmaps
except Exception:  # pragma: no cover
    compute_spatial_heatmaps = None

try:
    from .report.pdf_report import generate_pdf_report
except Exception:  # pragma: no cover
    generate_pdf_report = None

logger = logging.getLogger(__name__)


def _discover_feature_modules() -> Dict[str, Any]:
    """
    Discover and import modules under the cveda.features package.

    Returns a mapping of feature short name to the imported module object.
    Modules that fail to import are skipped and logged.
    """
    modules: Dict[str, Any] = {}
    try:
        import cveda.features as features_pkg  # type: ignore
    except Exception:
        logger.debug("No cveda.features package found", exc_info=True)
        return modules

    prefix = features_pkg.__name__ + "."
    for finder, module_name, ispkg in pkgutil.iter_modules(features_pkg.__path__, prefix):
        short = module_name.split(".")[-1]
        try:
            m = importlib.import_module(module_name)
            modules[short] = m
        except Exception:
            logger.debug("Failed to import feature module %s", module_name, exc_info=True)
    return modules


def _index_fingerprint(index: Dict[str, Any], keep_keys: Tuple[str, ...] = ("file_name", "abs_path", "width", "height"), max_items: int = 1000) -> str:
    """
    Compute a small deterministic fingerprint for the index to use for caching.

    The fingerprint uses a stable JSON serialization of a limited number of
    index entries and then SHA256 hashes the result.
    """
    try:
        items = []
        i = 0
        for fname, rec in (index or {}).items():
            if i >= max_items:
                break
            small = {k: rec.get(k) for k in keep_keys}
            items.append((fname, small))
            i += 1
        s = json.dumps(items, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(repr(index).encode("utf-8")).hexdigest()


def _cache_load(cache_dir: Path, key: str):
    path = cache_dir / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        logger.debug("Cache load failed for %s", str(path), exc_info=True)
        return None


def _cache_save(cache_dir: Path, key: str, payload):
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        logger.debug("Failed to save cache for %s", key, exc_info=True)


def _sanitize_for_json(obj):
    """
    Recursively convert numpy scalars and nd arrays and pathlib Paths to native
    Python types. This keeps the final audit result JSON serializable.
    """
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if _np is not None:
        if hasattr(_np, "ndarray") and isinstance(obj, _np.ndarray):
            try:
                return _sanitize_for_json(obj.tolist())
            except Exception:
                return str(obj)
        if hasattr(_np, "generic") and isinstance(obj, _np.generic):
            try:
                return obj.item()
            except Exception:
                return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    # fallback
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _feature_worker(module_name: str, runner_name: str, index: Dict[str, Any], cfg_slice: Dict[str, Any]):
    """
    This function runs inside a worker process. It imports the requested module
    by name, resolves the runner callable then executes it.

    The returned value is a JSON friendly dict or an error dict on failure.
    """
    try:
        mod = importlib.import_module(module_name)
        runner = getattr(mod, runner_name, None)
        if not callable(runner):
            runner = getattr(mod, "run", None)
        if not callable(runner):
            return {"status": "no-runner", "note": f"No runner {runner_name} or run callable found in {module_name}"}
        out = runner(index, cfg_slice or {})
        return _sanitize_for_json(out)
    except Exception as e:
        import traceback
        logger.debug("Feature worker exception for %s", module_name, exc_info=True)
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


class CVEDA:
    """
    High level dataset auditor.

    Example
    -------
    loader = ImageCollectionLoader("/path/to/dataset", recursive=True)
    cveda = CVEDA(loader)
    result = cveda.run_audit(out_pdf="report.pdf")

    Attributes
    ----------
    loader
        The ImageCollectionLoader instance or equivalent that builds the canonical index.
    _feature_modules
        Mapping of discovered feature short name to module object
    """

    def __init__(self, loader_or_root: Any):
        """
        Accept either:
        - an ImageCollectionLoader instance
        - any object with a callable build_index method
        - a string path to dataset root

        This allows tests to pass fake loader objects.
        """
        # If object already looks like a loader (duck typing)
        if hasattr(loader_or_root, "build_index") and callable(getattr(loader_or_root, "build_index")):
            self.loader = loader_or_root
        else:
            # treat string or path-like as dataset root
            self.loader = ImageCollectionLoader(str(loader_or_root), recursive=True)
        self._feature_modules = _discover_feature_modules()

    def run_audit(self, out_pdf: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the full audit pipeline and return a structured dictionary with results.

        Parameters
        ----------
        out_pdf : Optional[str]
            If provided the function will attempt to write a PDF report to this path.
            If PDF generation fails the error is recorded and execution continues.
        config : Optional[Dict[str, Any]]
            Optional configuration dictionary. Supported keys:
            - features_parallel : bool default True
            - max_workers : int default number of CPUs
            - feature_timeout : int seconds default 300
            - feature_cache : bool default True
            - cache_dir : str default ".cveda_cache"
            - features_to_run : list of feature short names to run
            - features : dict mapping feature_name to feature specific config

        Returns
        -------
        result : dict
            A dictionary containing:
            - index : the canonical index returned by the loader
            - checks : results of core checks
            - distributions : results of distribution computations
            - splits : information about dataset splits if discovered
            - features : mapping feature_name -> feature output dict
            - report_path : path to generated PDF or None
        """
        cfg = config or {}
        result: Dict[str, Any] = OrderedDict()

        # 1) Build canonical index
        index = {}
        try:
            logger.info("Building dataset index")
            index = self.loader.build_index()
        except Exception as e:
            logger.exception("Index build failed")
            raise

        result["index"] = index

        # 2) Run core checks and distribution modules with per step protection
        checks: Dict[str, Any] = {}
        distributions: Dict[str, Any] = {}

        # completeness
        if callable(run_completeness_audit):
            try:
                checks["completeness"] = run_completeness_audit(index, cfg.get("completeness", {}))
            except Exception:
                logger.exception("Completeness audit failed")
                checks["completeness"] = {"status": "error", "error": "completeness audit failed"}
        else:
            checks["completeness"] = {"status": "unavailable", "note": "completeness check not installed"}

        # bbox sanity
        if callable(run_bbox_sanity_checks):
            try:
                checks["bbox_sanity"] = run_bbox_sanity_checks(index, cfg.get("bbox", {}))
            except Exception:
                logger.exception("BBox sanity audit failed")
                checks["bbox_sanity"] = {"status": "error", "error": "bbox sanity failed"}
        else:
            checks["bbox_sanity"] = {"status": "unavailable", "note": "bbox sanity check not installed"}

        # distributions
        if callable(compute_class_distribution):
            try:
                distributions["class_distribution"] = compute_class_distribution(index, cfg.get("distribution", {}))
            except Exception:
                logger.exception("Class distribution failed")
                distributions["class_distribution"] = {"status": "error", "error": "class distribution failed"}
        else:
            distributions["class_distribution"] = {"status": "unavailable", "note": "class distribution not installed"}

        if callable(compute_bbox_statistics):
            try:
                distributions["bbox_statistics"] = compute_bbox_statistics(index, cfg.get("distribution", {}))
            except Exception:
                logger.exception("BBox statistics failed")
                distributions["bbox_statistics"] = {"status": "error", "error": "bbox statistics failed"}
        else:
            distributions["bbox_statistics"] = {"status": "unavailable", "note": "bbox statistics not installed"}

        if callable(compute_spatial_heatmaps):
            try:
                distributions["spatial_heatmaps"] = compute_spatial_heatmaps(index, cfg.get("distribution", {}))
            except Exception:
                logger.exception("Spatial heatmaps failed")
                distributions["spatial_heatmaps"] = {"status": "error", "error": "spatial heatmaps failed"}
        else:
            distributions["spatial_heatmaps"] = {"status": "unavailable", "note": "spatial heatmaps not installed"}

        result["checks"] = checks
        result["distributions"] = distributions

        # 3) Discover dataset splits and build per split indices if present
        try:
            splits = discover_splits(self.loader.root)
            if splits:
                indices_by_split = {}
                for name, path in splits.items():
                    try:
                        indices_by_split[name] = build_index_for_split(str(path), recursive=True)
                    except Exception:
                        logger.exception("Failed building index for split %s", name)
                        indices_by_split[name] = {}
                result["splits"] = {"found": True, "names": list(indices_by_split.keys()), "counts": {k: len(v) for k, v in indices_by_split.items()}}
            else:
                result["splits"] = {"found": False}
        except Exception:
            logger.exception("Split discovery failed")
            result["splits"] = {"status": "error", "error": "split discovery failed"}

        # 4) Run discovered feature modules with parallel execution, timeout and caching
        features_out: Dict[str, Any] = {}
        features_cfg = cfg.get("features", {}) or {}

        parallel_cfg = {
            "enabled": bool(cfg.get("features_parallel", True)),
            "max_workers": int(cfg.get("max_workers", cpu_count())),
            "feature_timeout": int(cfg.get("feature_timeout", 300)),
            "feature_cache": bool(cfg.get("feature_cache", True)),
            "cache_dir": str(cfg.get("cache_dir", ".cveda_cache"))
        }

        use_parallel = parallel_cfg["enabled"]
        max_workers = max(1, parallel_cfg["max_workers"])
        feature_timeout = max(1, parallel_cfg["feature_timeout"])
        cache_enabled = bool(parallel_cfg["feature_cache"])
        cache_dir = Path(parallel_cfg["cache_dir"])

        # allow running subset of features if requested
        requested_features = cfg.get("features_to_run")
        feature_items = list(self._feature_modules.items())
        if requested_features:
            feature_items = [(name, self._feature_modules[name]) for name in requested_features if name in self._feature_modules]

        index_fp = _index_fingerprint(index)

        if use_parallel and feature_items:
            tasks = []
            for name, mod in feature_items:
                module_name = getattr(mod, "__name__", f"cveda.features.{name}")
                runner_name = f"run_{name}"
                tasks.append((name, module_name, runner_name, features_cfg.get(name, {})))

            # submit tasks into process pool
            with ProcessPoolExecutor(max_workers=max_workers) as exec_:
                future_to_meta = {}
                for feat_name, module_name, runner_name, feat_cfg in tasks:
                    cache_key = None
                    if cache_enabled:
                        cfg_ser = json.dumps(feat_cfg or {}, sort_keys=True, default=str)
                        raw_key = f"{feat_name}:{index_fp}:{cfg_ser}"
                        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
                        cached = _cache_load(cache_dir, f"{feat_name}_{cache_key}")
                        if cached is not None:
                            features_out[feat_name] = cached
                            continue
                    fut = exec_.submit(_feature_worker, module_name, runner_name, index, feat_cfg)
                    future_to_meta[fut] = (feat_name, cache_key)

                # collect results with timeouts
                for fut in as_completed(list(future_to_meta.keys())):
                    feat_name, cache_key = future_to_meta[fut]
                    try:
                        res = fut.result(timeout=feature_timeout)
                    except TimeoutError:
                        logger.exception("Feature %s timed out after %s seconds", feat_name, feature_timeout)
                        res = {"status": "error", "error": "timeout"}
                    except Exception:
                        logger.exception("Feature %s failed during execution", feat_name)
                        res = {"status": "error", "error": "execution failed"}
                    features_out[feat_name] = res
                    if cache_enabled and cache_key:
                        try:
                            _cache_save(cache_dir, f"{feat_name}_{cache_key}", res)
                        except Exception:
                            logger.debug("Cache save failed for feature %s", feat_name, exc_info=True)
        else:
            # sequential execution fallback
            for name, mod in feature_items:
                try:
                    module_name = getattr(mod, "__name__", f"cveda.features.{name}")
                    feat_cfg = features_cfg.get(name, {})
                    cache_key = None
                    if cache_enabled:
                        cfg_ser = json.dumps(feat_cfg or {}, sort_keys=True, default=str)
                        raw_key = f"{name}:{index_fp}:{cfg_ser}"
                        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
                        cached = _cache_load(cache_dir, f"{name}_{cache_key}")
                        if cached is not None:
                            features_out[name] = cached
                            continue
                    runner = getattr(mod, f"run_{name}", None) or getattr(mod, "run", None)
                    if not callable(runner):
                        features_out[name] = {"status": "no-runner", "note": f"No run_{name} or run callable found in module {name}"}
                        continue
                    out = runner(index, feat_cfg)
                    out = _sanitize_for_json(out)
                    features_out[name] = out
                    if cache_enabled and cache_key:
                        _cache_save(cache_dir, f"{name}_{cache_key}", out)
                except Exception:
                    logger.exception("Feature %s execution failed", name)
                    features_out[name] = {"status": "error", "error": "feature execution failed"}

        result["features"] = features_out

        # 5) Lightweight health summary
        try:
            n_images = len(index)
            score = 100 if n_images > 0 else 0
            result["checks"].setdefault("summary", {"score": score, "n_images": n_images})
        except Exception:
            result["checks"].setdefault("summary", {"score": 0, "n_images": 0})

        # 6) Optionally generate PDF report gracefully
        report_path = None
        if out_pdf:
            if callable(generate_pdf_report):
                try:
                    report_path = generate_pdf_report(result, out_pdf, cfg.get("report", {}))
                except Exception:
                    logger.exception("PDF generation failed")
                    report_path = None
            else:
                logger.warning("PDF report generator not available, skipping PDF write")
                report_path = None

        result["report_path"] = report_path
        return result
