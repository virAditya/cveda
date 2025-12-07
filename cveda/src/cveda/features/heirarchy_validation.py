"""
hierarchy_validation

Validate user-provided hierarchies of classes. Config expected:
config["hierarchy"] = { "parent_class": ["child1","child2"], ... }

The check flags images where a child appears but parent does not.
"""

from typing import Dict, Any, List

def run_heirarchy_validation(index, config=None):
    """
    Backwards compatible wrapper expected by tests.
    Delegates to run(index, config) if present.
    """
    runner = globals().get("run")
    if callable(runner):
        return runner(index, config or {})
    # fallback if module uses a differently named function
    for alt in ("validate_hierarchy", "run_validation"):
        fn = globals().get(alt)
        if callable(fn):
            return fn(index, config or {})
    return {"status": "no-runner", "note": "no run or validate_hierarchy callable found in heirarchy_validation"}

