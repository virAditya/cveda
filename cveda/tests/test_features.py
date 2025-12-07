import importlib
import pkgutil
import cveda.features as features_pkg

def test_all_feature_runners_exist_and_return_dict():
    modules = [name for _, name, _ in pkgutil.iter_modules(features_pkg.__path__)]
    assert modules, "No feature modules found under cveda.features"
    for name in modules:
        mod = importlib.import_module(f"cveda.features.{name}")
        runner_name = f"run_{name}"
        runner = getattr(mod, runner_name, None)
        assert callable(runner), f"{name} missing callable {runner_name}"
        out = runner({}, {})
        assert isinstance(out, dict), f"{name} runner should return dict"
        assert "feature" in out or out.get("status") is not None
