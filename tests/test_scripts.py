"""Every runner in ``scripts/experiments`` must at least import.

Importing a runner executes its module body -- its imports, its module-level
constants, its default arguments -- without running the experiment. That is
cheap and it is exactly the check that was missing: ``s004`` pointed at a module
whose file had been renamed to ``... copy.py`` and failed on import, unnoticed,
because nothing in the suite ever loaded the scripts.
"""
import importlib.util
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")                # no display, and no window on import

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts" / "experiments")
                 .glob("s*.py"))


def test_runners_are_present():
    assert SCRIPTS, "no runner scripts found under scripts/experiments"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_runner_imports(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main"), f"{path.name} has no main()"
