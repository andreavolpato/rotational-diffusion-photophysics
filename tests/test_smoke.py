"""Shallow smoke tests: do the modules import, do the models construct, and
do the predefined systems run end-to-end without error?

These are intentionally broad and cheap. They won't catch numerical
regressions (that is ``test_regression.py``), but they fail loudly if a
refactor breaks imports, wiring, or public constructors.
"""
import importlib
import pkgutil
from dataclasses import replace

import numpy as np
import pytest

import rotational_diffusion_photophysics.experiments as _experiments

# Every experiment module in the package, discovered rather than listed: a new
# experiment is covered the day it is added, and a module that cannot be
# imported (a stray filename, a bad import) fails here instead of rotting.
EXPERIMENT_MODULES = sorted(
    m.name for m in pkgutil.iter_modules(_experiments.__path__)
    if m.name.startswith("exp"))

CORE_MODULES = [
    "rotational_diffusion_photophysics",
    "rotational_diffusion_photophysics.core",
    "rotational_diffusion_photophysics.engine_s2",
    "rotational_diffusion_photophysics.engine_so3",
    "rotational_diffusion_photophysics.utils.common",
    "rotational_diffusion_photophysics.utils.signals",
    "rotational_diffusion_photophysics.models.detection",
    "rotational_diffusion_photophysics.models.diffusion",
    "rotational_diffusion_photophysics.models.fluorophore",
    "rotational_diffusion_photophysics.models.illumination",
    "rotational_diffusion_photophysics.store",
] + [f"rotational_diffusion_photophysics.experiments.{name}"
     for name in EXPERIMENT_MODULES]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_modules_import(module):
    assert importlib.import_module(module) is not None


def test_predefined_fluorophores_exist():
    from rotational_diffusion_photophysics.models import fluorophore

    for name in ("rsEGFP2_4states", "rsEGFP2_8states", "atto647N"):
        obj = getattr(fluorophore, name)
        assert obj.nspecies >= 1
        assert np.isclose(np.sum(obj.starting_populations), 1.0)


@pytest.mark.parametrize("modname", EXPERIMENT_MODULES)
def test_experiments_run(modname):
    """Every experiment wires up and produces finite signals.

    Run at coarse resolution (few time points, small basis): what is under test
    is the wiring, not the numerics -- those are pinned in ``test_regression``.
    Full-resolution runs of every experiment would cost minutes.
    """
    mod = importlib.import_module(
        f"rotational_diffusion_photophysics.experiments.{modname}")
    params = replace(mod.experiment(), n_time=16, lmax=2)
    res = params.run()
    assert res.signals.shape[1] == res.t.size
    assert np.all(np.isfinite(res.signals))


def test_starss3_method_runs():
    from rotational_diffusion_photophysics.experiments.exp003_starss_method3 import experiment

    res = experiment(delay=10e-6).run()
    assert np.isfinite(res.normalized_count)
    assert res.normalized_count > 0
