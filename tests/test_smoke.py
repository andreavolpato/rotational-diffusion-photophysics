"""Shallow smoke tests: do the modules import, do the models construct, and
do the predefined systems run end-to-end without error?

These are intentionally broad and cheap. They won't catch numerical
regressions (that is ``test_regression.py``), but they fail loudly if a
refactor breaks imports, wiring, or public constructors.
"""
import importlib

import numpy as np
import pytest

CORE_MODULES = [
    "rotational_diffusion_photophysics",
    "rotational_diffusion_photophysics.engine",
    "rotational_diffusion_photophysics.utils.common",
    "rotational_diffusion_photophysics.utils.signals",
    "rotational_diffusion_photophysics.models.detection",
    "rotational_diffusion_photophysics.models.diffusion",
    "rotational_diffusion_photophysics.models.fluorophore",
    "rotational_diffusion_photophysics.models.illumination",
    "rotational_diffusion_photophysics.experiments.exp001_starss_method1",
    "rotational_diffusion_photophysics.experiments.exp002_starss_method2",
    "rotational_diffusion_photophysics.experiments.exp003_starss_method3",
    "rotational_diffusion_photophysics.experiments.exp010_steady_state",
    "rotational_diffusion_photophysics.experiments.exp020_first_time_is_special",
    "rotational_diffusion_photophysics.experiments.exp030_starss_sted",
]

# experiment modules whose experiment().run() returns a result with `.signals`
EXPERIMENT_MODULES = [
    "exp001_starss_method1",
    "exp002_starss_method2",
    "exp003_starss_method3",
    "exp010_steady_state",
    "exp020_first_time_is_special",
    "exp030_starss_sted",
]


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
    mod = importlib.import_module(
        f"rotational_diffusion_photophysics.experiments.{modname}")
    res = mod.experiment().run()
    assert res.signals.shape[1] == res.t.size
    assert np.all(np.isfinite(res.signals))


def test_starss3_method_runs():
    from rotational_diffusion_photophysics.experiments.exp003_starss_method3 import experiment

    res = experiment(delay=10e-6).run()
    assert np.isfinite(res.normalized_count)
    assert res.normalized_count > 0
