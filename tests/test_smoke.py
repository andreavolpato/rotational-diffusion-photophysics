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
    "rotational_diffusion_photophysics.common",
    "rotational_diffusion_photophysics.models.detection",
    "rotational_diffusion_photophysics.models.diffusion",
    "rotational_diffusion_photophysics.models.fluorophore",
    "rotational_diffusion_photophysics.models.illumination",
    "rotational_diffusion_photophysics.models.starss",
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


@pytest.mark.parametrize("name", ["starss1", "starss2", "starss_sted"])
def test_predefined_systems_run(name):
    from rotational_diffusion_photophysics.models import starss

    system = getattr(starss, name)
    t = np.linspace(0, 1e-3, 10)
    signals = system.detector_signals(t)

    assert signals.shape[1] == t.size
    assert np.all(np.isfinite(signals))


def test_starss3_observable_runs():
    from rotational_diffusion_photophysics.models import starss

    s, _ = starss.starss3_detector_signals(np.array([1e-6, 10e-6]))
    assert s.shape == (2,)
    assert np.all(np.isfinite(s))
