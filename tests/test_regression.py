"""Golden / characterization tests for safe refactoring.

These compare current detector output against committed baselines captured
from known-good code (see ``generate_baselines.py``). A refactor that changes
the numerics — intentionally or not — will fail here. If the change is
intended, regenerate the baseline and review the diff before committing.
"""
import os

import numpy as np
import pytest

from rotational_diffusion_photophysics.models import starss
from tests.generate_baselines import CASES, STARSS3_DELAYS

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "data", "baselines.npz")

# Tight enough to catch real behavior changes, loose enough to tolerate
# platform/BLAS floating-point jitter.
RTOL = 1e-6
ATOL = 1e-12


@pytest.fixture(scope="module")
def baselines():
    if not os.path.exists(BASELINE_PATH):
        pytest.skip(
            f"Baseline {BASELINE_PATH} missing; run tests/generate_baselines.py"
        )
    return np.load(BASELINE_PATH)


@pytest.mark.parametrize("name", list(CASES))
def test_detector_signals_match_baseline(name, baselines):
    system, t = CASES[name]
    np.testing.assert_allclose(t, baselines[f"{name}__time"])

    signals = system.detector_signals(t)
    np.testing.assert_allclose(
        signals, baselines[f"{name}__signals"], rtol=RTOL, atol=ATOL
    )


def test_starss3_observable_matches_baseline(baselines):
    np.testing.assert_allclose(STARSS3_DELAYS, baselines["starss3__delays"])

    s, _ = starss.starss3_detector_signals(STARSS3_DELAYS)
    np.testing.assert_allclose(
        s, baselines["starss3__signal"], rtol=RTOL, atol=ATOL
    )
