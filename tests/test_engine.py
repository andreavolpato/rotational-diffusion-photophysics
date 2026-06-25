"""End-to-end tests for the rotational-diffusion / photophysics engine.

These exercise the analytic diffusion-kinetics solver through the public
``System`` interface and assert physical invariants rather than golden
numbers, so they stay meaningful if the numerics are refactored.
"""
import numpy as np
import pytest

from rotational_diffusion_photophysics.engine import System
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.utils.signals import anisotropy


@pytest.fixture
def system():
    """A small, fast STARSS-like system shared across tests."""
    numerical_aperture = 1.4
    refractive_index = 1.518
    dt = 5e-3

    detectors = PolarizedDetection(
        polarization=["x", "y"],
        numerical_aperture=numerical_aperture,
        refractive_index=refractive_index,
    )
    lasers = ModulatedLasers(
        power_density=[200, 1000],
        wavelength=[405, 488],
        polarization=["xy", "xy"],
        modulation=[[0, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        time_windows=np.ones(5) * dt,
        time0=0,
        numerical_aperture=numerical_aperture,
        refractive_index=refractive_index,
    )
    diffusion = IsotropicDiffusion(diffusion_coefficient=1 / (6 * 100e-6))

    return System(
        illumination=lasers,
        fluorophore=rsEGFP2_8states,
        diffusion=diffusion,
        detection=detectors,
        lmax=6,
    )


def test_detector_signals_shape_and_finite(system):
    t = np.linspace(0, 25e-3, 100)
    signals = system.detector_signals(t)

    # One row per detector (x and y), one column per requested time point.
    assert signals.shape == (2, t.size)
    assert np.all(np.isfinite(signals))


def test_detector_signals_non_negative(system):
    t = np.linspace(0, 25e-3, 100)
    signals = system.detector_signals(t)

    # Fluorescence intensity cannot be negative (allow tiny numerical noise).
    assert np.all(signals >= -1e-9)


def test_total_population_is_conserved(system):
    t = np.linspace(0, 25e-3, 50)
    system.solve(t)

    # The l=0, m=0 spherical-harmonic coefficient (index 0) summed over all
    # species is the total population, which must stay constant in time:
    # photophysics moves molecules between states but creates/destroys none.
    total_population = system._c[:, 0, :].sum(axis=0)
    expected = float(np.sum(system.fluorophore.starting_populations))

    np.testing.assert_allclose(total_population, expected, rtol=1e-6, atol=1e-9)


def test_anisotropy_within_physical_bounds(system):
    t = np.linspace(0, 25e-3, 100)
    signals = system.detector_signals(t)

    # Anisotropy is only defined where there is fluorescence; before the
    # excitation pulses the signal is ~0 and (s_x - s_y) / (s_x + 2 s_y)
    # is an undefined 0/0. Restrict the check to time points with signal.
    total = signals[0] + 2 * signals[1]
    has_signal = total > 1e-6 * np.max(total)
    assert has_signal.any()

    r = anisotropy(signals)[has_signal]

    # Single-photon fluorescence anisotropy is bounded to [-0.2, 0.4].
    assert np.all(np.isfinite(r))
    assert np.all(r >= -0.2 - 1e-6)
    assert np.all(r <= 0.4 + 1e-6)
