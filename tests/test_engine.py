"""End-to-end tests for the rotational-diffusion / photophysics engine.

These exercise the analytic diffusion-kinetics solver through the public
``System`` interface and assert physical invariants rather than golden
numbers, so they stay meaningful if the numerics are refactored.
"""
import numpy as np
import pytest
from scipy.special import erf

from rotational_diffusion_photophysics.core import solve_evolution
from rotational_diffusion_photophysics.engine_s2 import SystemS2 as System
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


# ---------------------------------------------------------------------------
# Real-SH product correctness (regression guard for the m!=0 multiplication bug)
# ---------------------------------------------------------------------------
# The light-matter "multiply by the photoselection function" operator is built
# from a real-spherical-harmonic triple-product table. A former implementation
# used the *complex* Gaunt rule (single output channel m1+m2) and silently
# dropped the |m1-m2| channel, corrupting every non-axial (x/y linear)
# excitation once the density carried m!=0 content (saturation / sequential
# pulses). See study a40 n030 sec.13. These two tests pin the correct physics.

class _SaturationDye:
    """Minimal 2-state dye, 0 -> 1 by light only (no decay, no switching)."""
    nspecies = 2
    quantum_yield_fluo = np.array([0.0, 1.0])
    starting_populations = [1.0, 0.0]
    wavelength = np.array([488.0])
    cross_section = 1e-16

    def kinetics_matrix(self):
        k = np.zeros((2, 2, 2))
        k[1][1, 0] = self.cross_section
        return k

    def dipole_orientations(self):
        return np.zeros((self.nspecies, 2))


def _ground_after_saturation(polarization, a=3.0, lmax=6):
    """Total ground-state population after dose a = sigma * flux * t."""
    laser = ModulatedLasers(
        power_density=[1e-2], wavelength=[488], polarization=[polarization],
        modulation=[[1]], time_windows=[1e9], time0=0.0,
        numerical_aperture=0.01, refractive_index=1.518,
    )
    detector = PolarizedDetection(
        polarization=["z"], numerical_aperture=0.01, refractive_index=1.518)
    dye = _SaturationDye()
    system = System(fluorophore=dye, diffusion=IsotropicDiffusion(0.0),
                    illumination=laser, detection=detector, lmax=lmax)
    flux = laser.photon_flux[0]
    system.solve(np.array([0.0, a / (dye.cross_section * flux)]))
    return float(system._c[0, 0, -1].real)


@pytest.mark.parametrize("polarization", ["x", "y"])
@pytest.mark.parametrize("dose", [3.0, 30.0])
def test_even_l_reduction_is_exact_under_saturation(polarization, dose):
    # engine_s2.solve restricts the eigenproblem to the even-l subspace. That is
    # exact -- not a weak-field linearization -- because the photoselection
    # function has only even rank and parity is preserved order-by-order by the
    # matrix exponential, so odd l is never reached from an l=0 start at ANY
    # power. Checked adversarially here, deep into saturation and on a non-axial
    # (x/y) polarization, which is where the reduction would break first.
    # The SO(3) analogue is test_engine_so3.test_even_m_reduction_is_exact_under_saturation.
    laser = ModulatedLasers(
        power_density=[1e-2], wavelength=[488], polarization=[polarization],
        modulation=[[1]], time_windows=[1e9], time0=0.0,
        numerical_aperture=1.4, refractive_index=1.518,
    )
    detector = PolarizedDetection(
        polarization=["x", "y"], numerical_aperture=1.4, refractive_index=1.518)
    dye = _SaturationDye()
    system = System(fluorophore=dye, diffusion=IsotropicDiffusion(0.0),
                    illumination=laser, detection=detector, lmax=6)

    M = system.diffusion_kinetics_matrix()
    c0 = np.zeros((dye.nspecies, system._l.size))
    c0[:, 0] = dye.starting_populations
    t = np.array([0.0, dose / (dye.cross_section * laser.photon_flux[0])])

    even = system._l % 2 == 0
    c_full, _, _ = solve_evolution(M[0], c0, t, keep_mask=None)
    c_even, _, _ = solve_evolution(M[0], c0, t, keep_mask=even)

    # The dropped subspace stays empty in the unreduced solution ...
    assert np.abs(c_full[:, ~even, :]).max() < 1e-10
    # ... so reducing the eigenproblem changes nothing.
    np.testing.assert_allclose(c_even, c_full, atol=1e-10)


@pytest.mark.parametrize("polarization", ["x", "y", "z"])
def test_saturation_total_population_is_rotation_invariant(polarization):
    # Orientational hole-burning by a single linear laser depletes the ground
    # state; the *total* remaining ground population is a scalar and therefore
    # cannot depend on the lab-frame polarization direction. It must match the
    # closed form N0(a) = sqrt(pi) erf(sqrt(a)) / (2 sqrt(a)).
    a = 3.0
    analytic = np.sqrt(np.pi) * erf(np.sqrt(a)) / (2 * np.sqrt(a))
    n0 = _ground_after_saturation(polarization, a=a)
    np.testing.assert_allclose(n0, analytic, rtol=2e-4)
