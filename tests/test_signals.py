"""Tests for the signal/SNR helpers in utils.signals.

The variance formulas are validated against a brute-force Monte-Carlo of
independent Poisson channels (the physical model: PBS + Poisson emission ->
independent Poisson counts), both with and without an averaged background.
"""
import numpy as np
import pytest

from rotational_diffusion_photophysics.utils import signals


# (signal_par, signal_perp) expected detected counts, no background
CASES = [(1000, 1000), (2000, 500), (500, 2000), (300, 100), (5000, 4000)]
# (S_par, S_perp, b_par, b_perp, background_averages) with background
BG_CASES = [
    (1000, 800, 200, 200, 1),
    (1000, 800, 200, 200, 10),
    (2000, 500, 500, 500, 4),
    (500, 2000, 300, 300, 2),
    (400, 100, 50, 80, 20),
]
NDRAWS = 2_000_000


def _mc_ratio(observable, mu_M, mu_B, n_bg, seed):
    """Monte-Carlo variance of an observable of the background-subtracted signal.

    Draws measured channels M ~ Poisson(mu_M) and background B ~ Poisson(mu_B)
    (accumulated over n_bg acquisitions), forms S_hat = M - B/n_bg, applies the
    observable, returns its empirical variance.
    """
    rng = np.random.default_rng(seed)
    M = rng.poisson(np.array(mu_M)[:, None], (2, NDRAWS)).astype(float)
    B = rng.poisson(np.array(mu_B)[:, None], (2, NDRAWS)).astype(float)
    S_hat = M - B / n_bg
    return np.nanvar(observable(S_hat))


@pytest.mark.parametrize("mu_par, mu_perp", CASES)
def test_anisotropy_variance_matches_montecarlo(mu_par, mu_perp):
    var_mc = _mc_ratio(lambda s: (s[0] - s[1]) / (s[0] + 2 * s[1]),
                       [mu_par, mu_perp], [0, 0], 1, seed=1)
    var_f = signals.anisotropy_variance([mu_par, mu_perp])
    assert var_f > 0
    assert var_mc / var_f == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("mu_par, mu_perp", CASES)
def test_polarization_variance_matches_montecarlo(mu_par, mu_perp):
    var_mc = _mc_ratio(lambda s: (s[0] - s[1]) / (s[0] + s[1]),
                       [mu_par, mu_perp], [0, 0], 1, seed=2)
    var_f = signals.polarization_variance([mu_par, mu_perp])
    assert var_f > 0
    assert var_mc / var_f == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("S_par, S_perp, b_par, b_perp, n_bg", BG_CASES)
def test_anisotropy_variance_with_background(S_par, S_perp, b_par, b_perp, n_bg):
    # measured channel mean = signal + background; background accumulated n_bg times
    mu_M = [S_par + b_par, S_perp + b_perp]
    mu_B = [n_bg * b_par, n_bg * b_perp]
    var_mc = _mc_ratio(lambda s: (s[0] - s[1]) / (s[0] + 2 * s[1]),
                       mu_M, mu_B, n_bg, seed=3)
    var_f = signals.anisotropy_variance(mu_M, mu_B, n_bg)
    assert var_f > 0
    assert var_mc / var_f == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize("S_par, S_perp, b_par, b_perp, n_bg", BG_CASES)
def test_polarization_variance_with_background(S_par, S_perp, b_par, b_perp, n_bg):
    mu_M = [S_par + b_par, S_perp + b_perp]
    mu_B = [n_bg * b_par, n_bg * b_perp]
    var_mc = _mc_ratio(lambda s: (s[0] - s[1]) / (s[0] + s[1]),
                       mu_M, mu_B, n_bg, seed=4)
    var_f = signals.polarization_variance(mu_M, mu_B, n_bg)
    assert var_f > 0
    assert var_mc / var_f == pytest.approx(1.0, rel=0.03)


def test_background_subtract():
    channels = np.array([1000.0, 800.0])
    backgrounds = np.array([200.0, 160.0])
    s, var = signals.background_subtract(channels, backgrounds, background_averages=10)
    np.testing.assert_allclose(s, channels - backgrounds / 10)
    np.testing.assert_allclose(var, channels + backgrounds / 100)


def test_zero_background_recovers_pure_poisson():
    channels = np.array([500.0, 1500.0])
    s, var = signals.background_subtract(channels)     # backgrounds=None -> zeros
    np.testing.assert_allclose(s, channels)
    np.testing.assert_allclose(var, channels)         # Var = M when no background


def test_detected_counts_scaling():
    sig_in = np.array([[2.0, 1.0], [1.0, 0.5]])
    counts = signals.detected_counts(sig_in, n_molecules=1e6,
                                     collection_efficiency=0.1, lifetime=2e-9, dt=1e-6)
    np.testing.assert_allclose(counts, 1e6 * 0.1 * sig_in * 1e-6 / 2e-9)


def test_counts_snr_pure_shot_noise():
    channels = np.array([100.0, 400.0, 10000.0])
    np.testing.assert_allclose(signals.counts_snr(channels), np.sqrt(channels))


def test_polarization_value():
    ch = np.array([3.0, 1.0])
    assert signals.polarization(ch) == pytest.approx((3 - 1) / (3 + 1))
