"""Signal-to-noise ratio helpers for simulated fluorescence experiments.

The detected fluorescence is a stream of photons with Poisson statistics, so the
SNR of any observable follows from (i) converting the engine's arbitrary-unit
``detector_signals`` into expected photon counts and (ii) propagating Poisson
noise (Var = mean per channel) through the observable.

Count conversion
----------------
``System.detector_signals`` returns, per channel ``i``,

    s_i(t) = Phi_fluo * p_exc(t) * g_i(t)

i.e. the fluorescence quantum yield times the *excited-state population* times an
angular collection factor. This is a population-like quantity, not a flux. The
detected photon flux is the rate of the radiative de-excitation reaction
S1 -> S0, which is the excited population times the de-excitation rate 1/tau.
Hence the expected counts in a time bin of width ``dt`` from ``N`` molecules
collected with efficiency ``eta`` are

    mu_i(t) = N * eta * s_i(t) * dt / tau

where ``tau`` is the excited-state lifetime. ``Phi_fluo`` is already inside ``s_i``.

Background model
----------------
Every experiment measures two channels -- parallel and perpendicular -- and also
measures the background of each channel. The background may be averaged over
``background_averages`` acquisitions to reduce its noise. The SNR helpers ingest:

    channels            measured parallel and perpendicular channel counts
    backgrounds         measured parallel and perpendicular background counts
    background_averages number to divide the background counts by

All measurements are modelled as independent Poisson:

    M       measured channel (signal + background):           Var = M
    B       background counts:                                Var = B
    S_hat = M - B / background_averages          (background-subtracted signal)
    Var(S_hat) = M + B / background_averages**2   (measurement + background-estimate noise)

As ``background_averages -> inf`` the background estimate becomes noise-free and
Var -> M, but M still carries the shot noise of the background photons it contains.

Channel independence
--------------------
The parallel and perpendicular channels (e.g. behind a polarizing beam splitter)
are independent Poisson: Var = mean, Cov = 0 (Poisson thinning of a Poisson total).
Monte-Carlo confirms the variance formulas below to ~0.1%, including with
background. This is *not* what ``common.anisotropy_variance`` computes (it mixes a
multinomial covariance with Poisson marginals and takes abs of the numerator, both
bugs); we re-derive correctly here. See a30 n020.

All count inputs are in detected photons, with the two channels along axis 0:
``[parallel, perpendicular]``. Counts in distinct time bins are treated as
independent Poisson samples.
"""
import numpy as np

__all__ = [
    "detected_counts",
    "background_subtract",
    "counts_snr",
    "anisotropy",
    "anisotropy_variance",
    "anisotropy_snr",
    "polarization",
    "polarization_variance",
    "polarization_snr",
]


# ----------------------------------------------------------------------------
# Count conversion
# ----------------------------------------------------------------------------
def detected_counts(signals,
                    n_molecules, lifetime, dt,
                    collection_efficiency=0.05):
    """Expected Poisson photon counts per channel per time bin.

    Parameters
    ----------
    signals : ndarray, shape (nchannels, ntime)
        Output of ``System.detector_signals``.
    n_molecules : float
        Number of emitters in the probed volume.
    collection_efficiency : float
        Overall detection efficiency eta (optics x detector QE), in [0, 1].
        For a typical confocal microscope at high NA is about 2 to 10 percent.
    lifetime : float
        Excited-state lifetime tau [s] (the de-excitation flux rate is 1/tau).
        Use ``fluorophore.lifetime_on`` for the fluorescent state.
    dt : float
        Integration time per bin [s]. For a uniform time grid, ``t[1] - t[0]``.

    Returns
    -------
    ndarray, same shape as ``signals``
        Expected detected photon counts mu_i(t).
    """
    return n_molecules * collection_efficiency * signals * dt / lifetime


# ----------------------------------------------------------------------------
# Background subtraction
# ----------------------------------------------------------------------------
def background_subtract(channels, backgrounds, background_averages=1):
    """Background-subtracted signal and its Poisson variance, per channel.

    Parameters
    ----------
    channels : array_like
        Measured channel counts (signal + background), [parallel, perpendicular].
    backgrounds : array_like
        Measured background counts, [parallel, perpendicular].
    background_averages : float, optional
        Number to divide the background counts by (>= 1). Default 1.

    Returns
    -------
    (signal, variance) : tuple of ndarray
        ``signal   = channels - backgrounds / background_averages``
        ``variance = channels + backgrounds / background_averages**2``
    """
    channels = np.asarray(channels, dtype=float)
    backgrounds = np.asarray(backgrounds, dtype=float)
    signal = channels - backgrounds / background_averages
    variance = channels + backgrounds / background_averages ** 2
    return signal, variance


def _ratio_variance(value, variance, perp_factor):
    """Delta-method variance of (v_par - v_perp) / (v_par + perp_factor*v_perp)
    for independent channels, given per-channel signal ``value`` and Poisson
    ``variance`` (both arrays with channel along axis 0: [parallel, perpendicular]).
    """
    v_par, v_perp = value[0], value[1]
    var_par, var_perp = variance[0], variance[1]
    N = v_par - v_perp                       # signed numerator
    D = v_par + perp_factor * v_perp
    Var_N = var_par + var_perp
    Var_D = var_par + perp_factor ** 2 * var_perp
    Cov_N_D = var_par - perp_factor * var_perp
    return (
        Var_N / D ** 2
        - 2 * N * Cov_N_D / D ** 3
        + N ** 2 * Var_D / D ** 4
    )


# ----------------------------------------------------------------------------
# Counts SNR (per channel)
# ----------------------------------------------------------------------------
def counts_snr(channels, backgrounds, background_averages=1):
    """SNR of each detection channel after background subtraction.

    With zero background this is pure shot noise, ``SNR = sqrt(M)``.
    """
    signal, variance = background_subtract(channels, backgrounds, background_averages)
    return signal / np.sqrt(variance)


# ----------------------------------------------------------------------------
# Anisotropy  r = (I_par - I_perp) / (I_par + 2 I_perp)
# ----------------------------------------------------------------------------
def anisotropy(channels, backgrounds, background_averages=1):
    """Fluorescence anisotropy from background-subtracted channels."""
    signal, _ = background_subtract(channels, backgrounds, background_averages)
    return (signal[0] - signal[1]) / (signal[0] + 2 * signal[1])


def anisotropy_variance(channels, backgrounds, background_averages=1):
    """Poisson variance of the anisotropy (independent channels)."""
    signal, variance = background_subtract(channels, backgrounds, background_averages)
    return _ratio_variance(signal, variance, perp_factor=2)


def anisotropy_snr(channels, backgrounds, background_averages=1):
    """SNR of the fluorescence anisotropy."""
    r = anisotropy(channels, backgrounds, background_averages)
    var = anisotropy_variance(channels, backgrounds, background_averages)
    return r / np.sqrt(var)


# ----------------------------------------------------------------------------
# Polarization  p = (I_par - I_perp) / (I_par + I_perp)
# (anisotropy "without the 2" on the perpendicular channel)
# ----------------------------------------------------------------------------
def polarization(channels, backgrounds, background_averages=1):
    """Polarization from background-subtracted channels."""
    signal, _ = background_subtract(channels, backgrounds, background_averages)
    return (signal[0] - signal[1]) / (signal[0] + signal[1])


def polarization_variance(channels, backgrounds, background_averages=1):
    """Poisson variance of the polarization (independent channels)."""
    signal, variance = background_subtract(channels, backgrounds, background_averages)
    return _ratio_variance(signal, variance, perp_factor=1)


def polarization_snr(channels, backgrounds, background_averages=1):
    """SNR of the polarization."""
    p = polarization(channels, backgrounds, background_averages)
    var = polarization_variance(channels, backgrounds, background_averages)
    return p / np.sqrt(var)
