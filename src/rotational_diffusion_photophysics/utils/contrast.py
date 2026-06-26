"""Contrast (difference) between two conditions of the same experiment.

A *contrast* is the difference of a readout measured under two **independent**
conditions -- e.g. a freely rotating sample (rs2 in solution) vs a static sample
-- and we want the signal-to-noise ratio of that difference.

Because the two conditions are acquired independently, their (shot-noise)
variances add:

    delta      = value_a - value_b
    delta_std  = sqrt(std_a**2 + std_b**2)
    SNR(delta) = delta / delta_std

The helpers are readout-agnostic: pass any value with its std (anisotropy,
polarization, counts, ...). ``contrast`` works on scalars or arrays elementwise.

To compare conditions at equal photon cost, normalize the contrast SNR by the
photon budget -- the **mean** of the total photons emitted in the two conditions:

    SNR_per_photons = SNR(delta) / (0.5 * (photons_a + photons_b) / scale)
"""
from dataclasses import dataclass

import numpy as np

__all__ = [
    "contrast", "contrast_snr", "contrast_snr_per_photons", "Contrast", "contrast_of",
]


@dataclass
class Contrast:
    """The difference between two conditions and its uncertainty.

    ``budget`` (optional) is the mean total photons emitted across the two
    conditions, used to express the contrast SNR per photon budget.
    """
    delta: object        # value_a - value_b (scalar or array)
    delta_std: object    # sqrt(std_a**2 + std_b**2)
    budget: object = None  # mean total emitted photons of the two conditions

    @property
    def snr(self):
        """delta / delta_std (the SNR of the contrast)."""
        return self.delta / self.delta_std

    def snr_per_photons(self, scale=1e6):
        """Contrast SNR normalized by the photon budget (per ``scale`` photons).

        Uses ``budget`` = mean total emitted photons of the two conditions.
        """
        if self.budget is None:
            raise ValueError("no photon budget recorded for this contrast "
                             "(pass results with a total_emitted_photons attribute)")
        return self.snr / (self.budget / scale)

    @property
    def snr_per_Mph(self):
        """Contrast SNR per 1e6 emitted photons (budget = mean of the two)."""
        return self.snr_per_photons(1e6)


def contrast(value_a, std_a, value_b, std_b):
    """Contrast ``a - b`` and its std for two independent conditions.

    Parameters are the readout value and its std for each condition (any readout:
    anisotropy, polarization, counts, ...). Returns ``(delta, delta_std)``;
    operates elementwise on arrays.
    """
    delta = np.asarray(value_a, dtype=float) - np.asarray(value_b, dtype=float)
    delta_std = np.hypot(np.asarray(std_a, dtype=float),
                         np.asarray(std_b, dtype=float))
    return delta, delta_std


def contrast_snr(value_a, std_a, value_b, std_b):
    """SNR of the contrast ``a - b`` = delta / delta_std."""
    delta, delta_std = contrast(value_a, std_a, value_b, std_b)
    return delta / delta_std


def contrast_snr_per_photons(value_a, std_a, value_b, std_b,
                             photons_a, photons_b, scale=1e6):
    """Contrast SNR normalized by the photon budget (per ``scale`` photons).

    The budget is the mean of the total photons emitted in the two conditions,
    ``0.5 * (photons_a + photons_b)``.
    """
    snr = contrast_snr(value_a, std_a, value_b, std_b)
    budget = 0.5 * (photons_a + photons_b)
    return snr / (budget / scale)


def contrast_of(a, b, readout="anisotropy", budget_attr="total_emitted_photons"):
    """Contrast of a ``readout`` between two result-like objects.

    Reads ``getattr(obj, readout)`` and ``getattr(obj, readout + '_std')`` from
    each object -- the convention used by the experiment ``Result`` objects, e.g.
    ``readout='anisotropy'`` uses ``.anisotropy`` and ``.anisotropy_std``,
    ``readout='polarization'`` uses ``.polarization`` and ``.polarization_std``.

    If both objects expose ``budget_attr`` (default ``total_emitted_photons``),
    the resulting :class:`Contrast` records the mean of the two as its photon
    budget, enabling ``.snr_per_Mph`` / ``.snr_per_photons()``.
    """
    va, sa = getattr(a, readout), getattr(a, readout + "_std")
    vb, sb = getattr(b, readout), getattr(b, readout + "_std")
    delta, delta_std = contrast(va, sa, vb, sb)
    budget = None
    if budget_attr and hasattr(a, budget_attr) and hasattr(b, budget_attr):
        budget = 0.5 * (getattr(a, budget_attr) + getattr(b, budget_attr))
    return Contrast(delta, delta_std, budget)
