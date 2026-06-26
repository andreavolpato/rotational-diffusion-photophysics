"""Generate golden baseline outputs for the regression smoke tests.

Run this with the CURRENT (known-good) code to capture reference detector
signals, then commit ``tests/data/baselines.npz``. The regression tests in
``test_regression.py`` compare future output against this baseline, so any
behavior change introduced while refactoring is caught.

Re-run (and review the diff carefully) only when a numerical change is
*intended*:

    python tests/generate_baselines.py
"""
import os
from dataclasses import replace

import numpy as np

from rotational_diffusion_photophysics.experiments.exp001_starss_method1 import experiment as exp001
from rotational_diffusion_photophysics.experiments.exp002_starss_method2 import experiment as exp002
from rotational_diffusion_photophysics.experiments.exp003_starss_method3 import experiment as exp003
from rotational_diffusion_photophysics.experiments.exp030_starss_sted import experiment as exp030

# (name, engine System, time grid) for the single-pulse-scheme experiments.
CASES = {
    "starss1": (exp001().build(), np.linspace(0, 3e-3, 200)),
    "starss2": (exp002().build(), np.linspace(0, 3e-3, 200)),
    "starss_sted": (exp030().build(), np.linspace(-1e-9, 12e-9, 200)),
}

# Delays for the STARSS modality-3 derived observable.
STARSS3_DELAYS = np.array([1e-6, 10e-6, 100e-6])


def starss3_normalized_counts(delays, base: exp003 = None):
    """Method-3 normalized two-pulse/one-pulse counts for each delay."""
    if base is None:
        base = exp003()
    return np.array([replace(base, delay=float(d)).run().normalized_count
                     for d in delays])


def compute_baselines():
    data = {}
    for name, (system, t) in CASES.items():
        data[f"{name}__time"] = t
        data[f"{name}__signals"] = system.detector_signals(t)

    data["starss3__delays"] = STARSS3_DELAYS
    data["starss3__signal"] = starss3_normalized_counts(STARSS3_DELAYS)
    return data


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baselines.npz")
    np.savez_compressed(out_path, **compute_baselines())
    print(f"Wrote baselines to {out_path}")


if __name__ == "__main__":
    main()
