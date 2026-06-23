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

import numpy as np

from rotational_diffusion_photophysics.models import starss

# (name, system, time grid) for the directly-defined System objects.
CASES = {
    "starss1": (starss.starss1, np.linspace(0, 3e-3, 200)),
    "starss2": (starss.starss2, np.linspace(0, 3e-3, 200)),
    "starss_sted": (starss.starss_sted, np.linspace(-1e-9, 12e-9, 200)),
}

# Delays for the STARSS modality-3 derived observable.
STARSS3_DELAYS = np.array([1e-6, 10e-6, 100e-6])


def compute_baselines():
    data = {}
    for name, (system, t) in CASES.items():
        data[f"{name}__time"] = t
        data[f"{name}__signals"] = system.detector_signals(t)

    s, _ = starss.starss3_detector_signals(STARSS3_DELAYS)
    data["starss3__delays"] = STARSS3_DELAYS
    data["starss3__signal"] = s
    return data


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baselines.npz")
    np.savez_compressed(out_path, **compute_baselines())
    print(f"Wrote baselines to {out_path}")


if __name__ == "__main__":
    main()
