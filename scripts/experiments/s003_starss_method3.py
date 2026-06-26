"""s003 - plot STARSS method 3 (exp003).

Sweeps the delay between the two 405 nm activation pulses and plots the
normalized two-pulse/one-pulse fluorescence counts vs delay (the method-3
observable, ranging ~1 to 2). Also shows the readout decay for one delay.

Run:  python scripts/experiments/s003_starss_method3.py
"""
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from rotational_diffusion_photophysics.experiments.exp003_starss_method3 import experiment


def normalized_counts(delays, base: experiment = None):
    """Normalized two-pulse/one-pulse counts for each delay."""
    if base is None:
        base = experiment()
    return np.array([replace(base, delay=d).run().normalized_count for d in delays])


def main(base: experiment = None):
    if base is None:
        base = experiment()
    delays = np.array([0.5e-6, 1e-6, 10e-6, 100e-6, 200e-6, 499e-6])
    counts = normalized_counts(delays, base)

    plt.figure(figsize=(6, 4))
    plt.plot(delays, counts, 'o-')
    plt.xscale('log')
    plt.xlabel('Delay (s)')
    plt.ylabel('Normalized counts')
    plt.title('STARSS method 3 (rsEGFP2)')
    plt.tight_layout()

    # Example readout decay (two-pulse vs one-pulse) for a single delay.
    res = replace(base, delay=delays[-2]).run()
    sel = res.t <= base.integration_time
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.t[sel], res.signals[0, sel], label='two pulses')
    ax.plot(res.t[sel], res.signals_1pulse[0, sel], label='one pulse')
    ax.set_xlabel('Readout time (s)')
    ax.set_ylabel('Signal (a.u.)')
    ax.set_title(f'STARSS method 3 readout (delay = {delays[-2]*1e6:.0f} us)')
    ax.legend()
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
