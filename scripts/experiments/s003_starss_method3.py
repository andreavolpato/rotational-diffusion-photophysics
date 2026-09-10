"""s003 - plot STARSS method 3 (exp003).

Sweeps the delay between the two 405 nm activation pulses and plots the
normalized two-pulse/one-pulse fluorescence counts vs delay (the method-3
observable, ranging ~1 to 2). Also shows the readout decay for one delay.

Every delay is one stored run under ``$RDP_RUNS`` (see ``store``), so a repeated
sweep costs nothing and the sweep's own figures go to a session folder that
lists the runs behind them.

Run:  python scripts/experiments/s003_starss_method3.py
"""
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from rotational_diffusion_photophysics import store
from rotational_diffusion_photophysics.experiments.exp003_starss_method3 import experiment


def normalized_counts(delays, base: experiment = None):
    """Normalized two-pulse/one-pulse counts for each delay."""
    if base is None:
        base = experiment()
    variants = [replace(base, delay=d) for d in delays]
    return np.array([store.run_cached(p, label=f'delay-{d:.3g}s').normalized_count
                     for p, d in zip(variants, delays)])


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
    example = replace(base, delay=delays[-2])
    res = store.run_cached(example, label=f'delay-{delays[-2]:.3g}s')
    sel = res.t <= base.integration_time
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res.t[sel], res.signals[0, sel], label='two pulses')
    ax.plot(res.t[sel], res.signals_1pulse[0, sel], label='one pulse')
    ax.set_xlabel('Readout time (s)')
    ax.set_ylabel('Signal (a.u.)')
    ax.set_title(f'STARSS method 3 readout (delay = {delays[-2]*1e6:.0f} us)')
    ax.legend()
    fig.tight_layout()

    # The sweep figures span many runs, so they belong to a session, not to one
    # run folder; runs.txt records which runs they were built from.
    sess = store.session('s003_starss_method3')
    store.save_open_figures(sess)
    (sess / 'runs.txt').write_text('\n'.join(
        str(store.run_dir(replace(base, delay=d))) for d in delays))
    print(f'saved -> {sess}')
    plt.show()


if __name__ == '__main__':
    main()
