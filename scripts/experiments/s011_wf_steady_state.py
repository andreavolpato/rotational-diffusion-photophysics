"""s011 - widefield steady-state: sweep the 405 power and plot counts vs time.

Re-runs the ``exp011_wf_steady_state`` experiment (rsEGFP2, widefield pulse
scheme) for each factor in ``power_405_scale``; both 405 beams
(``power_405_1`` and ``power_405_2``) are scaled by that factor. The per-bin
detected counts of the two cross-polarized channels are collected for every
scale (kept in memory) and plotted against time, coloured by the 405 scale.

Run:  python scripts/experiments/s011_wf_steady_state.py
"""
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from rotational_diffusion_photophysics.experiments.exp011_wf_steady_state import experiment

power_405_scale = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def sweep_405(scales=power_405_scale, base: experiment = None):
    """Re-run the experiment with both 405 powers scaled by each factor in
    ``scales``. Returns (scales, t, counts) with counts of shape
    (n_scale, 2, n_time) [scale, channel(par/perp), time]."""
    if base is None:
        base = experiment()
    scales = np.asarray(scales, dtype=float)
    t = None
    counts = []
    for s in scales:
        res = replace(base,
                      power_405_1=base.power_405_1 * s,
                      power_405_2=base.power_405_2 * s).run()
        t = res.t
        counts.append(res.counts)
    return scales, t, np.array(counts)


def main(scales=power_405_scale, base: experiment = None):
    scales, t, counts = sweep_405(scales, base)
    
    print(counts.shape)

    norm = LogNorm(vmin=scales.min(), vmax=scales.max())
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ch, ax, title in zip((0, 1), axes, ('parallel (x)', 'perpendicular (y)')):
        for s, c in zip(scales, counts):
            ax.plot(t, c[ch], color=cmap(norm(s)))
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
    axes[0].set_ylabel('Detected counts / bin')
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label='405 power scale')
    fig.suptitle('s011 widefield steady-state — counts vs time over 405 power scale')
    plt.show()


if __name__ == '__main__':
    main()
