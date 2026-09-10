"""s020 - plot the "first time is special" population dynamics (exp020).

Runs ``exp020`` (alternating 405/488 pulse train on rsEGFP2) and plots the state
populations over time together with the pulse scheme. The run and its figures
are stored under ``$RDP_RUNS`` (see ``store``); an identical re-run is loaded
from the store instead of recomputed.

Run:  python scripts/experiments/s020_first_time_is_special.py
"""
import numpy as np
import matplotlib.pyplot as plt

from rotational_diffusion_photophysics import store
from rotational_diffusion_photophysics.experiments.exp020_first_time_is_special import experiment
from rotational_diffusion_photophysics.plot.plot_pulse_scheme import plot_pulse_scheme


def main(params: experiment = None, label='default'):
    if params is None:
        params = experiment()
    res = store.run_cached(params, label=label)

    plt.figure(1)
    plt.plot(res.t, res.populations.transpose())
    plt.legend(np.arange(res.populations.shape[0]))
    plt.xlabel('Time (s)')
    plt.ylabel('Population')
    plt.title('first time is special (pka << ph)')

    plt.figure(2)
    plot_pulse_scheme(res.system, xlim=[0, params.t_end], ylim=[0, 1500])
    plt.title('Pulse scheme')

    run = store.run_dir(params)
    store.save_open_figures(run)
    print(f'saved -> {run}')
    plt.show()


if __name__ == '__main__':
    main()
