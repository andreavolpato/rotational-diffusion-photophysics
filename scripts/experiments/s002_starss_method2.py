"""s002 - plot STARSS method 2 (exp002).

Runs ``exp002`` and plots the pulse scheme, the detector signals, and the
time-resolved anisotropy. The run and its figures are stored under ``$RDP_RUNS``
(see ``store``); an identical re-run is loaded from the store instead of
recomputed.

Run:  python scripts/experiments/s002_starss_method2.py
"""
import matplotlib.pyplot as plt

from rotational_diffusion_photophysics import store
from rotational_diffusion_photophysics.experiments.exp002_starss_method2 import experiment
from rotational_diffusion_photophysics.plot.plot_pulse_scheme import plot_pulse_scheme


def main(params: experiment = None, label='default'):
    if params is None:
        params = experiment()
    res = store.run_cached(params, label=label)
    t = res.t

    plt.figure(1)
    plot_pulse_scheme(res.system, xlim=[-1e-3, params.t_stop],
                      ylim=[1e2, 1e6], yscale='log')
    plt.title('STARSS method 2 - pulse scheme')

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax0.plot(t, res.signals[0], label='parallel')
    ax0.plot(t, res.signals[1], label='perpendicular')
    ax0.set_ylabel('Signal (a.u.)')
    ax0.legend()
    ax0.set_title('STARSS method 2 (rsEGFP2)')
    ax1.plot(t, res.anisotropy, color='k')
    ax1.set_ylabel('Anisotropy')
    ax1.set_xlabel('Time (s)')

    fig.tight_layout()

    run = store.run_dir(params)
    store.save_open_figures(run)
    print(f'saved -> {run}')
    plt.show()


if __name__ == '__main__':
    main()
