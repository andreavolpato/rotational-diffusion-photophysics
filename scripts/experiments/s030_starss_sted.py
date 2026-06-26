"""s030 - plot the STARSS-STED anisotropy decay (exp030).

Runs ``exp030`` (cross-polarized 640 nm excitation + 775 nm STED on ATTO647N)
and plots the pulse scheme, the time-resolved anisotropy, and the two detector
signals.

Run:  python scripts/experiments/s030_starss_sted.py
"""
import matplotlib.pyplot as plt

from rotational_diffusion_photophysics.experiments.exp030_starss_sted import experiment
from rotational_diffusion_photophysics.plot.plot_pulse_scheme import plot_pulse_scheme


def main(params: experiment = None):
    if params is None:
        params = experiment()
    res = params.run()
    t = res.t

    xlim = [params.t_start, params.t_stop]
    ylim = [1e3, 5e9]
    yscale = 'log'

    fig = plt.figure(1)
    gs = fig.add_gridspec(3, 1)
    ax0 = fig.add_subplot(gs[0, 0])
    plot_pulse_scheme(res.system, xlim=xlim, ylim=ylim, yscale=yscale)
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax1.plot(t, res.anisotropy)
    ax1.set_ylabel('Anisotropy')
    ax1.set_xlabel('Time (s)')
    ax1.sharex(ax0)
    ax2.plot(t, res.signals.transpose())
    ax2.set_ylabel('Signal (a.u.)')
    ax2.set_xlabel('Time (s)')
    ax2.sharex(ax0)

    plt.show()


if __name__ == '__main__':
    main()
