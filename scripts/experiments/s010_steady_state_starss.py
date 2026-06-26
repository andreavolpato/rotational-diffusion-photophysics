"""s010 - Cross-polarized steady-state experiment with rsEGFP2.

Two-phase pulse scheme. Phase durations are fractions of ``total_time``; any
leftover (1 - phase1_fraction - phase2_fraction) is dead time at the end (both
lasers off). A ``settling_fraction`` of ``total_time`` is cut from the start of
each phase, and the remaining part of the phase is the measurement window.
  Phase 1 (background): only 488 nm on. After settling, the measurement window
      is integrated to give the BACKGROUND counts per channel.
  Phase 2 (signal): 488 nm stays on and 405 nm is switched on too. After
      settling, the measurement window gives the CHANNEL counts per channel.

By default the two beams are cross-polarized to each other; fluorescence is
collected on two cross-polarized detectors (X, Y). The 405-induced signal is the
background-subtracted (channels - backgrounds), from which the anisotropy and
polarization and their photon-shot-noise SNR are computed.

The whole experiment is described by a single ``S010Params`` dataclass, so a
parameter sweep is just "vary one field". Use :func:`sweep`::

    base = S010Params()
    pw, snr = sweep("power_488", [0.5, 1, 2, 4],
                    observable=measured_anisotropy_snr)

This is a prototyping script (a20 "graduation" step 1): everything is built
inline against the engine. Once the parameters settle, ``S010Params`` lifts
directly into a registered ``Experiment`` factory in
``rotational_diffusion_photophysics.experiments``.

Run:  python scripts/experiments/s010_cross_polarized_steady_state.py
"""
from dataclasses import replace
from typing import Callable, Sequence

import numpy as np

from rotational_diffusion_photophysics.experiments.exp010_steady_state import experiment, results
from rotational_diffusion_photophysics.utils.contrast import contrast_of
from rotational_diffusion_photophysics.utils.format import _human


# ----------------------------------------------------------------------------
# Observables (reduce a single Result to the quantity you sweep)
# ----------------------------------------------------------------------------
def measured_anisotropy(res: results) -> float:
    return res.anisotropy


def measured_polarization(res: results) -> float:
    return res.polarization


def measured_anisotropy_snr(res: results) -> float:
    return res.anisotropy_snr


def measured_polarization_snr(res: results) -> float:
    return res.polarization_snr


def measured_signal_counts(res: results) -> float:
    """Total background-subtracted counts (both channels)."""
    return float(res.signal.sum())


# ----------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------
def sweep(param: str,
          values: Sequence,
          base: experiment = None,
          observable: Callable = measured_anisotropy):
    """Vary one ``S010Params`` field over ``values`` and reduce each run.

    Returns ``(values, results)`` as arrays. ``observable`` maps a single
    :class:`Result` to the scalar/array you want to collect.
    """
    if base is None:
        base = experiment()
    if not hasattr(base, param):
        raise AttributeError(f"S010Params has no field '{param}'")
    values = np.asarray(values)
    results = np.array([observable(replace(base, **{param: v}).run())
                        for v in values])
    return values, results


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def main(params: experiment = None):
    import matplotlib.pyplot as plt

    if params is None:
        params = experiment()
    res = params.run()
    print(f"background counts (x, y) = {res.backgrounds[0]:.4g}, {res.backgrounds[1]:.4g}")
    print(f"channel    counts (x, y) = {res.channels[0]:.4g}, {res.channels[1]:.4g}")
    print(f"anisotropy   r = {res.anisotropy:.4f} +/- {res.anisotropy_std:.4f} "
          f"(SNR {res.anisotropy_snr:.1f})")
    print(f"polarization p = {res.polarization:.4f} +/- {res.polarization_std:.4f} "
          f"(SNR {res.polarization_snr:.1f})")

    t = res.t
    fig, (ax0, ax_tot, ax1) = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: per-bin counts over the pulse scheme, with the two measurement windows.
    ax0.plot(t, res.counts[0], label='counts (parallel)')
    ax0.plot(t, res.counts[1], label='counts (perpendicular)')
    ax0.axvspan(t[res.background_window][0], t[res.background_window][-1],
                color='0.7', alpha=0.3, label='background window (488)')
    ax0.axvspan(t[res.channel_window][0], t[res.channel_window][-1],
                color='C2', alpha=0.2, label='channel window (488+405)')
    ax0.axvline(params.phase1_duration, color='k', lw=0.8, ls='--')
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Detected counts / bin')
    ax0.set_title('Pulse scheme & counts')
    ax0.legend(fontsize=8)

    # Middle: total integrated photons per channel, background vs channel window,
    # with Poisson (shot-noise) error bars sqrt(counts).
    x = np.arange(2)
    w = 0.38
    ax_tot.bar(x - w / 2, res.backgrounds, w, yerr=np.sqrt(res.backgrounds),
               capsize=5, color='0.6', label='background (488)')
    ax_tot.bar(x + w / 2, res.channels, w, yerr=np.sqrt(res.channels),
               capsize=5, color='C2', label='channel (488+405)')
    ax_tot.set_xticks(x)
    ax_tot.set_xticklabels(['parallel', 'perpendicular'])
    ax_tot.set_ylabel('total detected photons (window)')
    ax_tot.set_title('Total photons: channel vs background')
    ax_tot.legend(fontsize=8)

    # Right: the measured signal (anisotropy & polarization) as bars +/- std.
    labels = ['anisotropy r', 'polarization p']
    values = [res.anisotropy, res.polarization]
    errors = [res.anisotropy_std, res.polarization_std]
    snrs = [res.anisotropy_snr, res.polarization_snr]
    bars = ax1.bar(labels, values, yerr=errors, capsize=8,
                   color=['C0', 'C3'], alpha=0.8)
    ax1.axhline(0, color='0.7', lw=0.8)
    ax1.set_ylabel('value +/- std (shot noise)')
    ax1.set_title('Measured signal & SNR')
    # total fluorescence photons emitted over the whole experiment, shown in
    # parentheses next to each SNR.
    emitted = _human(res.total_emitted_photons)
    for bar, val, snr in zip(bars, values, snrs):
        ax1.annotate(f'SNR {snr:.0f} ({emitted} ph)',
                     (bar.get_x() + bar.get_width() / 2, val),
                     ha='center', va='bottom' if val >= 0 else 'top',
                     xytext=(0, 6 if val >= 0 else -6), textcoords='offset points')

    fig.suptitle('s010 two-phase cross-polarized experiment (rsEGFP2)')
    fig.tight_layout()
    plt.show()


def main_sweep(param='power_488', values=(0.5, 1.0, 2.0, 4.0, 8.0),
               observable=measured_anisotropy_snr):
    """Example sweep: a measured observable vs one parameter."""
    import matplotlib.pyplot as plt

    values, ys = sweep(param, values, observable=observable)
    plt.figure(figsize=(6, 4))
    plt.plot(values, ys, 'o-')
    plt.xlabel(param)
    plt.ylabel(observable.__name__)
    plt.title(f's010 sweep: {observable.__name__} vs {param}')
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXPERIMENT SET
# Define the experiments to simulate and compare here. Each entry is a
# label -> S010Params(...). Edit freely; main_set() runs them all and produces
# the three comparison figures.
# ============================================================================
EXPERIMENTS = {
    "rs2 488c + 405p":  experiment(tau=30e-9, pol_405='x', pol_488='xy'),
    "100nm 488c + 405p":  experiment(tau=100e-6, pol_405='x', pol_488='xy'),
    "static 488c + 405p":  experiment(pol_405='x', pol_488='xy'),
    "rs2 488p + 405c":  experiment(tau=30e-9, pol_405='xy', pol_488='x', power_405=5e3*0.05, power_488=5e3*0.2),
    "100nm 488p + 405c":  experiment(tau=100e-6, pol_405='xy', pol_488='x', power_405=5e3*0.05, power_488=5e3*0.2),
    "static 488p + 405c":  experiment(pol_405='xy', pol_488='x', power_405=5e3*0.05, power_488=5e3*0.2),
    "rs2 488p + 405p": experiment(tau=30e-9, pol_405='y', pol_488='x'),
    "100nm 488p + 405p": experiment(tau=100e-6, pol_405='y', pol_488='x'),
    "static 488p + 405p": experiment(pol_405='y', pol_488='x'),
}


def run_all(experiments: dict = None) -> dict:
    """Run every experiment in ``experiments`` (default EXPERIMENTS) -> {label: Result}."""
    if experiments is None:
        experiments = EXPERIMENTS
    return {label: params.run() for label, params in experiments.items()}


# ----------------------------------------------------------------------------
# Comparison plots across an experiment set
# ----------------------------------------------------------------------------
def _observable_bar_figure(results, value_attr, std_attr, snr_attr, ylabel, title, color):
    """Bar plot of a measured observable +/- std across experiments.

    On top of each bar: SNR, total emitted photons, and SNR per 1M emitted
    photons (a photon-budget efficiency figure of merit).
    """
    import matplotlib.pyplot as plt

    names = list(results)
    vals = [getattr(results[n], value_attr) for n in names]
    errs = [getattr(results[n], std_attr) for n in names]
    fig, ax = plt.subplots(figsize=(1.7 * len(names) + 2, 5))
    bars = ax.bar(names, vals, yerr=errs, capsize=6, color=color, alpha=0.8)
    ax.axhline(0, color='0.7', lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, n in zip(bars, names):
        res = results[n]
        val = getattr(res, value_attr)
        snr = getattr(res, snr_attr)
        emitted = res.total_emitted_photons
        snr_per_Mph = snr / (emitted / 1e6) if emitted else float('nan')
        up = val >= 0
        ax.annotate(f'{snr_per_Mph:.2f} SNR/1M ph\n(SNR {snr:.0f}, {_human(emitted)} ph)',
                    (bar.get_x() + bar.get_width() / 2, val),
                    ha='center', va='bottom' if up else 'top',
                    xytext=(0, 8 if up else -8), textcoords='offset points', fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout()
    return fig


def plot_polarization_bars(results: dict):
    """Bar plot of polarization +/- std, with SNR, photons, and SNR/1M ph on top."""
    return _observable_bar_figure(
        results, 'polarization', 'polarization_std', 'polarization_snr',
        'polarization p +/- std', 's010 polarization across experiments', 'C3')


def plot_anisotropy_bars(results: dict):
    """Bar plot of anisotropy +/- std, with SNR, photons, and SNR/1M ph on top."""
    return _observable_bar_figure(
        results, 'anisotropy', 'anisotropy_std', 'anisotropy_snr',
        'anisotropy r +/- std', 's010 anisotropy across experiments', 'C0')


def plot_counts_bars(results: dict):
    """Grouped bar plot: the four channel/background counts per experiment."""
    import matplotlib.pyplot as plt

    names = list(results)
    x = np.arange(len(names))
    w = 0.2
    bg_par = np.array([results[n].backgrounds[0] for n in names])
    bg_perp = np.array([results[n].backgrounds[1] for n in names])
    ch_par = np.array([results[n].channels[0] for n in names])
    ch_perp = np.array([results[n].channels[1] for n in names])

    fig, ax = plt.subplots(figsize=(2.0 * len(names) + 2, 5))
    ax.bar(x - 1.5 * w, bg_par, w, yerr=np.sqrt(bg_par), capsize=3,
           color='0.75', label='background parallel')
    ax.bar(x - 0.5 * w, bg_perp, w, yerr=np.sqrt(bg_perp), capsize=3,
           color='0.45', label='background perpendicular')
    ax.bar(x + 0.5 * w, ch_par, w, yerr=np.sqrt(ch_par), capsize=3,
           color='C0', label='channel parallel')
    ax.bar(x + 1.5 * w, ch_perp, w, yerr=np.sqrt(ch_perp), capsize=3,
           color='C1', label='channel perpendicular')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('integrated photons (window)')
    ax.set_title('s010 channels & backgrounds across experiments')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_time_evolution(results: dict):
    """One subplot per experiment: per-bin counts in the two channels vs time."""
    import math
    import matplotlib.pyplot as plt

    names = list(results)
    n = len(names)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax, name in zip(axes.flat, names):
        res = results[name]
        ax.plot(res.t, res.counts[0], label='parallel')
        ax.plot(res.t, res.counts[1], label='perpendicular')
        ax.axvspan(res.t[res.background_window][0], res.t[res.background_window][-1],
                   color='0.7', alpha=0.3)
        ax.axvspan(res.t[res.channel_window][0], res.t[res.channel_window][-1],
                   color='C2', alpha=0.2)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('counts / bin')
    for ax in axes.flat[n:]:           # hide unused axes
        ax.axis('off')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('s010 time evolution per experiment')
    fig.tight_layout()
    return fig


def plot_contrast_over_photons(results: dict, readout='polarization',
                               pairs=(('rs2', 'static'),
                                      ('rs2', '100nm'),
                                      ('100nm', 'static'))):
    """Contrast SNR per photon budget between conditions, for each configuration.

    Experiment labels are ``"<condition> <config>"`` (e.g. "rs2 488c + 405p").
    For every configuration and every requested condition pair, the contrast SNR
    per 1e6 emitted photons is computed (budget = mean emitted photons of the two
    conditions) and shown as grouped bars. Magnitude is used, so the pair order
    does not matter.
    """
    import matplotlib.pyplot as plt

    # Group results by configuration: {config: {condition: result}}.
    by_config = {}
    for label, res in results.items():
        cond, _, config = label.partition(' ')
        by_config.setdefault(config, {})[cond] = res

    configs = list(by_config)
    x = np.arange(len(configs))
    width = 0.8 / len(pairs)

    fig, ax = plt.subplots(figsize=(2.2 * len(configs) + 2, 5))
    for i, (cond_a, cond_b) in enumerate(pairs):
        heights = []
        for config in configs:
            cond_map = by_config[config]
            if cond_a in cond_map and cond_b in cond_map:
                c = contrast_of(cond_map[cond_a], cond_map[cond_b], readout=readout)
                heights.append(abs(c.snr_per_Mph))
            else:
                heights.append(np.nan)
        offset = (i - (len(pairs) - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=f'{cond_a} vs {cond_b}')
        for bar, h in zip(bars, heights):
            if np.isfinite(h):
                ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, h),
                            ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha='right')
    ax.set_ylabel(f'|contrast SNR| / 1M ph  ({readout})')
    ax.set_title('s010 contrast per photon budget across configurations')
    ax.legend(fontsize=8, title='condition pair')
    fig.tight_layout()
    return fig


def main_set(experiments: dict = None):
    """Run the experiment set and show the comparison figures."""
    import matplotlib.pyplot as plt

    results = run_all(experiments)
    for name, res in results.items():
        print(f"{name:>16}: p = {res.polarization:+.4f} +/- {res.polarization_std:.4f} "
              f"(SNR {res.polarization_snr:.0f}, {_human(res.total_emitted_photons)} ph)")
    plot_polarization_bars(results)
    plot_anisotropy_bars(results)
    plot_counts_bars(results)
    plot_contrast_over_photons(results, readout='polarization')
    plot_time_evolution(results)
    plt.show()


if __name__ == '__main__':
    main_set()
