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

from rotational_diffusion_photophysics.core import find_wavelenght
from rotational_diffusion_photophysics.experiments.exp011_wf_steady_state import experiment
from rotational_diffusion_photophysics.utils.signals import anisotropy, detected_counts

power_405_scale = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
power_405_scale = [0.1, 1, 10, 100]

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
        print(f"  405 scale {s:.3g}: counts = {res.counts.sum(axis=1)}")
    return scales, t, np.array(counts)


def main(scales=power_405_scale, base=experiment()):
    scales, t, counts = sweep_405(scales, base)
    
    plot_count_traces(scales, t, counts)
    plot_saturation(scales, counts, base)
    plot_excitation_split(base)

    plt.show()

def plot_count_traces(scales, t, counts):
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

def local_time(e: experiment):
    """Time axis of ONE instance, relative to the start of its readout: it runs
    from -(waiting + equilibration) to +readout_time, so ``local_time > 0``
    selects the readout part only (after the dark period and the settling under
    the lasers). Note ``t[0:n_time]`` is the ABSOLUTE axis (0 .. total), where
    ``> 0`` would keep the dark + equilibration transient too."""
    return np.linspace(-e.waiting_time - e.equilibration_time, e.readout_time,
                       e.n_time)


def readout_window_counts(counts, e: experiment):
    """Integrate the per-bin counts over the readout part of each of the 6
    instances. ``counts`` is (2, n_time*6) -> returns (2, 6) [channel, window]."""
    local = local_time(e)
    sig = np.reshape(counts, (2, 6, e.n_time))
    return np.sum(sig[:, :, local > 0], axis=2)


def check_errors(base: experiment = None, verbose=True):
    """Monitor the model's numerical error, using three independent probes.

    The time propagation itself is EXACT (``core.solve_evolution`` diagonalizes
    the diffusion-kinetics matrix and applies exp(L t) analytically), so there is
    no ODE time-stepping error. What can be wrong is:

    1. ``window_mismatch`` -- PHYSICAL self-consistency: instances 1 and 4 are
       both 488-only, so they must integrate to the same counts. Their relative
       difference measures whether ``waiting_time + equilibration_time`` is long
       enough to return to the 488 steady state (it is the only thing that makes
       instance 4, which follows a 405 pulse, differ from instance 1). It floors
       at the numerical noise level (~1e-8) once equilibration is sufficient.
    2. ``lmax_error`` -- ANGULAR truncation: relative change of every window when
       the spherical-harmonic cutoff is raised by 4.
    3. ``n_time_error`` -- COUNT QUADRATURE: relative change when the time grid
       is doubled. ``detected_counts`` multiplies by ``dt`` (a rectangle rule),
       so this is the true "integration error" of the observable, and it is
       usually the dominant one.
    4. ``xy_symmetry`` -- ANGULAR self-consistency (the strongest probe of the
       photoselection machinery): windows 2,3 use an x-polarized 405 and windows
       5,6 the same powers y-polarized. A 90 deg rotation about z maps one onto
       the other and swaps the two detection channels (the 488 is circular, so
       it is invariant), hence exactly
       ``counts[par, w2] == counts[perp, w5]`` and ``counts[perp, w2] ==
       counts[par, w5]`` (same for w3 <-> w6). This is a rotation-invariance
       test. Note it is a STRUCTURAL check, not a resolution one: it stays at
       ~1e-7 even at lmax=2, because truncation preserves the x<->y symmetry.
       It is the probe that catches a wrong spherical-harmonic product /
       photoselection coefficient (the class of bug that a lmax or n_time sweep
       cannot see, since a wrong-but-consistent model converges happily).

    Returns the four relative errors as a dict.
    """
    if base is None:
        base = experiment()

    w = readout_window_counts(base.run().counts, base)
    mismatch = np.max(np.abs(w[:, 0] - w[:, 3]) / np.abs(w[:, 0]))

    # x-polarized 405 windows (1, 2) vs y-polarized (4, 5), channels swapped.
    xy_symmetry = max(
        max(abs(w[0, xw] - w[1, yw]) / abs(w[0, xw]),
            abs(w[1, xw] - w[0, yw]) / abs(w[1, xw]))
        for xw, yw in ((1, 4), (2, 5)))

    fine_l = replace(base, lmax=base.lmax + 4)
    w_l = readout_window_counts(fine_l.run().counts, fine_l)
    lmax_error = np.max(np.abs(w_l - w) / np.abs(w))

    fine_t = replace(base, n_time=base.n_time * 2)
    w_t = readout_window_counts(fine_t.run().counts, fine_t)
    n_time_error = np.max(np.abs(w_t - w) / np.abs(w))

    if verbose:
        print(f"  window 1 vs 4 mismatch (488-only, equilibration): {mismatch:.2e}")
        print(f"  x/y 405 rotation symmetry (angular machinery):    {xy_symmetry:.2e}")
        print(f"  lmax {base.lmax} -> {base.lmax + 4} (angular truncation):    {lmax_error:.2e}")
        print(f"  n_time {base.n_time} -> {base.n_time * 2} (count quadrature):   {n_time_error:.2e}")

    return {'window_mismatch': mismatch, 'xy_symmetry': xy_symmetry,
            'lmax_error': lmax_error, 'n_time_error': n_time_error}


def photons_by_excitation(base: experiment = None):
    """Split the detected photons by WHICH LASER's absorption produced them.

    In rsEGFP2 the ON (fluorescent) state absorbs BOTH lasers -- its extinction
    is 5260 at 405 nm and 51560 at 488 nm -- so 405 does not only activate: it
    also *directly* excites fluorescence. The OFF state absorbs 405 strongly
    (22000) but does not emit, so those absorptions produce no photons directly;
    they only switch molecules ON, which is an indirect (and much larger) effect.

    This is an ATTRIBUTION, not a counterfactual. It answers "of the photons
    actually emitted in this experiment, which absorption event produced each
    one" -- exactly, because the emitting state is fed only by light, from a
    single ground state, so its source term is a sum of per-laser contributions
    and the (linear) emission splits with it. It is NOT "what you would measure
    with the other laser off": turning 405 off changes the ON/OFF equilibrium and
    hence the 488 signal too (that is the whole STARSS effect). For that
    question, re-run with the other power set to zero.

    Method: the emitting state is in quasi-steady state (tau ~ 1.6 ns << every
    pulse-scheme timescale), so its orientational distribution is
    ``tau_eff * feed_j(Omega)`` for each laser j, where ``feed_j`` is the
    absorption rate density (the engine's own photon-flux multiplication
    operator applied to the source species) and ``tau_eff`` is set by the TOTAL
    decay rate out of the excited state -- fluorescence + non-radiative +
    off-switching, i.e. ``tau / (1 + quantum_yield_on_to_off)``, 1.65% shorter
    than tau here. Contracting with the detector collection coefficients keeps
    the split CHANNEL-RESOLVED (405- and 488-excited photons have different
    angular distributions, so they divide differently over par/perp).

    Summed over lasers this reproduces ``System.detector_signals`` to ~1e-8.

    Returns ``(wavelength, counts)``: unique excitation wavelengths (nw,) and
    detected counts (nw, 2, ntime), which sum to the experiment's total counts.
    """
    if base is None:
        base = experiment()
    sys = base.build()
    t = base.time()
    sys.detector_signals(t)                  # solve; fills _c, _F, _c_det
    c, F, c_det = sys._c, sys._F, sys._c_det
    f, ill = sys.fluorophore, sys.illumination

    k = f.kinetics_matrix()
    idx = np.concatenate(([0], find_wavelenght(f.wavelength,
                                               np.atleast_1d(ill.wavelength)) + 1))
    k_sel = k[idx]                           # (nlasers+1, ns, ns) [out, in]
    qy = np.asarray(f.quantum_yield_fluo)
    fluo = np.nonzero(qy)[0]                 # emitting species
    tau_eff = 1.0 / k[0][:, fluo[0]].sum()   # total decay rate, not just 1/tau

    # Window index of every time point. A point exactly ON a window boundary
    # belongs to the EARLIER window (side='left'): the engine carries the state
    # continuously across the join, so the photons there were driven by the
    # window that just ended. Using 'right' mis-assigns those points and costs
    # ~3% of the total.
    time_lab = t + ill.time0
    edges = np.insert(np.cumsum(np.atleast_1d(ill.time_windows)), 0, 0)
    edges[-1] = max(edges[-1], time_lab.max())
    wi = np.clip(np.searchsorted(edges, time_lab, side='left') - 1,
                 0, ill.nwindows - 1)
    mod = np.atleast_2d(ill.modulation)

    nlasers, ndet = F.shape[0], c_det.shape[0]
    signals = np.zeros((nlasers, ndet, t.size))
    for j in range(nlasers):
        wgt = qy[fluo] @ k_sel[1 + j][fluo]          # rate into emitters x QY
        feed = np.einsum('s,lm,smt->lt', wgt, F[j], c) * mod[j][wi]
        for d in range(ndet):
            signals[j, d] = tau_eff * (c_det[d] @ feed)

    # Group the lasers by excitation wavelength and convert to detected counts.
    wl = np.atleast_1d(ill.wavelength)
    uniq = np.unique(wl)
    dt = t[1] - t[0]
    counts = np.array([
        detected_counts(signals[wl == w].sum(axis=0), base.n_molecules,
                        base.collection_efficiency, f.lifetime_on, dt)
        for w in uniq])
    return uniq, counts


def plot_excitation_split(base: experiment = None):
    """Detected counts split by which laser's absorption produced them."""
    if base is None:
        base = experiment()
    wl, counts = photons_by_excitation(base)
    t = base.time()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for d, (ax, name) in enumerate(zip(axes, ('parallel (x)', 'perpendicular (y)'))):
        for w, cw in zip(wl, counts):
            ax.plot(t, cw[d], label=f'{w:.0f} nm excitation')
        ax.set_yscale('log')
        ax.set_xlabel('Time (s)')
        ax.set_title(name)
    axes[0].set_ylabel('Detected counts / bin')
    axes[0].legend(fontsize=8)
    fig.suptitle('s011 — emission split by which laser was absorbed')
    fig.tight_layout()

    # Integrated over the readout windows, per excitation wavelength.
    per_win = np.array([readout_window_counts(cw, base) for cw in counts])
    total = per_win.sum(axis=0)
    print("readout-window counts by excitation wavelength (par+perp):")
    for w, pw in zip(wl, per_win):
        frac = pw.sum(axis=0) / total.sum(axis=0)
        print(f"  {w:.0f} nm: " + "  ".join(f"{v:10.1f} ({fr:6.3%})"
                                            for v, fr in zip(pw.sum(axis=0), frac)))
    return wl, counts


# Window layout of the exp011 pulse scheme (6 instances):
#   0: 488 only          -> background for windows 1, 2
#   1: 488 + 405_1 (x)      2: 488 + 405_2 (x)
#   3: 488 only          -> background for windows 4, 5
#   4: 488 + 405_1 (y)      5: 488 + 405_2 (y)
# Every 405 power is therefore measured TWICE, x- and y-polarized. The two are
# related by a 90 deg rotation about z that swaps the detection channels (the
# invariant checked in check_errors), so they are averaged here.
# Detection channels are (x, y) = (ch 0, ch 1).
PUMP_WINDOWS = (
    # (power attribute, x-pol window, y-pol window, their background windows)
    ('power_405_1', 1, 4, (0, 3)),
    ('power_405_2', 2, 5, (0, 3)),
)


def saturation_curve(scales, counts, base: experiment):
    """Background-subtracted [parallel, perpendicular] counts vs 405 power.

    Returns ``(power, channels)``: ``power`` (n,) in W/cm^2, sorted ascending,
    and ``channels`` (2, n) = [parallel, perpendicular]. "Parallel" is always
    relative to the 405 pump polarization, so for the y-polarized windows the
    detection channels are swapped before averaging with the x-polarized ones.

    Because ``power_405_2`` is a multiple of ``power_405_1``, the two pump
    windows sample 2 x ``len(scales)`` powers (and where they coincide you get a
    free reproducibility check: the points must overlay).
    """
    power, par, perp = [], [], []
    for s, c in zip(scales, counts):
        w = readout_window_counts(c, base)
        for attr, xw, yw, (bg_x, bg_y) in PUMP_WINDOWS:
            # x-pol pump: parallel = x channel. y-pol pump: parallel = y channel.
            par_x, perp_x = w[0, xw] - w[0, bg_x], w[1, xw] - w[1, bg_x]
            par_y, perp_y = w[1, yw] - w[1, bg_y], w[0, yw] - w[0, bg_y]
            power.append(getattr(base, attr) * s)
            par.append(0.5 * (par_x + par_y))
            perp.append(0.5 * (perp_x + perp_y))
    order = np.argsort(power)
    return (np.asarray(power)[order],
            np.array([np.asarray(par)[order], np.asarray(perp)[order]]))


def plot_saturation(scales, counts, base: experiment):
    """Two plots vs the 405 power: (1) background-subtracted counts of the
    parallel and perpendicular channels, (2) the resulting anisotropy."""
    power, channels = saturation_curve(scales, counts, base)
    par, perp = channels
    r = anisotropy(channels)          # (par - perp) / (par + 2 perp)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0.plot(power, par, 'o-', color='C0', label='parallel (to 405)')
    ax0.plot(power, perp, 's-', color='C1', label='perpendicular')
    ax0.plot(power, par + 2 * perp, '^--', color='0.5', label='total (par + 2 perp)')
    ax0.set_xscale('log')
    ax0.set_xlabel('405 power density (W/cm$^2$)')
    ax0.set_ylabel('counts over background (readout window)')
    ax0.set_title('405 saturation of the switched population')
    ax0.legend(fontsize=8)

    ax1.plot(power, r, 'o-', color='C3')
    ax1.set_xscale('log')
    ax1.axhline(0, color='0.7', lw=0.8)
    ax1.set_xlabel('405 power density (W/cm$^2$)')
    ax1.set_ylabel('anisotropy $r$')
    ax1.set_title('Anisotropy vs 405 power (photoselection washout)')

    fig.suptitle('s011 — 405 saturation')
    fig.tight_layout()
    return power, channels, r
    
    
if __name__ == '__main__':
    # print(experiment())
    main()