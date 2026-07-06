"""Experiment definitions for the rotational-diffusion-photophysics package.

Promoted from the s010 prototyping script (a20 "graduation"). An experiment is a
frozen parameter dataclass with a ``run()`` method that returns a results
dataclass. Frozen so a swept variant is created with ``dataclasses.replace``
rather than by mutating shared state.

Currently provides:
  - ``experiment``          two-phase steady-state STARSS experiment
  - ``results``  its run output (time series + integrated observables/SNR)
"""
from dataclasses import dataclass, field

import numpy as np

from rotational_diffusion_photophysics import core
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
from rotational_diffusion_photophysics.utils.signals import (
    detected_counts,
    background_subtract,
    anisotropy,
    anisotropy_variance,
    polarization,
    polarization_variance,
)

__all__ = ["experiment", "results"]


@dataclass
class results:
    """Everything computed for one ``experiment`` run.

    Time series span the whole pulse scheme; the measured observables are
    scalars from integrating counts over the two measurement windows.
    """
    t: np.ndarray
    signals: np.ndarray            # engine output (2, ntime), arbitrary units
    counts: np.ndarray             # per-bin detected counts (2, ntime)
    background_window: np.ndarray  # boolean mask over t (phase-1 measurement)
    channel_window: np.ndarray     # boolean mask over t (phase-2 measurement)
    backgrounds: np.ndarray        # integrated background counts (2,) [par, perp]
    channels: np.ndarray           # integrated channel counts (2,) [par, perp]
    signal: np.ndarray             # background-subtracted counts (2,) [par, perp]
    signal_std: np.ndarray         # shot-noise std of the signal (2,) [par, perp]
    anisotropy: float
    anisotropy_std: float
    anisotropy_snr: float
    polarization: float
    polarization_std: float
    polarization_snr: float
    total_emitted_photons: float   # fluorescence photons emitted over the whole experiment


@dataclass(frozen=True)
class experiment:
    """Parameters that define the s010 two-phase cross-polarized STARSS experiment.

    Two-phase pulse scheme. Phase durations are fractions of ``total_time``; any
    leftover (1 - phase1_fraction - phase2_fraction) is dead time at the end (both
    lasers off). A ``settling_fraction`` of ``total_time`` is cut from the start
    of each phase, and the remaining part of the phase is the measurement window.
    Phase 1 is 488 nm only (background); phase 2 adds 405 nm (signal).
    """
    tau: float = 30e9             # rotational correlation time [s]
    power_405: float = 5.0e3*0.01 # 405 nm power density [W/cm^2]
    power_488: float = 5.0e3      # 488 nm power density [W/cm^2]
    pol_405: str = 'x'            # 405 nm polarization: 'x', 'y', or 'xy' (circular)
    pol_488: str = 'xy'           # 488 nm polarization
    det_pol: tuple = ('x', 'y')   # cross-polarized detection channels
    na: float = 0.75              # numerical aperture
    ri: float = 1.000             # refractive index of immersion medium
    lmax: int = 6                 # spherical-harmonics cutoff
    # --- pulse scheme (phase durations as fractions of total_time) ---
    total_time: float = 2.0          # total experiment duration [s]
    phase1_fraction: float = 0.5     # fraction of total_time spent in phase 1 (488 only)
    phase2_fraction: float = 0.5     # fraction of total_time spent in phase 2 (488 + 405)
    #   any leftover (1 - phase1_fraction - phase2_fraction) is dead time at the end
    settling_fraction: float = 0.05  # fraction of total_time cut from the START of each
    #   phase (settling); the remaining part of each phase is the measurement window
    n_time: int = 100                # number of time points over the whole scheme
    fluorophore: object = field(default=rsEGFP2_8states)  # swappable fluorophore
    # --- photon-counting / SNR parameters ---
    n_molecules: float = 1e4          # number of emitters in the probed volume
    # one cell is 1e4 to 1e6 molecules
    collection_efficiency: float = 0.01   # overall detection efficiency eta
    background_averages: int = 1      # times the background phase was averaged

    @property
    def phase1_duration(self) -> float:
        return self.phase1_fraction * self.total_time

    @property
    def phase2_duration(self) -> float:
        return self.phase2_fraction * self.total_time

    @property
    def dead_time(self) -> float:
        """Leftover time at the end with both lasers off (>= 0)."""
        return max(0.0, (1.0 - self.phase1_fraction - self.phase2_fraction) * self.total_time)

    @property
    def settling_time(self) -> float:
        return self.settling_fraction * self.total_time

    @property
    def t_end(self) -> float:
        return self.total_time

    def time(self) -> np.ndarray:
        return np.linspace(0, self.total_time, self.n_time)

    def build(self):
        """Construct the engine System with the two-phase (+ dead time) pulse scheme."""
        diffusion = IsotropicDiffusion(diffusion_coefficient=1 / (6 * self.tau))

        # Windows (rows are [405, 488]): phase 1 = 488 only, phase 2 = 488 + 405,
        # then an optional dead-time window with both lasers off.
        time_windows = [self.phase1_duration, self.phase2_duration]
        mod_405 = [0, 1]
        mod_488 = [1, 1]
        if self.dead_time > 1e-12:
            time_windows.append(self.dead_time)
            mod_405.append(0)
            mod_488.append(0)

        lasers = ModulatedLasers(
            power_density=[self.power_405, self.power_488],
            wavelength=[405, 488],
            polarization=[self.pol_405, self.pol_488],
            modulation=[mod_405, mod_488],
            time_windows=time_windows,
            time0=0,
            numerical_aperture=self.na,
            refractive_index=self.ri,
        )
        detection = PolarizedDetection(
            polarization=list(self.det_pol),
            numerical_aperture=self.na,
            refractive_index=self.ri,
        )
        return core.System(
            fluorophore=self.fluorophore,
            diffusion=diffusion,
            illumination=lasers,
            detection=detection,
            lmax=self.lmax,
            representation='so3',
        )

    def run(self) -> results:
        """Run this experiment: pulse scheme -> windowed counts -> observables + SNR."""
        t = self.time()
        system = self.build()
        signals = system.detector_signals(t)

        # Convert the engine signal into expected detected photon counts per bin.
        dt = t[1] - t[0]
        counts = detected_counts(signals, self.n_molecules,
                                 self.collection_efficiency,
                                 self.fluorophore.lifetime_on, dt)

        # Total fluorescence photons EMITTED over the whole experiment (4*pi,
        # before collection efficiency): N * k_r * integral of the excited-state
        # population, with k_r = Phi_fluo / tau. The l=0,m=0 coefficient of the
        # fluorescent species (system._c) is its population -- physically real; the
        # SO(3) (complex Wigner-D) engine carries it as complex with a numerically
        # negligible imaginary part (roundoff), so take the real part explicitly.
        exc = int(np.nonzero(self.fluorophore.quantum_yield_fluo)[0][0])
        p_exc = system._c[exc, 0, :].real
        k_r = self.fluorophore.quantum_yield_fluo[exc] / self.fluorophore.lifetime_on
        total_emitted_photons = float(self.n_molecules * k_r * np.sum(p_exc) * dt)

        # Measurement windows: cut the settling time from the START of each phase,
        # then measure the remaining part of that phase.
        t1 = self.phase1_duration
        t2_end = self.phase1_duration + self.phase2_duration
        settling = self.settling_time
        bg_window = (t >= settling) & (t < t1)            # phase 1, 488 only
        ch_window = (t >= t1 + settling) & (t <= t2_end)  # phase 2, 488 + 405

        # Integrated counts per channel over each window.
        backgrounds = counts[:, bg_window].sum(axis=1)
        channels = counts[:, ch_window].sum(axis=1)
        navg = self.background_averages

        # Per-channel background-subtracted signal and its Poisson variance.
        sig, sig_var = background_subtract(channels, backgrounds, navg)

        r = anisotropy(channels, backgrounds, navg)
        p = polarization(channels, backgrounds, navg)
        r_var = anisotropy_variance(channels, backgrounds, navg)
        p_var = polarization_variance(channels, backgrounds, navg)

        return results(
            t=t, signals=signals, counts=counts,
            background_window=bg_window, channel_window=ch_window,
            backgrounds=backgrounds, channels=channels,
            signal=sig, signal_std=np.sqrt(sig_var),
            anisotropy=float(r), anisotropy_std=float(np.sqrt(r_var)),
            anisotropy_snr=float(r / np.sqrt(r_var)),
            polarization=float(p), polarization_std=float(np.sqrt(p_var)),
            polarization_snr=float(p / np.sqrt(p_var)),
            total_emitted_photons=total_emitted_photons,
        )
