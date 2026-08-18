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
from rotational_diffusion_photophysics.utils.signals import detected_counts

__all__ = ["experiment", "results"]


@dataclass
class results:
    """Output of one ``experiment`` run: the pulse-scheme time axis, the raw
    engine detector signals, and the expected detected photon counts per bin."""
    t: np.ndarray        # (ntime,) pulse-scheme time axis [s]
    signals: np.ndarray  # (2, ntime) engine detector signals [par, perp], a.u.
    counts: np.ndarray   # (2, ntime) expected detected photon counts per bin


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
    power_405_1: float = 4.0e3*0.01 # 405 nm power density [W/cm^2]
    power_405_2: float = 4.0e3*0.01*2 # 405 nm power density [W/cm^2]
    power_488: float = 4.0e3      # 488 nm power density [W/cm^2]
    det_pol: tuple = ('x', 'y')   # cross-polarized detection channels
    na: float = 0.75              # numerical aperture
    ri: float = 1.000             # refractive index of immersion medium
    lmax: int = 12                 # spherical-harmonics cutoff
    engine: str = 's2'           # engine representation ('so3' or 'wigner')
    # --- pulse scheme (phase durations as fractions of total_time) ---
    waiting_time: float = 2.0e-3        # time before phase 1 starts [s]
    equilibration_time: float = 2.0e-3  # time before phase 2 starts [s]
    readout_time: float = 20.0e-3       # time after phase 2 ends [s]
    n_time: int = 50                # number of time points over the whole scheme
    fluorophore: object = field(default=rsEGFP2_8states)  # swappable fluorophore
    # --- photon-counting / SNR parameters ---
    n_molecules: float = 1e4          # number of emitters in the probed volume
    # one cell is 1e4 to 1e6 molecules
    collection_efficiency: float = 0.01   # overall detection efficiency eta
    background_averages: int = 1      # times the background phase was averaged

    def time(self) -> np.ndarray:
        time = []
        time_zero = self.waiting_time + self.equilibration_time
        time_instance = self.waiting_time + self.equilibration_time + self.readout_time
        # zero = 0
        zero = - self.waiting_time - self.equilibration_time
        for i in np.arange(6):
            time.append(time_zero + np.linspace(zero, self.readout_time, self.n_time) + i*time_instance)
        return np.concatenate(time)

    def build(self):
        """Construct the engine System with the two-phase (+ dead time) pulse scheme."""
        diffusion = IsotropicDiffusion(diffusion_coefficient=1 / (6 * self.tau))

        # Windows (rows are [405, 488]): phase 1 = 488 only, phase 2 = 488 + 405,
        # then an optional dead-time window with both lasers off.
        time_windows = [self.waiting_time, self.equilibration_time+self.readout_time] * 6

        lasers = ModulatedLasers(
            power_density=[self.power_488, self.power_405_1, self.power_405_2,
                                           self.power_405_1, self.power_405_2],
            wavelength  =[ 488, 405, 405, 405, 405],
            polarization=["xy", "x", "x", "y", "y"],
            modulation=[ [0,1, 0,1, 0,1, 0,1, 0,1, 0,1,],
                         [0,0, 0,1, 0,0, 0,0, 0,0, 0,0,],
                         [0,0, 0,0, 0,1, 0,0, 0,0, 0,0,],
                         [0,0, 0,0, 0,0, 0,0, 0,1, 0,0,],
                         [0,0, 0,0, 0,0, 0,0, 0,0, 0,1,],
            ],
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
            representation=self.engine,
        )

    def run(self) -> results:
        """Run this experiment: pulse scheme -> engine signals -> detected counts."""
        t = self.time()
        system = self.build()
        signals = system.detector_signals(t)

        # Convert the engine signal into expected detected photon counts per bin.
        dt = t[1] - t[0]
        counts = detected_counts(signals, self.n_molecules,
                                 self.collection_efficiency,
                                 self.fluorophore.lifetime_on, dt)

        return results(t=t, signals=signals, counts=counts)
