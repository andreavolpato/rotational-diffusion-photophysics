"""exp020 - "first time is special": alternating 405/488 pulse train on rsEGFP2.

A train of square-wave pulses (405 and 488 nm interleaved) is shined on a slowly
diffusing rsEGFP2 sample and the state populations are followed over time. The
first activation pulse drives a fresh population and behaves differently from the
later ones, hence the name. Same ``experiment``/``results`` structure as exp010.
"""
from dataclasses import dataclass, field

import numpy as np

from rotational_diffusion_photophysics.engine import System
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion

__all__ = ["experiment", "results"]


@dataclass
class results:
    """Everything computed for one ``experiment`` run."""
    t: np.ndarray            # time axis [s]
    signals: np.ndarray      # detector signals (ndetectors, ntime)
    populations: np.ndarray  # state populations (nspecies, ntime)
    system: System           # the solved engine System (e.g. for pulse-scheme plots)


@dataclass(frozen=True)
class experiment:
    """Alternating 405/488 pulse train; observable is the population dynamics.

    The pulse scheme is a sequence of equal-duration windows; ``modulation_405``
    and ``modulation_488`` switch each laser on/off per window. Frozen so a swept
    variant is made with ``dataclasses.replace``.
    """
    tau: float = 100e-6           # rotational correlation time [s]
    power_405: float = 200.0      # 405 nm power density [W/cm^2]
    power_488: float = 1000.0     # 488 nm power density [W/cm^2]
    pol_405: str = 'xy'           # 405 nm polarization ('x', 'y', or 'xy' circular)
    pol_488: str = 'xy'           # 488 nm polarization
    modulation_405: tuple = (0, 0, 1, 0, 1)  # 405 on/off per window
    modulation_488: tuple = (0, 1, 0, 1, 0)  # 488 on/off per window
    window_duration: float = 5e-3  # duration of each pulse window [s]
    det_pol: tuple = ('xy',)      # detection channels (single circular by default)
    na: float = 1.4               # numerical aperture
    ri: float = 1.518             # refractive index of immersion medium
    lmax: int = 6                 # spherical-harmonics cutoff
    fluorophore: object = field(default=rsEGFP2_8states)
    n_time: int = 1000            # number of time points over the whole scheme

    @property
    def n_windows(self) -> int:
        return len(self.modulation_405)

    @property
    def t_end(self) -> float:
        return self.n_windows * self.window_duration

    def time(self) -> np.ndarray:
        return np.linspace(0, self.t_end, self.n_time)

    def build(self) -> System:
        """Construct the engine System with the alternating pulse train."""
        diffusion = IsotropicDiffusion(diffusion_coefficient=1 / (6 * self.tau))
        lasers = ModulatedLasers(
            power_density=[self.power_405, self.power_488],
            wavelength=[405, 488],
            polarization=[self.pol_405, self.pol_488],
            modulation=[list(self.modulation_405), list(self.modulation_488)],
            time_windows=list(np.ones(self.n_windows) * self.window_duration),
            time0=0,
            numerical_aperture=self.na,
            refractive_index=self.ri,
        )
        detection = PolarizedDetection(
            polarization=list(self.det_pol),
            numerical_aperture=self.na,
            refractive_index=self.ri,
        )
        return System(
            illumination=lasers,
            fluorophore=self.fluorophore,
            diffusion=diffusion,
            detection=detection,
            lmax=self.lmax,
        )

    def run(self) -> results:
        """Run the experiment and return the population dynamics."""
        t = self.time()
        system = self.build()
        signals = system.detector_signals(t)
        return results(t=t, signals=signals, populations=system._p, system=system)
