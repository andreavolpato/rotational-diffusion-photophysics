"""exp001 - STARSS method 1 (rsEGFP2, 8-state model).

A short polarized 405 nm activation pulse on-switches an oriented sub-population,
read out continuously by circularly polarized 488 nm light; the cross-polarized
fluorescence gives an anisotropy decay that tracks rotational diffusion. A single
pulse scheme produces the experimental observable directly. Promotes the former
``models.starss.starss1`` into the ``experiment``/``results`` structure.
"""
from dataclasses import dataclass, field

import numpy as np

from rotational_diffusion_photophysics.engine import System
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
from rotational_diffusion_photophysics.utils.signals import anisotropy

__all__ = ["experiment", "results"]


@dataclass
class results:
    """Everything computed for one ``experiment`` run."""
    t: np.ndarray            # time axis [s]
    signals: np.ndarray      # detector signals (2, ntime), [parallel, perpendicular]
    anisotropy: np.ndarray   # time-resolved anisotropy r(t)
    system: System           # the solved engine System (e.g. for pulse-scheme plots)


@dataclass(frozen=True)
class experiment:
    """STARSS method 1: 405 nm activation pulse + continuous 488 nm readout.

    Pulse scheme windows are [488-preconditioning, 405-activation, readout]; the
    405 beam is linearly polarized, the 488 beam circular. Frozen so a swept
    variant is made with ``dataclasses.replace``.
    """
    tau: float = 100e-6           # rotational correlation time [s]
    power_405: float = 176.7e3    # 405 nm power density [W/cm^2]
    power_488: float = 17.6e3 / 6 # 488 nm power density [W/cm^2]
    pol_405: str = 'x'            # 405 nm polarization
    pol_488: str = 'xy'           # 488 nm polarization (circular)
    # pulse scheme: [488 preconditioning, 405 activation, readout]
    time_windows: tuple = (10e-3, 250e-9, 3e-3)
    modulation_405: tuple = (0, 1, 0)
    modulation_488: tuple = (1, 1, 1)
    det_pol: tuple = ('x', 'y')   # cross-polarized detection channels
    na: float = 1.4               # numerical aperture
    ri: float = 1.518             # refractive index of immersion medium
    lmax: int = 6                 # spherical-harmonics cutoff
    fluorophore: object = field(default=rsEGFP2_8states)
    t_start: float = 0.0          # first readout time point [s]
    t_stop: float = 3e-3          # last readout time point [s]
    n_time: int = 1000            # number of time points

    @property
    def time0(self) -> float:
        """Lab time zero: start of the readout (end of the activation pulse)."""
        return self.time_windows[0] + self.time_windows[1]

    @property
    def t_end(self) -> float:
        return self.t_stop

    def time(self) -> np.ndarray:
        return np.linspace(self.t_start, self.t_stop, self.n_time)

    def build(self) -> System:
        """Construct the engine System with the STARSS-1 pulse scheme."""
        diffusion = IsotropicDiffusion(diffusion_coefficient=1 / (6 * self.tau))
        lasers = ModulatedLasers(
            power_density=[self.power_405, self.power_488],
            wavelength=[405, 488],
            polarization=[self.pol_405, self.pol_488],
            modulation=[list(self.modulation_405), list(self.modulation_488)],
            time_windows=list(self.time_windows),
            time0=self.time0,
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
        """Run the experiment and return the time-resolved anisotropy."""
        t = self.time()
        system = self.build()
        signals = system.detector_signals(t)
        return results(t=t, signals=signals,
                       anisotropy=anisotropy(signals), system=system)
