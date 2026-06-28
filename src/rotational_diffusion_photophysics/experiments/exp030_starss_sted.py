"""exp030 - STARSS-STED: cross-polarized 640 nm excitation + 775 nm STED on ATTO647N.

A short polarized 640 nm excitation pulse is followed by a cross-polarized 775 nm
STED pulse; the time-resolved cross-polarized fluorescence reports the
sub-nanosecond rotational dynamics. Same ``experiment``/``results`` structure as
exp010 (this promotes the former ``models.starss.starss_sted`` demo).
"""
from dataclasses import dataclass, field

import numpy as np

from rotational_diffusion_photophysics import core
from rotational_diffusion_photophysics.models.fluorophore import atto647N
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
    system: object           # the solved engine (SystemS2 / SystemSO3)


@dataclass(frozen=True)
class experiment:
    """STARSS-STED pulse scheme on ATTO647N; observable is the anisotropy decay.

    The five windows are [pre-delay, excitation, gap, STED, readout]; the
    excitation and STED lasers are cross-polarized. Frozen so a swept variant is
    made with ``dataclasses.replace``.
    """
    tau: float = 1e-9             # rotational correlation time [s]
    power_exc: float = 1e5        # 640 nm excitation power density [W/cm^2]
    power_sted: float = 5e8       # 775 nm STED power density [W/cm^2]
    wl_exc: int = 640             # excitation wavelength [nm]
    wl_sted: int = 775            # STED wavelength [nm]
    pol_exc: str = 'x'            # excitation polarization
    pol_sted: str = 'y'           # STED polarization (cross-polarized to excitation)
    # pulse scheme: [pre-delay, excitation, gap, STED, readout]
    time_windows: tuple = (1e-9, 170e-12, 100e-12, 500e-12, 12e-9)
    modulation_exc: tuple = (0, 1, 0, 0, 0)   # excitation on during window 1
    modulation_sted: tuple = (0, 0, 0, 1, 0)  # STED on during window 3
    det_pol: tuple = ('x', 'y')   # cross-polarized detection channels
    na: float = 1.4               # numerical aperture
    ri: float = 1.518             # refractive index of immersion medium
    lmax: int = 6                 # spherical-harmonics cutoff
    fluorophore: object = field(default=atto647N)
    t_start: float = -1e-9        # first time point [s]
    t_stop: float = 12e-9         # last time point [s]
    n_time: int = 1000            # number of time points

    @property
    def time0(self) -> float:
        """Lab time zero: the centre of the excitation pulse."""
        return self.time_windows[0] + self.time_windows[1] / 2

    @property
    def t_end(self) -> float:
        return self.t_stop

    def time(self) -> np.ndarray:
        return np.linspace(self.t_start, self.t_stop, self.n_time)

    def build(self):
        """Construct the engine System with the STARSS-STED pulse scheme."""
        diffusion = IsotropicDiffusion(diffusion_coefficient=1 / (6 * self.tau))
        lasers = ModulatedLasers(
            power_density=[self.power_exc, self.power_sted],
            wavelength=[self.wl_exc, self.wl_sted],
            polarization=[self.pol_exc, self.pol_sted],
            modulation=[list(self.modulation_exc), list(self.modulation_sted)],
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
        return core.System(
            fluorophore=self.fluorophore,
            diffusion=diffusion,
            illumination=lasers,
            detection=detection,
            lmax=self.lmax,
            representation='s2',
        )

    def run(self) -> results:
        """Run the experiment and return the time-resolved anisotropy."""
        t = self.time()
        system = self.build()
        signals = system.detector_signals(t)
        return results(t=t, signals=signals,
                       anisotropy=anisotropy(signals), system=system)
