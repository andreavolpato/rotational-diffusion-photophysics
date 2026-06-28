"""exp003 - STARSS method 3 (rsEGFP2, 8-state model).

A pair of saturating 405 nm pulses separated by a tunable ``delay`` activates the
sample; rotational diffusion between the pulses changes how many molecules the
second pulse can still on-switch, so the integrated 488 nm readout depends on the
delay. The observable for one ``delay`` is the two-pulse fluorescence normalized
by the one-pulse fluorescence (ranges ~1 to 2); sweeping ``delay`` traces the
rotational dynamics. Promotes the former ``models.starss.starss3`` /
``starss3_detector_signals`` into the ``experiment``/``results`` structure.
"""
from dataclasses import dataclass, field

import numpy as np

from rotational_diffusion_photophysics import core
from rotational_diffusion_photophysics.models.fluorophore import rsEGFP2_8states
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion

__all__ = ["experiment", "results"]


@dataclass
class results:
    """Everything computed for one ``experiment`` run (a single ``delay``)."""
    t: np.ndarray              # time axis [s] (readout grid + a long-delay background point)
    signals: np.ndarray        # two-pulse detector signal (1, ntime)
    signals_1pulse: np.ndarray # one-pulse detector signal (1, ntime)
    normalized_count: float    # integrated two-pulse / one-pulse counts (~1..2)
    system: object             # the (two-pulse) solved engine (SystemS2 / SystemSO3)


@dataclass(frozen=True)
class experiment:
    """STARSS method 3: a pair of 405 nm activation pulses with a tunable delay.

    Pulse scheme windows are [pre, prep-delay, pulse, delay, pulse, wait, readout]
    (405 fires in the two ``pulse`` windows, 488 is on in ``pre`` and ``readout``).
    The headline observable from ``run()`` is ``normalized_count`` for this
    ``delay``; sweep ``delay`` (with ``dataclasses.replace``) to trace the dynamics.
    """
    delay: float = 50e-6          # delay between the two 405 pulses [s]
    tau: float = 100e-6           # rotational correlation time [s]
    power_405: float = 7.12e6 / 6 / 2  # 405 nm power density [W/cm^2]
    power_488: float = 14e3 / 6   # 488 nm power density [W/cm^2]
    pol_405: str = 'x'            # 405 nm polarization
    pol_488: str = 'xy'           # 488 nm polarization (circular)
    # pulse-scheme window durations (the delay is inserted between the two pulses)
    pre_time: float = 10e-3       # 488 preconditioning
    prep_time: float = 0.6e-3     # total pre-pulse preparation (window 1 = prep - delay)
    pulse_time: float = 50e-9     # each 405 activation pulse
    wait_time: float = 0.2e-3     # wait before readout
    readout_time: float = 3e-3    # 488 readout window
    det_pol: tuple = ('xy',)      # circular (unpolarized) detection
    na: float = 1.4               # numerical aperture
    ri: float = 1.518             # refractive index of immersion medium
    lmax: int = 6                 # spherical-harmonics cutoff
    fluorophore: object = field(default=rsEGFP2_8states)
    integration_time: float = 1e-3  # integrate the readout over [0, integration_time]
    background_time: float = 10e-3  # long-delay time point used as background
    n_time: int = 1000            # number of readout time points

    @property
    def time0(self) -> float:
        """Lab time zero: start of the readout window (delay-independent)."""
        return (self.pre_time + self.prep_time
                + 2 * self.pulse_time + self.wait_time)

    @property
    def t_end(self) -> float:
        return self.background_time

    def time(self) -> np.ndarray:
        # Readout grid plus one long-delay point sampling the steady background.
        t = np.linspace(0, self.integration_time, self.n_time)
        return np.append(t, self.background_time)

    def build(self, first_pulse: bool = True):
        """Construct the engine System; ``first_pulse=False`` drops the first 405 pulse."""
        diffusion = IsotropicDiffusion(diffusion_coefficient=1 / (6 * self.tau))
        time_windows = [
            self.pre_time,
            self.prep_time - self.delay,
            self.pulse_time,
            self.delay,
            self.pulse_time,
            self.wait_time,
            self.readout_time,
        ]
        mod_405 = [0, 0, 1, 0, 1, 0, 0] if first_pulse else [0, 0, 0, 0, 1, 0, 0]
        mod_488 = [1, 0, 0, 0, 0, 0, 1]
        lasers = ModulatedLasers(
            power_density=[self.power_405, self.power_488],
            wavelength=[405, 488],
            polarization=[self.pol_405, self.pol_488],
            modulation=[mod_405, mod_488],
            time_windows=time_windows,
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
        """Run both pulse schemes for this delay and return the normalized count."""
        t = self.time()
        sys2 = self.build(first_pulse=True)
        sys1 = self.build(first_pulse=False)
        sig2 = sys2.detector_signals(t)
        sig1 = sys1.detector_signals(t)

        # Background-subtract using the long-delay point, then integrate the readout.
        sig2_bg = sig2 - sig2[0, -1]
        sig1_bg = sig1 - sig1[0, -1]
        sel = t <= self.integration_time
        norm = float(np.sum(sig2_bg[0, sel]) / np.sum(sig1_bg[0, sel]))

        return results(t=t, signals=sig2, signals_1pulse=sig1,
                       normalized_count=norm, system=sys2)
