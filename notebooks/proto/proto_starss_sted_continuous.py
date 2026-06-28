import numpy as np
import rotational_diffusion_photophysics as rdp
import matplotlib.pyplot as plt
from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
from rotational_diffusion_photophysics.models.fluorophore import atto647N
from rotational_diffusion_photophysics.models.detection import PolarizedDetection
from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
from rotational_diffusion_photophysics.engine_s2 import SystemS2 as System
from rotational_diffusion_photophysics.plot.plot_pulse_scheme import plot_pulse_scheme
from rotational_diffusion_photophysics.common import anisotropy


def make_exp(tauR=50e-9, sted_pw=1e8, exc_pw=1e3):
    light = ModulatedLasers(wavelength=  [640, 775],
                            polarization=['x', 'y'],
                            power_density=[exc_pw, sted_pw],
                            time_windows=[1e-9, 100e-9],
                            modulation=[[1,1],[0,1]],
                            numerical_aperture=0.8,
                            refractive_index=1.0)

    det = PolarizedDetection(numerical_aperture=0.8,
                             refractive_index=1.0)
    diff = IsotropicDiffusion(diffusion_coefficient=1/(6*tauR))
    exp = System(fluorophore=atto647N, diffusion=diff, detection=det, illumination=light)
    return exp

xlim = [-1e-9, 100e-9]
ylim = [1, 5e9]
yscale = 'log'
# plot_pulse_scheme(exp, xlim=xlim, ylim=ylim, yscale=yscale)

fig = plt.figure(1)
gs = fig.add_gridspec(3,1)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[2,0])
t = np.linspace(-1e-9,100e-9,1000)


taus = [0.01e-9, 0.1e-9, 1e-9, 10e-9, 100e-9, 1000e-9, 10000e-9]
for tauR in taus:
    exp = make_exp(tauR=tauR, sted_pw=5e0, exc_pw=1e4)
    signals = exp.detector_signals(t)
    ax1.plot(t, anisotropy(signals))
    ax2.plot(t, signals.transpose())

ax1.set_ylabel('Anisotropy')
ax1.set_xlabel('Time (s)')
ax1.sharex(ax0)
ax1.legend(taus)
ax2.set_ylabel('Signal (a.u.)')
ax2.set_xlabel('Time (s)')
ax2.sharex(ax0)
plt.show()
