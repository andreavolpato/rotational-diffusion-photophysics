import numpy as np
import matplotlib.pyplot as plt
from rotational_diffusion_photophysics.models.starss import starss_sted
from rotational_diffusion_photophysics.plot.plot_pulse_scheme import plot_pulse_scheme
from rotational_diffusion_photophysics.common import anisotropy

t = np.linspace(-1e-9,12e-9,1000)
signals = starss_sted.detector_signals(t)
population = starss_sted._p

xlim = [-1e-9, 12e-9]
ylim = [1e3, 5e9]
yscale = 'log'


fig = plt.figure(1)
gs = fig.add_gridspec(3,1)
ax0 = fig.add_subplot(gs[0,0])
plot_pulse_scheme(starss_sted, xlim=xlim, ylim=ylim, yscale=yscale)
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[2,0])
ax1.plot(t, anisotropy(signals))
ax1.set_ylabel('Anisotropy')
ax1.set_xlabel('Time (s)')
ax1.sharex(ax0)
ax2.plot(t, signals.transpose())
ax2.set_ylabel('Signal (a.u.)')
ax2.set_xlabel('Time (s)')
ax2.sharex(ax0)
# plt.show()
