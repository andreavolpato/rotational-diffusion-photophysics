import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, 'C:/Users/andre/Documents/GitHub/rotational_diffusion_photophysics')
import rotational_diffusion_photophysics as rdp
import colour

def plot_pulse_scheme(exp, xlim=[-.1e-3, 1e-3], ylim=[0, 6e4], yscale='linear'):

    # Values for testing
    # xlim = [-.1e-3, 1e-3]
    # ylim = [0, 6e4]
    # yscale = 'linear'

    # Get all the necessary variables from the illumination class
    pd = exp.illumination.power_density
    tw = exp.illumination.time_windows
    t0 = exp.illumination.time0
    mod = exp.illumination.modulation
    wl = exp.illumination.wavelength
    pol = exp.illumination.polarization

    # Prepare the time axis of the sqare waves
    # Two data points are necessary to plot the edges of the wave.
    t = np.cumsum(tw)
    t = np.repeat(t, 2)
    t = t[0:-1]
    t = np.insert(t, 0, 0)
    t = t - t0

    # Prepare the power density array
    p = np.zeros(mod.shape)
    for i in np.arange(wl.size):
        p[i] = mod[i] * pd[i]
    p = np.repeat(p, 2, axis=1)

    def wl2rgb(wl):
        xyz = colour.wavelength_to_XYZ(wl)
        rgb = colour.XYZ_to_sRGB(xyz)
        rgb[rgb<0] = 0
        rgb[rgb>1] = 1
        return rgb

    labels = []
    for i in np.arange(wl.size):
        color = wl2rgb(wl[i])
        plt.fill_between(t, p[i], edgecolor=color, facecolor=np.concatenate((color, [0.1])))
        labels.append(np.array_str(wl[i])+' nm ('+pol[i]+')')
    plt.yscale(yscale)
    plt.ylabel('Power Density (W/cm$^{2}$)')
    plt.xlabel('Time (s)')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(labels)
    return None


if __name__ == '__main__':
    from rotational_diffusion_photophysics.models.starss import starss1

    xlim = [-.1e-3, 1e-3]
    ylim = [0, 6e4]
    yscale = 'linear'
    plot_pulse_scheme(starss1, xlim=xlim, ylim=ylim, yscale=yscale)
    plt.show()