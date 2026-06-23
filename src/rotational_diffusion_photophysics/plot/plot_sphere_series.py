import numpy as np
import matplotlib.pyplot as plt
import sys; sys.path.insert(0, 'C:/Users/andre/Documents/GitHub/rotational_diffusion_photophysics')
import rotational_diffusion_photophysics as rdp
from plot_data_sphere import plot_data_sphere
from rotational_diffusion_photophysics_models import starss2, starss1

def plot_sphere_series(experiment, t, species=[0,1], vmaxs=[1,[]], cmaps=[plt.cm.Blues, plt.cm.Greens], scale=3, plot_colorbar=True):
    # This functions plot a series of data spheres using plot_data_sphere(),
    # to illustrate the time evolution of the whole experiment

    # Compute the experimental signals and SH coefficients
    s = experiment.detector_signals(t)  # detector signals
    c = experiment._c  # SH coefficients

    # Number of species and time points
    nspecies = np.size(species)
    ntimes = np.size(t)

    # Create a figure with a suitable size
    fig = plt.figure()
    fig.set_dpi(300)
    fig.set_figwidth(3*ntimes)
    fig.set_figheight(3*nspecies)

    # Plot all the spheres
    for i in range(nspecies):
        for j in range(ntimes):
            ax = fig.add_subplot(nspecies, ntimes, (j+1) + (i)*ntimes, projection='3d')
            cij = c[species[i],:,j]
            gridij = rdp.vec2grid(cij, lmax=16)
            plot_data_sphere(ax, gridij.data, scale=scale, plot_colorbar=True, vmax=vmaxs[i], cmap=cmaps[i])
            if i == 0:
                ax.set_title("%.2e" % t[j] + ' s')
    return