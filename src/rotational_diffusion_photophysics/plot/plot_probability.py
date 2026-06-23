import numpy as np
import matplotlib.pyplot as plt  # used only for plotting tools
from common import make_angles

################################################################################
# Plotting functions
################################################################################

def plot_prob_sphere(grid):
    prob = grid.data
    omega = make_angles(grid.lmax)
    cmap = plt.cm.Blues

    x = np.cos(omega[0]) * np.cos(omega[1])
    y = np.cos(omega[0]) * np.sin(omega[1])
    z = np.sin(omega[0])

    ax = plt.subplot(111, projection='3d')
    ax._axis3don = False
    ax.plot_surface(x, y, z,
                    cstride=1,
                    rstride=1,
                    facecolors=cmap(prob))
    ax.view_init(elev=30, azim=150)
    ax.set_xticks(np.linspace(-1,1,5))
    ax.set_yticks(np.linspace(-1,1,5))
    ax.set_zticks(np.linspace(-1,1,5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    return

def plot_proj(grid, clims=[0, 1], cmap=plt.cm.Blues):
    prob = grid.data

    if clims == []:
        clims = [np.min(prob), np.max(prob)]

    im1 = plt.imshow(prob, extent=(0, 2, -0.5, 0.5),
                    vmin=clims[0], vmax=clims[1],
                    cmap=cmap, alpha=1)
    plt.xlabel('$\phi$ $(\pi)$')
    plt.ylabel('$\\theta$ $(\pi)$')
    plt.xticks([0, 0.5, 1, 1.5, 2])
    plt.yticks([0.5, 0.25, 0, -0.25, -0.5])
    plt.colorbar(ax=plt.gca())
    return