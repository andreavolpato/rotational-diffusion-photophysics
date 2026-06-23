import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
import quaternionic

def equal_aspect_3d_axes(ax, position=[0,0,1,1], elev=45, azim=45):
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    ax.set_position(position)
    ax.patch.set_alpha(0)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    # Set aspect ratio
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)

    # Remove ticks to gain space and remove axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    return

def plot_circular_axes(ax2, azim=45, elev=45, scale=3,
                       linewidth=0.5, plot_arrows=True):
    azim = azim/180*np.pi
    elev = elev/180*np.pi

    # half equator
    N = 256
    r = np.zeros((3,N))
    phi = np.linspace(-np.pi/2, np.pi/2, 256)
    theta = np.pi/2
    r[0] = np.cos(phi) * np.sin(theta)
    r[1] = np.sin(phi) * np.sin(theta)
    r[2] = np.cos(theta)
    q = quaternionic.array([np.cos(azim/2), 0, 0, np.sin(azim/2)])  # Create a new array
    R = q.to_rotation_matrix
    r = R.dot(r)
    ax2.plot(r[0], r[1], r[2], linewidth=linewidth, color=[0,0,0], zorder=10)

    # x vertical
    r = np.zeros((3,N))
    phi = 0
    theta = np.linspace(0, np.pi, N)
    r[0] = np.cos(phi) * np.sin(theta) 
    r[1] = np.sin(phi) * np.sin(theta) 
    r[2] = np.cos(theta)
    q = quaternionic.array([np.cos(-1.29*elev/2), 0, np.sin(-1.29*elev/2), 0])  # Create a new array
    R = q.to_rotation_matrix
    r = R.dot(r)
    ax2.plot(r[0], r[1], r[2], linewidth=linewidth, color=[0,0,0], zorder=10)

    # y vertical
    r = np.zeros((3,N))
    phi = np.pi/2
    theta = np.linspace(0, np.pi, N)
    r[0] = np.cos(phi) * np.sin(theta) 
    r[1] = np.sin(phi) * np.sin(theta) 
    r[2] = np.cos(theta)
    q = quaternionic.array([np.cos(1.29*elev/2), np.sin(1.29*elev/2), 0, 0])  # Create a new array
    R = q.to_rotation_matrix
    r = R.dot(r)
    ax2.plot(r[0], r[1], r[2], linewidth=linewidth, color=[0,0,0], zorder=10)

    # outer circle
    r = np.zeros((3,N))
    phi = np.linspace(0, 2*np.pi, N)
    theta = np.pi/2
    r[0] = np.cos(phi) * np.sin(theta) 
    r[1] = np.sin(phi) * np.sin(theta) 
    r[2] = np.cos(theta)
    q = quaternionic.array([np.cos(np.pi/4-elev/2), 
                            -np.sin(azim)*np.sin(np.pi/4-elev/2), 
                            np.cos(azim)*np.sin(np.pi/4-elev/2),
                             0])  # Create a new array
    R = q.to_rotation_matrix
    r = R.dot(r)
    ax2.plot(r[0], r[1], r[2], linewidth=linewidth, color=[0,0,0], zorder=10)

    if plot_arrows:
        # Arrows
        arrowprops = dict(mutation_scale=5*scale,
                          linewidth=linewidth*2,
                          arrowstyle='-|>',
                          color='k')
        arrow_length = 0.55
        arrow_pos = 0.98
        # x arrow
        a = Arrow3D([arrow_pos, arrow_pos+arrow_length], [0,0], [0,0], **arrowprops)
        ax2.add_artist(a)
        # y arrow
        a = Arrow3D([0, 0], [arrow_pos, arrow_pos+arrow_length], [0,0], **arrowprops)
        ax2.add_artist(a)   
        # z arrow
        a = Arrow3D([0, 0], [0,0], [arrow_pos, arrow_pos+arrow_length], **arrowprops)
        ax2.add_artist(a)
        # Arrow labels
        ax2.text(1.66, 0.21, 0, 'x', size=4*scale)
        ax2.text(0.27, 1.57, 0, 'y', size=4*scale)
        ax2.text(0, 0.14, 1.39, 'z', size=4*scale)
    return

# For arrows, see
# https://stackoverflow.com/questions/29188612/arrows-in-matplotlib-using-mplot3d
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_data_sphere(ax, data,
                     cmap=plt.cm.Greens,
                     scale=5,
                     elev=35.2644,
                     azim=45,
                     plot_arrows=True,
                     vmin=0,
                     vmax=1,
                     plot_colorbar=True,
                     ax2=[],
                     cbax=[]):
    # Plot a spherical surface
    # It takes roughly 1 second per plot.

    #TODO: interp data so we have multiple of 4 and 2 for nphi-1 and ntheta-1
    # nphi=181 ntheta=91 for example
    # use scipy.interpolate.interp2d for the interpolation
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp2d.html

    # Normalize data
    if np.size(vmin) == 0:
        vmin = np.min(data)
    if np.size(vmax) == 0:
        vmax = np.max(data)
    data_norm = (data-vmin)/vmax

    # Compute angles and coordinates
    ntheta = data.shape[0]  # should be odd
    nphi = data.shape[1]  # should be odd
    r = np.zeros((3, ntheta, nphi))
    phi = np.linspace(0, 2*np.pi, nphi)
    theta = np.linspace(0, np.pi, ntheta)
    r[0] = np.outer(np.sin(theta), np.cos(phi))
    r[1] = np.outer(np.sin(theta), np.sin(phi))
    r[2] = np.outer(np.cos(theta), np.ones(np.size(phi)))

    # Plot the spherical surface
    surf = ax.plot_surface(r[0], r[1], r[2], facecolors=cmap(data_norm),
                    shade=False, antialiased=False,
                    alpha=1, linewidth=0,
                    rstride=1, cstride=1, vmin=vmin, vmax=vmax)
    # plt.colorbar(surf, ax=ax, shrink=0.4, orientation='horizontal')
    equal_aspect_3d_axes(ax, elev=elev, azim=azim, position=ax.get_position())


    if ax2 == []:
        ax2 = plt.axes(ax.get_position(), projection='3d')
    plot_circular_axes(ax2, azim=azim, elev=elev,
                       scale=scale, 
                       linewidth=0.5, 
                       plot_arrows=plot_arrows)
    equal_aspect_3d_axes(ax2, elev=elev, azim=azim, position=ax.get_position())

    if plot_colorbar:
        norm = Normalize(vmin=vmin, vmax=vmax)
        box = ax.get_position()
        if cbax == []:
            cbax = plt.axes(ax.get_position())
        cbax.set_position([box.x0+box.width*0.24,
                        box.y0+box.height*0.075,
                        box.width*0.56,
                        box.height*0.025])
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbax, orientation='horizontal')
    return surf

if __name__ == '__main__':

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    data = np.random.uniform(size=(181,91))
    plot_data_sphere(ax, data, cmap=plt.cm.Blues, scale=3, vmin=0, vmax=1)

    plt.show()

    
    