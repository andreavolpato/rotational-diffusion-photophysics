import numpy as np
import pyshtools as sht  # used in spherical calculations

################################################################################
# Anisotropy
################################################################################

def anisotropy(signals):
    return (signals[0] - signals[1]) / (signals[0] + 2*signals[1])

def anisotropy_variance(signals, signals_minus_background):
    il_wobg = signals_minus_background[0]; # Intensity parallel w/o background
    ip_wobg = signals_minus_background[1]; # Intensity perpendicular w/o background
    E_il = signals[0]; # Intensity parallel
    E_ip = signals[1]; # Intensity perpendicular
    
    # Expectations, variances and covariances
    Cov_il_ip = - E_il * E_ip / (E_il + E_ip)
    E_N = np.abs(il_wobg - ip_wobg)
    E_D = np.abs(il_wobg + 2*ip_wobg)
    Var_N = E_il + E_ip - 2*Cov_il_ip
    Var_D = E_il + 4*E_ip + 4*Cov_il_ip
    Cov_N_D = E_il - 2*E_ip + Cov_il_ip
    
    # Anisotropy Variance (see DP 20200224) and reference
    Var_r = (
        Var_N/np.power(E_D,2) 
        - 2*E_N*Cov_N_D/np.power(E_D,3) 
        + Var_D/np.power(E_D,4)
        )
    return Var_r

################################################################################
# Auxiliary functions
################################################################################

def photon_flux(power_density, wavelength):    
    # Compute the photon flux from power density and wavelength.
    # The wavelength is used to compute the photon energy
    # photon_energy = h*c/wavelength
    # The wavelength is coverted in meters.
    light_speed = 299792457  # [m/s]
    h_planck = 6.62607015e-34  # [J*s]
    photon_energy = h_planck * light_speed / (wavelength*1e-9)  # [J]
    photon_flux = power_density / photon_energy  # [photons/cm2]
    return photon_flux

# TODO Check that photon flux is conserved changing polarization state

################################################################################
# Axelrod and NA corrections
################################################################################

def na_corrected_linear_coeffs(l, m,
                               polarization='x', 
                               numerical_aperture=1.4,
                               refractive_index=1.518,
                               norm='4pi'):
    # Compute Axelrod high-NA correction coefficients
    k = axelrod_correction_k(numerical_aperture, refractive_index)

    # Get the SH expansion coefficients for x, y, and z photoselections.
    cx = linear_light_matter_coeffs(l, m, polarization='x', norm=norm)
    cy = linear_light_matter_coeffs(l, m, polarization='y', norm=norm)
    cz = linear_light_matter_coeffs(l, m, polarization='z', norm=norm)

    # Compute the SH expansion coefficients
    # We are assuming that the propagation direction is z.
    # If z polarization is chosen, the propagation direction is assumed as x.
    if polarization == 'x':
        c = k[0]*cz + k[1]*cy + k[2]*cx

    if polarization == 'y':
        c = k[0]*cz + k[1]*cx + k[2]*cy

    if polarization == 'z':
        c = k[0]*cx + k[1]*cy + k[2]*cz

    if polarization == 'xy':
        c = k[0]*cz + (k[1]+k[2])/2*cy + (k[1]+k[2])/2*cx

    # c are the SH coefficients of the expanded angular function.
    return c

def axelrod_correction_k(numerical_aperture, refractive_index):
    # Normalized Axelrod correction cartesian coefficients from [Fisz2005] eq.(2).
    # For a linear polarized beam:
    # - k[0] refers to the propagation direction
    # - k[1] refers to the perpendicular direction
    # - k[2] refers to the parallel direction
    # In the same paper the correction for the spherical coordinates problem are
    # discussed. It will be useful when adding the photoswithing angle.
    # Note that k[0] + k[1] + k[2] = 1. In other words, for a randomly oriented
    # sample in linear excitation, the total number of excited molecules is the
    # same regardless of the NA.

    # The maximum angle of the focalized beam with respect to the
    # propagation direction of the beam is computed.
    # NA = n*sin(theta), thus theta = arcsin(NA/n).
    max_ray_angle = np.arcsin(numerical_aperture/
                              refractive_index)

    k = np.zeros(3)
    costh = np.cos(max_ray_angle)
    k[0] = 1/6  * (2 -3*costh             +costh**3) / (1 - costh)
    k[1] = 1/24 * (1 -3*costh +3*costh**2 -costh**3) / (1 - costh)
    k[2] = 1/8  * (5 -3*costh   -costh**2 -costh**3) / (1 - costh)
    return k

def linear_light_matter_coeffs(l, m, polarization='x', 
                                    norm='4pi'):
    # This function computes the spherical harmonics expansion for linear 
    # light matter interactions photoselection functions.
    # For example, the function (r dot x)^2 is expanded and has only three non 
    # zero coefficients, in particular l=0, m=0 and l=2, m=0,2. r and x are 
    # versors. In this function few polarizations are expressed in terms of
    # their SH coefficients.
    c = np.zeros(l.shape)
    if polarization == 'x':
        c[np.logical_and(l==0, m==0)] = 1/3
        c[np.logical_and(l==2, m==0)] = -np.sqrt(1/45)
        c[np.logical_and(l==2, m==2)] = np.sqrt(1/15)
    if polarization == 'y':
        c[np.logical_and(l==0, m==0)] = 1/3
        c[np.logical_and(l==2, m==0)] = -np.sqrt(1/45)
        c[np.logical_and(l==2, m==2)] = -np.sqrt(1/15)
    if polarization == 'z':
        c[np.logical_and(l==0, m==0)] = 1/3
        c[np.logical_and(l==2, m==0)] = np.sqrt(4/45)
    if polarization == 'xy':  # circular polarization in the xy plane
        c[np.logical_and(l==0, m==0)] = 1/3
        c[np.logical_and(l==2, m==0)] = -np.sqrt(1/45)
    if norm == 'ortho':
        c = c/np.sqrt(4*np.pi)

    # Normalize the coefficients so the total integral of the function is 1
    # c = c/c[0]  # For now it is not on, because we loose the probability
    # connection
    return c

################################################################################
# Angular auxiliary functions
################################################################################

def make_angles(lmax):
    # Create angles for computing your functions of theta and phy
    cos2 = sht.SHGrid.from_zeros(lmax=lmax, kind='real')
    # cos2 = sht.shclasses.DHComplexGrid.from_zeros(lmax=2)
    theta = cos2.lats()
    phi = cos2.lons()
    omega = np.meshgrid(theta,phi)
    omega[0] = omega[0].transpose()*np.pi/180
    omega[1] = omega[1].transpose()*np.pi/180
    return omega

def make_grid(farray, lmax):
    # Make a sht grid from array data and expand in sh up to lmax
    fgrid = sht.SHGrid.from_zeros(lmax=lmax, kind='real')
    # fgrid = sht.SHGrid.from_array(farray)
    fgrid.data = farray
    fcilm = fgrid.expand(normalization='4pi')
    fvec = sht.shio.SHCilmToVector(fcilm.coeffs, lmax)

    # Remove small coefficients from errors
    fvec[np.abs(fvec)<1e-6] = 0 
    return fgrid, fvec, fcilm

def vecs2grids(cvecs):
    # Convert an array of vectors of SH coefficients to an array of SH grids
    cgridarray = []
    ntimes = cvecs.shape[1]
    for i in range(ntimes):
        cvec = cvecs[:,i]
        cgrid = vec2grid(cvec)
        cgridarray.append(cgrid)
    return cgridarray

def vec2grid(cvec, lmax=32):
    # Convert a vector of SH coefficients to an SH grid
    cilm = sht.shio.SHVectorToCilm(cvec)
    carr = sht.expand.MakeGridDH(cilm, sampling=2, extend=1, norm=1, lmax=lmax) # 4 orthonorm, 1 4pi
    cgrid = sht.shclasses.SHGrid.from_array(carr)
    return cgrid


################################################################################
# Spherical harmonics coefficients operations
################################################################################

def kinetic_prod_block(kvec, cgp):
    # Create a kinetic constant block for multiplication using cg coeffs.
    kblock = np.transpose(cgp, [2, 1, 0]).dot(kvec)
    return kblock
