import numpy as np
from rotational_diffusion_photophysics.common import na_corrected_linear_coeffs, photon_flux, kinetic_prod_block

######################
# Illumination classes
######################
class SingleLaser:
    def __init__(self,
                 power_density,
                 polarization='x',
                 wavelength=488, 
                 numerical_aperture=1.4,
                 refractive_index=1.518):
        # Compute the photon flux from power density and wavelength
        self.power_density = power_density  # [W/cm2]
        self.wavelength = wavelength  # [nm]
        self.photon_flux = photon_flux(self.power_density,
                                       self.wavelength)  # [photons/cm2]

        # Polarization of the beam
        # It can be 'x', 'y', or 'c' for circular
        self.polarization = polarization

        # Numerical aperture of excitation light beam
        self.numerical_aperture = numerical_aperture

        # Refractive index of the immersion medium of the objective
        self.refractive_index = refractive_index

        # Time windows settings for compatibility
        self.modulation = np.array([[1]], dtype='float')
        self.time_windows = np.array([0], dtype='float')
        self.nwindows = 1
        self.time0 = 0
        return None

    def photon_flux_prod_coeffs(self, l, m, wigner_3j_prod_coeffs):
        # Compute the prod_matrix for the angular dependence of the photon flux.
        # When the photon flux is multiplied by the absorbtion cross-section
        # the kinetic rate of absortion is obtained.
        c_exc = np.zeros((1, l.size))
        c_exc[0] = na_corrected_linear_coeffs(l, m,
                polarization = self.polarization,
                numerical_aperture = self.numerical_aperture,
                refractive_index = self.refractive_index,
                ) * self.photon_flux

        # Return the matrix F as a 3 dimensional array, this will be helpfull
        # when more than one laser is used to excite the sample.
        F = np.zeros((1, l.size, l.size))
        F[0] = kinetic_prod_block(c_exc[0], wigner_3j_prod_coeffs)
        return F, c_exc

class ModulatedLasers:
    def __init__(self,
                 power_density,
                 polarization=['x', 'xy'],
                 wavelength=[405, 488],
                 modulation=[[1, 0], [1, 1]],
                 time_windows=[250e-9, 1e-3],
                 time0=0,
                 numerical_aperture=1.4,
                 refractive_index=1.518):
        # Compute the photon flux from power density and wavelength
        self.power_density = np.array(power_density)  # [W/cm2]
        self.wavelength = np.array(wavelength)  # [nm]
        self.photon_flux = photon_flux(self.power_density,
                                       self.wavelength)  # [photons/cm2]

        # Polarization of the beam
        # It can be 'x', 'y', or 'c' for circular. 
        self.polarization = polarization

        # Time Modulation properties
        self.modulation = np.array(modulation, dtype='float')
        self.time_windows = np.array(time_windows, dtype='float')
        self.nwindows = self.time_windows.shape[0]
        self.time0 = time0

        # Numerical aperture of excitation light beam
        # Refractive index of the immersion medium of the objective
        self.numerical_aperture = numerical_aperture
        self.refractive_index = refractive_index

        return None

    def photon_flux_prod_coeffs(self, l, m, wigner_3j_prod_coeffs):
        # Compute the prod_matrix for the angular dependence of the photon flux.
        # When the photon flux is multiplied by the absorbtion cross-section
        # the kinetic rate of absortion is obtained.

        # Initialize arrays for SH photoselection coefficients in c, and 
        # product coefficients in F.
        nlasers = np.size(self.polarization)
        c_exc = np.zeros((nlasers, l.size))
        F = np.zeros((nlasers, l.size, l.size))
        for i in np.arange(nlasers):
            c_exc[i] = na_corrected_linear_coeffs(l, m,
                    polarization = self.polarization[i],
                    numerical_aperture = self.numerical_aperture,
                    refractive_index = self.refractive_index,
                    ) * self.photon_flux[i]
            F[i] = kinetic_prod_block(c_exc[i], 
                        wigner_3j_prod_coeffs)

        # F is a 3 dimensional array. The firt index is for the lasers, the last
        # indexes are for lm of SH.
        return F, c_exc
