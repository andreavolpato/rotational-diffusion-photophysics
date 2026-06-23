import numpy as np
from rotational_diffusion_photophysics.common import na_corrected_linear_coeffs

###################
# Detection classes
###################
class PolarizedDetection:
    def __init__(self,
                 polarization=['x', 'y'],
                 numerical_aperture=1.4,
                 refractive_index=1.518,
                 ):
        # Numerical aperture of the collection objective
        self.numerical_aperture = numerical_aperture

        # Refractive index of the immersion medium
        self.refractive_index = refractive_index

        # Polarization directions of the detection channels
        self.polarization = polarization
        return None

    def detector_coeffs(self, l, m):
        # Compute the SH expansion for the angular collection functions of the
        # detectors
        ndetectors = np.size(self.polarization)
        c = np.zeros((ndetectors, l.size))

        for i in np.arange(ndetectors):
            c[i] = na_corrected_linear_coeffs(l, m,
                polarization=self.polarization[i],
                numerical_aperture=self.numerical_aperture,
                refractive_index=self.refractive_index,
                )
        return c
