import numpy as np

##########################
# Diffusion Models Classes
##########################
class IsotropicDiffusion:
    def __init__(self,
                 diffusion_coefficient=14e-9,  # GFP rotational diffusion
                 ):
        # Rotational diffusion coefficient in Hertz
        self.diffusion_coefficient = diffusion_coefficient  # [Hz]

    def diffusion_matrix(self, l, m, nspecies):
        # Compute the diffusion matrix
        # Here an isotropic diffusion model is employed, every state has also
        # the same rotational diffusion properties.
        D = isotropic_diffusion_matrix(l, m,
                self.diffusion_coefficient,
                nspecies)
        return D

def isotropic_diffusion_matrix(l, m, diffusion_coefficient, nspecies):
    # Make all the diffusion matrices for all the species assuming isotropic
    # rotational diffusion and the same rotational diffusion coefficient for
    # every species.
    D = np.zeros( (nspecies, l.size, l.size) )
    D[0] = isotropic_diffusion_block(l, m, diffusion_coefficient)
    for i in np.arange(1,nspecies):
        D[i] = D[0]
    return D

def isotropic_diffusion_block(l, m, diffusion_coefficient):
    # Make the diffusion diagonal block for isotropic rotational diffusion
    ddiag = -l*(l+1)*diffusion_coefficient
    dblock = np.zeros([l.size, l.size])
    np.fill_diagonal(dblock, ddiag)
    return dblock
