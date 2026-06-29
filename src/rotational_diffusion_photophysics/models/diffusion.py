import numpy as np

from rotational_diffusion_photophysics.utils.common import sh_multiplication_operator

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


class OrderingPotentialDiffusion:
    """Restricted rotational diffusion in an axial Maier-Saupe ordering potential

        U(theta)/kT = -lam * P2(cos theta),

    i.e. the transition-dipole axis wobbles in a soft "cone" whose tightness is
    set by the order strength ``lam`` (a50 n010/n020). The equilibrium dipole
    distribution is c_eq ~ exp(lam P2); the second-rank order parameter
    S2 = <P2>_eq and the residual (plateau) anisotropy is r_inf = r0 * S2^2.
    ``lam = 0`` recovers free isotropic diffusion exactly.

    This subsumes wobbling-in-cone / Lipari-Szabo / MOMD / the strong-ordering
    limit of SRLS in a single operator (n010).

    Implementation: returns the *bare* rotational-Smoluchowski operator acting on
    the density c, built as  Gamma = T+ . L~ . T-  where L~ is the symmetrized
    operator (M0, n020)  L~ = Dr[Lap_S2 - V_eff],
        V_eff = 3 lam^2/10 + (3 lam^2/14 - 3 lam) P2 - (18 lam^2/35) P4,
    and T+- multiply by c_eq^{+-1/2} = exp(+-lam P2/2). Only the multiplication
    machinery is used (no first-derivative operator). The block is even-l,
    m-conserving (parity-reduction compatible).

    IMPORTANT: the equilibrium of Gamma is the anisotropic c_eq, so a
    *macroscopically aligned* sample must START at c_eq (n020 M2.5), else the dark
    dynamics show a spurious isotropic->c_eq quench. For an isotropic ensemble
    (vesicles) use the Plan-A analytic anisotropy multiplier instead. l_max must
    grow with ``lam`` (n020 M0 result): ~8 for lam<=2, ~20 for lam=5.
    """

    def __init__(self,
                 diffusion_coefficient=14e-9,  # GFP rotational diffusion [Hz]
                 lam=0.0,                       # ordering strength (Maier-Saupe)
                 ):
        self.diffusion_coefficient = diffusion_coefficient
        self.lam = lam

    def diffusion_matrix(self, l, m, nspecies):
        block = ordering_potential_block(l, m, self.diffusion_coefficient, self.lam)
        return np.repeat(block[None, :, :], nspecies, axis=0)

    def equilibrium_coeffs(self, l, m):
        """SH coefficients of the equilibrium dipole distribution c_eq ~ exp(lam P2),
        normalized so the l=0 (population) coefficient is 1. This is the operator's
        null mode; use it as the initial condition for an aligned sample (n020 M2.5)."""
        c = _axial_coeffs(lambda x: np.exp(self.lam * _P2(x)), l, m)
        return c / c[np.logical_and(l == 0, m == 0)][0]

    def order_parameter(self, ngauss=256):
        """Second-rank order parameter S2 = <P2>_eq (Maier-Saupe), c_eq ~ exp(lam P2)."""
        from numpy.polynomial.legendre import leggauss
        xg, wg = leggauss(ngauss)
        ceq = np.exp(self.lam * _P2(xg))
        return float(np.sum(wg * _P2(xg) * ceq) / np.sum(wg * ceq))


def _P2(x):
    return (3 * x**2 - 1) / 2


def _axial_coeffs(func, l, m, ngauss=256):
    # 4pi real-SH coefficients of an axial function func(x), x = cos(theta):
    #   c[l, 0] = sqrt(2l+1)/2 * int_{-1}^1 func(x) P_l(x) dx ;  c[l, m!=0] = 0.
    from numpy.polynomial.legendre import leggauss
    from scipy.special import eval_legendre
    xg, wg = leggauss(ngauss)
    fx = func(xg)
    c = np.zeros(l.size)
    for i in range(l.size):
        if m[i] == 0:
            c[i] = np.sqrt(2 * l[i] + 1) / 2 * np.sum(wg * fx * eval_legendre(int(l[i]), xg))
    return c


def ordering_potential_block(l, m, diffusion_coefficient, lam):
    # Bare Smoluchowski operator Gamma = T+ . L~ . T- for U/kT = -lam P2 (see
    # OrderingPotentialDiffusion). Exact analytic V_eff coefficients; the
    # c_eq^{+-1/2} factors via Gauss-Legendre projection.
    Dr = diffusion_coefficient
    # V_eff = a0 P0 + a2 P2 + a4 P4  ->  SH coeff of P_l is a_l / sqrt(2l+1).
    a0, a2, a4 = 3 * lam**2 / 10, 3 * lam**2 / 14 - 3 * lam, -18 * lam**2 / 35
    cV = np.zeros(l.size)
    cV[np.logical_and(l == 0, m == 0)] = a0
    cV[np.logical_and(l == 2, m == 0)] = a2 / np.sqrt(5)
    cV[np.logical_and(l == 4, m == 0)] = a4 / 3.0    # sqrt(2*4+1) = 3
    Ltilde = np.diag(-Dr * l * (l + 1)) - Dr * sh_multiplication_operator(cV, l, m)
    Tplus = sh_multiplication_operator(
        _axial_coeffs(lambda x: np.exp(lam * _P2(x) / 2), l, m), l, m)
    Tminus = sh_multiplication_operator(
        _axial_coeffs(lambda x: np.exp(-lam * _P2(x) / 2), l, m), l, m)
    return Tplus @ Ltilde @ Tminus
