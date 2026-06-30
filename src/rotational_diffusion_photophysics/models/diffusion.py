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

    def diffusion_matrix_so3(self, l, m, n, nspecies):
        # SO(3) (Wigner-D) blocks: isotropic diffusion is diagonal with eigenvalue
        # -l(l+1)D_r, independent of the body index n.
        block = np.diag((-l * (l + 1) * self.diffusion_coefficient).astype(complex))
        return np.repeat(block[None, :, :], nspecies, axis=0)


class AnisotropicDiffusion:
    """Free rotational diffusion of an axially-symmetric top (symmetry axis = the
    body z-axis, i.e. the Wigner-D body index n), with diffusion coefficients
    D_parallel (about the symmetry axis) and D_perp (tumbling of the axis).

    SO(3)-only: the body-frame tensor is meaningful only when the full orientation
    is tracked (the S^2 dipole-axis engine cannot represent it). The operator is
    diagonal in the (l, m, n) basis:
        eigenvalue = -[ l(l+1) D_perp  +  n^2 (D_parallel - D_perp) ]
    (Favro 1960; Woessner 1962). The rank-2 (l=2) rates are the classic symmetric-
    top trio 6 D_perp (n=0), 5 D_perp + D_par (n=+-1), 2 D_perp + 4 D_par (n=+-2),
    giving the up-to-3-exponential anisotropy decay when the transition dipole is
    tilted off the symmetry axis. D_parallel = D_perp recovers isotropic diffusion.
    Free rotor -> the anisotropy decays fully (no plateau).
    """

    def __init__(self,
                 diffusion_coefficient_parallel=14e-9,  # D_par about symmetry axis [Hz]
                 diffusion_coefficient_perp=14e-9,       # D_perp tumbling [Hz]
                 ):
        self.diffusion_coefficient_parallel = diffusion_coefficient_parallel
        self.diffusion_coefficient_perp = diffusion_coefficient_perp
        # An effective isotropic coefficient, for any code path that asks for one.
        self.diffusion_coefficient = diffusion_coefficient_perp

    def diffusion_matrix_so3(self, l, m, n, nspecies):
        Dpar, Dperp = (self.diffusion_coefficient_parallel,
                       self.diffusion_coefficient_perp)
        ddiag = -(l * (l + 1) * Dperp + n**2 * (Dpar - Dperp))
        block = np.diag(ddiag.astype(complex))
        return np.repeat(block[None, :, :], nspecies, axis=0)

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

    i.e. the transition-dipole axis is ordered relative to the director (theta from
    the director). The order strength and SIGN are set by ``lam`` (a50 n010/n020):

      lam > 0 : "prolate" / along-director order -- dipoles align WITH the director
                (wobble in a cone about it). S2 = <P2>_eq in (0, 1].
      lam = 0 : free isotropic diffusion (recovered exactly).
      lam < 0 : "oblate" / in-plane order -- dipoles lie in the plane PERPENDICULAR
                to the director (repelled from it, e.g. a dye lying in the membrane
                plane while the director is the membrane normal). S2 in [-1/2, 0).

    The equilibrium dipole distribution is c_eq ~ exp(lam P2); the second-rank order
    parameter S2 = <P2>_eq (positive prolate, negative oblate) and the residual
    (plateau) anisotropy is r_inf = r0 * S2^2 (>= 0 for either sign). Limits:
    lam -> +inf gives S2 -> 1 (fully along director); lam -> -inf gives S2 -> -1/2
    (fully in-plane).

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
    grow with |lam| (n020 M0): ~8 for lam<=2, ~20 for lam=5; STRONG in-plane order
    (lam very negative) concentrates c_eq into an equatorial ring and needs even
    higher l_max (e.g. lam=-8 is not converged by l_max=14) -- use moderate |lam| or
    the symmetrized (psi-space) form for tight oblate order.
    """

    def __init__(self,
                 diffusion_coefficient=14e-9,  # GFP rotational diffusion [Hz]
                 lam=0.0,                       # ordering strength (Maier-Saupe)
                 director=(0.0, 0.0),           # (theta, phi) of the ordering axis [rad]
                 ):
        self.diffusion_coefficient = diffusion_coefficient
        self.lam = lam
        # Lab-frame orientation (theta, phi) of the preferred-alignment axis
        # (the director, e.g. the membrane normal). Default (0,0) = along lab-z (the
        # optical axis). For an ISOTROPIC ensemble the anisotropy decay is
        # rotation-invariant -> the director is irrelevant; it matters for a
        # MACROSCOPICALLY ALIGNED sample (oriented membrane), where it sets the
        # equilibrium's lab orientation and hence the polarized signal levels. A
        # tilted director makes the potential non-axial (m != 0): on S^2 the even-l
        # reduction still holds; on SO(3) it breaks even-m (use even_m_reduction=False).
        self.director = director

    def diffusion_matrix(self, l, m, nspecies):
        block = ordering_potential_block(l, m, self.diffusion_coefficient, self.lam,
                                         self.director)
        return np.repeat(block[None, :, :], nspecies, axis=0)

    def diffusion_matrix_so3(self, l, m, n, nspecies):
        """SO(3) (Wigner-D) blocks of the same ordering potential, restricting a
        body axis to the lab director (U/kT = -lam P2(cos beta), beta = polar Euler
        angle). Built like the S^2 case (Gamma = T+ L~ T-) but with the exact
        complex-Wigner-D product (engine_so3.multiplication_operator via CG): the
        beta-only V_eff and c_eq^{+-1/2} factors enter as (L, 0, 0) coefficients,
        coupling l (Delta l <= +-4) while preserving m and n. For a dipole along the
        body axis (n=0) this reduces to the S^2 operator; the n!=0 couplings (via the
        body-index CG) are the genuinely-SO(3) content, relevant when the dipole
        reorients within an ordered barrel. Needs a higher l_max than S^2 (the
        T+ L~ T- construction; ~12 for lam=2)."""
        from rotational_diffusion_photophysics.engine_so3 import multiplication_operator
        lmax = int(np.max(l))
        Dr, lam = self.diffusion_coefficient, self.lam
        a0, a2, a4 = 3 * lam**2 / 10, 3 * lam**2 / 14 - 3 * lam, -18 * lam**2 / 35
        cV = _beta_wignerd_coeffs({0: a0, 2: a2, 4: a4}, l, m, n)
        Ltilde = (np.diag((-Dr * l * (l + 1)).astype(complex))
                  - Dr * multiplication_operator(l, m, n, cV, lmax))
        Tplus = multiplication_operator(l, m, n, _beta_wignerd_coeffs(
            _legendre_coeffs(lambda x: np.exp(lam * _P2(x) / 2), lmax), l, m, n), lmax)
        Tminus = multiplication_operator(l, m, n, _beta_wignerd_coeffs(
            _legendre_coeffs(lambda x: np.exp(-lam * _P2(x) / 2), lmax), l, m, n), lmax)
        block = Tplus @ Ltilde @ Tminus
        return np.repeat(block[None, :, :], nspecies, axis=0)

    def equilibrium_coeffs(self, l, m):
        """SH coefficients of the equilibrium dipole distribution c_eq ~ exp(lam P2),
        oriented along the director, normalized so the (l=0,m=0) (population)
        coefficient is 1. The operator's null mode; the aligned-sample IC (n020 M2.5)."""
        if self.director[0] == 0.0:
            c = _axial_coeffs(lambda x: np.exp(self.lam * _P2(x)), l, m)
        else:
            c = _oriented_coeffs(lambda x: np.exp(self.lam * _P2(x)), l, m, self.director)
        return c / c[np.logical_and(l == 0, m == 0)][0]

    def equilibrium_coeffs_so3(self, l, m, n):
        """Wigner-D coefficients of the SO(3) equilibrium c_eq ~ exp(lam P2(cos beta))
        at (l, 0, 0), normalized so the (0,0,0) (population) coefficient is 1. The
        operator's null mode -> the aligned-sample initial condition (n020 M2.5)."""
        c = _beta_wignerd_coeffs(
            _legendre_coeffs(lambda x: np.exp(self.lam * _P2(x)), int(np.max(l))),
            l, m, n)
        return c / c[np.logical_and.reduce((l == 0, m == 0, n == 0))][0]

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


def _legendre_coeffs(func, lmax, ngauss=256):
    # Legendre coefficients g_L of a function func(x) of x=cos(beta):
    #   func = sum_L g_L P_L,  g_L = (2L+1)/2 int_{-1}^1 func P_L dx.
    from numpy.polynomial.legendre import leggauss
    from scipy.special import eval_legendre
    xg, wg = leggauss(ngauss)
    fx = func(xg)
    return {L: (2 * L + 1) / 2 * np.sum(wg * fx * eval_legendre(L, xg))
            for L in range(lmax + 1)}


def _beta_wignerd_coeffs(gL, l, m, n):
    # Wigner-D coefficients of a beta-only function g(beta) = sum_L g_L P_L(cos beta).
    # Since D^L_{0,0}(a,beta,c) = P_L(cos beta), the coefficients sit at (L, 0, 0).
    c = np.zeros(l.size, dtype=complex)
    for i in range(l.size):
        if m[i] == 0 and n[i] == 0:
            c[i] = gL.get(int(l[i]), 0.0)
    return c


def _P4(x):
    return (35 * x**4 - 30 * x**2 + 3) / 8


def _oriented_coeffs(func, l, m, director):
    # 4pi real-SH coefficients of func(cos gamma), gamma = angle to the director
    # (theta_d, phi_d): cos gamma = cos th cos th_d + sin th sin th_d cos(ph - ph_d).
    # Evaluated on a Driscoll-Healy grid and expanded (same pyshtools/'4pi' csphase=1
    # convention as sh_multiplication_operator). Reduces to _axial_coeffs at director
    # (0,0). A tilted director populates m != 0 (still even-l for an even func).
    import pyshtools as sht
    theta_d, phi_d = director
    lmax = int(np.max(l))
    grid = sht.SHGrid.from_zeros(lmax=2 * lmax + 2, kind='real')
    th = np.radians(90.0 - grid.lats())[:, None]   # colatitude
    ph = np.radians(grid.lons())[None, :]
    cosg = (np.cos(th) * np.cos(theta_d)
            + np.sin(th) * np.sin(theta_d) * np.cos(ph - phi_d))
    grid.data = func(cosg)
    cilm = grid.expand(normalization='4pi', csphase=1)
    return sht.shio.SHCilmToVector(cilm.coeffs, lmax)


def ordering_potential_block(l, m, diffusion_coefficient, lam, director=(0.0, 0.0)):
    # Bare Smoluchowski operator Gamma = T+ . L~ . T- for U/kT = -lam P2(cos gamma),
    # gamma measured from the director (see OrderingPotentialDiffusion). For an axial
    # director (theta_d=0) the V_eff coefficients are exact analytic (m=0); a tilted
    # director uses the grid expansion (m != 0), the V_eff is the same P0/P2/P4 but
    # oriented along the director.
    Dr = diffusion_coefficient
    a0, a2, a4 = 3 * lam**2 / 10, 3 * lam**2 / 14 - 3 * lam, -18 * lam**2 / 35
    if director[0] == 0.0:
        # V_eff = a0 P0 + a2 P2 + a4 P4 -> 4pi-SH coeff of P_l is a_l / sqrt(2l+1).
        cV = np.zeros(l.size)
        cV[np.logical_and(l == 0, m == 0)] = a0
        cV[np.logical_and(l == 2, m == 0)] = a2 / np.sqrt(5)
        cV[np.logical_and(l == 4, m == 0)] = a4 / 3.0    # sqrt(2*4+1) = 3
        cTp = _axial_coeffs(lambda x: np.exp(lam * _P2(x) / 2), l, m)
        cTm = _axial_coeffs(lambda x: np.exp(-lam * _P2(x) / 2), l, m)
    else:
        cV = _oriented_coeffs(lambda x: a0 + a2 * _P2(x) + a4 * _P4(x), l, m, director)
        cTp = _oriented_coeffs(lambda x: np.exp(lam * _P2(x) / 2), l, m, director)
        cTm = _oriented_coeffs(lambda x: np.exp(-lam * _P2(x) / 2), l, m, director)
    Ltilde = np.diag(-Dr * l * (l + 1)) - Dr * sh_multiplication_operator(cV, l, m)
    Tplus = sh_multiplication_operator(cTp, l, m)
    Tminus = sh_multiplication_operator(cTm, l, m)
    return Tplus @ Ltilde @ Tminus
