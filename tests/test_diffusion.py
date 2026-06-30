"""Tests for diffusion models, in particular the Maier-Saupe ordering-potential
operator (a50 n020 M1). These are the M0 acceptance tests at the operator level:
lam->0 reduces to isotropic; the equilibrium c_eq is the operator's null mode; the
spectrum is sensible; and the new general multiplication operator agrees with the
engine's existing (degree-2) product machinery.
"""
import numpy as np
import pytest

from rotational_diffusion_photophysics.engine_s2 import (
    quantum_numbers, real_sh_product_coeffs,
)
from rotational_diffusion_photophysics.engine_so3 import quantum_numbers_so3
from rotational_diffusion_photophysics.models.diffusion import (
    IsotropicDiffusion, OrderingPotentialDiffusion, ordering_potential_block,
    AnisotropicDiffusion,
)
from rotational_diffusion_photophysics.utils.common import (
    kinetic_prod_block, sh_multiplication_operator, linear_light_matter_coeffs,
)


def test_sh_multiplication_operator_matches_product_table():
    # The new general builder must equal the existing degree-2 path
    # kinetic_prod_block(g, real_sh_product_coeffs) for a degree-2 multiplier
    # (here the x-pol photoselection function, l in {0,2}).
    l, m = quantum_numbers(6)
    g = linear_light_matter_coeffs(l, m, polarization="x")
    new = sh_multiplication_operator(g, l, m)
    old = kinetic_prod_block(g, real_sh_product_coeffs(l, m))
    np.testing.assert_allclose(new, old, atol=1e-10)


def test_sh_multiplication_operator_is_symmetric():
    # multiply-by-(real function) is self-adjoint: M[k,j] = (1/4pi) int Y_k g Y_j.
    l, m = quantum_numbers(6)
    g = linear_light_matter_coeffs(l, m, polarization="x")
    M = sh_multiplication_operator(g, l, m)
    np.testing.assert_allclose(M, M.T, atol=1e-12)


@pytest.mark.parametrize("lmax", [6, 8])
def test_ordering_potential_reduces_to_isotropic_at_zero_lambda(lmax):
    # lam = 0 must reproduce free isotropic diffusion exactly.
    l, m = quantum_numbers(lmax)
    Dr = 3.0
    iso = IsotropicDiffusion(diffusion_coefficient=Dr).diffusion_matrix(l, m, 1)[0]
    op = OrderingPotentialDiffusion(diffusion_coefficient=Dr, lam=0.0
                                    ).diffusion_matrix(l, m, 1)[0]
    np.testing.assert_allclose(op, iso, atol=1e-10)


def test_ordering_potential_block_is_even_l_parity_preserving():
    # All multipliers (V_eff, c_eq^{+-1/2}) are even -> the operator must not
    # couple even-l to odd-l (so the engine's even-l reduction stays valid).
    l, m = quantum_numbers(8)
    G = ordering_potential_block(l, m, 1.0, 2.0)
    even, odd = (l % 2 == 0), (l % 2 == 1)
    np.testing.assert_allclose(G[np.ix_(odd, even)], 0.0, atol=1e-9)
    np.testing.assert_allclose(G[np.ix_(even, odd)], 0.0, atol=1e-9)


# l_max must grow with the order strength lam (n020 M0 result): the peaked c_eq
# needs more harmonics. These (lam, l_max) pairs are converged.
@pytest.mark.parametrize("lam,lmax,tol", [(1.0, 16, 1e-6), (2.0, 20, 1e-5)])
def test_equilibrium_is_the_operator_null_mode(lam, lmax, tol):
    # Gamma . c_eq = 0  (the anisotropic Boltzmann distribution is the steady
    # state of the ordering-potential diffusion operator).
    l, m = quantum_numbers(lmax)
    model = OrderingPotentialDiffusion(diffusion_coefficient=2.5, lam=lam)
    G = model.diffusion_matrix(l, m, 1)[0]
    ceq = model.equilibrium_coeffs(l, m)
    residual = np.linalg.norm(G @ ceq) / np.linalg.norm(ceq)
    assert residual < tol, f"||G c_eq|| / ||c_eq|| = {residual:.2e}"


def test_ordering_potential_spectrum_is_physical():
    # In the engine-relevant even-l, even-m sector (the rank-2 wobble physics; the
    # odd-l rank-1 sector is dropped by the even-l reduction): real eigenvalues,
    # all <= 0, with exactly one isolated null mode (the equilibrium). The slowest
    # nonzero mode is the l=2 wobble (~ -6 Dr at this lam), not near zero.
    l, m = quantum_numbers(12)
    G = ordering_potential_block(l, m, 1.0, 2.0)
    sel = (l % 2 == 0) & (m % 2 == 0)
    ev = np.sort(np.linalg.eigvals(G[np.ix_(sel, sel)]).real)[::-1]
    assert abs(ev[0]) < 1e-6                      # isolated null mode (equilibrium)
    assert ev[1] < -1.0                           # next mode is a genuine decay
    assert np.all(ev < 1e-6)                      # no growth


@pytest.mark.parametrize("lam,expected", [(0.0, 0.0), (2.0, 0.4393), (5.0, 0.7771)])
def test_order_parameter(lam, expected):
    # S2 = <P2>_eq, the Maier-Saupe second-rank order parameter.
    s2 = OrderingPotentialDiffusion(lam=lam).order_parameter()
    assert abs(s2 - expected) < 1e-3


def test_in_plane_ordering_negative_lambda():
    # lam < 0 -> oblate / in-plane order: S2 negative, in [-1/2, 0); the limits are
    # S2 -> +1 (lam->+inf, along director) and S2 -> -1/2 (lam->-inf, in-plane).
    assert OrderingPotentialDiffusion(lam=-2.0).order_parameter() < 0
    assert -0.5 < OrderingPotentialDiffusion(lam=-2.0).order_parameter() < 0
    assert OrderingPotentialDiffusion(lam=-50.0).order_parameter() < -0.48   # -> -1/2
    assert OrderingPotentialDiffusion(lam=+50.0).order_parameter() > 0.97    # -> +1


def test_in_plane_operator_null_mode():
    # The operator works for in-plane ordering too (moderate |lam| converges).
    l, m = quantum_numbers(14)
    model = OrderingPotentialDiffusion(1.0, lam=-2.0)
    G = model.diffusion_matrix(l, m, 1)[0]
    ceq = model.equilibrium_coeffs(l, m)
    assert np.linalg.norm(G @ ceq) / np.linalg.norm(ceq) < 1e-3
    # equilibrium is oblate: its l=2,m=0 coefficient (the alignment) is NEGATIVE.
    i20 = np.nonzero((l == 2) & (m == 0))[0][0]
    assert ceq[i20] < 0


@pytest.mark.parametrize("lam,lmax", [(2.0, 16)])
def test_rank2_anisotropy_decays_to_plateau(lam, lmax):
    # M2: the rank-2 correlation C2(t)=<P2(mu0.mu_t)>, propagated by the engine,
    # decays from 1 to the residual plateau S2^2 (wobbling-in-cone / Lipari-Szabo).
    # C2 is rotation-invariant, so this is the isotropic-ensemble anisotropy decay
    # r(t)=r0 C2(t) (r_inf = r0 S2^2) without any powder average. See a50 s020.
    from rotational_diffusion_photophysics.core import solve_evolution
    l, m = quantum_numbers(lmax)
    model = OrderingPotentialDiffusion(diffusion_coefficient=1.0, lam=lam)
    G = model.diffusion_matrix(l, m, 1)
    Mc = sh_multiplication_operator(model.equilibrium_coeffs(l, m), l, m)
    idx2 = {int(m[i]): i for i in range(l.size) if l[i] == 2}
    times = np.array([0.0, 60.0])
    total = np.zeros(times.size)
    for mm in range(-2, 3):
        i2 = idx2[mm]
        c, _, _ = solve_evolution(G[None], Mc[:, i2][None, :], times,
                                  keep_mask=(l % 2 == 0), real_output=True)
        total += c[0, i2, :]
    c2 = total / total[0]
    assert abs(c2[0] - 1.0) < 1e-9
    assert abs(c2[1] - model.order_parameter()**2) < 1e-4


# ---------------------------------------------------------------------------
# M2.5: aligned-sample equilibrium initial condition
# ---------------------------------------------------------------------------
def test_equilibrium_coeffs_isotropic_at_zero_lambda():
    # lam=0 -> c_eq is isotropic (only l=0), so the IC matches free diffusion.
    l, m = quantum_numbers(6)
    ceq = OrderingPotentialDiffusion(lam=0.0).equilibrium_coeffs(l, m)
    expected = np.zeros(l.size)
    expected[0] = 1.0
    np.testing.assert_allclose(ceq, expected, atol=1e-9)


def test_equilibrium_coeffs_even_l_m0_only():
    # c_eq lives in the even-l, m=0 subspace (a subset of the engine's kept
    # even-l/even-m space) -> the parity reductions stay valid with this IC.
    l, m = quantum_numbers(8)
    ceq = OrderingPotentialDiffusion(lam=3.0).equilibrium_coeffs(l, m)
    assert np.max(np.abs(ceq[l % 2 == 1])) < 1e-9      # no odd-l
    assert np.max(np.abs(ceq[m != 0])) < 1e-9          # axial (m=0)


def test_aligned_equilibrium_ic_no_quench():
    # M2.5: an aligned sample starts at the operator's equilibrium c_eq, so a dark
    # (no-light) evolution shows NO relaxation transient (Gamma c_eq = 0). If the
    # engine wrongly started isotropic, the l=2 coefficient would grow from 0
    # toward c_eq (a quench). Here it must stay constant at c_eq's value.
    from rotational_diffusion_photophysics.engine_s2 import SystemS2
    from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
    from rotational_diffusion_photophysics.models.detection import PolarizedDetection

    class _DarkDye:
        nspecies = 1
        quantum_yield_fluo = np.array([1.0])
        starting_populations = [1.0]
        wavelength = np.array([488.0])

        def kinetics_matrix(self):
            return np.zeros((2, 1, 1))     # no transitions

        def dipole_orientations(self):
            return np.zeros((1, 2))

    lam, lmax = 2.0, 16
    diff = OrderingPotentialDiffusion(diffusion_coefficient=1.0, lam=lam)
    lasers = ModulatedLasers(power_density=[0.0], wavelength=[488],
                             polarization=["x"], modulation=[[0]],
                             time_windows=[1e9], time0=0.0,
                             numerical_aperture=0.01, refractive_index=1.518)
    det = PolarizedDetection(polarization=["x", "y"], numerical_aperture=0.01,
                             refractive_index=1.518)
    sysm = SystemS2(_DarkDye(), diff, lasers, det, lmax=lmax)
    t = np.linspace(0.0, 5.0, 20)
    sysm.solve(t)

    l, m = quantum_numbers(lmax)
    i20 = np.nonzero((l == 2) & (m == 0))[0][0]
    ceq = diff.equilibrium_coeffs(l, m)
    c20 = sysm._c[0, i20, :]
    assert abs(c20[0] - ceq[i20]) < 1e-9    # started at c_eq (not isotropic 0)
    assert ceq[i20] > 0.1                   # c_eq genuinely anisotropic at lam=2
    assert np.std(c20) < 1e-6               # constant in the dark: no quench


# ---------------------------------------------------------------------------
# M5: axially-symmetric diffusion tensor (D_par, D_perp) on SO(3)
# ---------------------------------------------------------------------------
def test_anisotropic_l2_rates_are_woessner():
    # The l=2 relaxation rates are the symmetric-top trio (Favro/Woessner):
    #   |n|=0: 6 D_perp ; |n|=1: 5 D_perp + D_par ; |n|=2: 2 D_perp + 4 D_par.
    l, mm, n = quantum_numbers_so3(4)
    Dpar, Dperp = 3.0, 1.0
    D = AnisotropicDiffusion(Dpar, Dperp).diffusion_matrix_so3(l, mm, n, 1)[0]
    rates = -np.diag(D).real
    for nn, expected in [(0, 6*Dperp), (1, 5*Dperp + Dpar), (2, 2*Dperp + 4*Dpar)]:
        sel = (l == 2) & (np.abs(n) == nn)
        np.testing.assert_allclose(rates[sel], expected, atol=1e-9)
    assert np.max(np.abs(D - np.diag(np.diag(D)))) < 1e-12     # diagonal


def test_anisotropic_reduces_to_isotropic_when_equal():
    l, mm, n = quantum_numbers_so3(4)
    iso = IsotropicDiffusion(diffusion_coefficient=2.0).diffusion_matrix_so3(l, mm, n, 1)
    aniso = AnisotropicDiffusion(2.0, 2.0).diffusion_matrix_so3(l, mm, n, 1)
    np.testing.assert_allclose(aniso, iso, atol=1e-12)


def test_anisotropic_engine_reduces_to_isotropic():
    # SystemSO3 with AnisotropicDiffusion(D, D) must reproduce IsotropicDiffusion(D)
    # end-to-end (confirms the diffusion_matrix_so3 engine path).
    pytest.importorskip("spherical")
    from rotational_diffusion_photophysics.engine_so3 import SystemSO3
    from rotational_diffusion_photophysics.models.fluorophore import NegativeSwitcher
    from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
    from rotational_diffusion_photophysics.models.detection import PolarizedDetection

    fl = NegativeSwitcher(
        extinction_coeff_on=[5260, 51560], extinction_coeff_off=[22000, 60],
        wavelength=[405, 488], lifetime_on=1.6e-9, quantum_yield_on_fluo=0.35,
        quantum_yield_on_to_off=1.65e-2,
        dipole_orientation_cis_anionic=(35.0, 0.0),
        dipole_orientation_trans=(0.0, 0.0),
        flurophore_type='rsFP_negative_4states')
    las = ModulatedLasers(power_density=[1e2, 1e2], wavelength=[405, 488],
                          polarization=['x', 'xy'], modulation=[[1], [1]],
                          time_windows=[1.0], time0=0.0,
                          numerical_aperture=1.4, refractive_index=1.518)
    det = PolarizedDetection(polarization=['x', 'y'], numerical_aperture=1.4,
                             refractive_index=1.518)
    D = 1 / (6 * 100e-6)
    t = np.linspace(0, 1.0, 15)
    iso = SystemSO3(fl, IsotropicDiffusion(D), las, det, lmax=4).detector_signals(t)
    ani = SystemSO3(fl, AnisotropicDiffusion(D, D), las, det, lmax=4).detector_signals(t)
    np.testing.assert_allclose(ani, iso, rtol=1e-9, atol=1e-12)


# ---------------------------------------------------------------------------
# SO(3) ordering potential (the genuinely-SO(3) ordering operator)
# ---------------------------------------------------------------------------
def test_so3_ordering_reduces_to_isotropic_at_zero_lambda():
    l, mm, n = quantum_numbers_so3(4)
    iso = IsotropicDiffusion(2.5).diffusion_matrix_so3(l, mm, n, 1)[0]
    op = OrderingPotentialDiffusion(2.5, lam=0.0).diffusion_matrix_so3(l, mm, n, 1)[0]
    np.testing.assert_allclose(op, iso, atol=1e-7)


def test_so3_ordering_n0_block_matches_s2():
    # The (m=0, n=0) sub-block (dipole along the body axis) reproduces the S^2
    # ordering operator's relaxation spectrum -- same physics on the P_l functions.
    lam, lmax = 2.0, 6
    l3, m3, n3 = quantum_numbers_so3(lmax)
    G3 = OrderingPotentialDiffusion(1.0, lam).diffusion_matrix_so3(l3, m3, n3, 1)[0]
    sel3 = (m3 == 0) & (n3 == 0) & (l3 % 2 == 0)
    ev3 = np.sort(np.linalg.eigvals(G3[np.ix_(sel3, sel3)]).real)[::-1]
    l2, m2 = quantum_numbers(lmax)
    G2 = ordering_potential_block(l2, m2, 1.0, lam)
    sel2 = (m2 == 0) & (l2 % 2 == 0)
    ev2 = np.sort(np.linalg.eigvals(G2[np.ix_(sel2, sel2)]).real)[::-1]
    np.testing.assert_allclose(ev3, ev2, atol=1e-6)


def test_so3_ordering_equilibrium_null_mode():
    # Gamma_SO3 . c_eq = 0 (c_eq ~ exp(lam P2(cos beta))). lam=1 @ lmax=10 is
    # converged for the T+ L~ T- construction (needs higher lmax than S^2).
    lam, lmax, tol = 1.0, 10, 1e-3
    l, mm, n = quantum_numbers_so3(lmax)
    model = OrderingPotentialDiffusion(2.5, lam)
    G = model.diffusion_matrix_so3(l, mm, n, 1)[0]
    ceq = model.equilibrium_coeffs_so3(l, mm, n)
    res = np.linalg.norm(G @ ceq) / np.linalg.norm(ceq)
    assert res < tol, f"||G c_eq|| / ||c_eq|| = {res:.2e}"


# ---------------------------------------------------------------------------
# Director orientation of the ordering potential (aligned/membrane samples)
# ---------------------------------------------------------------------------
def test_director_default_matches_axial():
    # director=(0,0) (the default) is the lab-z axial operator.
    l, m = quantum_numbers(8)
    axial = OrderingPotentialDiffusion(2.0, lam=2.0).diffusion_matrix(l, m, 1)
    explicit = OrderingPotentialDiffusion(2.0, lam=2.0,
                                          director=(0.0, 0.0)).diffusion_matrix(l, m, 1)
    np.testing.assert_allclose(explicit, axial, atol=1e-12)


def test_director_rates_are_rotation_invariant():
    # Tilting the director must not change the relaxation spectrum (the rates are
    # rotation-invariant; only the equilibrium's lab orientation changes).
    l, m = quantum_numbers(8)
    G_axial = OrderingPotentialDiffusion(1.0, lam=2.0).diffusion_matrix(l, m, 1)[0]
    G_tilt = OrderingPotentialDiffusion(1.0, lam=2.0, director=(np.radians(40), np.radians(25))
                                        ).diffusion_matrix(l, m, 1)[0]
    # even-l subspace (the reachable one); compare sorted eigenvalues
    sel = (l % 2 == 0)
    ev0 = np.sort(np.linalg.eigvals(G_axial[np.ix_(sel, sel)]).real)
    ev1 = np.sort(np.linalg.eigvals(G_tilt[np.ix_(sel, sel)]).real)
    np.testing.assert_allclose(ev1, ev0, atol=1e-6)


def test_director_tilted_equilibrium_null_mode_and_orientation():
    # The tilted operator still annihilates its (tilted) equilibrium, and that
    # equilibrium is oriented along the director: its rank-2 part has m!=0 content
    # (it is not axial about lab-z).
    l, m = quantum_numbers(10)
    director = (np.radians(50), np.radians(30))
    model = OrderingPotentialDiffusion(1.0, lam=1.0, director=director)
    G = model.diffusion_matrix(l, m, 1)[0]
    ceq = model.equilibrium_coeffs(l, m)
    assert np.linalg.norm(G @ ceq) / np.linalg.norm(ceq) < 1e-3      # null mode
    l2m = (l == 2) & (m != 0)
    assert np.max(np.abs(ceq[l2m])) > 0.05      # genuinely tilted (m!=0 rank-2)
