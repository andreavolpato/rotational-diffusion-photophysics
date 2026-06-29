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
from rotational_diffusion_photophysics.models.diffusion import (
    IsotropicDiffusion, OrderingPotentialDiffusion, ordering_potential_block,
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
