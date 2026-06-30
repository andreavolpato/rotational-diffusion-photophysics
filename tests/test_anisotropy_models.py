"""Tests for the reference anisotropy-decay models (utils.anisotropy_models):
the Plan-A closed form, the cone<->order-parameter helpers, and the uncoupled
two-tier composition (exact local wobble correlation x slow global). These are
reference/validation tools, not diffusion operators dispatched into the engine.
"""
import numpy as np

from rotational_diffusion_photophysics.models.diffusion import OrderingPotentialDiffusion
from rotational_diffusion_photophysics.utils.anisotropy_models import (
    two_component_anisotropy, order_parameter_from_cone_angle,
    cone_angle_from_order_parameter, wobble_correlation, two_tier_anisotropy,
)


# --- Plan A: analytic two-component (wobble + global) anisotropy ---
def test_two_component_anisotropy_initial_value():
    t = np.linspace(0, 1e-3, 10)
    r = two_component_anisotropy(t, r0=0.4, order_parameter=0.6,
                                 wobble_time=1e-5, global_diffusion=1e3)
    assert abs(r[0] - 0.4) < 1e-12          # r(0) = r0 regardless of S, tau, D_g


def test_two_component_rigid_limit_is_pure_global():
    # S = 1 (rigid internal): r(t) = r0 exp(-6 D_g t), no wobble term.
    t = np.linspace(0, 2e-3, 20)
    Dg = 800.0
    r = two_component_anisotropy(t, r0=0.4, order_parameter=1.0,
                                 wobble_time=1e-6, global_diffusion=Dg)
    np.testing.assert_allclose(r, 0.4 * np.exp(-6 * Dg * t), atol=1e-12)


def test_two_component_plateau_is_r0_S2():
    # D_g = 0: anisotropy decays to the residual plateau r_inf = r0 S^2.
    S = 0.7
    r = two_component_anisotropy(np.array([1e9]), r0=0.4, order_parameter=S,
                                 wobble_time=1e-6, global_diffusion=0.0)
    assert abs(r[0] - 0.4 * S**2) < 1e-9


def test_cone_angle_order_parameter_roundtrip():
    assert abs(order_parameter_from_cone_angle(0.0) - 1.0) < 1e-12     # rigid
    assert abs(order_parameter_from_cone_angle(90.0) - 0.0) < 1e-12    # free
    for thc in (10.0, 35.0, 60.0, 80.0):
        S = order_parameter_from_cone_angle(thc)
        np.testing.assert_allclose(cone_angle_from_order_parameter(S), thc, atol=1e-6)


def test_planA_shares_exact_plateau_with_operator():
    # Plan A's plateau r0 S^2, with S the Maier-Saupe order parameter of the
    # operator, equals the operator's residual anisotropy r0 * C2(inf) (=S^2).
    S = OrderingPotentialDiffusion(lam=2.0).order_parameter()
    plateau_A = two_component_anisotropy(np.array([1e9]), r0=0.4, order_parameter=S,
                                         wobble_time=1.0, global_diffusion=0.0)[0]
    assert abs(plateau_A - 0.4 * S**2) < 1e-12


# --- Two-tier: fast restricted local x slow global free diffusion (uncoupled) ---
def test_wobble_correlation_starts_at_one_decays_to_plateau():
    # C2_local(0)=1, plateau = S^2 (the exact multi-exponential local wobble).
    model = OrderingPotentialDiffusion(1.0, lam=2.0)
    c2 = wobble_correlation(model, np.array([0.0, 60.0]), lmax=16)
    assert abs(c2[0] - 1.0) < 1e-9
    assert abs(c2[1] - model.order_parameter()**2) < 1e-4


def test_two_tier_anisotropy_composition():
    # r(t) = r0 * C2_local(t) * exp(-6 Dg t): r(0)=r0; with Dg>0 decays to 0;
    # equals r0 * wobble_correlation * global decay (the uncoupled factorization).
    model = OrderingPotentialDiffusion(1.0, lam=2.0)
    Dg = 0.2
    t = np.linspace(0, 30.0, 40)
    r = two_tier_anisotropy(t, model, global_diffusion=Dg, r0=0.4, lmax=16)
    assert abs(r[0] - 0.4) < 1e-9                       # r(0) = r0
    expected = 0.4 * wobble_correlation(model, t, lmax=16) * np.exp(-6 * Dg * t)
    np.testing.assert_allclose(r, expected, atol=1e-12)
    assert r[-1] < 1e-3                                 # global tumble -> 0


def test_two_tier_matches_plan_a_at_shared_plateau():
    # With Dg=0 the two-tier (exact local) and Plan A (single-exp local) share the
    # exact plateau r0 * S^2.
    model = OrderingPotentialDiffusion(1.0, lam=2.0)
    S = model.order_parameter()
    big = np.array([0.0, 80.0])
    r_tier = two_tier_anisotropy(big, model, global_diffusion=0.0, r0=0.4, lmax=16)
    r_planA = two_component_anisotropy(big, r0=0.4, order_parameter=S,
                                       wobble_time=1.0, global_diffusion=0.0)
    assert abs(r_tier[-1] - r_planA[-1]) < 1e-4         # shared plateau r0 S^2
    assert abs(r_tier[-1] - 0.4 * S**2) < 1e-4
