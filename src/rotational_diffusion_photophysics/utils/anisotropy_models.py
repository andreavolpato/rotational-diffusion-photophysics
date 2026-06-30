"""Reference forward-models of fluorescence anisotropy decay (a50 n020).

These are NOT diffusion operators consumed by ``core.System`` -- the engine bakes
diffusion into the kinetics+diffusion matrix (see ``models.diffusion``). These are
closed-form / correlation-level reference tools for restricted-diffusion anisotropy:
Plan A (the wobbling-in-cone / Lipari-Szabo closed form) and the uncoupled two-tier
composition. They live here (out of ``models.diffusion``) for quick estimates,
validation, and cross-checks. The real two-tier model is the decoupled-SRLS mode
designed in a50 n030 (solved by the engine), which supersedes the post-processing
composition below.
"""
import numpy as np


def two_component_anisotropy(time, r0=0.4, order_parameter=1.0,
                             wobble_time=np.inf, global_diffusion=0.0):
    """Time-resolved anisotropy of a fast wobble + slow global tumbling:

        r(t) = r0 * [ (1 - S^2) exp(-t / tau_w) + S^2 ] * exp(-6 D_g t)

    Parameters
    ----------
    time : array        times [s]
    r0 : float          fundamental anisotropy at t=0 (0.4 for collinear abs/em).
    order_parameter : S second-rank order parameter; residual (plateau) anisotropy is
                        r_inf = r0 S^2 (when D_g = 0). S=1 rigid (no wobble loss),
                        S=0 free internal wobble. (S may be negative for in-plane/
                        oblate order; r_inf = r0 S^2 is still >= 0.)
    wobble_time : tau_w fast internal (wobble) correlation time [s]; np.inf = rigid.
    global_diffusion :  global isotropic rotational diffusion coefficient D_g [Hz];
                        the slow tumble multiplies in as exp(-6 D_g t).

    The order parameter relates to a wobbling cone of semi-angle theta_c by
    S = 1/2 cos(theta_c)(1 + cos(theta_c)) (see order_parameter_from_cone_angle).
    """
    time = np.asarray(time, dtype=float)
    s2 = order_parameter**2
    wobble = (1.0 - s2) * np.exp(-time / wobble_time) + s2
    return r0 * wobble * np.exp(-6.0 * global_diffusion * time)


def order_parameter_from_cone_angle(cone_angle_deg):
    """Wobbling-in-cone order parameter S = 1/2 cos(thc)(1 + cos(thc)) (Kinosita
    1977). thc=0 -> S=1 (rigid); thc=90 deg -> S=0 (free). Prolate cone only."""
    c = np.cos(np.radians(cone_angle_deg))
    return 0.5 * c * (1.0 + c)


def cone_angle_from_order_parameter(order_parameter):
    """Inverse of order_parameter_from_cone_angle: cos(thc) = (-1 + sqrt(1+8S))/2.
    Defined for prolate order S in [0, 1]."""
    c = (-1.0 + np.sqrt(1.0 + 8.0 * np.asarray(order_parameter, dtype=float))) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def wobble_correlation(model, time, lmax=None):
    """Exact rank-2 correlation C2(t) = <P2(mu(0).mu(t))> of the *restricted local*
    motion of an ``OrderingPotentialDiffusion`` ``model``, propagated by the engine
    (``core.solve_evolution``). Normalized C2(0)=1, decaying to the plateau
    S2 = model.order_parameter()^2 (multi-exponential, not the single-exp Plan A).
    Compose with slow global free tumbling (uncoupled, timescale-separated limit) via
    ``two_tier_anisotropy``: r(t) = r0 C2(t) exp(-6 Dg t)."""
    from rotational_diffusion_photophysics.core import solve_evolution
    from rotational_diffusion_photophysics.engine_s2 import quantum_numbers
    from rotational_diffusion_photophysics.utils.common import sh_multiplication_operator
    if lmax is None:
        lmax = int(np.clip(6 + 3 * abs(model.lam), 8, 24))
    time = np.atleast_1d(np.asarray(time, dtype=float))
    l, m = quantum_numbers(lmax)
    G = model.diffusion_matrix(l, m, 1)                  # (1, nc, nc)
    Mc = sh_multiplication_operator(model.equilibrium_coeffs(l, m), l, m)
    idx2 = {int(m[i]): i for i in range(l.size) if l[i] == 2}
    total = np.zeros(time.size)
    for mm in range(-2, 3):
        c, _, _ = solve_evolution(G[None], Mc[:, idx2[mm]][None, :], time,
                                  keep_mask=(l % 2 == 0), real_output=True)
        total += c[0, idx2[mm], :]
    return total / total[0]


def two_tier_anisotropy(time, local_model, global_diffusion, r0=0.4, lmax=None):
    """Anisotropy of a fast restricted LOCAL motion uncoupled from a slow GLOBAL
    free isotropic tumbling, in the timescale-separated limit:

        r(t) = r0 * C2_local(t) * exp(-6 D_global t).

    ``local_model`` is an ``OrderingPotentialDiffusion`` whose *exact* (multi-
    exponential) rank-2 correlation C2_local(t) (-> plateau S^2) is taken from the
    operator (``wobble_correlation``), then multiplied by the slow global decay.
    Higher fidelity than ``two_component_anisotropy`` (single effective wobble
    exponential). Valid only when the two motions are UNCOUPLED: local fast vs global
    slow, isotropic global tumbling -- the correlations then factorize. (Comparable /
    coupled timescales need the two-body SRLS treatment; see n030.)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    c2_local = wobble_correlation(local_model, time, lmax=lmax)
    return r0 * c2_local * np.exp(-6.0 * global_diffusion * time)
