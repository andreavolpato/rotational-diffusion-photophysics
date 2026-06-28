"""Tests for the SO(3) Wigner-D engine (engine A, a40 n030).

First milestones: the (l, m, n) basis and isotropic rotational diffusion. The
diffusion-decay test also validates the *complex* path through the shared
``core.solve_evolution`` and the even-l reduction.
"""
import numpy as np
import pytest

from rotational_diffusion_photophysics import core
from rotational_diffusion_photophysics.engine_so3 import (
    quantum_numbers_so3,
    isotropic_diffusion_matrix_so3,
    multiplication_operator,
    absorption_coeffs,
    so3_overlap,
    _rotor_to_direction,
    SystemSO3,
)


class _TwoStateDye:
    """Minimal fluorophore for engine tests: ground (0) absorbs, excited (1)
    emits, collinear dipoles along z (so r0 = 0.4). No switching."""
    nspecies = 2
    quantum_yield_fluo = np.array([0.0, 1.0])
    starting_populations = [1.0, 0.0]
    wavelength = np.array([488.0])

    def __init__(self, lifetime=1.0, cross_section=1.0):
        self.lifetime = lifetime
        self.cross_section = cross_section

    def kinetics_matrix(self):
        K = np.zeros((2, 2, 2))           # (nwavelength+1, nspecies, nspecies)
        K[0][0, 1] = 1.0 / self.lifetime  # 1 -> 0 emission/decay (non-light)
        K[1][1, 0] = self.cross_section   # 0 -> 1 absorption at 488
        return K

    def dipole_orientations(self):
        return np.zeros((2, 2))           # both dipoles along z


def _unit(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def _index(l, m, n, li, mi, ni):
    return int(np.nonzero((l == li) & (m == mi) & (n == ni))[0][0])


def test_basis_sizes():
    for lmax in range(5):
        l, m, n = quantum_numbers_so3(lmax)
        expected = sum((2 * li + 1) ** 2 for li in range(lmax + 1))
        assert l.size == m.size == n.size == expected
        assert np.all(np.abs(m) <= l) and np.all(np.abs(n) <= l)
        for li in range(lmax + 1):
            assert np.count_nonzero(l == li) == (2 * li + 1) ** 2


@pytest.mark.parametrize(
    "lval, rate_factor, reduce_even",
    [(2, 6, True),   # even l: rate l(l+1)=6, with the even-l reduction on
     (1, 2, False)],  # odd l: rate 2, reduction must be off
)
def test_free_diffusion_decay(lval, rate_factor, reduce_even):
    # A single rank-l Wigner-D coefficient decays as exp(-l(l+1) Dr t) under pure
    # isotropic rotational diffusion; the l=0 (population) coefficient is constant.
    lmax = 2
    Dr = 1.0e4
    a0 = 0.3
    l, m, n = quantum_numbers_so3(lmax)
    ncoeff = l.size

    D = isotropic_diffusion_matrix_so3(l, Dr, 1)              # (1, ncoeff, ncoeff)
    K = np.zeros((1, 1, ncoeff, ncoeff), dtype=complex)       # no kinetics
    M = core.diffusion_kinetics_matrix(D, K)                  # (1, 1, ncoeff, ncoeff)

    i0 = _index(l, m, n, 0, 0, 0)
    il = _index(l, m, n, lval, 0, 0)
    c0 = np.zeros((1, ncoeff), dtype=complex)
    c0[0, i0] = 1.0      # population
    c0[0, il] = a0       # some rank-l content

    time = np.linspace(0.0, 5e-5, 50)
    l_arg = l if reduce_even else None
    c, _, _ = core.solve_evolution(M, c0, time, l=l_arg, real_output=False)

    assert c.shape == (1, ncoeff, time.size)
    # l=0 population conserved.
    np.testing.assert_allclose(c[0, i0, :], 1.0, atol=1e-9)
    # rank-l coefficient decays at the analytic rate.
    np.testing.assert_allclose(
        c[0, il, :], a0 * np.exp(-rate_factor * Dr * time), atol=1e-7)
    # evolution stays real here (real diagonal M, real c0).
    assert np.max(np.abs(c.imag)) < 1e-9


def _eval_density(l, m, n, coeffs, wig, R):
    """f(R) = Σ_k coeffs[k] D^{l_k}_{m_k n_k}(R), via spherical."""
    Dflat = wig.D(R)
    val = 0j
    for k in np.nonzero(coeffs)[0]:
        val += coeffs[k] * Dflat[wig.Dindex(int(l[k]), int(m[k]), int(n[k]))]
    return val


def test_multiplication_operator_matches_function_product():
    # The CG-built multiplication operator W must satisfy, for every rotation R,
    #   Σ (W @ c)_k D_k(R)  ==  w(R) · f(R),
    # where w (func_coeffs) and f (c) are densities in the Wigner-D basis. Checked
    # numerically against spherical's D — an independent evaluator. This validates
    # the CG product formula, the CG values, and the convention all at once.
    spherical = pytest.importorskip("spherical")
    quaternionic = pytest.importorskip("quaternionic")

    lmax = 4  # so D^2 * D^2 -> l=4 products are captured, not truncated
    l, m, n = quantum_numbers_so3(lmax)
    ncoeff = l.size

    def put(arr, li, mi, ni, val):
        arr[_index(l, m, n, li, mi, ni)] = val

    # Function w: only L in {0,2} (the optics multipoles).
    w = np.zeros(ncoeff, dtype=complex)
    put(w, 0, 0, 0, 0.7)
    put(w, 2, 0, 0, 0.5)
    put(w, 2, 1, -1, 0.3 + 0.1j)
    put(w, 2, -2, 2, -0.2j)

    # Input density f: a spread of low-l content.
    c = np.zeros(ncoeff, dtype=complex)
    put(c, 0, 0, 0, 1.0)
    put(c, 1, 0, 0, 0.4)
    put(c, 2, -1, 1, 0.6 - 0.2j)
    put(c, 2, 2, -2, 0.25)

    W = multiplication_operator(l, m, n, w, lmax)
    new_c = W @ c

    wig = spherical.Wigner(lmax)
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(8):
        R = quaternionic.array(rng.standard_normal(4)).normalized
        lhs = _eval_density(l, m, n, new_c, wig, R)
        rhs = (_eval_density(l, m, n, w, wig, R)
               * _eval_density(l, m, n, c, wig, R))
        max_err = max(max_err, abs(lhs - rhs))
    assert max_err < 1e-10, f"max|W@c - w*f| = {max_err:.2e}"


def test_rotor_to_direction_maps_zaxis():
    quaternionic = pytest.importorskip("quaternionic")
    for theta, phi in [(0.0, 0.0), (0.7, 1.1), (2.3, -0.5), (np.pi / 2, np.pi)]:
        R = _rotor_to_direction(theta, phi)
        z = quaternionic.array([0, 0, 0, 1.0])
        got = (R * z * R.conjugate()).vector
        np.testing.assert_allclose(got, _unit(theta, phi), atol=1e-12)


def test_absorption_coeffs_match_dipole_projection():
    # The built coefficients must reconstruct the exact rate function
    #   w(R) = |e . (R d)|^2
    # for arbitrary field direction e and body dipole direction d. Checked at
    # random rotations against spherical (independent evaluator). This is where
    # the dipole direction enters the optics (n030 §4) — the physics appears.
    spherical = pytest.importorskip("spherical")
    quaternionic = pytest.importorskip("quaternionic")

    lmax = 2
    l, m, n = quantum_numbers_so3(lmax)
    wig = spherical.Wigner(lmax)
    rng = np.random.default_rng(2)

    cases = [
        ((0.0, 0.0), (0.0, 0.0)),       # canonical: field & dipole along z
        ((0.0, 0.0), (np.radians(35), 0.0)),   # z field, 35deg dipole (rsEGFP2-like)
        ((0.6, 1.2), (1.3, -0.7)),      # both tilted
        ((np.pi / 2, 0.0), (np.pi / 2, np.pi / 2)),  # orthogonal in-plane
    ]
    for field_dir, dip_dir in cases:
        b = absorption_coeffs(l, m, n, field_dir, dip_dir, wig)
        e = _unit(*field_dir)
        d = _unit(*dip_dir)
        max_err = 0.0
        for _ in range(8):
            R = quaternionic.array(rng.standard_normal(4)).normalized
            recon = _eval_density(l, m, n, b, wig, R)
            dq = quaternionic.array([0, *d])
            Rd = (R * dq * R.conjugate()).vector
            exact = (e @ Rd) ** 2
            max_err = max(max_err, abs(recon - exact))
        assert max_err < 1e-10, f"{field_dir},{dip_dir}: max err {max_err:.2e}"


def test_two_dipole_cone_anisotropy():
    # Classic photoselection anisotropy: excite (field || z) selecting on the
    # absorption dipole d_a, read out emission on dipole d_b tilted by beta. The
    # steady photoselected anisotropy must be r = r0 * P2(cos beta) with r0 = 0.4.
    # This is the a40 depolarization, and being a ratio it is independent of the
    # overall normalization. Validates absorption + detection + the SO(3) overlap.
    spherical = pytest.importorskip("spherical")
    lmax = 2
    l, m, n = quantum_numbers_so3(lmax)
    wig = spherical.Wigner(lmax)

    z_field = (0.0, 0.0)          # excitation and parallel detection
    x_field = (np.pi / 2, 0.0)    # perpendicular detection
    d_a = (0.0, 0.0)             # absorption dipole along z

    b_exc = absorption_coeffs(l, m, n, z_field, d_a, wig)  # excited-state density
    for beta_deg in (0, 30, 54.7356, 60, 90):
        beta = np.radians(beta_deg)
        d_b = (beta, 0.0)        # emission dipole tilted by beta from d_a
        b_par = absorption_coeffs(l, m, n, z_field, d_b, wig)
        b_perp = absorption_coeffs(l, m, n, x_field, d_b, wig)
        I_par = so3_overlap(l, m, n, b_par, b_exc, lmax).real
        I_perp = so3_overlap(l, m, n, b_perp, b_exc, lmax).real
        r = (I_par - I_perp) / (I_par + 2 * I_perp)
        P2 = (3 * np.cos(beta) ** 2 - 1) / 2
        assert abs(r - 0.4 * P2) < 1e-10, (
            f"beta={beta_deg}: r={r:.6f}, expected {0.4 * P2:.6f}")


def test_system_free_anisotropy_decay_rate():
    # Full SystemSO3 pipeline: excite (window 1, 488 || z) then a dark window.
    # In the dark the excited-state l=2 anisotropy decays purely by rotational
    # diffusion at rate 6 D_r (the fluorescence lifetime is isotropic and cancels
    # in the ratio), so r(t) ∝ exp(-6 D_r t). Checks excitation + kinetics +
    # diffusion + time-resolved par/perp detection + the pulse-window loop.
    pytest.importorskip("spherical")
    from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
    from rotational_diffusion_photophysics.models.detection import PolarizedDetection
    from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion

    Dr = 0.1
    dye = _TwoStateDye(lifetime=1.0, cross_section=1e-16)
    lasers = ModulatedLasers(
        power_density=[1e-4], wavelength=[488], polarization=['z'],
        modulation=[[1, 0]], time_windows=[0.5, 10.0], time0=0.0,
        numerical_aperture=0.01, refractive_index=1.518)  # near-ideal optics
    # NB: with high-NA detection the par/perp ratio is not exactly the l=2/l=0
    # ratio, so r(t) is not a pure single exponential at 6 D_r (both engines agree
    # on that, it is real optics). Near-ideal NA recovers the clean closed form.
    det = PolarizedDetection(polarization=['z', 'x'],  # parallel, perpendicular
                             numerical_aperture=0.01, refractive_index=1.518)
    system = SystemSO3(fluorophore=dye,
                       diffusion=IsotropicDiffusion(diffusion_coefficient=Dr),
                       illumination=lasers, detection=det, lmax=2)

    time = np.linspace(0.6, 4.5, 40)          # inside the dark window
    s = system.detector_signals(time)
    r = (s[0] - s[1]) / (s[0] + 2 * s[1])
    assert np.all(r > 0)
    slope = np.polyfit(time, np.log(r), 1)[0]
    assert abs(slope - (-6 * Dr)) < 1e-3, f"decay rate {slope:.5f}, want {-6 * Dr}"


def _rsegfp2_4state(cis_trans_deg):
    from rotational_diffusion_photophysics.models.fluorophore import NegativeSwitcher
    return NegativeSwitcher(
        extinction_coeff_on=[5260, 51560], extinction_coeff_off=[22000, 60],
        wavelength=[405, 488], lifetime_on=1.6e-9, quantum_yield_on_fluo=0.35,
        quantum_yield_on_to_off=1.65e-2,
        dipole_orientation_cis_anionic=(cis_trans_deg, 0.0),
        dipole_orientation_trans=(0.0, 0.0),
        flurophore_type='rsFP_negative_4states')


def _starss1_spec(fluorophore, power_scale=1.0):
    """Common (diffusion, illumination, detection) for a STARSS-1-like run.

    `power_scale<1` keeps the 405 activation non-saturating (l<=2 exact), where
    the S^2 and SO(3) engines provably agree.
    """
    from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
    from rotational_diffusion_photophysics.models.detection import PolarizedDetection
    from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
    twin = (1e-3, 250e-9, 1e-3)
    lasers = ModulatedLasers(
        power_density=[176.7e3 * power_scale, 17.6e3 / 6 * power_scale],
        wavelength=[405, 488],
        polarization=['x', 'xy'], modulation=[[0, 1, 0], [1, 1, 1]],
        time_windows=list(twin), time0=twin[0] + twin[1],
        numerical_aperture=1.4, refractive_index=1.518)
    det = PolarizedDetection(polarization=['x', 'y'],
                             numerical_aperture=1.4, refractive_index=1.518)
    diff = IsotropicDiffusion(diffusion_coefficient=1 / (6 * 100e-6))
    return diff, lasers, det


def test_reduction_full_system_signals_match_s2():
    # REDUCTION TEST (full pipeline): with no dipole reorientation (all dipoles
    # along z) the SO(3) engine must reproduce the S^2 engine's detector_signals,
    # not just anisotropy -- this confirms the population-0..1 normalization is
    # shared. Same fluorophore / diffusion / illumination / detection objects.
    pytest.importorskip("spherical")
    from rotational_diffusion_photophysics.engine_s2 import SystemS2 as S2System

    fl = _rsegfp2_4state(0.0)            # cis == trans -> no reorientation
    # Non-saturating activation: l<=2 is exact, the regime where the two engines
    # provably agree. (They DIFFER under strong/saturating excitation -- a real,
    # lmax-converged discrepancy, documented in a40 n030; SO(3)'s operators are
    # rigorously verified, the S^2 w3jp has empirical factors -> needs the MC
    # oracle to settle which is correct.)
    diff, lasers, det = _starss1_spec(fl, power_scale=1e-3)
    s2 = S2System(fluorophore=fl, diffusion=diff, illumination=lasers,
                  detection=det, lmax=2)
    so3 = SystemSO3(fluorophore=fl, diffusion=diff, illumination=lasers,
                    detection=det, lmax=2)
    t = np.linspace(0, 1e-3, 30)
    sig2 = s2.detector_signals(t)
    sig3 = so3.detector_signals(t)
    np.testing.assert_allclose(sig3, sig2, rtol=1e-5, atol=1e-12)


def test_core_system_selector():
    # core.System dispatches: 'auto' -> S^2 when dipoles are uniform, SO(3) when
    # they reorient; explicit 's2'/'so3' force the backend.
    pytest.importorskip("spherical")
    from rotational_diffusion_photophysics import core
    from rotational_diffusion_photophysics.engine_s2 import SystemS2 as S2System

    rigid = _rsegfp2_4state(0.0)
    reorient = _rsegfp2_4state(35.0)
    diff, lasers, det = _starss1_spec(rigid)

    auto_rigid = core.System(rigid, diff, lasers, det, representation="auto")
    auto_reorient = core.System(reorient, diff, lasers, det, representation="auto")
    assert isinstance(auto_rigid, S2System)               # uniform -> cheap S^2
    assert type(auto_reorient).__name__ == "SystemSO3"    # reorient -> SO(3)
    # explicit override
    assert type(core.System(rigid, diff, lasers, det,
                            representation="so3")).__name__ == "SystemSO3"
    assert isinstance(core.System(reorient, diff, lasers, det,
                                  representation="s2"), S2System)


def test_reduction_optics_na_matches_old_engine():
    # REDUCTION TEST: engine A's NA-corrected photoselection function must equal
    # the old S^2 engine's optics (utils.common.na_corrected_linear_coeffs) at any
    # NA / polarization. Both are the Axelrod-weighted sum
    #   k0|z.d|^2 + k1|y.d|^2 + k2|x.d|^2  (indices permuted per polarization).
    # We check engine A reconstructs that closed form AND that the old real-SH
    # coefficients evaluate to the same function (via pyshtools MakeGridPoint).
    spherical = pytest.importorskip("spherical")
    quaternionic = pytest.importorskip("quaternionic")
    import pyshtools as sht
    from rotational_diffusion_photophysics.engine_s2 import quantum_numbers
    from rotational_diffusion_photophysics.utils.common import (
        na_corrected_linear_coeffs, axelrod_correction_k)
    from rotational_diffusion_photophysics.engine_so3 import (
        na_corrected_absorption_coeffs)

    lmax = 2
    l, m, n = quantum_numbers_so3(lmax)
    wig = spherical.Wigner(lmax)
    l2, m2 = quantum_numbers(lmax)            # old S^2 (l, m)
    ri = 1.518
    rng = np.random.default_rng(7)

    for pol in ('x', 'y', 'z', 'xy'):
        for na in (0.8, 1.2, 1.4):
            k = axelrod_correction_k(na, ri)
            b = na_corrected_absorption_coeffs(l, m, n, pol, (0.0, 0.0), wig, na, ri)
            c_old = na_corrected_linear_coeffs(l2, m2, polarization=pol,
                                               numerical_aperture=na,
                                               refractive_index=ri)
            cilm = sht.shio.SHVectorToCilm(c_old)
            for _ in range(4):
                R = quaternionic.array(rng.standard_normal(4)).normalized
                dq = quaternionic.array([0, 0, 0, 1.0])
                d = (R * dq * R.conjugate()).vector       # dipole dir = R z
                if pol == 'x':
                    f_cf = k[0] * d[2]**2 + k[1] * d[1]**2 + k[2] * d[0]**2
                elif pol == 'y':
                    f_cf = k[0] * d[2]**2 + k[1] * d[0]**2 + k[2] * d[1]**2
                elif pol == 'z':
                    f_cf = k[0] * d[0]**2 + k[1] * d[1]**2 + k[2] * d[2]**2
                else:
                    f_cf = k[0] * d[2]**2 + (k[1] + k[2]) / 2 * (d[0]**2 + d[1]**2)
                f_new = _eval_density(l, m, n, b, wig, R).real
                colat = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
                lon = np.degrees(np.arctan2(d[1], d[0]))
                f_old = sht.expand.MakeGridPoint(cilm, 90 - colat, lon)
                assert abs(f_new - f_cf) < 1e-10, f"engine A {pol},NA={na}"
                assert abs(f_old - f_cf) < 1e-10, f"old engine {pol},NA={na}"


# --- absolute rate validation (independent closed-form checks) ----------------
class _ExciteOnlyDye:
    """2-state fluorophore, 0 -> 1 by 488 light only (no decay/switching). The
    ground population then probes the bare excitation rate / saturation."""
    nspecies = 2
    quantum_yield_fluo = np.array([0.0, 1.0])
    starting_populations = [1.0, 0.0]
    wavelength = np.array([488.0])

    def __init__(self, cross_section):
        self.cross_section = cross_section

    def kinetics_matrix(self):
        K = np.zeros((2, 2, 2))
        K[1][1, 0] = self.cross_section   # 0 -> 1; K[0] = 0 (no decay)
        return K

    def dipole_orientations(self):
        return np.zeros((2, 2))


def _excite_only_system(pol, na, rep, lmax, cross_section=1e-16, power=1e-2):
    from rotational_diffusion_photophysics.models.illumination import ModulatedLasers
    from rotational_diffusion_photophysics.models.detection import PolarizedDetection
    from rotational_diffusion_photophysics.models.diffusion import IsotropicDiffusion
    las = ModulatedLasers(power_density=[power], wavelength=[488], polarization=[pol],
                          modulation=[[1]], time_windows=[1e9], time0=0.0,
                          numerical_aperture=na, refractive_index=1.518)
    det = PolarizedDetection(polarization=['z'], numerical_aperture=na, refractive_index=1.518)
    sysm = core.System(_ExciteOnlyDye(cross_section),
                       IsotropicDiffusion(diffusion_coefficient=0.0),
                       las, det, lmax=lmax, representation=rep)
    sF = cross_section * las.photon_flux[0]      # absolute excitation rate scale
    return sysm, sF


@pytest.mark.parametrize("rep", ["s2", "so3"])
@pytest.mark.parametrize("pol,na", [("z", 1.4), ("x", 1.4), ("xy", 1.4),
                                    ("z", 0.01), ("xy", 0.01)])
def test_isotropic_excitation_rate(rep, pol, na):
    # The INITIAL total excitation rate of an isotropic ensemble equals sF/3 for
    # ANY polarization (linear x/y/z or circular) and ANY NA -- because
    # <|e.mu|^2>_iso = 1/3 and the Axelrod NA correction conserves the total
    # (k0+k1+k2=1). This pins the ABSOLUTE rate, independent of the angular optics.
    pytest.importorskip("spherical")
    sysm, sF = _excite_only_system(pol, na, rep, lmax=6)
    dt = 1e-4 / sF                                # so sF*dt << 1 (initial slope)
    sysm.solve(np.array([0.0, dt]))
    n_ground = sysm._c[0, 0, :].real             # total ground population N0(t)
    rate = -(n_ground[1] - n_ground[0]) / dt
    np.testing.assert_allclose(rate, sF / 3, rtol=1e-3)


@pytest.mark.parametrize("rep", ["s2", "so3"])
def test_saturation_matches_closed_form(rep):
    # Continuous ideal z-polarized excitation, no decay: each orientation depletes
    # as exp(-sF cos^2(theta) t), so the isotropic ground population is exactly
    #   N0(t) = sqrt(pi) erf(sqrt(a)) / (2 sqrt(a)),  a = sF t
    # (orientational hole-burning). This validates the ABSOLUTE saturation
    # dynamics; both engines must converge to it as lmax rises.
    import math
    pytest.importorskip("spherical")
    sysm, sF = _excite_only_system("z", 0.01, rep, lmax=6)
    t = np.linspace(0.0, 2.0 / sF, 12)
    a = sF * t
    analytic = np.ones_like(a)
    nz = a > 0
    analytic[nz] = [math.sqrt(math.pi) * math.erf(math.sqrt(ai)) / (2 * math.sqrt(ai))
                    for ai in a[nz]]
    sysm.solve(t)
    n_ground = sysm._c[0, 0, :].real
    np.testing.assert_allclose(n_ground, analytic, atol=1e-4)
