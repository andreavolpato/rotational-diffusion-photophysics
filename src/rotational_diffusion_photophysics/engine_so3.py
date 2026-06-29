"""SO(3) Wigner-D rotational-diffusion + photophysics engine (engine A).

Work in progress. Design: a40 study note n030. Where the existing ``engine.py``
expands the orientational density of the *dipole axis* on S^2 in real spherical
harmonics ``(l, m)``, this engine expands the full *barrel* orientation on SO(3)
in complex Wigner-D functions ``D^l_{mn}`` ``(l, m, n)`` (m = lab index, n = body
index). Each state carries a body-frame dipole direction; the reorientation on
photoswitching lives in the optics, not a jump operator (n030 §4.1).

Shared, basis-agnostic primitives (propagator, M assembly) come from ``core.py``;
this module adds the SO(3)-specific basis and angular operators.

Implemented so far (milestones, n030 §9):
  1. ``quantum_numbers_so3`` — the (l, m, n) basis.
  2. ``isotropic_diffusion_matrix_so3`` — diagonal -l(l+1)D_r over (m, n).
Still to come: the tilted light-matter operator (§4), the (m,n) D-product (§5),
detection, and the System orchestration.
"""
from functools import lru_cache

import numpy as np

from rotational_diffusion_photophysics.core import (
    find_wavelenght,
    solve_evolution,
    diffusion_kinetics_matrix,
)


@lru_cache(maxsize=None)
def _clebsch_gordan(j1, m1, j2, m2, j3, m3):
    """⟨j1 m1; j2 m2 | j3 m3⟩, cached. Exact, via sympy.

    Used to build the Wigner-D product (§5). The build is one-time per System and
    tiny at the `l<=2..6` we need, so sympy's exactness beats its speed cost; it
    can be swapped for a pyshtools 3j path (n030 §7) if high-`l_max` build time
    ever matters. CG are real, so the product identity below is convention-robust
    (holds for spherical's conjugated D with the same coefficients).
    """
    from sympy.physics.wigner import clebsch_gordan
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))


def quantum_numbers_so3(lmax):
    """Quantum-number arrays (l, m, n) for the Wigner-D basis up to ``lmax``.

    For each l in 0..lmax, all (m, n) with m, n in -l..l are listed, so the basis
    size is sum_{l=0}^{lmax} (2l+1)^2. Ordering is (l, then m ascending, then n
    ascending); it only needs to be internally consistent.
    """
    l, m, n = [], [], []
    for li in range(lmax + 1):
        for mi in range(-li, li + 1):
            for ni in range(-li, li + 1):
                l.append(li)
                m.append(mi)
                n.append(ni)
    return np.array(l, dtype=np.int32), np.array(m, dtype=np.int32), np.array(n, dtype=np.int32)


def isotropic_diffusion_matrix_so3(l, diffusion_coefficient, nspecies):
    """Isotropic rotational-diffusion blocks for each species.

    On SO(3) the isotropic diffusion operator is diagonal in the Wigner-D basis
    with eigenvalue -l(l+1)D_r (same as on S^2, now replicated over the body
    index n). Returns an array of shape (nspecies, ncoeff, ncoeff); complex so it
    composes with the complex Wigner-D kinetics blocks.
    """
    ncoeff = l.size
    block = np.diag((-l * (l + 1) * diffusion_coefficient).astype(complex))
    return np.repeat(block[None, :, :], nspecies, axis=0)


def multiplication_operator(l, m, n, func_coeffs, lmax):
    """Matrix of "multiply the density by the function `func_coeffs`" in the
    Wigner-D basis (§5 of n030).

    The product of two Wigner-D functions is the Clebsch-Gordan series
        D^L_{MN} D^l_{mn} = Σ_{l'} ⟨L M l m | l' m'⟩ ⟨L N l n | l' n'⟩ D^{l'}_{m'n'},
    with m' = M+m, n' = N+n. So multiplying a density `f = Σ c_{lmn} D^l_{mn}` by a
    function `w = Σ b_{LMN} D^L_{MN}` (here `func_coeffs`, aligned to the `(l,m,n)`
    basis) gives new coefficients `W @ c` with
        W_{(l'm'n'),(lmn)} = Σ_{LMN} b_{LMN} ⟨L M l m|l' m'⟩ ⟨L N l n|l' n'⟩.

    The `m`-CG and `n`-CG factorize over the same `l`-triples (the key structure).
    Output `l'` is truncated to `lmax` (the deliberate band-limit, n030 §5).
    NOTE: an overall per-`l` normalization (population convention, n030 §2) is
    applied where operators are assembled, not here; this is the bare-D product.
    """
    ncoeff = l.size
    index = {(int(l[k]), int(m[k]), int(n[k])): k for k in range(ncoeff)}
    W = np.zeros((ncoeff, ncoeff), dtype=complex)

    for f in np.nonzero(func_coeffs)[0]:
        L, M, N = int(l[f]), int(m[f]), int(n[f])
        b = func_coeffs[f]
        for k in range(ncoeff):
            li, mi, ni = int(l[k]), int(m[k]), int(n[k])
            mo, no = M + mi, N + ni
            for lo in range(abs(L - li), min(L + li, lmax) + 1):
                if abs(mo) > lo or abs(no) > lo:
                    continue
                ko = index.get((lo, mo, no))
                if ko is None:
                    continue
                W[ko, k] += b * (_clebsch_gordan(L, M, li, mi, lo, mo)
                                 * _clebsch_gordan(L, N, li, ni, lo, no))
    return W


def so3_overlap(l, m, n, coeffs_f, coeffs_g, lmax):
    """Proportional to the SO(3) integral ∫ f(R) g(R) dR of two real functions
    expanded in the Wigner-D basis.

    Returns the `D^0_{00}` coefficient of the product `f·g` (the integral is
    `8π²` times this). This is the detection inner product (collection function
    overlapped with the emitting-state density) and the building block for
    normalization-free observables like anisotropy, where the `8π²` cancels.
    Only the `(0,0,0)` output is needed, so `lmax` need only cover the inputs.
    """
    W = multiplication_operator(l, m, n, coeffs_f, lmax)
    i000 = int(np.nonzero((l == 0) & (m == 0) & (n == 0))[0][0])
    return (W @ coeffs_g)[i000]


def _rotor_to_direction(theta, phi):
    """Rotor R taking the body z-axis to the lab direction (theta, phi).

    ZYZ with gamma=0, matching the convention pinned in s010.
    """
    import quaternionic
    return quaternionic.array.from_euler_angles(phi, theta, 0.0)


# One-photon photoselection rank weights (no NA/Axelrod yet; those replace these).
# Linear field e: |e.u|^2 = 1/3 + (2/3) P2(e.u).
_LINEAR_DIPOLE_WEIGHTS = {0: 1.0 / 3.0, 2: 2.0 / 3.0}
# Circular field about propagation axis k: (1 - (k.u)^2)/2 = 1/3 - (1/3) P2(k.u).
# Same (M,N) structure as linear about k, only the rank-2 weight differs.
_CIRCULAR_DIPOLE_WEIGHTS = {0: 1.0 / 3.0, 2: -1.0 / 3.0}
_POLARIZATION_WEIGHTS = {"linear": _LINEAR_DIPOLE_WEIGHTS,
                         "circular": _CIRCULAR_DIPOLE_WEIGHTS}


def absorption_coeffs(l, m, n, field_direction, dipole_direction, wigner,
                      weights=None):
    """Wigner-D coefficients of the absorption-rate function |e.(R d)|^2.

    `field_direction` (e, lab frame) and `dipole_direction` (d, body frame) are
    each (theta, phi) in radians. The rank-L weight is spread over the lab index
    M by the field direction and over the body index N by the dipole direction:

        b_{L,M,N} = a_L * (-1)^M * D^L_{-M,0}(R_e) * D^L_{N,0}(R_d),

    where R_e, R_d rotate the z-axis to e and d. The dipole enters the body index
    N as the plain D^L_{N0}(R_d); the field enters the lab index M with the
    (-1)^M / -M twist because (addition theorem) e appears against a *conjugated*
    D(R) (in spherical's convention conj D^L_{MN} = (-1)^{M-N} D^L_{-M,-N}). The
    two forms agree at M=0 (e along z). Verified against the closed form
    |e.(R d)|^2 in test_engine_so3 (n030 milestone 4). `weights` overrides the
    default one-photon {0:1/3, 2:2/3} (NA-corrected weights later). `wigner` is a
    spherical.Wigner instance.
    """
    if weights is None:
        weights = _LINEAR_DIPOLE_WEIGHTS
    De = wigner.D(_rotor_to_direction(*field_direction))
    Dd = wigner.D(_rotor_to_direction(*dipole_direction))
    index = {(int(l[k]), int(m[k]), int(n[k])): k for k in range(l.size)}
    b = np.zeros(l.size, dtype=complex)
    for L, aL in weights.items():
        for M in range(-L, L + 1):
            for N in range(-L, L + 1):
                k = index.get((L, M, N))
                if k is None:
                    continue
                b[k] = (aL * ((-1) ** M)
                        * De[wigner.Dindex(L, -M, 0)]
                        * Dd[wigner.Dindex(L, N, 0)])
    return b


def na_corrected_absorption_coeffs(l, m, n, polarization, dipole_direction,
                                   wigner, numerical_aperture, refractive_index):
    """NA/Axelrod-corrected photoselection coefficients in the Wigner-D basis.

    Mirrors ``utils.common.na_corrected_linear_coeffs`` (the S^2 engine's optics):
    a high-NA focused beam is a weighted sum of x/y/z linear photoselections with
    the Axelrod weights k = [propagation, perpendicular, parallel]. Built here as
    the same weighted sum of `absorption_coeffs` over the three lab axes (so it
    spreads correctly over the body index from the dipole direction). Lets engine
    A reproduce the old engine at any NA (reduction test).
    """
    from rotational_diffusion_photophysics.utils.common import axelrod_correction_k
    k = axelrod_correction_k(numerical_aperture, refractive_index)
    Z, Y, X = (0.0, 0.0), (np.pi / 2, np.pi / 2), (np.pi / 2, 0.0)
    cz = absorption_coeffs(l, m, n, Z, dipole_direction, wigner)
    cy = absorption_coeffs(l, m, n, Y, dipole_direction, wigner)
    cx = absorption_coeffs(l, m, n, X, dipole_direction, wigner)
    if polarization == 'x':
        return k[0] * cz + k[1] * cy + k[2] * cx
    if polarization == 'y':
        return k[0] * cz + k[1] * cx + k[2] * cy
    if polarization == 'z':
        return k[0] * cx + k[1] * cy + k[2] * cz
    if polarization == 'xy':  # circular in the xy plane
        return k[0] * cz + (k[1] + k[2]) / 2 * cy + (k[1] + k[2]) / 2 * cx
    raise ValueError(f"unknown polarization '{polarization}'")


def realifying_transform(l, m, n, keep_mask=None):
    """Unitary T (angular block) mapping the *complex* Wigner-D coefficients of a
    REAL density to real numbers, so that ``T M T^H`` is real and the eig can run
    in real (dgeev) instead of complex (zgeev) arithmetic (~2x; see
    ``core.solve_evolution``). A real function on SO(3) has coefficients obeying
    ``c_{l,-m,-n} = (-1)^{m-n} conj(c_{lmn})``; pairing (m,n)<->(-m,-n) with that
    phase gives the standard cos/sin real combination. Built over the kept angular
    basis (``keep_mask``, or all coefficients if None). Returns the angular
    unitary, or None if that basis is not closed under (m,n)->(-m,-n) (then the
    caller keeps the complex eig). This is SO(3) physics (the Wigner-D reality
    constraint), hence it lives here, not in the basis-agnostic core."""
    ang = np.arange(l.size) if keep_mask is None else np.nonzero(keep_mask)[0]
    la, ma, na = l[ang], m[ang], n[ang]
    ka = ang.size
    pos = {(int(la[a]), int(ma[a]), int(na[a])): a for a in range(ka)}
    T = np.zeros((ka, ka), dtype=complex)
    done = np.zeros(ka, dtype=bool)
    r2 = 1.0 / np.sqrt(2.0)
    for a in range(ka):
        if done[a]:
            continue
        partner = pos.get((int(la[a]), -int(ma[a]), -int(na[a])))
        if partner is None:
            return None                      # basis not closed -> cannot realify
        if partner == a:
            T[a, a] = 1.0                     # self-paired (m=n=0): already real
            done[a] = True
        else:
            i, j = (a, partner) if a < partner else (partner, a)
            ei = (-1.0) ** (int(ma[i]) - int(na[i]))
            T[i, i] = r2;        T[i, j] = ei * r2
            T[j, i] = -1j * r2;  T[j, j] = 1j * ei * r2
            done[i] = done[j] = True
    return T


class SystemSO3:
    """SO(3) Wigner-D engine: rotational diffusion + photophysics with a
    per-state body-frame transition dipole (engine A, a40 n030).

    Same constructor/API as ``engine_s2.SystemS2`` (fluorophore, diffusion,
    illumination, detection, lmax) so the two are interchangeable behind
    ``core.System``. It consumes the *same* model objects (``ModulatedLasers`` /
    ``SingleLaser`` / ``PolarizedDetection``, with polarization strings + NA), but
    expands the *barrel* orientation on SO(3) in complex Wigner-D functions and
    lets each fluorophore state carry its own transition dipole
    (``fluorophore.dipole_orientations()``); the reorientation on switching lives
    in the optics, not a jump.

    Normalization matches the S^2 engine: with ``c_000 = population`` (0..1) the
    detector signals agree for non-reorienting fluorophores (reduction test).
    """

    def __init__(self, fluorophore, diffusion, illumination, detection=None,
                 lmax=2, even_m_reduction=True, real_eig=True):
        import spherical
        self.fluorophore = fluorophore
        self.diffusion = diffusion
        self.illumination = illumination
        self.detection = detection
        self.lmax = lmax
        # Drop odd lab-order (m) coefficients from the eigenproblem (exact for the
        # implemented even-lab-order optics; ~4-5x faster). Disable for fields with
        # odd lab order (out-of-xy-plane-tilted linear, or elliptical/rotating),
        # which populate m = +-1. See core.System docstring and n030 sec.14.
        self.even_m_reduction = even_m_reduction
        # Run the eig in real arithmetic via a unitary realifying transform of the
        # (complex) Wigner-D matrix (~2x on the dominant eig; exact, with a
        # self-check + complex fallback). See core.solve_evolution / n030 sec.14.
        self.real_eig = real_eig

        self.l, self.m, self.n = quantum_numbers_so3(lmax)
        self.wigner = spherical.Wigner(lmax)
        self._i000 = int(np.nonzero(
            (self.l == 0) & (self.m == 0) & (self.n == 0))[0][0])

        # Precompute the parity-reduction mask and the realifying transform once
        # (they depend only on the basis, not on M). Engine physics: linear/
        # circular light-matter is parity-preserving, so starting from (l,m)=(0,0)
        # only the even-l (and, for even-lab-order optics, even-m) subspace is ever
        # reached -- exact, see n030 sec.14. core.solve_evolution just applies them.
        self._keep_mask = (self.l % 2 == 0)
        if even_m_reduction:
            self._keep_mask = self._keep_mask & (self.m % 2 == 0)
        self._transform = (realifying_transform(self.l, self.m, self.n,
                                                self._keep_mask)
                           if real_eig else None)
        return None

    def _absorption_operators(self):
        # W[laser, absorbing_state] = photon_flux * "multiply by the NA-corrected
        # photoselection |e.(R d_s)|^2" for that laser's polarization/NA.
        ill = self.illumination
        ns = self.fluorophore.nspecies
        dip = self.fluorophore.dipole_orientations()
        ncoeff = self.l.size
        polarization = np.atleast_1d(ill.polarization)
        photon_flux = np.atleast_1d(ill.photon_flux)
        nlasers = polarization.size
        W = np.zeros((nlasers, ns, ncoeff, ncoeff), dtype=complex)
        for j in range(nlasers):
            for s in range(ns):
                b = na_corrected_absorption_coeffs(
                    self.l, self.m, self.n, polarization[j], tuple(dip[s]),
                    self.wigner, ill.numerical_aperture, ill.refractive_index)
                W[j, s] = photon_flux[j] * multiplication_operator(
                    self.l, self.m, self.n, b, self.lmax)
        return W

    def diffusion_kinetics_matrix(self):
        ns = self.fluorophore.nspecies
        ncoeff = self.l.size
        ill = self.illumination
        modulation = np.atleast_2d(ill.modulation)
        nwindows = modulation.shape[1]
        nlasers = modulation.shape[0]

        D = isotropic_diffusion_matrix_so3(
            self.l, self.diffusion.diffusion_coefficient, ns)

        k = self.fluorophore.kinetics_matrix()  # (nwl+1, ns, ns) scalars
        wavelength_indexes = np.concatenate(
            ([0], find_wavelenght(self.fluorophore.wavelength,
                                  np.atleast_1d(ill.wavelength)) + 1))
        k_sel = k[wavelength_indexes]           # (nlasers+1, ns, ns)

        Wabs = self._absorption_operators()     # (nlasers, ns, ncoeff, ncoeff)
        eye = np.eye(ncoeff, dtype=complex)

        M = np.zeros((nwindows, ns, ns, ncoeff, ncoeff), dtype=complex)
        for w in range(nwindows):
            # Fi[laser_incl_nonlight, in_state]: angular operator. Layer 0 is the
            # non-light transitions (identity: switching does not rotate the
            # barrel, n030 §4.1); light layers carry the absorbing state's dipole
            # operator, scaled by this window's modulation.
            Fi = np.zeros((nlasers + 1, ns, ncoeff, ncoeff), dtype=complex)
            Fi[0] = eye
            for j in range(nlasers):
                Fi[1 + j] = modulation[j, w] * Wabs[j]
            # K[out,in] = sum_laser k[laser,out,in] * Fi[laser,in]  (Fi keyed on
            # the absorbing/in state, so depletion is handled correctly).
            K = np.einsum('aoi,aicd->oicd', k_sel, Fi)
            M[w] = diffusion_kinetics_matrix(D, K)
        return M

    def solve(self, time):
        M = self.diffusion_kinetics_matrix()
        ns = self.fluorophore.nspecies
        ncoeff = self.l.size
        ill = self.illumination
        time_windows = np.atleast_1d(ill.time_windows).astype(float)
        nwindows = M.shape[0]

        c0 = np.zeros((ns, ncoeff), dtype=complex)
        c0[:, self._i000] = self.fluorophore.starting_populations

        time_lab = time + ill.time0
        time_mod = np.cumsum(time_windows)
        time_mod = np.insert(time_mod, 0, 0)
        time_mod[-1] = np.max(time_lab)

        c = np.zeros((ns, ncoeff, time.size), dtype=complex)
        for i in range(nwindows):
            sel = np.logical_and(time_lab >= time_mod[i], time_lab <= time_mod[i + 1])
            ti = np.append(time_lab[sel], time_mod[i + 1]) - time_mod[i]
            ci, _, _ = solve_evolution(M[i], c0, ti, keep_mask=self._keep_mask,
                                       transform=self._transform,
                                       real_output=False)
            c[:, :, sel] = ci[:, :, :-1]
            c0 = ci[:, :, -1]
        self._c = c
        return c

    def detector_signals(self, time):
        c = self.solve(time)
        qy = self.fluorophore.quantum_yield_fluo
        dip = self.fluorophore.dipole_orientations()
        det = self.detection
        polarization = np.atleast_1d(det.polarization)
        ndet = polarization.size
        s = np.zeros((ndet, time.size))
        for d in range(ndet):
            sig = np.zeros(time.size, dtype=complex)
            for sp in range(self.fluorophore.nspecies):
                if qy[sp] == 0:
                    continue
                bcol = na_corrected_absorption_coeffs(
                    self.l, self.m, self.n, polarization[d], tuple(dip[sp]),
                    self.wigner, det.numerical_aperture, det.refractive_index)
                Wd = multiplication_operator(self.l, self.m, self.n, bcol, self.lmax)
                sig += qy[sp] * (Wd @ c[sp])[self._i000]
            s[d] = sig.real
        return s
