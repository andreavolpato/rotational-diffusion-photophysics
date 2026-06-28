"""Core solver primitives shared by the engines.

These are **basis-agnostic**: they operate on a flat (species x coefficient)
state and on block diffusion/kinetics matrices, without knowing whether the
angular basis is real spherical harmonics on S^2 (``engine_s2.py``) or complex
Wigner-D functions on SO(3) (``engine_so3.py``). Keeping them here lets both
engines reuse the same propagator and matrix assembly.
"""
import numpy as np


def _has_dipole_reorientation(fluorophore):
    """True if the fluorophore's states have non-identical body-frame dipoles
    (so the orientation cannot be reduced to a single dipole axis -> needs SO(3))."""
    if not hasattr(fluorophore, "dipole_orientations"):
        return False
    try:
        d = np.asarray(fluorophore.dipole_orientations(), dtype=float)
    except Exception:  # noqa: BLE001
        return False
    return not np.allclose(d, d[0])


def System(fluorophore, diffusion, illumination, detection=None, lmax=None,
           representation="auto"):
    """Unified entry point: build the right engine for the fluorophore.

    Both backends consume the same spec (fluorophore, diffusion model,
    illumination, detection) and the same population normalization
    (``c_000 = population``), so they are interchangeable.

      representation = 's2'   -> spherical-harmonic engine (``engine_s2.SystemS2``):
                                 dipole-axis on S^2; cheapest. Correct when the
                                 transition dipole does not reorient between states.
      representation = 'so3'  -> Wigner-D engine (``engine_so3.SystemSO3``): full
                                 barrel orientation on SO(3) with a per-state dipole;
                                 needed when switching reorients the dipole.
      representation = 'auto' -> 'so3' if the fluorophore's state dipoles differ
                                 (``dipole_orientations``), else 's2' (cheap).

    ``lmax`` defaults to each backend's own default (6 for S^2, 2 for SO(3)).
    """
    rep = representation
    if rep == "auto":
        rep = "so3" if _has_dipole_reorientation(fluorophore) else "s2"

    if rep == "s2":
        from rotational_diffusion_photophysics.engine_s2 import SystemS2
        kw = {} if lmax is None else {"lmax": lmax}
        return SystemS2(fluorophore, diffusion, illumination, detection, **kw)
    if rep == "so3":
        from rotational_diffusion_photophysics.engine_so3 import SystemSO3
        kw = {} if lmax is None else {"lmax": lmax}
        return SystemSO3(fluorophore, diffusion, illumination, detection, **kw)
    raise ValueError(f"unknown representation '{representation}' (use s2/so3/auto)")


def find_wavelenght(wavelength, laser):
    # Map each laser wavelength to its index in the fluorophore's wavelength list.
    wavelength = np.array(wavelength)
    laser = np.array(laser)

    laser_to_use = np.isin(laser, wavelength)
    assert np.all(laser_to_use), "Fluorophore data at one or more laser wavelenghts is missing."

    wavelength_indexes = np.zeros(laser.shape)
    for i, laseri in enumerate(laser):
        wavelength_indexes[i] = np.where(wavelength == laseri)[0]
    wavelength_indexes = np.int32(wavelength_indexes)
    return wavelength_indexes


def solve_evolution(M, c0, time, l=None, real_output=True):
    # Analitically solve the diffusion-kinetic problem by matrix exp of M.
    # We compute p(t) = U exp(L t) U^-1 p0, where (L, U) diagonalize M.
    #
    # real_output=True (default) discards the imaginary part of the result, which
    # is correct for the real spherical-harmonic engine (the density is real and
    # M is real, so any imaginary part is rounding error). The complex Wigner-D
    # SO(3) engine must pass real_output=False to keep the genuine complex
    # coefficients.
    nspecies = c0.shape[0]
    ncoeffs = c0.shape[1]

    # Variables with unique index for species and angular-basis coefficients
    c0 = c0.flatten()
    M = np.transpose(M, axes=[0, 2, 1, 3])
    M = np.reshape(M, [nspecies * ncoeffs, nspecies * ncoeffs])

    # Restrict the eigenproblem to the even-l subspace.
    # Rotational diffusion is diagonal in l, linear light-matter interaction
    # couples only l1=0 and l1=2 (parity preserving), and the initial condition
    # lives entirely in l=0. Even-l and odd-l coefficients therefore never mix,
    # and the odd-l block stays exactly zero for all time. We can drop it from
    # the (cubic-cost) eig/inv and scatter zeros back afterwards. This is exact
    # for the implemented linear interaction, not an approximation.
    # NOTE: if an orientation-dependent process involving odd l1 is ever added
    # (or a non-l=0 initial condition), pass l=None to disable this reduction.
    if l is not None:
        keep = np.nonzero(np.tile(l % 2 == 0, nspecies))[0]
        M = M[np.ix_(keep, keep)]
        c0 = c0[keep]

    # Diagonalize the (reduced) M and invert the eigenvector matrix.
    L, U = np.linalg.eig(M)
    Uinv = np.linalg.inv(U)

    # Evaluate p(t) = U exp(L t) U^-1 p0 at all requested times at once.
    # Project the initial condition onto the eigenbasis, a = U^-1 p0, then
    # the diagonal propagator exp(L t) is just an elementwise factor on a.
    # This replaces the per-time-point python loop (and the np.diag it built
    # for every time point) with two vectorized BLAS calls.
    a = Uinv.dot(c0)                        # eigenbasis amplitudes
    propagator = np.exp(np.outer(L, time))  # exp(L t), shape (Nkeep, time.size)
    propagated = U.dot(a[:, None] * propagator)
    cr = np.real(propagated) if real_output else propagated

    # Scatter the solved coefficients back into the full basis (odd-l rows
    # stay zero), then reshape into a 3D array separating the species.
    if l is not None:
        c = np.zeros((nspecies * ncoeffs, time.size), dtype=cr.dtype)
        c[keep] = cr
    else:
        c = cr
    c = np.reshape(c, (nspecies, ncoeffs, time.size))
    return c, L, U


def diffusion_kinetics_matrix(D, K):
    # Assemble the full diffusion-kinetics block matrix M from the per-species
    # diffusion blocks D and the kinetics blocks K. Basis-agnostic.
    nspecies = D.shape[0]
    M = np.zeros(K.shape, dtype=np.promote_types(D.dtype, K.dtype))

    for i in range(nspecies):
        for j in range(nspecies):
            # Add the rotational diffusion blocks on the diagonal
            if i == j:
                M[i, i] = D[i]
            # Add the kinetics blocks in all the slots
            M[i, j] = M[i, j] + K[i, j]

    # Add on the diagonal the kinetics contributions that deplete the states
    for i in range(nspecies):
        for j in range(nspecies):
            if i != j:
                M[i, i] = M[i, i] - M[j, i]
    return M
