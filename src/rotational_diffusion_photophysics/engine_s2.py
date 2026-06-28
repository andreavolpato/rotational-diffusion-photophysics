'''
Andrea Volpato (2023)
andrea.volpato@outlook.com

This module solves analytically the rotational diffusion and kinetics
in flurescence experiments with complex photophysics.
An arbitrary kinetic scheme can be implemented using fluorophore classes.
The illumination of the experiment is implemented as a pulse scheme with
square wave modulations.
Detection and illumination can be done in arbitrary NA objectives, using 
the Axelrod correction.
'''

import numpy as np
import pyshtools as sht  # used in many spherical calculations
# import spherical as sf  # currently used only for CG coefficients

# Basis-agnostic solver primitives shared with engine_so3 (see core.py).
from rotational_diffusion_photophysics.core import (
    find_wavelenght,
    solve_evolution,
    diffusion_kinetics_matrix,
)

################################################################################
# Classes for programming experiments
################################################################################

class SystemS2:
    def __init__(self,
                 fluorophore,
                 diffusion,
                 illumination,
                 detection=None,
                 lmax=6,
                 norm='4pi',
                 ):
        # The simulation requires the expansion of all angular functions in 
        # spherical harmonics (SH). lmax is the cutoff for the maximum l quantum
        # number of the SH basis set, i.e. 0 <= l <= lmax.
        #
        # The '4pi' normalization is reccomended because the coefficients for
        # l=0 and m=0 for the angular probability distributions give directly
        # the state populations.
        self.lmax = lmax
        self.norm = norm

        # Compute arrays with quantum numbers l and m of the SH basis set.
        self._l, self._m = quantum_numbers(self.lmax)

        # Compute the real-SH triple-product table. These coefficients are
        # necessary for the evaluation of the product of angular functions from
        # the SH expansion coefficients (i.e. the "multiply by a function"
        # operator used for light-matter interaction).
        self._wigner3j_prod_coeffs = real_sh_product_coeffs(self._l, self._m)

        # Import the classes containing the parametrization and characteristics
        # of fluorophore, diffusion model, illumination, and detection.
        self.fluorophore = fluorophore
        self.diffusion = diffusion
        self.illumination = illumination
        self.detection = detection

        '''
        Small description of hidden variables. 
        These are used mainly for debugging.
        self._F is the photon flux prod matrix, when multiplied by the cross
            section gives the absorption rate prod matrix as a function of the 
            angle.
        self._c_exc are SH coefficients for the orientational functions behind F. 
            It can be interpred as the number of photons that would be absorbed 
            by a molecule with cross section 1cm2 as a function of the 
            orientation of the dipole.
        self._D Diffusion matrix rate model for each species.
        self._K Kinetic matrix rate model for each time window and each pair 
            of species.
        self._M = M Full diffusion-kinetics matrix for each time window and each 
            pair of species.
        self._c0 SH coeffs of starting orientational populations
        self._c SH coeffs of populations computed at the derired times
        self._c_det SH coeffs of the detectors collection functions
        self._s fluorescent signal for each detector
        '''
        return None

    def detector_signals(self, time):
        # Compute the flurescence signal registered in the detectors
        # Firt, solve the diffusion/kinetics problem and get the SH coefficients
        # of populations.
        c = self.solve(time)

        # Multiply the populations by the quantum yields of fluorescence.
        # Usually, only the fluorescent state will have a quantum yield 
        # different from zero.
        c_fluo = c * self.fluorophore.quantum_yield_fluo[:,None,None]

        # Compute the SH coefficients for the collection functions of 
        # the detectors
        c_det = self.detection.detector_coeffs(self._l, self._m)
        ndetectors = c_det.shape[0]

        # Initialize signal array and compute signals.
        # s[i] is the bare SH-coefficient dot product sum_lm c_det[i]_lm c_fluo_lm.
        # In the '4pi' convention this is already the orientational integral of
        # (molecular density x per-molecule collection efficiency): the 4*pi from
        # the SH inner product (int f g dOmega = 4*pi sum c_f c_g) cancels the
        # 1/(4*pi) in the density normalization (c_00 = population = orientational
        # mean). So no extra scaling is applied. Cf. notes.tex eq. (highna_detection)
        # with the experimental constant A = 1; e.g. an isotropic excited state
        # gives s = (1/3) Phi P, the cos^2 average a linear detector collects.
        s = np.zeros( (ndetectors, time.size) )
        for i in range(ndetectors):
            s[i] = np.sum(c_det[i][None,:,None] * c_fluo, axis=(0,1))

        # Save variables, mainly for debugging.
        self._c_det = c_det
        self._s = s
        return s

    def solve(self, time):
        # Solve the time evolution of the system including the pulse scheme.
        
        # Compute the diffusion kinetics matrix
        # Matrix M contains all the information about the time evolution of the 
        # system in every time window.
        M = self.diffusion_kinetics_matrix()

        # Get the initial conditions for the experiment from the fluorophore
        # class.
        c0 = np.zeros( (self.fluorophore.nspecies, self._l.size) )
        c0[:,0] = self.fluorophore.starting_populations

        # Optimization, remove all odd l value coefficients from c0 and M.
        # We will need to add them back, after the solution.
        # This semplification is usefull when only light matter interaction can
        # affect the orientational probablities.None
        # print(M.shape)
        # M, c0 = remove_odd_coeffs(M, c0, self._l)
        # print(M.shape)

        # Shift the time with time0, so it is represented in the laboratory
        # time (starting with the time zero of pulse sequence).
        time_lab = time + self.illumination.time0
        time_mod = np.cumsum(self.illumination.time_windows)

        # Insert a zero at the beginning of the modulation time sequence and 
        # extend last time window to fit the whole requested times to sovle.
        time_mod = np.insert(time_mod, 0, 0 )
        time_mod[-1] = np.max(time_lab)

        # Solve evolution in every time window
        c = np.zeros((self.fluorophore.nspecies,
                      self._l.size,
                      time.size))
        for i in np.arange(self.illumination.nwindows):
            # Selection of time axis inside the current time window
            time_sel = np.logical_and(time_lab >= time_mod[i],
                                      time_lab <= time_mod[i+1])

            # Select the time points inside the current time window
            time_i = time_lab[time_sel]

            # Add to the time points the end of the time window
            # so we can comute the starting coefficients for the next window.
            time_i = np.append(time_i, time_mod[i+1])

            # Shift the local time axis so the zero concides with the beginning
            # of the current time window
            time_i = time_i - time_mod[i]

            # Solve the time evolution
            c_i, _, _ = solve_evolution(M[i], c0, time_i, l=self._l)

            # Save results and update initial conditions for the next window
            c[:,:,time_sel] = c_i[:,:,:-1]
            c0 = c_i[:,:,-1]
        
        # Add back zeros in place of odd l coeffs
        # M, c0, c = add_odd_coeffs_zeros(M, c0, c, self._l)

        # Save variables, mainly for debugging.
        self._c0 = c0
        self._c = c
        self._p = np.sum(c, axis=1)
        return c

    def diffusion_kinetics_matrix(self):
        # Preliminary computations based on the illumination class.
        # Photon flux product coefficients based on wigner3j symbols.
        # F has dimensions: [nlansers, l_size, l_size].
        F, c_exc = self.illumination.photon_flux_prod_coeffs(
                                                    self._l,
                                                    self._m,
                                                    self._wigner3j_prod_coeffs)

        # Compute the rotational diffusion matrix, it will not change with the
        # time windows of laser modulation.
        # The diffusion model is provided by the fluorophore class.
        # In the future, if more complex diffusion models will be implemented,
        # it might be convinient to create a separate diffusion class.
        D = self.diffusion.diffusion_matrix(self._l, self._m,
                self.fluorophore.nspecies)

        # Prepare the rotational diffusion matrix for each time window
        nwindows = self.illumination.nwindows
        nspecies = self.fluorophore.nspecies
        M = np.zeros( (nwindows,
                       nspecies, nspecies,
                       self._l.size, self._l.size) )
        K = np.zeros( (nwindows,
                       nspecies, nspecies,
                       self._l.size, self._l.size) )

        # Calcualte the kinetic matrix, k.
        # Light driven transition are expressed as corss sections.
        # Transition induced by a certain wavelength are groupped
        # in separate layers of k.
        # k has three dimensions: [nwavelenght+1, nspecies, nspecies].
        # k[0] is reserved for all non-light-induced transitions.
        k = self.fluorophore.kinetics_matrix()
        # Select the right lasers to use in the calculations
        nlasers = np.size(self.illumination.wavelength)
        lasers = self.illumination.wavelength
        wavelength = self.fluorophore.wavelength
        wavelength_indexes = np.concatenate( ([0], find_wavelenght(wavelength, lasers)+1) )
        # Allocate memory for the photon flux matrices for each window.
        # +1 accounts for the transitions not induced by light.
        Fi = np.zeros((nlasers+1, self._l.size, self._l.size))
        Fi[0] = np.eye(self._l.size, self._l.size)
        for i in np.arange(nwindows):
            Fi[1:] = F * self.illumination.modulation[:,i][:,None,None]
            K[i] = np.einsum('ijk,ilm->jklm', k[wavelength_indexes], Fi)
            M[i] = diffusion_kinetics_matrix(D, K[i])

        # Save variables, mainly for debugging.
        self._c_exc = c_exc
        self._F = F
        self._D = D
        self._K = K
        self._M = M
        return M


################################################################################
# Engine of the rotational-diffusion and kinetics solver
################################################################################

def quantum_numbers(lmax):
    # Generate arrays with quantum numbers l and m
    l = np.array([])
    m = np.array([])
    for li in np.arange(lmax+1):
        # This is the way pyshtools is exporting vectors of sh expansions
        l = np.concatenate((l, np.ones(2*li+1)*li))
        m = np.concatenate((m, np.arange(li+1)))
        m = np.concatenate((m, np.arange(-1,-li-1,-1)))
    l = np.int32(l)
    m = np.int32(m)
    return l, m

def real_sh_product_coeffs(l, m):
    # 3D table for the product of two real spherical harmonics (4pi norm):
    #   w3jp[i,j,k] = coefficient of R_k in the product  R_i * R_j ,
    # so that "multiply by a function g" becomes the matrix
    #   F[k,j] = sum_i w3jp[i,j,k] * g_i      (see kinetic_prod_block).
    #
    # The table is computed exactly by quadrature on a Driscoll-Healy grid:
    # synthesise R_i and R_j, multiply them pointwise, and analyse the product
    # back to SH coefficients. This is convention-consistent with the rest of
    # the engine (pyshtools, '4pi' norm, csphase=1) by construction.
    #
    # WHY NOT WIGNER-3j: the previous implementation used the *complex* SH
    # Gaunt rule (a single output channel m3 = m1+m2). For pyshtools *real*
    # harmonics the product of two m!=0 harmonics also has a |m1-m2| channel
    # (e.g. cos2phi*cos2phi = 1/2 + 1/2 cos4phi -> m=0 AND m=4). Dropping it
    # corrupted every non-axial (x/y linear) excitation as soon as the density
    # carried m!=0 content (saturation / sequential pulses): total excited
    # population stopped being rotation invariant. See study a40 n030 sec.13.
    #
    # As before, only multiplier indices i with l in {0,2} are filled: linear
    # light-matter photoselection functions have degree <= 2.
    n = l.size
    lmax = int(np.max(l))
    grid_lmax = 2 * lmax + 2  # resolve products up to degree lmax+2 (no aliasing)
    w3jp = np.zeros([n, n, n])

    def synth(vec):
        cilm = sht.shio.SHVectorToCilm(vec)
        return sht.expand.MakeGridDH(cilm, sampling=2, norm=1, csphase=1,
                                     lmax=grid_lmax)

    def analyze(grid):
        cilm = sht.expand.SHExpandDH(grid, sampling=2, norm=1, csphase=1,
                                     lmax_calc=lmax)
        return sht.shio.SHCilmToVector(cilm, lmax)

    # Synthesise every basis harmonic R_j once.
    basis_grid = []
    for j in np.arange(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        basis_grid.append(synth(ej))

    l1_allowed_indexes = np.arange(n)[np.logical_or(l == 0, l == 2)]
    for i in l1_allowed_indexes:
        for j in np.arange(n):
            w3jp[i, j, :] = analyze(basis_grid[i] * basis_grid[j])
    return w3jp

# Backwards-compatible alias (the old name was a misnomer: see above, the table
# is a real-SH product, not a bare Wigner-3j symbol).
wigner_3j_prod_3darray = real_sh_product_coeffs

################################################################################
# Unused functions - still possibly useful
################################################################################

# def clebsch_gordan_prod_3darray(l, m):
#     # Currently not used because slow
#     # 3D array with all the clebsh gordan coefficients for sh multiplication
#     n = l.size
#     cgp = np.zeros([n, n, n])
#     for i in np.arange(n):
#         for j in np.arange(n):
#             for k in np.arange(n):
#                     cgp[i,j,k] = np.sqrt( (2*l[i] + 1) * (2*l[j] + 1) / 
#                                         ( np.pi*4    * (2*l[k] + 1) ) ) * (
#                                 sf.clebsch_gordan(l[i], 0,
#                                                 l[j], 0,
#                                                 l[k], 0) * 
#                                 sf.clebsch_gordan(l[i], m[i],
#                                                 l[j], m[j],
#                                                 l[k], m[k])
#                                 )
    
#     # Multiply constant due to normalization issues
#     cgp = cgp*np.sqrt(4*np.pi) 
#     return cgp

# def kinetics_diffusion_matrix_lmax(Dvec, Kmatrix, lmax):
#     # Create the full kinetic diffusion matrix expanded in sh staring 
#     # from scratch.
#     # In this routine the expansion in l,m are included.
#     assert len(Dvec) == len(Kmatrix)
#     # Compute quantum number arrays and cg coefficients
#     # The main simplification is that linear light matter interaction 
#     # limits l1 to 0 and 2.
#     l, m = quantum_numbers(lmax)
#     #TODO Compute only cg coefficients that are needed, partially done
#     # cgp = clebsch_gordan_prod_3darray(l, m)
#     cgp = wigner_3j_prod_3darray(l, m)

#     ncoeff = l.size  # number of spherical harmonics expansion coefficients
#     nspecies = len(Dvec)  # number of species

#     M = np.array( [ [np.zeros([ncoeff,ncoeff])]*nspecies ]*nspecies )
#     for i in range(nspecies):
#         for j in range(nspecies):
#             if i == j:
#                 M[i,j] = isotropic_diffusion_block(l, m, Dvec[i])
#             # Create blocks of the kinetic matrix rates
#             else:
#                 k = Kmatrix[i][j]
#                 if np.size(k) == 1:
#                     M[i,j] = np.eye(ncoeff).dot(k)
#                 else:
#                     M[i,j] = kinetic_prod_block(k, cgp)

#     for i in range(nspecies):
#         for j in range(nspecies):
#             if i != j:
#                 M[i,i] = M[i,i]-M[j,i]
#     return M

# def remove_odd_coeffs(M, c0, l):
#     l_sel = l%2 == 0
#     c0 = c0[:, l_sel]
#     M = M[:, :, :, l_sel, l_sel]
#     return M, c0

# def add_odd_coeffs_zeros(M, c0, c, l):
#     c0_out = np.zeros((c0.shape[0], l.size))
#     c_out = np.zeros((c0.shape[0], l.size, c0.shape[2]))
#     M_out = np.zeros((M.shape[0], M.shape[1], M.shape[2], l.size, l.size))
#     l_sel = l%2 == 0
#     c0_out[:,l_sel] = c0
#     c_out[:,l_sel,:] = c
#     M_out[:,:,:,l_sel,l_sel] = M
#     return M_out, c0_out, c_out


if __name__ == "__main__":
    a = None
    # from codetiming import Timer

    # rsEGFP2 = NegativeSwitcher(cross_section_on_blue=1e-10,
    #                            lifetime_on=3.6e-9,
    #                            quantum_yield_on_to_off=0.001,
    #                            diffusion_coefficient=1/(6*100e-6) )

    # tau = 100e-6 # us anisotropy decay
    # D = 1/(tau*6) # Hz
    # yield_off = 0.001 
    # tau_off = 80e-6 # time constant off switching
    # tau_on_exc = 3.6e-9 # lifetime of excited state
    # k21 = 1 / (tau_off * yield_off)
    # k12 = 1 / tau_on_exc
    # k32 = (1/tau_off) / yield_off

    # lmax = 8
    # omega = make_angles(lmax)
    # k21a = (np.sin(omega[0])**2) * k21
    # k21grid, k21c, k21cilm = make_grid(k21a, lmax)
    # plot_proj(k21grid, clims=[])

    # # Test product using cg coeff
    # l, m = quantum_numbers(lmax)
    # t = Timer()
    # t.start()
    # cgp = clebsch_gordan_prod_3darray(l, m)
    # t.stop()
    # # k21prod = kinetic_prod_block(k21c, cgp)
    # # kp = np.cos(omega[0])**2 * np.cos(omega[1]+np.pi)**2
    # # kpgrid, kpc = make_grid(kp, lmax)
    # # k21kpc = k21prod.dot(kpc)
    # # k21kpcilm = sht.shio.SHVectorToCilm(k21kpc)
    # # k21kparray = sht.expand.MakeGridDH(k21kpcilm, sampling=2)
    # # k21kpgrid = sht.shclasses.SHGrid.from_array(k21kparray)
    # # k21kpgrid.plot3d()

    # # Array with all the diffusion tensors/scalar for every specie.
    # # In this case every specie diffuse with the same rate.
    # Dvec = [D, D, D]

    # # Array with kinetic constants connecting the states.
    # Kmatrix = [[   0, k12, 0],
    #            [k21a,   0, 0],
    #            [   0, k32, 0]]


    # # # Simplified kinetic scheme
    # # # Array with all the diffusion tensors/scalar for every specie.
    # # # In this case every specie diffuse with the same rate.
    # # Dvec = [D, D]

    # # # Array with kinetic constants connecting the states.
    # # Kmatrix = [[   0,   0],
    # #            [k21a,   0]]

    # t = Timer()
    # t.start()
    # M = kinetics_diffusion_matrix(Dvec, Kmatrix, lmax)
    # t.stop()

    # # initial conditions
    # c0a = omega[0]*0 +1
    # c0grid, c0vec, c0cilm = make_grid(c0a, lmax)
    # c0 = np.zeros(((lmax+1)**2,3))
    # c0[:,0] = c0vec
    # # p0_1 = 1 + np.cos(omega[0]) * 0
    # # p0_1grid, c0_1 = make_grid(p0_1, lmax)
    # # c0[:,0] =c0_1

    # time = np.logspace(-11,-3,128)
    # # time = np.linspace(0,1e-3,1000)
    # t.start()
    # c, L, U =solve_evolution(M, c0, time)
    # t.stop()
    # # plt.imshow(np.real(Dblock))
    
    # Uinv = np.linalg.inv(U)
    # ci = np.matmul(U, np.diag(np.exp(L*100e-6)))
    # ci = np.matmul(ci, Uinv)
    # plt.imshow(np.real(ci))


    # t.start()
    # np.linalg.inv(U)
    # t.stop()

    # cplot = c[:,0,:]
    # cplotgrid = vecs2grids(cplot)

    # plt.figure()
    # plt.plot(time, cplot.T)
    # plt.xscale('log')

    # plot_proj(cplotgrid[100])

    # # cplotgrid[10].plot()
    # # plt.imshow(M)
    # # plt.show()
    # # plt.imshow(cplotgrid[20].data, vmin=0, vmax=1)
