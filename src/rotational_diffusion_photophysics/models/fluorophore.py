import numpy as np

################################################################################
# Reversibly switchable fluorescence proteins (RSFP) classes
################################################################################

## Negative switchers

class NegativeSwitcher:
    def __init__(self,
                 extinction_coeff_on=[0, 0],
                 extinction_coeff_off=[0, 0],
                 wavelength=[405, 488],
                 lifetime_on=3e-9,
                 lifetime_off=16e-12,
                 quantum_yield_on_to_off=0.001,
                 quantum_yield_off_to_on=0.2,
                 quantum_yield_on_fluo=1,
                 starting_populations=[1,0,0,0],
                 deprotonation_time_off = 15e-6,  # for 6 states model
                 protonation_time_on = 150e-6,  # for 6 states model
                 quantum_yield_trans_to_cis_anionic=0,  # for 8 states model
                 quantum_yield_cis_to_trans_neutral=0, # for 8 states model
                 quantum_yield_cis_anionic_bleaching=0,
                 quantum_yield_trans_anionic_bleaching=0,
                 flurophore_type='rsfp_negative_4states'):
        #TODO: make simpler init passing only variables to self.
        #      Afterwards in the kinetic_matrix() method compute the necessary parameters.
        #      Make a method called update_params() to do so.
        # Cross section in cm2 of absorptions
        epsilon2sigma = 3.825e-21  # [Tkachenko2007, page 5]
        self.extinction_coeff_on = np.array(extinction_coeff_on)  # [M-1 cm-1]
        self.extinction_coeff_off = np.array(extinction_coeff_off)  # [M-1 cm-1]
        self.cross_section_on = self.extinction_coeff_on * epsilon2sigma  # [cm2]
        self.cross_section_off = self.extinction_coeff_off * epsilon2sigma  # [cm2]
        self.wavelength = np.array(wavelength)  # [nm]

        # Lifetime of the on excited state in seconds
        # Assumption: lifetime on and off are the same for the same protonation
        # state. This might not be the case, especially because cis_neutral
        # species is not fluorescent, so most likely it will have a shorter
        # lifetime. For small enough excitation kinetic rates, if we don't have
        # accumulation of exited state species, then it doesn't matter much.
        self.lifetime_on = lifetime_on  # [s]
        self.lifetime_off = lifetime_off  # [s]

        # Quantum yield of an off-switching event from the on excited state and
        # fluorescence from the on state
        self.quantum_yield_on_fluo = quantum_yield_on_fluo
        self.quantum_yield_on_to_off = quantum_yield_on_to_off  # cis_to_trans_anionic 
        self.quantum_yield_off_to_on = quantum_yield_off_to_on  # trans_to_cis_neutra

        # Quantum yeilds and Protonation and deprotonation times for 6 and 8
        # states models.
        self.quantum_yield_cis_to_trans_neutral = quantum_yield_cis_to_trans_neutral
        self.quantum_yield_trans_to_cis_anionic = quantum_yield_trans_to_cis_anionic
        self.protonation_time_on = protonation_time_on
        self.deprotonation_time_off = deprotonation_time_off

        # Bleaching quantum yields, assuming only a bleaching channel from 
        # excitaiton of the on state.
        self.quantum_yield_cis_anionic_bleaching = quantum_yield_cis_anionic_bleaching
        self.quantum_yield_trans_anionic_bleaching = quantum_yield_trans_anionic_bleaching

        # Label describing the fluorophore type
        # Number of states in the kinetic model
        self.nspecies = np.size(starting_populations)
        self.fluorophore_type = flurophore_type #TODO make the kinetic matrix based on the fluorophore type

        # Index of the fluorescent state
        # Here, only one fluorescent state is assumed.
        # This is not necessary in general and could be extended in the future.
        self.quantum_yield_fluo = np.zeros((self.nspecies))
        self.quantum_yield_fluo[1] = self.quantum_yield_on_fluo

        # Population at the beginning of the experiment
        self.starting_populations = starting_populations
        return None

    def kinetics_matrix(self):
        # Compute the kinetics matrix 

        # Initialize arrays
        nwavelengths = self.wavelength.size
        K = np.zeros( (nwavelengths+1, self.nspecies, self.nspecies))

        # Put the kinetic constants connecting the right species
        # K[1,0] is the kinetic constant for the proces 1 <- 0.
        if  self.fluorophore_type == 'rsFP_negative_4states':
            # Light independent transitions
            K[0][0,1] = 1 / self.lifetime_on
            K[0][2,1] = 1 / self.lifetime_on  * self.quantum_yield_on_to_off
            K[0][2,3] = 1 / self.lifetime_off
            K[0][0,3] = 1 / self.lifetime_off * self.quantum_yield_off_to_on
            # Light driven transition
            for j in np.arange(nwavelengths):
                K[j+1][1,0] = self.cross_section_on[j]
                K[j+1][3,2] = self.cross_section_off[j]

        if self.fluorophore_type == 'rsFP_negative_6states':
            # Light independent transitions
            K[0][0,1] = 1 / self.lifetime_on
            K[0][2,1] = 1 / self.lifetime_on  * self.quantum_yield_on_to_off
            K[0][3,2] = 1 / self.protonation_time_on
            K[0][3,4] = 1 / self.lifetime_off
            K[0][5,4] = 1 / self.lifetime_off * self.quantum_yield_off_to_on
            K[0][0,5] = 1 / self.deprotonation_time_off
            # Light driven transition
            for j in np.arange(nwavelengths):
                K[j+1][1,0] = self.cross_section_on[j]
                K[j+1][4,3] = self.cross_section_off[j]

        if self.fluorophore_type == 'rsFP_negative_8states':
            # On-branch
            K[0][0,1] = 1 / self.lifetime_on
            K[0][2,1] = 1 / self.lifetime_on  * self.quantum_yield_on_to_off
            K[0][2,3] = 1 / self.lifetime_on
            K[0][0,3] = 1 / self.lifetime_on  * self.quantum_yield_trans_to_cis_anionic
            K[0][4,2] = 1 / self.protonation_time_on
            # Off-branch
            K[0][4,5] = 1 / self.lifetime_off
            K[0][6,5] = 1 / self.lifetime_off * self.quantum_yield_off_to_on
            K[0][6,7] = 1 / self.lifetime_off
            K[0][4,7] = 1 / self.lifetime_off * self.quantum_yield_cis_to_trans_neutral
            K[0][0,6] = 1 / self.deprotonation_time_off
            # Light driven
            for j in np.arange(nwavelengths):
                K[j+1][1,0] = self.cross_section_on[j]  # cis anionic
                K[j+1][3,2] = self.cross_section_on[j]  # trans anionic
                K[j+1][5,4] = self.cross_section_off[j]  # trans neutral
                K[j+1][7,6] = self.cross_section_off[j]  # cis neutral

        if self.fluorophore_type == 'rsFP_negative_9states_bleaching':
            # On-branch
            K[0][0,1] = 1 / self.lifetime_on
            K[0][2,1] = 1 / self.lifetime_on  * self.quantum_yield_on_to_off
            K[0][2,3] = 1 / self.lifetime_on
            K[0][0,3] = 1 / self.lifetime_on  * self.quantum_yield_trans_to_cis_anionic
            K[0][4,2] = 1 / self.protonation_time_on
            # Off-branch
            K[0][4,5] = 1 / self.lifetime_off
            K[0][6,5] = 1 / self.lifetime_off * self.quantum_yield_off_to_on
            K[0][6,7] = 1 / self.lifetime_off
            K[0][4,7] = 1 / self.lifetime_off * self.quantum_yield_cis_to_trans_neutral
            K[0][0,6] = 1 / self.deprotonation_time_off
            # Bleaching channels from teh on-branch
            K[0][8,1] = 1 / self.lifetime_on * self.quantum_yield_cis_anionic_bleaching
            K[0][8,3] = 1 / self.lifetime_on * self.quantum_yield_trans_anionic_bleaching
            # Light driven
            for j in np.arange(nwavelengths):
                K[j+1][1,0] = self.cross_section_on[j]  # cis anionic
                K[j+1][3,2] = self.cross_section_on[j]  # trans anionic
                K[j+1][5,4] = self.cross_section_off[j]  # trans neutral
                K[j+1][7,6] = self.cross_section_off[j]  # cis neutral

        return K

# rsEGFP2 Models
rsEGFP2_4states = NegativeSwitcher(extinction_coeff_on=[5260, 51560],
                                    extinction_coeff_off=[22000, 60],
                                    wavelength=[405, 488],
                                    lifetime_on=1.6e-9,
                                    quantum_yield_on_fluo=0.35,
                                    quantum_yield_on_to_off=1.65e-2,
                                    flurophore_type='rsFP_negative_4states',
                                    )

rsEGFP2_8states = NegativeSwitcher(extinction_coeff_on=  [  5260, 51560],
                                    extinction_coeff_off=[ 22000,    60],
                                    wavelength=          [   405,   488],
                                    lifetime_on=1.6e-9,
                                    lifetime_off=20e-12,
                                    quantum_yield_on_to_off=1.65e-2,
                                    quantum_yield_off_to_on=0.33,
                                    quantum_yield_on_fluo=0.35,
                                    starting_populations=[1,0,0,0,0,0,0,0],
                                    deprotonation_time_off=5.1e-6,
                                    protonation_time_on=48e-6,
                                    quantum_yield_trans_to_cis_anionic=0.0165,
                                    quantum_yield_cis_to_trans_neutral=0.33,
                                    flurophore_type='rsFP_negative_8states',
                                    )

rsEGFP2_9states = NegativeSwitcher(extinction_coeff_on= [  5260, 51560],
                                   extinction_coeff_off=[ 22000,    60],
                                   wavelength=          [   405,   488],
                                   lifetime_on=1.6e-9,
                                   lifetime_off=20e-12,
                                   quantum_yield_on_to_off=1.65e-2,
                                   quantum_yield_off_to_on=0.33,
                                   quantum_yield_on_fluo=0.35,
                                   starting_populations=[1,0,0,0,0,0,0,0,0],
                                   deprotonation_time_off=5.1e-6,
                                   protonation_time_on=48e-6,
                                   quantum_yield_trans_to_cis_anionic=0.0165,
                                   quantum_yield_cis_to_trans_neutral=0.33,
                                   flurophore_type='rsFP_negative_9states',
                                   )

## Generic switchers

class GenericSwitcher:
    def __init__(self,
                 extinction_coeff_anionic=[0, 0],
                 extinction_coeff_neutral=[0, 0],
                 wavelength=[405, 488],
                 lifetime_anionic=3e-9, # on state
                 lifetime_neutral=16e-12, # off state
                 quantum_yield_cis_to_trans_anionic=0.0165, # off switch quantum yield
                 quantum_yield_trans_to_cis_neutral=0.33, # on switch quantum yield
                 quantum_yield_trans_to_cis_anionic=0.33, # off switch reversed
                 quantum_yield_cis_to_trans_neutral=0.0165, # on switch reversed
                 deprotonation_time_cis_neutral = 5e-6,  # on switch
                 protonation_time_trans_anionic = 50e-6,  # off switch
                 quantum_yield_cis_anionic_fluo=1,
                 starting_populations=[1,0,0,0,0,0,0,0],
                 pka_cis=5.9,
                 pka_trans=10.0,
                 ph_buffer=7.5,
                 flurophore_type='rsfp_generic_8states'):
        # Cross section in cm2 of absorptions
        epsilon2sigma = 3.825e-21  # [Tkachenko2007, page 5]
        self.extinction_coeff_anionic = np.array(extinction_coeff_anionic)  # [M-1 cm-1]
        self.extinction_coeff_neutral = np.array(extinction_coeff_neutral)  # [M-1 cm-1]
        self.cross_section_anionic = self.extinction_coeff_anionic * epsilon2sigma  # [cm2]
        self.cross_section_neutral = self.extinction_coeff_neutral * epsilon2sigma  # [cm2]
        self.wavelength = np.array(wavelength)  # [nm]

        # Lifetime of the on excited state in seconds
        # Assumption: lifetime on and off are the same for the same protonation
        # state. This might not be the case, especially because cis_neutral
        # species is not fluorescent, so most likely it will have a shorter
        # lifetime. For small enough excitation kinetic rates, if we don't have
        # accumulation of exited state species, then it doesn't matter much.
        self.lifetime_anionic = lifetime_anionic  # [s]
        self.lifetime_neutral = lifetime_neutral  # [s]

        # Quantum yields and protonation/deprotonation
        self.quantum_yield_cis_anionic_fluo = quantum_yield_cis_anionic_fluo
        self.quantum_yield_cis_to_trans_anionic = quantum_yield_cis_to_trans_anionic  # cis_to_trans_anionic 
        self.quantum_yield_trans_to_cis_neutral = quantum_yield_trans_to_cis_neutral  # trans_to_cis_neutra
        self.quantum_yield_cis_to_trans_neutral = quantum_yield_cis_to_trans_neutral
        self.quantum_yield_trans_to_cis_anionic = quantum_yield_trans_to_cis_anionic
        self.protonation_time_trans_anionic = protonation_time_trans_anionic
        self.deprotonation_time_cis_neutral = deprotonation_time_cis_neutral

        # Adic-base properties
        self.pka_cis = pka_cis
        self.pka_trans = pka_trans
        self.ph_buffer = ph_buffer

        # Label describing the fluorophore type
        # Number of states in the kinetic model
        self.nspecies = np.size(starting_populations)
        self.fluorophore_type = flurophore_type

        # Index of the fluorescent state
        # Here, only one fluorescent state is assumed.
        # This is not necessary in general and could be extended in the future.
        self.quantum_yield_fluo = np.zeros((self.nspecies))
        self.quantum_yield_fluo[1] = self.quantum_yield_cis_anionic_fluo

        # Population at the beginning of the experiment
        self.starting_populations = starting_populations
        return None

    def kinetics_matrix(self):
        # Compute the kinetics matrix

        # Initialize arrays
        nwavelengths = self.wavelength.size
        K = np.zeros( (nwavelengths+1, self.nspecies, self.nspecies))

        # Put the kinetic constants connecting the right species
        # K[1,0] is the kinetic constant for the proces 1 <- 0.
        if self.fluorophore_type == 'rsFP_generic_8states':
            # On-branch
            K[0][0,1] = 1 / self.lifetime_anionic
            K[0][2,1] = 1 / self.lifetime_anionic  * self.quantum_yield_cis_to_trans_anionic
            K[0][2,3] = 1 / self.lifetime_anionic
            K[0][0,3] = 1 / self.lifetime_anionic  * self.quantum_yield_trans_to_cis_anionic
            K[0][4,2] = 1 / self.protonation_time_trans_anionic
            K[0][2,4] = 1 / self.protonation_time_trans_anionic * 10**(self.ph_buffer - self.pka_trans)
            # Off-branch
            K[0][4,5] = 1 / self.lifetime_neutral
            K[0][6,5] = 1 / self.lifetime_neutral * self.quantum_yield_trans_to_cis_neutral
            K[0][6,7] = 1 / self.lifetime_neutral
            K[0][4,7] = 1 / self.lifetime_neutral * self.quantum_yield_cis_to_trans_neutral
            K[0][0,6] = 1 / self.deprotonation_time_cis_neutral
            K[0][6,0] = 1 / self.deprotonation_time_cis_neutral * 10**(self.pka_cis - self.ph_buffer)
            # Light driven
            for j in np.arange(nwavelengths):
                K[j+1][1,0] = self.cross_section_anionic[j]  # cis anionic
                K[j+1][3,2] = self.cross_section_anionic[j]  # trans anionic
                K[j+1][5,4] = self.cross_section_neutral[j]  # trans neutral
                K[j+1][7,6] = self.cross_section_neutral[j]  # cis neutral
        return K

rsEGFP2_8states_pka = GenericSwitcher(extinction_coeff_anionic=[  5260, 51560],
                                      extinction_coeff_neutral=[ 22000,    60],
                                      wavelength=              [   405,   488],
                                      lifetime_anionic=1.6e-9,
                                      lifetime_neutral=20e-12,
                                      quantum_yield_cis_to_trans_anionic=0.0165, # off switch quantum yield
                                      quantum_yield_trans_to_cis_neutral=0.33, # on switch quantum yield
                                      quantum_yield_trans_to_cis_anionic=0.0165, # off switch reversed
                                      quantum_yield_cis_to_trans_neutral=0.33, # on switch reversed
                                      quantum_yield_cis_anionic_fluo=0.35,
                                      starting_populations=[1,0,0,0,0,0,0,0],
                                      deprotonation_time_cis_neutral=5.1e-6,
                                      protonation_time_trans_anionic=48e-6,
                                      flurophore_type='rsFP_generic_8states',
                                      pka_cis=5.9,
                                      pka_trans=10.0,
                                      ph_buffer=7.5,
                                      )



################################################################################
# STED dye classes
################################################################################

class STEDDye:
    def __init__(self,
                 extinction_coeff_exc=[0, 0],
                 extinction_coeff_sted=[0, 0],
                 wavelength=[640, 775],
                 lifetime=3e-9,
                 quantum_yield_fluo=1,
                 starting_populations=[1,0],
                 nspecies=2):
        self.fluorophore_type = 'sted_dye'
        # Cross section in cm2 of absorptions
        epsilon2sigma = 3.825e-21  # [Tkachenko2007, page 5]
        self.extinction_coeff_exc = np.array(extinction_coeff_exc)  # [M-1 cm-1]
        self.extinction_coeff_sted = np.array(extinction_coeff_sted)  # [M-1 cm-1]
        self.cross_section_exc = self.extinction_coeff_exc * epsilon2sigma  # [cm2]
        self.cross_section_sted = self.extinction_coeff_sted * epsilon2sigma  # [cm2]
        self.wavelength = np.array(wavelength)  # [nm]

        # Lifetime of the on excited state in seconds
        self.lifetime = lifetime  # [s]

        self.quantum_yield_fluo = quantum_yield_fluo

        # Label describing the fluorophore type
        # Number of states in the kinetic model
        self.nspecies = nspecies

        # Quantum yield of fluorescence
        # Here, only one fluorescent state is assumed.
        # This is not necessary in general and could be extended in the future.
        self.quantum_yield_fluo = np.zeros((nspecies))
        self.quantum_yield_fluo[1] = quantum_yield_fluo

        # Population at the beginning of the experiment
        self.starting_populations = starting_populations

    def kinetics_matrix(self):
        # Compose the kinetic matrix, k.
        # Light driven transition are expressed as corss sections.
        # Transition induced by a certain wavelength are groupped
        # in separate layers of k.
        # k has three dimensions: [nwavelenght+1, nspecies, nspecies].
        # k[0] is reserved for all non-light-induced transitions.

        # Initialize arrays
        nwavelengths = self.wavelength.size
        K = np.zeros( (nwavelengths+1, self.nspecies, self.nspecies))

        # Put the kinetic constants connecting the right species
        # K[1,0] is the kinetic constant for the proces 1 <- 0.
        K[0][0,1] = 1 / self.lifetime
        for i in np.arange(nwavelengths):
            K[i+1][1,0] = self.cross_section_exc[i]
            K[i+1][0,1] = self.cross_section_sted[i]
        return K

# atto647N
# parameterization from: J. Oracz, V. Westphal, C. Radzewicz, S. J. Sahl, and S. W. Hell, Scientific Reports 7, 1 (2017)
# https://doi.org/10.1021/acs.nanolett.7b00468

atto647N = STEDDye(extinction_coeff_exc=[25950, 0],
                   extinction_coeff_sted=[0, 1250],
                   wavelength=[640, 775],
                   quantum_yield_fluo=0.65,
                   lifetime=3.5e-9,
                   )
