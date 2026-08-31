# The problem, once

This chapter describes the model behind `rotational-diffusion-photophysics`: polarized
fluorescence experiments on photo-reactive systems in solution, and in particular the
signals produced by reversibly switchable fluorescent proteins (rsFPs). The physics is
one master equation --- rotational diffusion coupled to orientation-dependent
photochemistry --- and the whole method consists of choosing a basis in which that
equation becomes a matrix exponential.

The configuration of a molecule is a rotation $R \in SO(3)$. Everything a one-photon
polarized experiment can see, however, couples only to the laboratory direction of a
transition dipole, $\hat{\mathbf{u}} = R \mathbf{d}$, where $\mathbf{d}$ is that
dipole's direction in the molecular frame. That observation splits the problem in two:

- if **every species shares the same** $\mathbf{d}$, the dipole axis
  $\hat{\mathbf{u}} \in S^2$ is a faithful coordinate, and the orientational density can
  be expanded in spherical harmonics --- the cheap representation of chapter 2;
- if species differ in $\mathbf{d}$ --- as they do when photoswitching isomerizes the
  chromophore --- a rotation of the molecule changes *which* dipole points where, the
  reduction to $S^2$ is no longer available, and the density must be carried on the full
  rotation group and expanded in Wigner-$D$ functions.

The present chapter states the problem once, in a form that does not depend on that
choice; chapter 2 develops the $S^2$ representation. The structure mirrors the code,
which separates the basis-agnostic propagator (`core.py`) from the two engines
(`engine_s2.py`, `engine_so3.py`).

We use a rigid isotropic model for rotational diffusion, and assume the diffusive
properties are shared by all species in the photophysical scheme. We seek an analytical
solution, suitable for fitting experimental data.

## Master equation for diffusion and orientation-dependent kinetics

The conditional probability of a species $\epsilon$, starting with orientation $(\theta_0, \phi_0)$ at time $t=0$ from any possible species, and being at orientations $(\theta, \phi)$ at time $t$, is specified as $p_\epsilon(t,\theta,\phi|t=0,\theta_0, \phi_0)$, or for shortness $p_\epsilon(t,\theta,\phi)$. Definitions of the angles are illustrated in the sketch in figure [1](#fig:angles){reference-type="ref" reference="fig:angles"}.
Following the theory developed in [@Fisz1996a; @Fisz1996b], the time evolution of the orientational conditional probabilities can be specified using a set of differential equations as
\begin{equation}
\label{eq:time_propagation}
    \frac{\partial p_\epsilon(t,\theta,\phi)}{\partial t} =
    D_R \nabla^2 p_\epsilon(t,\theta,\phi)
    + \sum_\eta k_{\epsilon\eta} (\theta,\phi) p_\eta(t,\theta,\phi),
\end{equation}
where $D_R$ is the rotational diffusion coefficient for a rigid isotropic system, $\nabla^2$ is the square gradient in the angular space, defined as
\begin{equation}
    \nabla^2 p_\epsilon(t,\theta,\phi) =
    \frac{1}{\sin(\theta)} \frac{\partial}{\partial \theta} \left(
        \sin(\theta)
        \frac{\partial p_\epsilon(t,\theta,\phi) }{\partial \theta} \right)
    + \frac{1}{\sin^2(\theta)}
    \frac{\partial^2 p_\epsilon(t,\theta,\phi) }{\partial^2 \phi},
\end{equation}
and $k_{\epsilon\eta}$ is the kinetic constant for the reaction that transform the $\eta$-th species into the $\epsilon$-th species when $\eta \neq \epsilon$, i.e. $\eta \rightarrow \epsilon$. The kinetic constant $k_{\epsilon\epsilon}$ where both indexes are the same is different than the others. It refers to the depopulation of state $\epsilon$, and it is computed summing all the possible rates that can remove population from that state
\begin{equation}
\label{eq:k_diagonal}
    k_{\epsilon\epsilon}(\theta,\phi) = 
    - \sum_{\eta \neq  \epsilon} k_{\eta\epsilon}(\theta,\phi).
\end{equation}
In our problem, kinetic constants might depend on the orientation of the fluorophore, in particular this happens if the transition is induced by light-matter interaction. Moreover, in eq. \eqref{eq:time_propagation}, we can recognize two main terms: the first term models the free isotropic rotational diffusion of the system, while the second term models the kinetics and photo-physics of the system.

\begin{figure}[h!]
    \centering
    \includegraphics[width=0.5\textwidth]{figures/angles_theta_phi.png}
    \caption[Dipole orientation and laboratory frame]
    {Illustration of the transition dipole moment orientation and the laboratory fixed
    three dimensional reference frame.}
    \label{fig:angles}
\end{figure}

## Time propagation and pulse windows

The SH expansions in eq. \eqref{eq:sh_expansion} and the kinetic term in eq. \eqref{eq:coupling_simplified} can be directly plugged in eq. \eqref{eq:time_propagation} and a new set of differential equation for the full problem is is retrieved,
\begin{equation}
\label{eq:time_evoution_coeffs}
        \frac{\partial p^\epsilon_{lm}(t)}{\partial t} =
        -D_R \ l(l+1) \ p^\epsilon_{lm}(t)
        +\sum_{l_2m_2} 
        k^{\epsilon\eta}_{lm,l_2m_2} p^{\eta}_{l_2m_2}(t),
\end{equation}
that is derived analogously to eq. \eqref{eq:diffusion_expansion_coeffs}, reducing the full set of differential equations to differential equations only on the SH coefficients of the orientational probabilities.

We can collect all the rotational diffusion rates on the diagonal of matrix $\mathbf{D}$, indexed in the same way as matrix $\mathbf{K}$, such that $D_{ii} = -D_R l(l+1)$. Analogously, we can collect all the SH probability coefficients $p^\epsilon_{lm}$ in a vector $\mathbf{p}$ sharing the same indexing of matrices $\mathbf{D}$ and $\mathbf{K}$. In this way we can simplify the notation rewriting eq. \eqref{eq:time_evoution_coeffs} as
\begin{equation}
\begin{aligned}
    \frac{d \mathbf{p}(t) }{dt} &=
    \mathbf{D}\mathbf{p} + \mathbf{K}\mathbf{p} \\
    \frac{d \mathbf{p}(t)}{dt} &=
    \mathbf{M}\mathbf{p},
\end{aligned}
\end{equation}
where the matrix $\mathbf{M} = \mathbf{D} + \mathbf{K}$ collects all the rotational diffusion rates and kinetic rates. Finally, we can write the solution of this equation, in analogy to eq. \eqref{eq:diffusion_solution}, as
\begin{equation}
\label{eq:full_solution}
    \mathbf{p}(t) = e^{\mathbf{M}t} \mathbf{p}_0,
\end{equation}
where $\mathbf{p}_0$ is a vector that collects all the coefficients $p^\epsilon_{lm,0}$ defining the starting conditions of the system. Note that the matrix exponential can be conveniently computed by diagonalizing matrix $\mathbf{M}$, obtaining
\begin{equation}
    e^{\mathbf{M}t} = \mathbf{U} e^{\mathbf{\Lambda} t} \mathbf{U}^{-1}, 
    \qquad \text{with} \quad
    \mathbf{M} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^{-1},
\end{equation}
where the exponential of the diagonal matrix $\mathbf{\Lambda}t$ is again a diagonal matrix with elements $e^{\Lambda_{ii}t}$, $\mathbf{U}$ is the eigenvectors matrix, and $\mathbf{\Lambda}$ is the diagonal eigenvalues matrix. From an implementation point of view, solving the full problem only requires to construct a suitable matrix $\mathbf{M}$ based on the properties of the system of interestr, and generate a suitable starting condition for the experiment, from which we will compute $\mathbf{p}_0$. The orientational probability of the fluorophores at any time delays is then promptly computed with eq. \eqref{eq:full_solution}.

The simulation of light-modulated experiments, in which laser powers are controlled with square waves, can be seen as a simple extension of the solution here proposed. The way to approach this issue it to divide the experiment in a series of time windows in which the laser powers are constant. Then for each time window a rate matrix $\mathbf{M}$ is constructed and used to propagate the probabilities. The first time window will propagate the initial conditions of the experiment, whereas the following time windows will propagate the ending conditions of the previous time window.

## Initial conditions

Propagation needs a starting vector $\mathbf{p}_0$, and the correct choice is not
"isotropic" but **the equilibrium of the diffusion operator**. For free rotational
diffusion the two coincide: the equilibrium is the uniform distribution, only the
$l=0$ coefficient of each species is populated, and it carries that species' starting
population. For hindered diffusion in an ordering potential they do not: the
equilibrium $c_\text{eq}$ is anisotropic, and starting from a uniform distribution
would simulate a physical process that is not happening --- a relaxation from isotropy
into the ordered equilibrium, superimposed on the photophysics.

The engines therefore ask the diffusion model for its equilibrium when it defines one
and start there, so that detailed balance holds and the dark dynamics are stationary
before the first laser pulse. The equilibrium coefficients are normalized so that the
$l=0$ (respectively $l=m=n=0$) coefficient is unity, which keeps the population
interpretation of the leading coefficient described below.

## Observables and the normalization contract

Every observable in these experiments is an overlap between the orientational density of
an emitting state and an angular collection function --- the detector's efficiency as a
function of molecular orientation. Written as an integral, the signal of a detector
whose collection function is $w(\theta,\phi)$ is

\begin{equation}\label{eq:observable}
    s(t) = \Phi_\text{fluo}^\epsilon
    \int w(\theta,\phi)\, p_\epsilon(t,\theta,\phi) \, \sin\theta \, d\theta \, d\phi ,
\end{equation}

and in an orthogonal basis it is simply an inner product of expansion coefficients.
That last step hides a normalization convention, and because the convention decides
whether simulated signals are absolute numbers or arbitrary units, we state it
explicitly.

**The normalization contract.** Angular functions are expanded in
"$4\pi$"-normalized real spherical harmonics, for which $Y_{00} = 1$ and
$\int f g \, d\Omega = 4\pi \sum_{lm} f_{lm} g_{lm}$. Orientational densities are
normalized so that

\begin{equation}\label{eq:normalization}
    p^{\epsilon}_{00}(t) = \text{population of species } \epsilon \text{ at time } t ,
\end{equation}

i.e. the leading coefficient *is* the population, and the density is a population per
unit solid angle divided by $4\pi$. With that convention the $4\pi$ of the inner
product cancels the $1/4\pi$ of the density, and the signal is the bare coefficient sum

\begin{equation}\label{eq:observable_coeffs}
    s(t) = \Phi_\text{fluo}^\epsilon \sum_{lm} w_{lm}\, p^{\epsilon}_{lm}(t) ,
\end{equation}

with no residual geometric factor. The same contract holds in the Wigner-$D$
representation, where $p^{\epsilon}_{000}$ is the population and the $8\pi^2$ of the
$SO(3)$ inner product plays the role of the $4\pi$.

The contract has a simple, checkable consequence. For an **isotropic** population $P$ of
a state with fluorescence quantum yield $\Phi$, any linear detector collects

\begin{equation}\label{eq:isotropic_signal}
    s = \tfrac{1}{3}\, \Phi \, P ,
\end{equation}

the $\cos^2$ average of a dipole over the sphere --- independent of the numerical
aperture, because the high-NA mixing coefficients of section 2.6 sum to one, and
independent of the molecular-frame dipole direction, because the ensemble is isotropic.
Both engines reproduce equation \eqref{eq:isotropic_signal} exactly and agree with each
other to machine precision, which is what makes the two representations quantitatively
interchangeable rather than merely analogous.

# Representation I --- the dipole axis on $S^2$

## When the dipole axis is enough

The representation developed in this chapter rests on one condition, which we state
plainly because it is a condition and not a law of nature:

> **All species in the kinetic scheme share the same transition-dipole direction in the
> molecular frame.**

When it holds, the lab direction $\hat{\mathbf{u}} = R\mathbf{d}$ of that single dipole
determines every rate and every observable in the experiment. The remaining degree of
freedom --- rotation of the molecule about its own dipole axis --- is invisible, so the
configuration space collapses from $SO(3)$ to the sphere $S^2$, and it suffices to
propagate a density $p_\epsilon(t,\theta,\phi)$ of dipole orientations. The saving is
large: the basis has $(l_\text{max}+1)^2$ functions instead of
$\sum_l (2l+1)^2$ --- 49 against 455 at $l_\text{max} = 6$ --- and since the cost of a
simulation is dominated by an eigendecomposition, that is roughly three orders of
magnitude.

The condition fails exactly where the photochemistry is most interesting. In green
negative rsFPs the on--off switching *is* a cis--trans isomerization of the chromophore,
which swings the transition dipole by tens of degrees; the same molecular orientation
therefore corresponds to different laboratory dipole directions before and after
switching, and no density on $S^2$ can represent that. Anisotropy losses associated with
the angle between absorption and emission dipoles have the same origin. Such systems
require the Wigner-$D$ representation on the full rotation group, where each state
carries its own molecular-frame dipole.

The two representations are not alternatives with different accuracies: the $S^2$ engine
is the exact restriction of the $SO(3)$ engine to the case of a shared dipole, and the
two produce identical detector signals in that limit. Where the dipole does reorient the
difference is quantitative and not small --- simulating STARSS method 1 for rsEGFP2 on
$SO(3)$ rather than $S^2$ changes the detector signals by about 14%.

## Spherical harmonics expansion

All the angular functions can be expanded using a spherical harmonics (SH) basis. This approach has several advantages because SH are well behaving functions and they can simplify many aspect of the solution of the kinetic-diffusive problem. In particular the SH basis set is orthogonal and complete, and there are simple expressions to calculate products of SH functions [@CourantHilbertV1; @Wieczorek2018].
Orientational probabilities and kinetic constants can be expanded as
\begin{equation}
\label{eq:sh_expansion}
\begin{aligned}
    p_\epsilon(t,\theta,\phi) &= \sum_{l=0}^{\infty} \sum_{m=-l}^{+l} p^{\epsilon}_{lm}(t) Y_{lm}(\theta,\phi), \\
    k_{\epsilon\eta}(\theta,\phi) &= \sum_{l=0}^{\infty} \sum_{m=-l}^{+l} k^{\epsilon\eta}_{lm} Y_{lm}(\theta,\phi)
\end{aligned}
\end{equation}
where $Y_{lm}(\theta,\phi)$ are the spherical harmonics functions of order $l$ and $m$, $p^{\epsilon}_{lm}(t)$ are the SH coefficients for the expansion of the orientational probability of state $\epsilon$, and $k^{\epsilon\eta}_{lm}$ are the coefficients for the expansion of the kinetic constants for the $\eta \rightarrow \epsilon$ reaction.
The expansions in eq. \eqref{eq:sh_expansion} separate the angular dependence that is fully embedded in the SH basis set, from the temporal dependence in the coefficients that weight the SH basis set.
In general, if a function contains sharp angular shapes higher $l$ values will be needed to fully map its content on the SH basis set. There is an analogy with the expansion in Fourier series, in which function with higher frequency content will require higher order sinusoidal functions to be expanded completely. In this sense SH are mapping higher angular frequency content for higher $l$ values. Since the SH basis set is complete, any orientational probability can be expanded. Unconventional orientational probabilities with very sharp shapes could be achieved for example through saturation of photo-physics transitions.

Linear light-matter interaction introduces a substantial simplification on the expansion of kinetic constants of light driven processes, e.g. the transition from ground state to first excited state of a fluorophore. For all linear processes the probability of transition is proportional to the square of the scalar product of the electric field vector of radiation and the transition dipole moment of the fluorophore [@LakowiczBook]. This dependence directly implies that only $l=0$ and $l=2$ are involved in the expasion in eq. \eqref{eq:sh_expansion}. Moreover, if a kinetic constant does not depend on molecular orientation the expansion collapse just on the $l=0$ and $m=0$ term. Thus we can get these simplified expressions
\begin{equation}
\begin{aligned}
    k_{\epsilon\eta}(\theta,\phi) &= \sum_{l=0,2} \sum_{m=-l}^{+l} k^{\epsilon\eta}_{lm} Y_{lm}(\theta,\phi), \qquad &\text{(light-induced)} \\
    k_{\epsilon\eta}(\theta,\phi) &= k_{00}^{\epsilon\eta} \ Y_{00}(\theta,\phi) = k_{\epsilon\eta}. \qquad &\text{(orientation-independent)}
\end{aligned}
\end{equation}
Note that when using "$4\pi$" normalized SH functions $Y_{00}(\phi,\theta)=1$, and $k_{00}^{\epsilon\eta}=k_{\epsilon\eta}$.
This simplification greatly reduces the computation needed when solving eq. \eqref{eq:time_propagation}. If higher order light-matter interaction are present analogous simplifications are possible, e.g. two photon absorption only involve SH terms with $l=0,2,4$.

## Solution of rigid isotropic rotational diffusion

Spherical harmonics expansion is convenient in the context of finding a solution for rotational diffusion problems [@Fisz1996b], and in particular for isotropic rotational diffusion, i.e. the rotational diffusion of a rigid sphere without preferential axis of rotation. Spherical harmonics are eigen-functions of the rigid rotational diffusion operator, thus if we rewrite the free rotational diffusion problem in eq. \eqref{eq:time_propagation} with the expansion in eq. \eqref{eq:sh_expansion}, excluding for now the kinetics induced by photo-physics, we get
\begin{equation}
\label{eq:diffusion_expansion}
\begin{aligned}
    \frac{\partial p_\epsilon(t,\theta,\phi)}{\partial t} 
    &=
    D_R \nabla^2 p_\epsilon(t,\theta,\phi) \\
    \sum_{lm}
    \frac{\partial p^\epsilon_{lm}(t)}{\partial t} Y_{lm}(\theta,\phi) 
    &=
    \sum_{lm}
        -D_R \ l(l+1) \ p^\epsilon_{lm}(t) Y_{lm}(\theta,\phi).
\end{aligned}
\end{equation}
It can be shown that to solve this set of differential equations it is only necessary to solve the differential equations on the expansion coefficients [@Fisz1996a]. Thus we can rewrite a new set of differential equations
\begin{equation}
\label{eq:diffusion_expansion_coeffs}
    \frac{d p^\epsilon_{lm}(t)}{d t} =
    -D_R \ l(l+1) \ p^\epsilon_{lm}(t).
\end{equation}
From this set of equations we can compute the solution for the SH probability coefficients, obtaining an exponential decay of each $p^\epsilon_{lm}(t)$ coefficient starting from the initial conditions $p^\epsilon_{lm,0}=p^\epsilon_{lm}(t=0)$,
\begin{equation}
\label{eq:diffusion_solution}
    p^\epsilon_{lm}(t) =
    p^\epsilon_{lm,0} \ e^{-t/\tau_R},
\end{equation}
where the relaxation times of the diffusion terms are specified as $\tau_R = 1/D_R \ l(l+1)$. A consequence coming from this solution, and the fact that linear light-matter interaction involves only SH terms with $l=0,2$, is that free rotational diffusion of an out of equilibrium orientational state prepared in linear regime exhibits a decay time of $\tau_R = 1/D_R2(2+1) = 1/6D_R$ and a stationary component.

## High-NA photo-selection

The angular dependence of $k_{\epsilon\eta}$ is the way we can create a photo-selected state in which the excited molecules have a preferential orientation. If we shine light polarized along the $\hat{\mathbf{x}}$ direction of the laboratory fixed coordinate system, we can write the following expression for the kinetic constant in the low-NA limit [@Fisz2005]
\begin{equation}
\label{eq:k_low_na}
\begin{aligned}
    k_{\epsilon\eta}(\theta,\phi)
    &= 
    | \hat{\mathbf{r}}(\theta,\phi) \cdot \hat{\mathbf{x}} |^2
    \sigma_{\epsilon\eta} \rho \\
    &=
    F_x \sigma_{\epsilon\eta} \rho
\end{aligned}
\end{equation}
where $\hat{\mathbf{r}}$ is a unitary vector pointing in the direction of the absorption transition dipole moment, $\hat{\mathbf{x}}$ is the unitary vector pointing in the direction of the electric field of the electromagnetic radiation, $F_x$ is the function encoding the pure angular dependence of the kinetic constant trough the scalar product of the two versors, $\sigma_{\epsilon\eta}$ is the cross-section for the transition $\eta\rightarrow\epsilon$ expressed in cm$^2$, and $\rho$ is the photon flux expressed in cm$^{-2}$s$^{-1}$. Cross-section and photon flux can be also expressed in terms of other widely used quantities as [@TkachenkoBook]
\begin{equation}
    \sigma_{\epsilon\eta} = \frac{2303}{N_A} \varepsilon,
    \qquad \text{and} \quad
    \rho = I \frac{\lambda}{h c},
\end{equation}
where $N_A$ is the number of Avogadro, $\varepsilon$ is molar extinction coefficient expressed in M$^{-1}$cm$^{-1}$, $I$ is the power density of the electromagnetic radiation expressed in W/cm$^{2}$, $\lambda$ is the wavelength, $h$ is the Planck constant, and $c$ is the speed of light.

When using high-NA objectives, eq. \eqref{eq:k_low_na} must be corrected taking in consideration the polarization mixing introduced by the focusing of the light [@Fisz2005]. The correction is obtained by integrating all the contribution to the excitation coming from the rays inside the focusing cone of the objective. The angle of the cone of the rays, $\alpha_0$, is obtained from the numerical aperture NA and the refractive index of the medium $n$, i.e. through the equation NA$= n\sin\alpha_0$. The corrected kinetic constant is expressed as
\begin{equation}
\label{eq:high_na_k_pre}
    k_{\epsilon\eta}(\theta,\phi) =
    \left(K_a F_z + K_b F_y + K_c F_x\right)
    \sigma_{\epsilon\eta} \rho
\end{equation}
where we assumed $\hat{\mathbf{z}}$ as the propagation direction, $\hat{\mathbf{y}}$ as the perpendicular direction, and where $K_a$, $K_b$, and $K_c$ are the normalized Axelrod coefficients [@Fisz2005], specified as
\begin{equation}
\label{eq:high_na_k}
\begin{aligned}
    K_a &= 1/6 \ (2 - 3\cos\alpha_0 +\cos^3\alpha_0)/(1 - \cos\alpha_0)\\
    K_b &= 1/24 \ (1 - 3\cos\alpha_0 + 3\cos^2\alpha_0 -\cos^3\alpha_0)/(1 - \cos\alpha_0) \\
    K_c &= 1/8 \ (5 - 3\cos\alpha_0 - \cos^2\alpha_0 -\cos^3\alpha_0)/(1 - \cos\alpha_0). \\
\end{aligned}
\end{equation}
The correction coefficients are normalized such that $K_a + K_b + K_c = 1$, and in the low-NA limits they tend to $K_c=1$ and $K_a=K_b=0$.

## The product operator: coupling diffusion to the kinetics

The coupling between rotational diffusion and photochemistry sits in the product
"rate $\times$ population" of equation \eqref{eq:time_propagation}. Both factors are
angular functions, so in the coefficient representation the kinetic term is the linear
operator *"multiply the density by the rate function"*:

\begin{equation}\label{eq:coupling}
    k_{\epsilon\eta}(\theta,\phi)\, p_\eta(t,\theta,\phi) =
    \sum_{lm} \left[ \sum_{l_2m_2}
    k^{\epsilon\eta}_{lm,l_2m_2}\, p^{\eta}_{l_2m_2}(t) \right] Y_{lm}(\theta,\phi) .
\end{equation}

The coefficient $k^{\epsilon\eta}_{lm,l_2m_2}$ is the rate connecting the $l_2m_2$
coefficient of the reagent $\eta$ to the $lm$ coefficient of the product $\epsilon$;
collected over all species and coefficients these form the kinetics matrix
$\mathbf{K}$ of the propagation above, whose blocks $\mathbf{K}^{\epsilon\eta}$ carry
one transition each.
Building them requires the expansion coefficients of a **product of two basis
functions**,

\begin{equation}\label{eq:triple_product}
    k^{\epsilon\eta}_{lm,l_2m_2} = \sum_{l_1m_1} k^{\epsilon\eta}_{l_1m_1}
    \, G^{lm}_{l_1m_1,\,l_2m_2} ,
    \qquad
    G^{lm}_{l_1m_1,\,l_2m_2} = \frac{1}{4\pi}\int
    Y_{l_1m_1} Y_{l_2m_2} Y_{lm} \, d\Omega ,
\end{equation}

with $G$ the triple-product (Gaunt) table of the basis.

**The real-harmonic product has two channels.** For *complex* spherical harmonics the
product $Y_{l_1m_1}Y_{l_2m_2}$ has a single azimuthal output, $m = m_1+m_2$, and $G$ is
given in closed form by Wigner-$3j$ symbols. The engine, however, expands in **real**
spherical harmonics --- the natural choice here, because the densities and rates are
real functions and the resulting matrix $\mathbf{M}$ is real. Real harmonics are
$\cos m\phi$ and $\sin m\phi$ combinations, and the product of two of them produces
**two** azimuthal channels, $m_1+m_2$ *and* $|m_1-m_2|$. The elementary identity

\begin{equation}\label{eq:two_channels}
    \cos 2\phi \, \cos 2\phi = \tfrac{1}{2} + \tfrac{1}{2}\cos 4\phi
\end{equation}

shows both: an $m=2$ function multiplied by an $m=2$ function has weight at $m=4$ *and*
at $m=0$. The complex-basis $3j$ rule, applied verbatim to real harmonics, keeps only
the first and silently discards the second. It is therefore **not** the correct
prescription in this basis, and using it is not a harmless approximation: it removes
population from the $m$-difference channel, which is precisely the channel that a
linearly polarized beam along $x$ or $y$ feeds as soon as the density carries $m \neq 0$
content --- under saturation, or after a sequence of pulses. The observable symptom is
that the total excited population stops being invariant under rotation of the laboratory
frame, a quantity that cannot physically depend on the polarization direction. That
invariance is now a regression test of the engine.

**How the table is actually built.** Rather than a closed form, the engine computes $G$
by exact quadrature on the sphere, which is convention-consistent with the rest of the
code by construction. Each basis function is synthesized on a Driscoll--Healy grid,
pairs are multiplied pointwise, and the product is analysed back into coefficients:

\begin{equation}\label{eq:quadrature}
    G^{lm}_{l_1m_1,\,l_2m_2} =
    \mathcal{A}_{lm}\Big[\, \mathcal{S}[Y_{l_1m_1}] \cdot \mathcal{S}[Y_{l_2m_2}] \,\Big] ,
\end{equation}

where $\mathcal{S}$ denotes synthesis onto the grid and $\mathcal{A}$ analysis back to
coefficients. The grid is chosen with $l_\text{max}^\text{grid} = 2 l_\text{max}+2$ so
that products of degree up to $l_\text{max}+2$ are resolved without aliasing, which
makes the table exact to numerical precision for the harmonics in play. Only multiplier
indices with $l_1 \in \{0,2\}$ are needed and filled, since one-photon photoselection
functions have degree at most two (section 2.4).

Two limits are worth recording. If a rate does not depend on orientation, only
$l_1 = m_1 = 0$ survives and equation \eqref{eq:triple_product} collapses to

\begin{equation}\label{eq:coupling_simplified}
    k^{\epsilon\eta}_{lm,l_2m_2} = k_{\epsilon\eta}\, \delta_{l l_2}\delta_{m m_2} ,
    \qquad\text{i.e.}\qquad
    \mathbf{K}^{\epsilon\eta} = \mathbf{1}\, k_{\epsilon\eta} ,
\end{equation}

so orientation-independent processes --- spontaneous decay, thermal relaxation,
protonation --- contribute diagonal blocks and do not mix coefficients. And in a
*complex* basis the $3j$ route is the right one: that is exactly what the Wigner-$D$
engine uses, where the product of two basis functions has a single output channel and
the coupling coefficients are Clebsch--Gordan products.

## High-NA polarized detection

The fluorescence signal emitted by the sample and recorded from a polarized detector is derived with a similar formalism to the high-NA photo-selection correction [@Fisz2005]. For example, if we are observing with two cross polarized detectors, measuring the polarization along $\hat{\mathbf{x}}$ and $\hat{\mathbf{y}}$, the fluorescence signals are computed as
\begin{equation}
\label{eq:highna_detection}
\begin{aligned}
    I_x(t) &= A \ \Phi_\text{fluo}^\epsilon
    \int\int 
    p_\epsilon(t,\theta,\phi)
    \left(K_a F_z + K_b F_y + K_c F_x\right)
    \sin\theta d\theta d\phi \\
    I_y(t) &= A \ \Phi_\text{fluo}^\epsilon
    \int\int 
    p_\epsilon(t,\theta,\phi)
    \left(K_a F_z + K_b F_x + K_c F_y\right)
    \sin\theta d\theta d\phi \\
\end{aligned}
\end{equation}
where $p_\epsilon(t,\theta,\phi)$ is the orientational probability of the fluorescent excited state, $A$ is a constant embedding all the experimental factors and the size of the sample, and $\Phi_\text{fluo}^\epsilon$ is the quantum yield of fluorescence of state $\epsilon$. Note that the constants $K_a$, $K_b$, and $K_c$ are related again to propagation direction, perpendicular direction and polarization direction respectively, as in eq. \eqref{eq:high_na_k}. We are also assuming that there is only one fluorescent state $\epsilon$. These expressions can be easily generalized for multiple fluorescent states. The integral in equations [\[eq:highna_detection\]](#eq:highna_detection){reference-type="ref" reference="eq:highna_detection"} can be easily computed exploiting the SH expansion of $p_\epsilon(t,\theta,\phi)$.

## The even-$l$ reduction

The eigendecomposition of $\mathbf{M}$ dominates the cost of a simulation, and its
dimension can be halved for free. One-photon photoselection functions contain only even
ranks, $l_1 \in \{0,2\}$; the triple product of section 2.5 therefore couples $l_2$ to
outputs of the same parity, and the initial condition of a freely diffusing sample lives
entirely at $l=0$. Odd-$l$ coefficients are consequently never populated, and the engine restricts
the eigenproblem to the even-$l$ subspace, reducing the basis from
$(l_\text{max}+1)^2$ to about half that.

The reduction is **exact, and exact at any power**. This deserves emphasis, because the
$l \in \{0,2\}$ statement is usually justified by the linearity of light--matter
interaction, which invites the reading that saturation would spoil it. It does not:
parity is preserved order by order in the expansion of the matrix exponential
$e^{\mathbf{M}t}$, so no amount of driving can populate an odd-$l$ coefficient from an
even-$l$ start. The claim is verified adversarially in the test suite --- deep into
saturation and with the non-axial polarizations where a parity leak would appear first
--- by checking that the odd-$l$ block of the *unreduced* solution stays at the level of
numerical noise, and that reducing the eigenproblem changes nothing.

The same argument applies to the lab index in the Wigner-$D$ representation and gives a
further reduction there; it does *not* apply to the molecular-frame index, which a
tilted dipole genuinely populates.

# Representation II --- the full orientation on $SO(3)$

## When the dipole axis is not enough

The $S^2$ representation rests on all species sharing one molecular-frame dipole. Green
negative rsFPs violate that condition by construction: the on--off switching *is* a
cis--trans isomerization of the chromophore, and the transition dipole of the trans
species points in a different molecular direction than that of the cis species. In the
parametrization used here the cis-anionic (on) dipole sits about $35^\circ$ from the
trans dipole, with the cis-neutral intermediate a further $13^\circ$ away.

The consequence is not a small correction to a mostly-correct picture; it is that the
state variable of the $S^2$ model no longer exists. A density over dipole directions
cannot answer the question "if this molecule switches, where will its new dipole point?",
because the answer depends on the molecular orientation about the old dipole axis --- the
one coordinate $S^2$ discards. What must be propagated instead is the orientation of the
molecule itself: the full rotation $R \in SO(3)$ that carries the molecular frame into
the laboratory frame, informally the orientation of the protein *barrel*. Each state then
carries its own body-frame dipole $\mathbf{d}_\epsilon$, and the laboratory dipole of a
molecule in state $\epsilon$ is $R\mathbf{d}_\epsilon$. Switching changes which dipole is
active without moving the molecule.

## The Wigner-$D$ basis

Functions on $SO(3)$ are expanded in Wigner-$D$ functions, the group's analogue of
spherical harmonics,

\begin{equation}\label{eq:wignerd_expansion}
    p_\epsilon(t,R) = \sum_{l=0}^{\infty} \sum_{m=-l}^{+l} \sum_{n=-l}^{+l}
    p^{\epsilon}_{lmn}(t) \, D^{l}_{mn}(R) ,
\end{equation}

which are orthogonal and complete on the group. Each rank $l$ now carries **two**
indices: $m$ transforms under rotations of the laboratory frame and $n$ under rotations
of the molecular frame. That second index is what makes a per-state dipole
representable --- it is the room in which the molecular geometry lives --- and it is also
the entire cost of the representation, since rank $l$ contributes $(2l+1)^2$ functions
instead of $2l+1$.

Two conventions carry over from chapter 1. The population normalization is the same,
$p^{\epsilon}_{000}$ is the population of species $\epsilon$, which is what makes signals
from the two engines directly comparable. The basis, however, is **complex**, and a real
density is not free to have arbitrary coefficients: reality imposes

\begin{equation}\label{eq:reality}
    p^{\epsilon}_{l,-m,-n} = (-1)^{m-n} \, \overline{p^{\epsilon}_{lmn}} ,
\end{equation}

a constraint that is exploited numerically in section 3.8.

## Rotational diffusion on the group

Isotropic rotational diffusion keeps the form it had on the sphere. Wigner-$D$ functions
are eigenfunctions of the Laplace--Beltrami operator on $SO(3)$ with the same eigenvalue
as before, now degenerate over the molecular index,

\begin{equation}\label{eq:so3_isotropic_diffusion}
    \frac{d p^{\epsilon}_{lmn}(t)}{dt} = -D_R\, l(l+1)\, p^{\epsilon}_{lmn}(t) ,
\end{equation}

so free diffusion is diagonal and the rank-2 correlation time is again $1/6D_R$.

The group representation also admits diffusion models that the sphere cannot express. For
an axially symmetric rotor with distinct diffusion coefficients parallel and
perpendicular to its symmetry axis, the operator is still diagonal but the molecular
index now appears explicitly,

\begin{equation}\label{eq:symmetric_top}
    \frac{d p^{\epsilon}_{lmn}(t)}{dt} =
    -\left[\, l(l+1) D_\perp + n^2 \left(D_\parallel - D_\perp\right) \right]
    p^{\epsilon}_{lmn}(t) ,
\end{equation}

which reproduces the classical symmetric-top relaxation rates at rank two,
$6D_\perp$, $5D_\perp + D_\parallel$ and $2D_\perp + 4D_\parallel$, and therefore an
anisotropy decay with up to three exponentials for a dipole tilted from the symmetry
axis. Setting $D_\parallel = D_\perp$ recovers equation \eqref{eq:so3_isotropic_diffusion}
exactly. Restricted diffusion in an ordering potential is available in the same way.

## Photo-selection with a tilted dipole

The absorption rate of a molecule in orientation $R$, illuminated by light with electric
field direction $\hat{\mathbf{e}}$, is proportional to the same squared projection as
before, with the state's own dipole,

\begin{equation}\label{eq:so3_photoselection}
    k_{\epsilon\eta}(R) \;\propto\;
    \left| \hat{\mathbf{e}} \cdot \left( R\, \mathbf{d}_\eta \right) \right|^2 ,
\end{equation}

a function on the group rather than on the sphere. Expanded in the Wigner-$D$ basis it
contains only ranks $L \in \{0,2\}$, as in the $S^2$ case, but the rank-2 weight is now
spread over **both** indices: over the laboratory index $M$ according to the direction of
the field, and over the molecular index $N$ according to the direction of the dipole in
the molecular frame,

\begin{equation}\label{eq:so3_absorption_coeffs}
    b_{LMN} = a_L \, (-1)^{M} \,
    D^{L}_{-M,0}(R_{\hat{\mathbf{e}}}) \;
    D^{L}_{N,0}(R_{\mathbf{d}}) ,
\end{equation}

where $R_{\hat{\mathbf{e}}}$ and $R_{\mathbf{d}}$ are rotations taking the reference axis
onto the field and dipole directions respectively, and the rank weights are
$a_0 = 1/3$, $a_2 = 2/3$ for linear polarization and $a_0 = 1/3$, $a_2 = -1/3$ for
circular polarization about the propagation axis. The asymmetry between the two indices
in equation \eqref{eq:so3_absorption_coeffs} --- the sign and the reflected index on the
field factor --- comes from the field appearing against a conjugated $D$ in the addition
theorem; the two forms agree when the field lies along the reference axis.

High-NA focusing is handled exactly as in section 2.4: the focused beam is the Axelrod
weighted sum of photo-selections along the three laboratory axes, and because equation
\eqref{eq:so3_absorption_coeffs} is linear in the field direction, the same weights apply
to the coefficients. The molecular-index structure, which the dipole sets, is untouched
by the optics.

## The product operator: a single channel

The kinetic term again requires the expansion coefficients of a product of two basis
functions, and here the complex basis pays off. The product of two Wigner-$D$ functions
is the Clebsch--Gordan series [@AngularMomentumBook],

\begin{equation}\label{eq:d_product}
    D^{L}_{MN} \, D^{l}_{mn} = \sum_{l'}
    \langle L M\, l m \,|\, l' m' \rangle \;
    \langle L N\, l n \,|\, l' n' \rangle \;
    D^{l'}_{m'n'} ,
    \qquad m' = M+m, \quad n' = N+n ,
\end{equation}

with a **single** output channel in each index, in contrast with the two channels of the
real spherical-harmonic product of section 2.5. The two Clebsch--Gordan factors
factorize over the same set of $l'$ triples, one acting on the laboratory index and one
on the molecular index --- the structural reason the representation stays tractable
despite doubling the index count. Multiplying a density by a rate function with
coefficients $b_{LMN}$ therefore gives the operator

\begin{equation}\label{eq:so3_multiplication}
    W_{(l'm'n'),(lmn)} = \sum_{LMN} b_{LMN} \;
    \langle L M\, l m \,|\, l' m' \rangle \;
    \langle L N\, l n \,|\, l' n' \rangle ,
\end{equation}

truncated at the working rank $l_\text{max}$, which is the band-limit of the calculation.
The blocks $\mathbf{K}^{\epsilon\eta}$ are then assembled from these operators exactly as
in chapter 1, and the propagation is unchanged: same $\mathbf{M} = \mathbf{D}+\mathbf{K}$,
same matrix exponential, same pulse-window sequence.

## Polarized detection as a group overlap

Detection mirrors excitation. The collection function of a polarized detector is the same
squared projection, built with the *emitting* state's dipole, and the signal is the
overlap of that function with the density. In the coefficient representation the overlap
is read off a single coefficient of the product: if $w$ is the collection function and
$p_\epsilon$ the density, then

\begin{equation}\label{eq:so3_overlap}
    \int w(R) \, p_\epsilon(t,R) \, dR = 8\pi^2 \left[ W p_\epsilon(t) \right]_{000} ,
\end{equation}

the $D^{0}_{00}$ component of the product, with $W$ the multiplication operator of
equation \eqref{eq:so3_multiplication}. Under the population normalization of section 1.4
the $8\pi^2$ cancels, exactly as the $4\pi$ did on the sphere, and the detector signal is
again an absolute number: an isotropic population $P$ with quantum yield $\Phi$ gives
$\Phi P / 3$, whatever the numerical aperture and whatever the molecular-frame dipole
direction.

## Switching does not rotate the molecule

One modelling choice deserves to be stated explicitly, because it is easy to implement
the opposite by accident. When a molecule switches from state $\eta$ to state $\epsilon$,
its orientation $R$ does not change --- the barrel is where it was, on the timescale of
the photochemistry. What changes is *which* body-frame dipole is active, and that
information already sits in the optics through $\mathbf{d}_\epsilon$ in equation
\eqref{eq:so3_absorption_coeffs}. Accordingly, non-radiative and orientation-independent
transitions contribute **identity** blocks in the angular indices, and the apparent
reorientation of the dipole on switching is a consequence of the excitation and detection
operators, not of a jump operator acting on the density. Modelling the isomerization as a
rotation of the density would double-count it.

## Parity reductions and the reality constraint

The eigendecomposition again dominates the cost, and three exact reductions apply.

The even-$l$ argument of section 2.7 carries over unchanged: photo-selection has even
rank, the initial condition sits at $l=0$, and odd ranks are never populated. A second,
independent reduction applies to the **laboratory** index: the photo-selection functions
built from principal-axis linear polarizations, and from circular polarization about the
propagation axis, contain only even laboratory orders $M \in \{0,\pm 2\}$, so odd $m$ is
never reached either. Together these cut the basis from $455$ to $152$ functions per
species at $l_\text{max} = 6$. Both are exact at any power, for the parity reason given
in section 2.7 --- not weak-field approximations --- and both have been verified deep
into saturation.

The **molecular** index admits no such reduction. A tilted dipole spreads rank-2 weight
over all $N \in \{0,\pm1,\pm2\}$ through equation \eqref{eq:so3_absorption_coeffs}, so odd
$n$ is genuinely populated. That is the irreducible price of modelling dipole
reorientation, and it is worth recognising as physics rather than inefficiency.

Finally, the reality constraint \eqref{eq:reality} means the complex coefficient vector
of a real density carries only as much information as a real vector of the same length.
Pairing $(m,n)$ with $(-m,-n)$ through that constraint gives a fixed unitary transform
under which the propagation matrix becomes real, so the eigendecomposition can be
performed in real arithmetic. Combined, the reductions and the real eigenproblem shorten
a representative eight-state calculation from about 39 s to 5.8 s, with results identical
to ten significant figures.

The even-$m$ reduction is the only one with a caveat, and it is a physical one: a field
with **odd** laboratory order breaks it. That means a linear polarization tilted out of
the plane perpendicular to the propagation axis, or an elliptical or rotating field. A
linear field anywhere within that plane --- at $45^\circ$ between $x$ and $y$, for
instance --- remains even-order and is safe.

## Exact reduction to the $S^2$ representation

The two representations are not independent models to be compared for plausibility. When
all species share a dipole direction, the molecular index becomes a redundant label, and
the $SO(3)$ engine reproduces the $S^2$ engine exactly: same populations, same detector
signals, to numerical precision. This is enforced as a test rather than argued, and it is
the strongest available statement about both --- the cheap representation is a special
case of the expensive one, so agreement in the shared-dipole limit tests the whole chain,
from photo-selection through the product operator to detection, in both engines at once.

# Choosing a representation

## The decision

The choice follows from one question: **do the species of the kinetic scheme share a
transition-dipole direction in the molecular frame?** If they do, use the spherical
representation; if they do not, the spherical representation cannot express the model and
the group representation is required. The code applies exactly this rule when asked to
choose automatically, selecting $SO(3)$ when the declared per-state dipoles differ and
$S^2$ otherwise.

Beyond the dipole question, the group representation is also required for physics that
lives in the molecular frame: anisotropic (symmetric-top) diffusion, where the rate
depends on the orientation of the rotor's symmetry axis relative to the dipole; and
two-tier models in which a local motion is defined in a frame attached to the molecule.

| | $S^2$ | $SO(3)$ |
|---|---|---|
| configuration | dipole axis $\hat{\mathbf{u}}$ | full orientation $R$ |
| basis | real $Y_{lm}$ | complex $D^{l}_{mn}$ |
| indices | one, laboratory | two, laboratory and molecular |
| size at $l_\text{max}=6$ | 49 | 455 (152 after reductions) |
| dipole | one, shared | one per state |
| product operator | real triple product, two channels | Clebsch--Gordan, one channel |
| diffusion | isotropic, ordering potential | isotropic, symmetric top, ordering potential |
| exact reductions | even $l$ | even $l$, even $m$; $n$ irreducible |
| valid when | all species share one dipole | always |

## What it costs

Essentially all of the computation time is the eigendecomposition of $\mathbf{M}$, once
per pulse window; assembling the operators is negligible. Cost therefore scales as the
cube of the basis size times the number of species, which is what makes the difference
between the representations concrete: at $l_\text{max}=6$ the spherical basis has 49
functions per species against 455 before reductions, and the reductions of section 3.8
recover a large part but not all of that gap.

In practice the two are used at different band-limits. The spherical representation is
cheap enough to run at $l_\text{max}=6$ routinely, which resolves sharp orientational
features such as those produced by saturation. The group representation defaults to
$l_\text{max}=2$, sufficient whenever the density stays within the rank-2 content that
one-photon photo-selection generates, and should be raised deliberately --- when
saturation sharpens the distribution, or when an ordering potential is strong --- with a
convergence check rather than by habit.

## Worked cases

**Time-resolved anisotropy of a rigidly labelled particle.** One fluorescent species,
one dipole, free isotropic diffusion. The spherical representation is exact and cheap;
the group representation would return the same numbers more slowly.

**STARSS with rsEGFP2.** The on and off states are cis and trans isomers with dipoles
about $35^\circ$ apart, so the group representation is required for a faithful model.
The difference is not academic: recomputing the STARSS method 1 simulation on $SO(3)$
rather than $S^2$ changes the simulated detector signals by roughly 14%.

**A dye in a membrane, or on a tumbling anisotropic body.** Restricted diffusion about a
director, or a symmetric-top rotor with a tilted dipole, needs the molecular index; the
sphere can carry the ordering potential but not the tilt.

**A first estimate during data analysis.** The spherical representation at
$l_\text{max}=6$ is fast enough for fitting loops; the group representation is the tool
for checking, at the end, how much the dipole reorientation moved the answer.

# Photophysical schemes

## Four states model for negative rsFPs

Photo-physics of rsFPs can be modeled as a 4 states system, with a ground on-state, a ground off-state, and the correspondent first excited states. The on- and off-state absorb in a different spectral region and we can use two laser with different wavelength to promote the interconversion among the states. The on-off switching happens through an excited state reaction, usually a cis-trans isomerization of the chromophore. Further processes might happen in the ground state, and in fast time-scales several intermediates can be revealed [@Laptenok2018; @Woodhouse2020]. Usually, in green negative rsFPs the great change of spectroscopical properties happens a after a proton transfer in the ground state [@Nienhaus2016].

Using the scheme in Figure [2](#fig:rsFP_kinetic_models){reference-type="ref" reference="fig:rsFP_kinetic_models"}(a), we can specify all the kinetic constants using parameters that are commonly found in rsFP literature
\begin{equation}
\label{eq:rsFP_k_4states}
\begin{aligned}
    k_{21} &= \sigma_{21}(\lambda) \rho(\lambda) \\
    k_{12} &= 1 / \tau_{\text{on}} \\
    k_{32} &= (1 / \tau_{\text{on}}) \ \Phi_{\text{on} \rightarrow \text{off}} \\
    k_{43} &= \sigma_{43}(\lambda) \rho(\lambda) \\
    k_{34} &= 1/ \tau_{\text{off}} \\
    k_{14} &= (1 / \tau_{\text{off}}) \ \Phi_{\text{off} \rightarrow \text{on}}\\
\end{aligned}
\end{equation}
where $\tau_{\text{on}}$ and $\tau_{\text{off}}$ are the lifetimes of the excited states, $\Phi_{\text{on} \rightarrow \text{off}}$ and $\Phi_{\text{off} \rightarrow \text{on}}$ are the quantum yield for the off- and on- switching processes respectively, and $\lambda$ the wavelength of the laser used to promote the ground to excited state transition.
We are assuming that the quantum yields do not depend on the wavelength of the exciting laser.

\begin{figure}[h!]
    \centering
    \includegraphics[width=1\textwidth]{figures/rsFP_4states_8states_kinetic_models.png}
    \caption[Kinetic models for negative rsFPs]
    {Kinetic models for negative rsFPs. (a) Four states model, and (b) eight states
    model. Light blue boxes highlight the anionic states, including the on-state,
    whereas gray boxes highlight the neutral states, which are all non-fluorescent.
    Fluorescence photons are emitted only from the state 2 (excited cis-anionic), when
    the pathway $1\rightarrow2$ is followed.}
    \label{fig:rsFP_kinetic_models}
\end{figure}

In order to implement the rsFP photo-physics in the full rotational diffusion and kinetics model, we need to compute the kinetic constants as in eq. \eqref{eq:triple_product}, thus we associate a block matrix $\mathbf{K}^{\epsilon\eta}$ to each kinetic rate. The block matrix include the details regarding high-NA photoselection, polarization state of the light, and the kinetic rate itself. For transitions that do not depend on the orientation of the fluorophores the block is simplified, such that $\mathbf{K}^{\epsilon\eta} = \mathbf{1}k_{\epsilon\eta}$. The block structure of the full matrix $\mathbf{K}$ can then be further specified as
\begin{equation}
\label{eq:K_4states}
    \mathbf{K} =
    \begin{bmatrix}
    \square & \mathbf{K}^{12} &   & \mathbf{K}^{14} \\
    \mathbf{K}^{21} & \square \\
        & \mathbf{K}^{32} & \square  & \mathbf{K}^{34} \\
        &   &   \mathbf{K}^{43}  & \square
    \end{bmatrix}
    =
    \begin{bmatrix}
    \square & \mathbf{1}k_{12} &   & \mathbf{1}k_{14} \\
    \mathbf{K}^{21} & \square \\
        & \mathbf{1}k_{32} & \square  & \mathbf{1}k_{34} \\
        &   &   \mathbf{K}^{43}  & \square
    \end{bmatrix}.
\end{equation}
The out of diagonal matrix blocks that are missing in this expression are zeros, or in other words there are not processes that interconvert the specific couple of species. The diagonal matrix blocks, that are represented with squares $\square$ for sake of simplicity of notation, are obtained from the out of diagonal ones as in eq. \eqref{eq:k_diagonal}, e.g. $\mathbf{K}^{22} = - \mathbf{1}k_{12} - \mathbf{1}k{32}$.
Note that when multiple laser excitation are used the same species might absorb at all wavelength, and thus the corresponding kinetic matrix block will be the sum of the contributions due to different wavelengths, e.g. $\mathbf{K}^{21} = \mathbf{K}^{21}(405\text{nm}) + \mathbf{K}^{21}(488\text{nm})$. This is generally referred as crosstalk of the switching processes of rsFPs.

## Eight states model for negative rsFPs

When the power densities employed for the switching are high enough, the excited state cis-trans isomerization might not be the rate determining step anymore. In this condition we can have the accumulation of concentration of intermediates. For example starting from the ground on-state, after the excited state flipping from cis- to trans-conformers, a protonation must occur to reach the final ground off-state. This process usually take several $\mu$s or tens of $\mu$s, and it might involve several intermediates. The simplest way to include this process in the photo-physics is to use an 8 states model. Two intermediate species are inserted between on- and off- states. The resulting scheme is represented in Figure \eqref{fig:rsFP_kinetic_models}(b).
For additional clarity, the states can be named after the conformation of the chromophore and the protonation state: cis or trans and anionic or neutral. The on-state is then also the cis-anionic state indexed with $\epsilon=1$, and the off-state is also the trans-neutral state indexed with $\epsilon=5$. The intermediate state during the off-switching process is the trans-anionic state indexed with $\epsilon=3$, whereas the intermediate state during the on-switching process is the cis-neutral state indexed with $\epsilon=7$. The excited states are respectively indexed with $\epsilon=2,4,6,\text{and }8$.
We can specify all the kinetic constants involved in this model as
\begin{equation}
\label{eq:rsFP_k_8states}
    \begin{aligned}
        k_{21} &= \sigma_{\text{on}}(\lambda) \rho(\lambda) \\
        k_{12} &= 1 / \tau_{\text{on}} \\
        k_{32} &= (1 / \tau_{\text{on}}) \ \Phi^{\text{anionic}}_{\text{cis} \rightarrow \text{trans}} \\    
        k_{43} &= \sigma_{\text{on}}(\lambda) \rho(\lambda) \\
        k_{34} &= 1 / \tau_{\text{on}} \\
        k_{14} &= (1 / \tau_{\text{on}}) \ \Phi^{\text{anionic}}_{\text{trans} \rightarrow \text{cis}} \\
        k_{53} &= 1 / \tau_{\text{prot}} \\
        k_{65} &= \sigma_{\text{off}}(\lambda) \rho(\lambda) \\
        k_{65} &= 1/ \tau_{\text{off}} \\
        k_{76} &= (1 / \tau_{\text{off}}) \ \Phi^{\text{neutral}}_{\text{trans} \rightarrow \text{cis}}\\
        k_{87} &= \sigma_{\text{off}}(\lambda) \rho(\lambda) \\
        k_{78} &= 1/ \tau_{\text{off}} \\
        k_{58} &= (1 / \tau_{\text{off}}) \ \Phi^{\text{neutral}}_{\text{cis} \rightarrow \text{trans}}\\
        k_{17} &= 1 / \tau_{\text{deprot}}, \\
    \end{aligned}
\end{equation}

where $\tau_\text{prot}$ and $\tau_\text{deprot}$ are the time constants for the ground state protonation and deprotonation processes.
In this parametrization we are assuming that the absorption spectrum and the lifetimes of cis and trans species are the same. This is a reasonable assumption in negative green rsFPs [@Yadav2015b], but the model can easily accept different properties for cis and trans conformers.

The full kinetic matrix is constructed in analogy to eq. \eqref{eq:K_4states} as
\begin{equation}
\label{eq:K_8states}
    \mathbf{K} = 
    \begin{bmatrix}
    \square & \mathbf{1}k_{12} &  & \mathbf{1}k_{14} & & & \mathbf{1}k_{17}  \\
    \mathbf{K}^{21} & \square \\
    & \mathbf{1}k_{32} & \square  & \mathbf{1}k_{34} \\
    & & \mathbf{K}^{43}  & \square \\
    & & \mathbf{1}k_{53} &  &  \square & \mathbf{1}k_{56} & & \mathbf{1}k_{58} \\
    & & & & \mathbf{K}_{65} & \square \\
    & & & & & \mathbf{1}k_{76} & \square & \mathbf{1}k_{78} \\
    & & & & & & \mathbf{K}_{87} & \square 
    \end{bmatrix}.
\end{equation}
Note that $\mathbf{K}^{21} =  \mathbf{K}^{43}$ and $\mathbf{K}^{65} =  \mathbf{K}^{87}$ because we are assuming the same absorption cross-sections for cis-anionic/trans-anionic states and trans-neutral/cis-neutral states respectively.

## Parametrization of rsEGFP2

The parametrization of the reversible switchable protein rsEFGP2 was mainly derived from [@Khatib2016].
In particular, the absorption properties of the on-state and off-state can be specified with the extinction coefficients [@Khatib2016],
\begin{equation}
    \begin{aligned}
        \epsilon_{\text{on}} (488 nm) &= 51560 \ \text{M}^{-1} \text{cm}^{-1} \\
        \epsilon_{\text{on}} (405 nm) &= 5260 \ \text{M}^{-1} \text{cm}^{-1} \\
        \epsilon_{\text{off}} (488 nm) &= 60 \ \text{M}^{-1} \text{cm}^{-1} \\
        \epsilon_{\text{off}} (405 nm) &= 22000 \ \text{M}^{-1} \text{cm}^{-1} \\
    \end{aligned}
\end{equation}
and then the cross-sections $\sigma$ in cm$^2$ can be computed accordingly using the equation [@TkachenkoBook]
\begin{equation}
    \sigma(\lambda) = \frac{2303}{N_A}\epsilon(\lambda),
\end{equation}
where $N_A$ is the Avogadro number.
The quantum yields of fluorescence and cis-trans isomerization are as follows [@Khatib2016]
\begin{equation}
    \begin{aligned}
        \Phi_\text{fluo} &= 0.35 \\
        \Phi^{\text{anionic}}_{\text{cis} \rightarrow \text{trans}} &= 1.65 \cdot 10^{-2} \\
        \Phi^{\text{neutral}}_{\text{trans} \rightarrow \text{cis}} &= 0.33. \\
    \end{aligned}
\end{equation}
The fluorescence lifetimes are $\tau_\text{on} = 1.6$ ns [@Testa2015] and $\tau_\text{off} \approx$ ps [@Woodhouse2020] (20 ps are assumed). Finally $\tau_\text{deprot} = 5.1$ us, in agreement to dynamics observed in transient absorption measurements [@Woodhouse2020], and $\tau_\text{prot} = 48$ $\mu$s was determined from high-power off-switching data.
The absorption properties of cis and trans conformers of the same protonation state were taken as equal, i.e. $\epsilon_\text{on}(\lambda) = \epsilon_\text{cis}^\text{anionic}(\lambda) \approx \epsilon_\text{trans}^\text{anionic}(\lambda)$ and $\epsilon_\text{off}(\lambda) = \epsilon_\text{trans}^\text{neutral}(\lambda) \approx \epsilon_\text{cis}^\text{neutral}(\lambda)$. This is a reasonable approximation given the fact that the absorption spectra are very close, see supporting information of [@Woodhouse2020].

Two remaining parameters are undefined, $\Phi^{\text{anionic}}_{\text{trans} \rightarrow \text{cis}}$ and $\Phi^{\text{neutral}}_{\text{cis} \rightarrow \text{trans}}$. These quantum yields enable the back-conversion from the intermediate states. $\Phi^{\text{anionic}}_{\text{trans} \rightarrow \text{cis}}$ will be fitted against STARSS method 2 data, whereas $\Phi^{\text{neutral}}_{\text{cis} \rightarrow \text{trans}}$ will be assumed as 0. Assuming a null quantum yield silences the associated process. This is a reasonable assumption if there is not an accumulation of intermediate in the on-switching process that is excited with 405 nm light. Further characterization will be needed to better characterize the back-conversion quantum yields.

<!-- TODO(a60): the SO(3) chapter and the choosing-a-representation chapter
attach here (n010 architecture sections 3 and 4). -->
