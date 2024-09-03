# Sediment Transport

Our treatment of the transport of sediment is based on a model developed by
Hairsine and Rose that accounts for size-selective sediment transport using a
particle size distribution.

## 2-D Hairsine-Rose (H-R) Equations

The H-R equations use a particle size distribution consisting of a set of
$I$ discrete particle/sediment size classes $i = 1, 2, ..., I$. Each size class
is represented by a sediment concentration $c_i$ and the mass $M_i$ of the layer
deposited by size class $i$ on the bed floor.

Each sediment concentration $c_i$ evolves in time according to its own transport
equation

\begin{equation}
\frac{\partial (hc)_{i}}{\partial t} + \nabla\cdot(h c_i \vec{u}) = e_i + e_{ri} + r_i + r_{ri} - d_i \tag{1}\label{1}
\end{equation}

where

* $h$ is the water height, as in the [shallow water equations](swe.md)
* $\vec{u} = (u, v)$ is the water flow velocity, along with which sediments are carried
* $e_i$ and $e_{ri}$ are the _rainfall-driven detachment_ and _re-detachment rates_
* $r_i$ and $r_{ri}$ are the _flow-induced entrainment and re-entrainment rates_
* $d_i$ is a _deposition rate_ for the size class, expressed as mass per unit area per unit time

and $\nabla\cdot\vec{F} = (\partial F_x/\partial x, \partial F_y/\partial y)$ is
the 2D divergence of the spatial vector $\vec{F}$.

The deposited layer mass $M_i$ for each size class accumulates according to an
ordinary differential equation involving its deposition, re-detachment, and
re-entrainment rates:

\begin{equation}
\frac{\partial M_i}{\partial t} = d_i - e_{ri} - r_{ri}\tag{2}\label{2}
\end{equation}

All size classes deposit their layers to the bed floor, changing the bed
elevation according to the ordinary differential equation

\begin{equation}
(1-\beta)\rho_{s}\frac{\partial z}{\partial t} = \sum_{i=1}^{I}(d_i - e_i - e_{ri} - r_{i} - r_{ri})\tag{3}\label{3}
\end{equation}

where 

* $\beta$ is the porosity of the soil in its original state
* $\rho_s$ is the density of solid sediment, assumed to be the same for all size classes.

### Source terms

[Hairsine and Rose, 1992] specify forms for each of the source terms appearing in
the H-R equations above.

#### Rainfall-driven detachment and re-detachment rates

\begin{eqnarray}
e_i &=& F_w (1 - H) p_i a_0 P \\
e_{ri} &=& F_w H \frac{M_i}{M_t} a_d P \tag{4}\label{4}
\end{eqnarray}

where

* $p_i$ is the time-dependent ratio of the proportion of sediment in size class $i$
  to its proportion in the soil's original state (i.e. $p_i(0) = 1)
* $a_0$ and $a_d$ are the detachability of uneroded and deposited soil, expressed
  in mass per unit volume
* P is the intensity of rainfall intensity expressed as the change in water height
  per unit time
* $M_t = \sum M_i$ is the total sediment mass in the deposited layer,
  expressed in mass per unit area
* $F_w$ is a _shield factor_ that attenuates the detachment and re-detachment
  rates under conditions where the water height is three times greater than the
  diameter of a "typical" raindrop.
* $H = min(M_t/(F_w M_t^*),1)$ is the proportion of shielding of the deposited
  layer, given in mass per unit area; here, $M_t^* is calibrated to the mass of
  deposited sediment needed to completely shield the soil in its original state.

The shield factor $F_w$ can be computed using a power law relation by [Proffitt et al. 1991]:

\begin{equation}
F_{w}=
\begin{cases}
  1             \quad & h \le h_{0} \\
  (h_{0}/h)^{b} \quad & h > h_{0}   \\
\end{cases} \tag{5}\label{5}
\end{equation}

where a threshold of $h_0 = 0.33D_R$ is used, and $D_R$ is the mean raindrop size.

The exponent $b$ in $\eqref{5}$ varies depending on the type of soil, and can be
obtained with a best fit using experimental data. For example, $b$ is 0.66 for
clay and 1.13 for loam.

#### Overland flow-driven entrainment and re-entrainment rates

\begin{eqnarray}
r_i &=& (1-H)p_{i}\frac{F(\Omega-\Omega_{cr})}{J} \\
r_{ri} &=& H\frac{M_{i}}{M_{t}}\frac{F(\Omega - \Omega_{cr})}{(\rho_{s}-\rho_{w})gh/\rho_{s}} \tag{6}\label{6}
\end{eqnarray}

where

* $\Omega = \rho_{w}gh S_f \sqrt{u^2+v^2}$ is the _stream power_ in mass per cubic unit time,
  with $S_f = n^2 (u^2 + v^2) h^{-4/3}$
* $\Omega_{cr}$ is the _critical stream power_, below which neither soil
  entrainment or re-entrainment occur
* $F$ is the _effective fraction of excess stream power_ in entrainment or
  re-entrainment, which accounts for thermal energy dissipation
* $J$ is the _specific energy of entrainment_ in energy per unit mass, which
  indicates e.g. the energy required for soil of a given mass to be entrained
* $\rho_{w}$ is the density of water.

#### Size class deposition rate

\begin{equation}
d_{i} = v_{i}c_{i} \tag{7}\label{7}
\end{equation}

where $v_{i}$ is the _settling velocity_ of each size class with concentration
$c_i$. This model assumes that

* the suspended load in the water column is completely mixed in the vertical direction
* the infiltration rate does not affect size class settling velocities.

## Coupling the H-R equations with the Shallow Water Equations

Equations $\eqref{1}$ can be coupled with the [shallow water equations](swe.md)
by augment the solution vector $\mathbf{U}$ with water-height-weighted
sediment size-class concentrations:

\begin{align}
\mathbf{U} =
  \begin{bmatrix}
    h      \\[.5em]
    uh     \\[.5em]
    vh     \\
    c_{1}h \\
    \vdots \\
    c_{I}h
  \end{bmatrix}.
\end{align}

We also augment the flux vectors $\mathbf{E}$ and $\mathbf{G}$ from the shallow
water equations with the flux terms for the sediment size class transport
equations:

\begin{align}
\mathbf{E} =
  \begin{bmatrix}
    uh                           \\[.5em]
    u^{2}h+\frac{1}{2}gh^{2}     \\[.5em]
    uvh                          \\
    c_{1}uh                      \\
    \vdots                       \\
    c_{I}uh
  \end{bmatrix},
\end{align}

\begin{align}
\mathbf{G} =
  \begin{bmatrix}
    vh                           \\[.5em]
    uvh                          \\[.5em]
    v^{2}h+\frac{1}{2}gh^{2}     \\
    c_{1}vh                      \\
    \vdots                       \\
    c_{I}vh
  \end{bmatrix}.
\end{align}

Finally, we augment the shallow water equation source vector $\mathbf{S}$ with
the (re)attachment, (re)entrainment, and deposition terms:

\begin{align}
\mathbf{S} =
  \begin{bmatrix}
    R
    -gh\frac{\partial z}{\partial x} - C_D u\sqrt{u^2 + v^2} \\[.5em]
    -gh\frac{\partial z}{\partial y} - C_D v\sqrt{u^2 + v^2} \\
    e_1 + e_{r1} + r_1 + r_{r1} - d_{1}                                 \\
    \vdots                                                          \\
    e_I + e_{rI} + r_I + r_{rI} - d_I 
  \end{bmatrix}.
\end{align}

with these augmentations, the H-R equations can be merged with the shallow
water equations to read

\begin{eqnarray}
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{E}}{\partial t} + \frac{\partial \mathbf{G}}{\partial t} &=& \mathbf{S}\\
\frac{\partial \mathbf{M}}{\partial t} &=& \mathbf{D} \tag{8}\label{8}
\end{eqnarray}

where we have defined a _deposited mass vector_ $\mathbf{M}$ and a
_net deposition vector_ $\mathbf{D}$:

\begin{align}
\mathbf{M} =
  \begin{bmatrix}
    M_{1}    \\[.5em]
    \vdots   \\
    M_{I} 
  \end{bmatrix},
\end{align}

\begin{align}
\mathbf{D} =
  \begin{bmatrix}
    d_{1}-e_{r1}-r_{r1} \\[.5em]
    \vdots              \\
    d_{I}-e_{rI}-r_{rI}  
  \end{bmatrix}.
\end{align}

### TELEMAC/GAIA equations

The sediment transport equations implemented in TELEMAC/GAIA are

\begin{equation}
\label{eqn:sd2d}
\frac{\partial hc_{i}}{\partial t}+\frac{\partial uhc_{i}}{\partial x} + \frac{\partial vhc_{i}}{\partial y} = E_{i} - D_{i},
\end{equation}

where $h$, $u$ and $v$ are water depth, velocities in horizontal and vertical directions, respectively, $i = 1, 2, \dots I$ is the sediment class, $c_{i}$ is the sediment concentration given as mass per unit volume $[M/L^{3}]$, 
$I$ is the number of sediment size classes, and $E_{i}$ and $D_{i}$ are erosion and deposition rate formulated as mass per unit area per unit time $[M/L^{2}/T]$.

According to GAIA, the erosion and deposition rates are calculated as:
\begin{equation}
E_i = M \left( \frac{\tau_b - \tau_{ce}}{\tau_{ce}} \right),
\end{equation}
\begin{equation}
D_i = w c_i \left[ 1 - \left( \frac{\tau_b}{\tau_{cd}} \right) \right],
\end{equation}

where, for each sediment class $i$, $M$ is the Krone-Partheniades erosion law constant [kg/m$^{2}$], or the erodibility coefficient, $w$ is the settling velocity for sediment class $i$ (m/s), $\tau_{ce}$ is critical shear stress for erosion (N/m$^2$), $\tau_{cd}$ is critical shear stress for deposition (N/m$^2$), $\tau_b = \rho C_D u\sqrt{u^2+v^2}$ is the bottom shear stress. 

Coupling sediment transport equation with Shallow Water Equations lead to:

\begin{equation}
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{E}}{\partial t} + \frac{\partial \mathbf{G}}{\partial t} = \mathbf{S},
\end{equation}

where $\mathbf{U}$ is the conservative variable vector, $\mathbf{E}$ and $\mathbf{G}$ are the flux vectors in x and y direction. $\mathbf{U}$, $\mathbf{E}$ and $\mathbf{G}$ are same as those in 2-D H-R equation. $\mathbf{S}$ is the source vector:


\begin{align}
\mathbf{S} 
=
    \begin{bmatrix}
    S_{r}                                                           \\[.5em]
    -gh\frac{\partial z_{b}}{\partial x} - C_{D}u\sqrt{u^{2}+v^{2}} \\[.5em]
    -gh\frac{\partial z_{b}}{\partial y} - C_{D}v\sqrt{u^{2}+v^{2}} \\
    E_{1}-D_{1}                                 \\
    \vdots                                                          \\
    E_{I}-D_{I} 
    \end{bmatrix}
\end{align}



## Spatial discretization

Integrating equation Eq (\ref{eqn:cpeqns}) over an arbitrary two-dimensional computational element $A$ with a boundary $\Gamma$, then the 
governing equations expressed in conservation form can be written as follows:

\begin{equation}
\frac{\partial}{\partial t}\int_{A}\mathbf{U}dA + \oint_{\Gamma}\mathbf{F} \cdot \mathbf{n}d\Gamma = \iint_{A}\mathbf{S}dA,
\end{equation}

where $\mathbf{F}$ is the flux vector, and $\mathbf{n}$ is the unit vector normal to boundary $\partial \Gamma$ and directed outward. The 
integrand $\mathbf{F} \cdot \mathbf{n} = F_{\perp}$ is the numerical flux normal to each cell face and defined as:

\begin{align}
\label{eqn:flux}
\mathbf{F} \cdot \mathbf{n} 
=
    \begin{bmatrix}
    hu_{\perp}                                                                      \\[.5em]
    huu_{\perp} + \frac{1}{2}gh^{2}cos \theta + \frac{1}{24}g\Delta h^{2}cos \theta \\
    hvu_{\perp} + \frac{1}{2}gh^{2}sin \theta + \frac{1}{24}g\Delta h^{2}sin \theta \\
    hc_{1}u_{\perp}                                                                 \\
    \vdots                                                                          \\
    hc_{I}u_{\perp}                                                                 \\
    \end{bmatrix}
\end{align}

where $u_{\perp}$ denotes the velocity normal to the cell interface and computed as $u_{\perp}=ucos \theta + vsin \theta$, and $\theta$ is an 
angle between the face normal vector and the x axis, $\Delta h$ is a variation of $h$ along the cell face. The last terms in the second and third
rows of equation (\ref{eqn:flux}) are the hydrostatic thrust correction terms suggested by Bradford and Sanders [2002]. The are necessary to balance 
the bed slope terms for the still water condition.

Roe's approximate Riemann solver [Roe, 1981] is implemented to compute the fluxes at the cell interface:

\begin{equation}
\mathbf{F} \cdot \mathbf{n} \approx \mathbf{F}_{\perp,f} = 
\frac{1}{2} \left(\mathbf{F}_{\perp,L} + \mathbf{F}_{\perp,R}-\mathbf{\hat{R}} |\mathbf{\hat{\Lambda}}| \mathbf{\Delta\hat{V}} \right)
\end{equation}

where the subscript $f$ denotes the interface between two adjacent cells, $L$ and $R$ denote left and right cell for the interface, and $\Delta$ 
denotes the finite difference across the interface. The terms $\mathbf{\hat{R}}$ and $\mathbf{\hat{\Lambda}}$ are the right eigenvector and the 
eigenvalue of the Jacobian of $\mathbf{F}_{\perp}$, and $\mathbf{\Delta}\mathbf{\hat{V}}=\hat{L}\Delta U$, denotes the wave strength, where $\hat{L}$ 
is the left eigenvector of the Jacobian of $\mathbf{F}_{\perp}$:

\begin{align}
\mathbf{\hat{R}}
=
    \begin{bmatrix}
    1                         & 0           & 1                         & 0      & \ldots & 0      \\
    \hat{u}-\hat{a}cos \theta & -sin \theta & \hat{u}+\hat{a}cos \theta & 0      & \ldots & 0      \\
    \hat{v}-\hat{a}sin \theta &  cos \theta & \hat{v}+\hat{a}sin \theta & 0      & \ldots & 0      \\
    \hat{c_{1}}               & 0           & \hat{c_{1}}               & 1      & \ldots & 0      \\
    \vdots                    & \vdots      & \vdots                    & \vdots & \ddots & \vdots \\ 
    \hat{c_{I}}               & 0           & \hat{c_{I}}               & 0      & \ldots & 1      \\
    \end{bmatrix}
\end{align}

\begin{align}
\mathbf{\hat{\Lambda}}
=
    \begin{bmatrix}
    |\hat{u_{\perp}}-\hat{a}|^{*} &                   &                               &                   &        &                   \\
                                  & |\hat{u_{\perp}}| &                               &                   &        &                   \\
                                  &                   & |\hat{u_{\perp}}+\hat{a}|^{*} &                   &        &                   \\
                                  &                   &                               & |\hat{u_{\perp}}| &        &                   \\
                                  &                   &                               &                   & \ddots &                   \\
                                  &                   &                               &                   &        & |\hat{u_{\perp}}| \\
    \end{bmatrix}
\end{align}

\begin{align}
\mathbf{\Delta}\mathbf{\hat{V}}=\hat{L}\Delta U
=
    \begin{bmatrix}
    \frac{1}{2} \left( \Delta h - \frac{\hat{h}\Delta u_\perp}{\hat{a}} \right) \\[.5em]
    \hat{h}u_\parallel                                                          \\[.5em]
    \frac{1}{2} \left( \Delta h + \frac{\hat{h}\Delta u_\perp}{\hat{a}} \right) \\[.5em]
    (c_{1}h)_{R} - (c_{1}h)_{L} - \hat{c_{1}}(h_{R}-h_{L})                      \\[.5em]
    \vdots                                                                      \\[.5em]
    (c_{I}h)_{R} - (c_{I}h)_{L} - \hat{c_{I}}(h_{R}-h_{L})                      \\[.5em]
    \end{bmatrix}
\end{align}

where $a$ denotes the celerity of a simple gravity wave, $u_{\parallel}$ denotes the vecloty components parallel to the cell interface and can be 
computed as:

\begin{equation}
u_{\parallel} = -usin \theta + vcos \theta
\end{equation}

The quantities with a hat denote the Roe averages, whch are calculated as the following:

\begin{eqnarray}
\hat{h}     &=& \sqrt{h_{L}h_{R}}                                                     \\
\hat{u}     &=& \frac{\sqrt{h_{L}}u_{L}+\sqrt{h_{R}}u_{R}}{\sqrt{h_{L}}+\sqrt{h_{R}}} \\
\hat{v}     &=& \frac{\sqrt{h_{L}}v_{L}+\sqrt{h_{R}}v_{R}}{\sqrt{h_{L}}+\sqrt{h_{R}}} \\
\hat{a}     &=& \sqrt{\frac{g}{2}(h_{L}+h_{R})}                                       \\
\hat{c_{i}} &=& \frac{\sqrt{h_{L}}c_{i,L}+\sqrt{h_{R}}c_{i,R}}{\sqrt{h_{L}}+\sqrt{h_{R}}} 
\end{eqnarray}

The asterisks denote that the eigenvalues $\hat{\lambda}_{1}=\hat{u}_{\perp}-\hat{a}$ and $\hat{\lambda}_{3}=\hat{u}_{\perp}+\hat{a}$ are adjusted since Roe's 
method does not provide correct flux for critical flow :

\begin{equation}
|\hat{\lambda}_{1}|^{*} = \frac{\hat{\lambda}_{1}^{2}}{\Delta \lambda} + \frac{\Delta \lambda}{4}
\quad if \quad -\Delta \lambda /2 < \hat{\lambda}_{1} < \Delta \lambda /2
\end{equation}

\begin{equation}
|\hat{\lambda}_{3}|^{*} = \frac{\hat{\lambda}_{3}^{2}}{\Delta \lambda} + \frac{\Delta \lambda}{4}
\quad if \quad -\Delta \lambda /2 < \hat{\lambda}_{3} < \Delta \lambda /2
\end{equation}

where $\Delta \lambda = 4(\lambda_{R}-\lambda_{L})$

## Source term

Please refer to shallow_water_equation.md. The H-R sediment source is

\begin{equation}
(e_{i}+e_{ri}+r_{i}+r_{ri}-d_{i})A
\end{equation}

The sediment source in GAIA is 
\begin{equation}
(E_{i}-D_{i})A
\end{equation}

## References

* Hairsine, P. B., and C. W. Rose (1991). Rainfall detachment and deposition:
Sediment transport in the absence of flow-driven processes, Soil Sci. Soc. Am. J., 55(2), 320–324.
* Hairsine, P. B., and C. W. Rose (1992). Modeling water erosion due to overland flow using physical principles: 1. Sheet flow, Water Resour. Res., 28(1), 237–243.
* Kim, J., V. Y. Ivanov, and N. D. Katopodes (2013). Modeling erosion and sedimentation coupled with hydrological and overland flow processes at the watershed scale, Water Resour. Res., 49, 5134–5154, doi:10.1002/wrcr.20373.
