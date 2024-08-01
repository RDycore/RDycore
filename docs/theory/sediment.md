# Sediment Transport

## 2-D H-R equation
The 2-D H-R equations are

\begin{equation}
\label{eqn:hr2d}
\frac{\partial hc_{i}}{\partial t}+\frac{\partial uhc_{i}}{\partial x} + \frac{\partial vhc_{i}}{\partial y} = e_{i} + e_{ri} + r_{i} + r_{ri} - d_{i},
\end{equation}

\begin{equation}
\frac{\partial M_{i}}{\partial t} = d_{i} - e_{ri} - r_{ri},
\end{equation}

\begin{equation}
(1-\beta)\rho_{s}\frac{\partial z_{b}}{\partial t} = \sum_{i=1}^{I}(d_{i}-e_{i}-e_{ri}-r_{i}-r_{ri}),
\end{equation}

where $i = 1, 2, \dots I$, $c_{i}$ is the sediment concentration given as mass per unit volume $[M/L^{3}]$, 
$M_{i}$ is the sediment mass of the deposited layer formulated as mass per unit area $[M/L^{2}]$, 
$I$ is the number of sediment size classes, and $e_{i}, e_{ri}, r_{i}, r_{ri}$, and $d_{i}$ are rainfall-driven
detachment and redetachment rates, flow-induced entrainment and reentrainment rates, and the deposition rate 
formulated as mass per unit area per unit time $[M/L^{2}/T]$. $\beta$ is the porosity of original soil and $\rho_{s}$
is the density of solids assumed to be uniform for all sediment classes.
According to [Hairsine and Rose, 1992], the detachment and redetachment rates due to rainfall are calculated as 
following:
\begin{equation}
e_{i} = F_{w}(1-H)p_{i}a_{0}P,
\end{equation}
\begin{equation}
e_{ri} = F_{w}H\frac{M_{i}}{M_{t}}a_{d}P,
\end{equation}
where $p_{i}$ is the ratio of the amount of sediment of class $i$ to that of the original soil, $a_{0}$ and $a_{d}$ 
represent detachability of uneroded and deposited soil as mass per unit volume $[M/L^{3}]$, P is rallfall intensity $[L/T]$,
and $M_{t} = \sum M_{i}$ is the total sediment mass in the deposited layer in mass per unit area $[M/L^2]$/.
$F_{w}$ is the shield factor, which can be formulated with the power law relation by Proffitt et al. [1991]:

\begin{equation}
F_{w}=\begin{cases}
       1             \quad & h \le h_{0} \\
       (h_{0}/h)^{b} \quad & h > h_{0}   \\
       \end{cases}
\end{equation}

where a threshold of $h_{0} = 0.33D_{R}$ is used, and $D_{R}$ is the mean raindrop size. The exponent b varies depending on the 
type of soil and can be obtained with a best fit using experimental data,  e.g., for clay, $b=0.66$, and for loam, $b=1.13$.
$H$ is the proportion of shielding of the deposited layer:
\begin{equation}
H = min(M_{t}/(F_{w}*M_{t}^{*}),1)
\end{equation}
where $M_{t}^{*}$ is a calibrated parameter denoting the mass of deposited sediment needed to completely sheild and original soil, 
given as mass per unit area $[M/L^{2}]$.
The entrainment and reentrainment rates due to overland flow can be estimated by:
\begin{equation}
r_{i} = (1-H)p_{i}\frac{F(\Omega-\Omega_{cr})}{J}
\end{equation}
\begin{equation}
r_{ri} = H\frac{M_{i}}{M_{t}}\frac{F(\Omega - \Omega_{cr})}{(\rho_{s}-\rho_{w})gh/\rho_{s}}
\end{equation}
where $\Omega$ is the stream power in units of $[M/^{3}]$:
\begin{equation}
\Omega = \rho_{w}ghS_{f}\sqrt{u^2+v^2}
\end{equation}
\begin{equation}
S_{f} = n^{2}(u^{2}+v^{2})h^{-4/3}
\end{equation}
and $\Omega_{cr}$ is the critical stream power, below which soil entrainment or reentrainment do not occur.
$F$ is the effective fraction of excess stream power in entrainment or reentrainment to account for energy dissipation due to heat.
$J$ is the specific energy of entrainment, for example, the energy required for soil to be entrained per unit mass of sediment $[ML^{2}/T^{2}/M]$.
$\rho_{w}$ is the density of water.
The deposition rate for a sediment class $i$ can be calculated as:
\begin{equation}
d_{i} = v_{i}c_{i}
\end{equation}
where $v_{i}$ represents the settling velocity of each sediment class $[L/T]$. Please note the assumptions behind this equation:

    *   The suspended load in the water column is completely mixed in the vertical direction.
    *   Infiltration rate does not affect settling velocities.

Coupling H-R equation with Shallow Water Equations lead to:

\begin{eqnarray}
\label{eqn:cpeqns}
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{E}}{\partial t} + \frac{\partial \mathbf{G}}{\partial t} &=& \mathbf{S}\\
\frac{\partial \mathbf{M}}{\partial t} &=& \mathbf{D}
\end{eqnarray}

where $\mathbf{U}$ is the conservative variable vector, $\mathbf{E}$ and $\mathbf{G}$ are the flux vectors in x and y direction, respectively, 
$\mathbf{S}$ is the source vector, $\mathbf{M}$ is a deposited mass vector, and $\mathbf{D}$ is the net deposition vector:

\begin{align}
\mathbf{U} 
=
    \begin{bmatrix}
    h      \\[.5em]
    uh     \\[.5em]
    vh     \\
    c_{1}h \\
    \vdots \\
    c_{I}h
    \end{bmatrix}
\end{align}

\begin{align}
\mathbf{E} 
=
    \begin{bmatrix}
    uh                           \\[.5em]
    u^{2}h+\frac{1}{2}gh^{2}     \\[.5em]
    uvh                          \\
    c_{1}uh                      \\
    \vdots                       \\
    c_{I}uh
    \end{bmatrix}
\end{align}

\begin{align}
\mathbf{G} 
=
    \begin{bmatrix}
    vh                           \\[.5em]
    uvh                          \\[.5em]
    v^{2}h+\frac{1}{2}gh^{2}     \\
    c_{1}vh                      \\
    \vdots                       \\
    c_{I}vh
    \end{bmatrix}
\end{align}

\begin{align}
\mathbf{S} 
=
    \begin{bmatrix}
    S_{r}                                                           \\[.5em]
    -gh\frac{\partial z_{b}}{\partial x} - C_{D}u\sqrt{u^{2}+v^{2}} \\[.5em]
    -gh\frac{\partial z_{b}}{\partial y} - C_{D}v\sqrt{u^{2}+v^{2}} \\
    e_{1}+e_{r1}+r_{1}+r_{r1}-d_{1}                                 \\
    \vdots                                                          \\
    e_{I}+e_{rI}+r_{I}+r_{rI}-d_{I} 
    \end{bmatrix}
\end{align}

\begin{align}
\mathbf{M} 
=
    \begin{bmatrix}
    M_{1}    \\[.5em]
    \vdots   \\
    M_{I} 
    \end{bmatrix}
\end{align}

\begin{align}
\mathbf{D} 
    =
    \begin{bmatrix}
    d_{1}-e_{r1}-r_{r1} \\[.5em]
    \vdots              \\
    d_{I}-e_{rI}-r_{rI}  
    \end{bmatrix}
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
