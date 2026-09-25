# Shallow Water Equations Coupled with the Heat Equation

RDycore can carry a depth-averaged water temperature alongside the
[shallow water equations](swe.md). The temperature is advected by the flow and
exchanges energy with the atmosphere through the water surface. The two
processes are advanced with an operator split: a transport solve that advances
the flow and carries the heat content as a passive tracer, followed by a source
solve that holds the flow fixed and applies the surface energy exchange.

## Governing equations

### Depth-averaged shallow water equations

The flow is governed by the two-dimensional shallow water equations in
conservative form,

$$
\frac{\partial \mathbf{U}_{\text{sw}}}{\partial t}
+ \frac{\partial \mathbf{E}_{\text{sw}}}{\partial x}
+ \frac{\partial \mathbf{G}_{\text{sw}}}{\partial y}
= \mathbf{S}_r + \mathbf{S}_b + \mathbf{S}_f,
\tag{1}
$$

with the flow solution vector and flux vectors

$$
\mathbf{U}_{\text{sw}} = \begin{bmatrix} h \\[.4em] hu \\[.4em] hv \end{bmatrix},
\qquad
\mathbf{E}_{\text{sw}} = \begin{bmatrix} hu \\[.4em] hu^2 + \tfrac{1}{2}gh^2 \\[.4em] huv \end{bmatrix},
\qquad
\mathbf{G}_{\text{sw}} = \begin{bmatrix} hv \\[.4em] huv \\[.4em] hv^2 + \tfrac{1}{2}gh^2 \end{bmatrix},
\tag{2}
$$

and the source vectors representing net runoff production, the bed elevation
slope, and bed friction,

$$
\mathbf{S}_r = \begin{bmatrix} Q_r \\[.4em] 0 \\[.4em] 0 \end{bmatrix},
\qquad
\mathbf{S}_b = \begin{bmatrix} 0 \\[.4em] -gh\dfrac{\partial z}{\partial x} \\[.6em] -gh\dfrac{\partial z}{\partial y} \end{bmatrix},
\qquad
\mathbf{S}_f = \begin{bmatrix} 0 \\[.4em] -C_D\,u\sqrt{u^2 + v^2} \\[.4em] -C_D\,v\sqrt{u^2 + v^2} \end{bmatrix}.
\tag{3}
$$

Here $h$ [m] is the flow depth, $\vec{u} = (u, v)$ [m s⁻¹] is the
depth-averaged velocity, $g$ is the acceleration due to gravity, $z$ is the bed
elevation, $Q_r$ is the net runoff production expressed as water height per unit
time, and $C_D = g n^2 h^{-1/3}$ is the drag coefficient formed from the Manning
roughness coefficient $n$. The [shallow water equations](swe.md) page derives
this system, its finite-volume discretization, and the Roe solver used to
evaluate the fluxes; everything below takes it as given.

### Prognostic heat variable

The prognostic heat variable is the conservative quantity $hT$, the product of
the water depth $h$ [m] and the depth-averaged water temperature $T$ [°C], with
units of m·°C. It is appended to $(2)$ as a trailing tracer component, with the
corresponding advective entries appended to the flux vectors:

$$
\mathbf{U} = \begin{bmatrix} h \\[.4em] hu \\[.4em] hv \\[.4em] hT \end{bmatrix},
\qquad
\mathbf{E} = \begin{bmatrix} hu \\[.4em] hu^2 + \tfrac{1}{2}gh^2 \\[.4em] huv \\[.4em] huT \end{bmatrix},
\qquad
\mathbf{G} = \begin{bmatrix} hv \\[.4em] huv \\[.4em] hv^2 + \tfrac{1}{2}gh^2 \\[.4em] hvT \end{bmatrix}.
\tag{4}
$$

When sediment classes or salinity are also active, their components precede $hT$
in the solution vector; the heat component is always the last tracer.

### Transport with surface energy exchange

The depth-integrated heat equation is

$$
\frac{\partial (hT)}{\partial t} + \vec{\nabla}\cdot\left(hT\,\vec{u}\right)
= \frac{Q_{\text{net}}}{\rho_w c_w},
\tag{5}
$$

where $\vec{u} = (u, v)$ is the depth-averaged flow velocity, $\rho_w$ is the
density of water, $c_w$ is the specific heat capacity of water, and
$Q_{\text{net}}$ [W m⁻²] is the net heat flux through the water surface, taken
positive into the water. Dividing the surface flux by $\rho_w c_w$ converts it
to a tendency of $hT$; note that it is independent of $h$, because the flux is
applied to the water column as a whole rather than per unit depth.

The model neglects horizontal diffusion of heat, heat exchange with the bed,
and any heat carried by runoff or by lateral inflows: the only non-advective
term in $(5)$ is the surface exchange.

### Net surface heat flux

$Q_{\text{net}}$ is the sum of four contributions,

$$
Q_{\text{net}}(T) = Q_{\text{sw}} + Q_{\text{lw}}(T) + Q_{\text{sh}}(T) + Q_{e}(T),
\tag{6}
$$

which are, respectively, the absorbed shortwave radiation, the net longwave
radiation, the sensible heat flux, and the latent heat flux:

$$
\begin{aligned}
Q_{\text{sw}} &= (1 - \alpha)\, S^{\downarrow}, \\[.5em]
Q_{\text{lw}} &= L^{\downarrow} - \varepsilon\,\sigma\,T_K^4, \\[.5em]
Q_{\text{sh}} &= \rho_a\, c_{p,a}\, C_H \left(T_a - T\right), \\[.5em]
Q_{e} &= \rho_a\, L_v\, C_H \left(q_a - q_{\text{sat}}(T)\right).
\end{aligned}
\tag{7}
$$

Here $T_K = T + T_0$ is the water temperature in kelvin, and the five
atmospheric forcing variables, supplied per cell, are

* $S^{\downarrow}$, the downwelling shortwave radiation [W m⁻²]
* $L^{\downarrow}$, the downwelling longwave radiation [W m⁻²]
* $U_w$, the wind speed [m s⁻¹]
* $T_a$, the air temperature [°C]
* $q_a$, the specific humidity of the air [kg kg⁻¹].

The turbulent fluxes $Q_{\text{sh}}$ and $Q_e$ are parameterized with a bulk
transfer velocity (the reciprocal of an aerodynamic resistance) that increases
linearly with wind speed,

$$
C_H = 0.2 + 0.1\, U_w \qquad [\text{m s}^{-1}],
\tag{8}
$$

with the constants carrying the units needed to make $C_H$ a velocity.

The saturation specific humidity at the water surface follows from the
Magnus–Tetens approximation to the saturation vapor pressure $e_{\text{sat}}$
[Pa]:

$$
e_{\text{sat}}(T) = 611.2 \exp\!\left(\frac{17.67\,T}{T + 243.5}\right),
\qquad
q_{\text{sat}}(T) = \frac{\epsilon_v\, e_{\text{sat}}(T)}{p_0 - (1 - \epsilon_v)\, e_{\text{sat}}(T)},
\tag{9}
$$

where $T$ is in degrees Celsius.

#### Constants

| Symbol | Value | Units | Description |
|--------|-------|-------|-------------|
| $\alpha$ | 0.08 | — | albedo of water |
| $\varepsilon$ | 0.97 | — | emissivity of water |
| $\sigma$ | $5.670374419\times10^{-8}$ | W m⁻² K⁻⁴ | Stefan–Boltzmann constant |
| $\rho_a$ | 1.225 | kg m⁻³ | density of air |
| $c_{p,a}$ | 1005 | J kg⁻¹ K⁻¹ | specific heat of air at constant pressure |
| $L_v$ | $2.5\times10^{6}$ | J kg⁻¹ | latent heat of vaporization |
| $\rho_w$ | 1000 | kg m⁻³ | density of water |
| $c_w$ | 4186 | J kg⁻¹ K⁻¹ | specific heat of water |
| $p_0$ | 101325 | Pa | standard air pressure |
| $\epsilon_v$ | 0.622 | — | ratio of molar masses of water vapor and dry air |
| $T_0$ | 273.15 | K | Celsius-to-kelvin offset |

### Evaporative mass and momentum exchange

The latent flux $Q_e$ moves water as well as energy: the mass it carries across
the surface leaves the water column. Written as a rate of change of depth,

$$
\dot{h}_e = \frac{Q_e}{\rho_w L_v},
\tag{10}
$$

which is negative while the cell evaporates and positive under condensation. In
the more familiar form, the evaporation rate is
$E = -\dot{h}_e = \rho_a C_H\left(q_{\text{sat}}(T) - q_a\right)/\rho_w$.

This adds a source vector to the coupled system, alongside the $\mathbf{S}_r$,
$\mathbf{S}_b$, and $\mathbf{S}_f$ of $(3)$:

$$
\mathbf{S}_e =
\begin{bmatrix} \dot{h}_e \\[.4em] u\,\dot{h}_e \\[.4em] v\,\dot{h}_e \\[.4em] 0 \end{bmatrix}.
\tag{11}
$$

The momentum entries carry the mass away at the local flow velocity, which
leaves $u = hu/h$ and $v = hv/h$ unchanged: evaporation thins the water column
without accelerating or retarding it. Omitting them would hold $hu$ fixed while
$h$ fell, spuriously accelerating a drying cell. RDycore uses the same
convention under condensation, taking the arriving vapor to join the flow at its
velocity rather than at rest.

The heat entry is zero, so $hT$ is conserved under the mass exchange and the
water that remains carries the heat of the water that left — $T = hT/h$ rises as
the cell thins. (The alternative convention removes the sensible enthalpy of the
departing vapor as well, adding $-T\dot{h}_e$ to the heat row and making $T$
independent of the mass loss; RDycore does not do this.) Sediment and salinity
are treated the same way, and so concentrate as water evaporates.

### Capping evaporation in shallow water

Over an implicit step of length $\Delta t$ the depth changes by
$Q_e\Delta t/(\rho_w L_v)$, so an evaporative demand exceeding the water present
would drive $h$ negative — and a negative depth propagates immediately into
$T = hT/h$ and into the next flow solve. RDycore therefore bounds the latent flux
below by the water available above the wet/dry threshold $h_{\min}$:

$$
Q_e \;\longleftarrow\; \max\!\left(Q_e,\; -\frac{\left(h - h_{\min}\right)\rho_w L_v}{\Delta t}\right).
\tag{12}
$$

The bound is non-positive and vanishes at $h = h_{\min}$, so condensation is
never limited. It applies to the energy budget as well as the mass budget —
latent heat is removed only for water that actually leaves — so a capped cell
also cools more slowly than the unlimited parameterization would have it.

Because the bound is enforced implicitly, a capped cell approaches $h_{\min}$
without reaching it. Substituting the bound, evaluated at the new state, into a
backward Euler step gives $h^{n+1} = h^{n} - \left(h^{n+1} - h_{\min}\right)$, so

$$
h^{n+1} = \tfrac{1}{2}\left(h^{n} + h_{\min}\right).
\tag{13}
$$

The excess over $h_{\min}$ halves on each capped step and the cell stays strictly
wet. The cap is not smooth where it activates, so Newton may need an extra
iteration on a cell that crosses it.

### Wetting and drying

The surface exchange acts only on cells holding water. Writing $h_{\min}$ for
the wet/dry threshold `physics.flow.tiny_h`, both the heat source of $(5)$ and the
mass and momentum sinks of $(11)$ are applied as

$$
\left(\frac{\partial \mathbf{U}}{\partial t}\right)_{\text{surface}} =
\begin{cases}
\mathbf{S}_e + \dfrac{Q_{\text{net}}(hT/h)}{\rho_w c_w}\,\mathbf{e}_{hT}, & h \ge h_{\min} \\[1em]
\mathbf{0}, & h < h_{\min},
\end{cases}
\tag{14}
$$

where $\mathbf{e}_{hT}$ is the unit vector selecting the heat component. Below the
threshold the cell exchanges nothing at all: its depth and heat content are both
frozen, so no temperature is derived from a vanishing depth and no water is drawn
from a cell that has none. This branch makes the residual discontinuous at
$h = h_{\min}$, which is worth knowing when comparing an analytic Jacobian against
finite differences near the threshold.

## Operator splitting

Over one coupling interval $[t^n, t^{n+1}]$ of length $\Delta t$, RDycore
advances $(5)$ with a first-order Lie split into a transport step and a source
step.

**Step 1 — transport.** The full shallow water system $(4)$ is advanced by the
flow `TS`, with $hT$ carried as a passive tracer and no surface exchange:

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{E}}{\partial x}
+ \frac{\partial \mathbf{G}}{\partial y} = \mathbf{S}_r + \mathbf{S}_b + \mathbf{S}_f,
\qquad \mathbf{U}(t^n) = \mathbf{U}^n
\;\;\longrightarrow\;\; \mathbf{U}^{*}.
\tag{15}
$$

The shallow water source terms $\mathbf{S}_r$, $\mathbf{S}_b$, and
$\mathbf{S}_f$ are those defined in the [shallow water equations](swe.md); the
heat component of the external source vector is zero in this step. The flow
`TS` may take several internal timesteps within the coupling interval.

**Step 2 — surface exchange.** The transported state is corrected by the surface
exchange, which introduces no spatial coupling and changes each cell through the
cell-local system

$$
\frac{d}{dt}
\begin{bmatrix} h \\[.4em] hu \\[.4em] hv \\[.4em] hT \end{bmatrix}
=
\begin{bmatrix}
\dot{h}_e \\[.4em] u\,\dot{h}_e \\[.4em] v\,\dot{h}_e \\[.4em] Q_{\text{net}}\!\left(hT/h\right)/(\rho_w c_w)
\end{bmatrix},
\qquad \mathbf{U}(t^n) = \mathbf{U}^{*}
\;\;\longrightarrow\;\; \mathbf{U}^{n+1}.
\tag{16}
$$

Note that this step is not confined to the heat DOF: because evaporation removes
water, it also moves $h$, $hu$, and $hv$. When the surface flux is instead
prescribed directly — the production `heat_flux` input, or the manufactured
source used by the MMS driver — there is no latent component to separate out, so
only the $hT$ row is active and the flow is left untouched.

This step runs on its own `TS` (options prefix `heat_`), which by default takes
a single backward Euler step across the whole coupling interval. The `TS`
operates on the complete state vector, with the implicit residual

$$
\mathbf{F}(\mathbf{U}, \dot{\mathbf{U}}) =
\begin{cases}
\dot{\mathbf{U}} - \mathbf{S}_e - \dfrac{Q_{\text{net}}\!\left(U_{hT}/h\right)}{\rho_w c_w}\,\mathbf{e}_{hT},
  & h \ge h_{\min} \\[1em]
\dot{\mathbf{U}}, & h < h_{\min},
\end{cases}
\tag{17}
$$

so that every component named by neither $\mathbf{S}_e$ nor $\mathbf{e}_{hT}$ —
the sediment and salinity tracers — carries the trivial residual $\dot{U}_c = 0$
and is left unchanged by the solve.

One property of $(17)$ is worth recording, because it is exactly what makes the
momentum rows safe. Backward Euler on the first two rows gives
$(hu)^{n+1} = (hu)^{*} + \Delta t\,u^{n+1}\dot{h}_e$ with $u^{n+1} = (hu)^{n+1}/h^{n+1}$,
and since $\Delta t\,\dot{h}_e = h^{n+1} - h^{*}$ the bracket collapses to
$h^{*}/h^{n+1}$, leaving

$$
u^{n+1} = u^{*}
$$

exactly, for any $\dot{h}_e$ and so independently of the parameterization that
produced it. The discrete scheme inherits the continuous statement that
evaporation does not accelerate the flow.

### Jacobian

The residual $(17)$ is pointwise, so its Jacobian is block diagonal with one dense
block per cell. It is no longer a scalar diagonal, though: evaporation couples the
$h$, $hu$, $hv$, and $hT$ rows, both through $T = hT/h$ and through the depth
dependence of the cap. With $\varsigma$ the `TS` shift
$\partial\dot{\mathbf{U}}/\partial\mathbf{U}$, the nonzero entries are

$$
\begin{aligned}
\frac{\partial F_h}{\partial h} &= \varsigma - \frac{1}{\rho_w L_v}\frac{\partial Q_e}{\partial h},
&\qquad
\frac{\partial F_h}{\partial (hT)} &= -\frac{1}{\rho_w L_v}\frac{\partial Q_e}{\partial (hT)}, \\[.6em]
\frac{\partial F_{hu}}{\partial h} &= \frac{u}{\rho_w L_v}\left(\frac{Q_e}{h} - \frac{\partial Q_e}{\partial h}\right),
&\qquad
\frac{\partial F_{hu}}{\partial (hu)} &= \varsigma - \frac{Q_e}{\rho_w L_v\,h}, \\[.6em]
\frac{\partial F_{hu}}{\partial (hT)} &= -\frac{u}{\rho_w L_v}\frac{\partial Q_e}{\partial (hT)},
&\qquad
\frac{\partial F_{hT}}{\partial h} &= -\frac{1}{\rho_w c_w}\frac{\partial Q_{\text{net}}}{\partial h}, \\[.6em]
\frac{\partial F_{hT}}{\partial (hT)} &= \varsigma - \frac{1}{\rho_w c_w}\frac{\partial Q_{\text{net}}}{\partial (hT)},
&&
\end{aligned}
\tag{18}
$$

with the $hv$ row following the $hu$ row under $u \rightarrow v$, and every other
diagonal entry equal to $\varsigma$.

The two branches of the cap enter only through the derivatives of $Q_e$, which is
why the net flux is carried in the code as a latent piece plus a non-latent
remainder $Q_{\text{nl}} = Q_{\text{sw}} + Q_{\text{lw}} + Q_{\text{sh}}$. Using
$\partial T/\partial h = -T/h$ and $\partial T/\partial (hT) = 1/h$,

$$
\begin{aligned}
\text{uncapped:} &\quad
\frac{\partial Q_e}{\partial h} = -\frac{T}{h}\frac{d Q_e}{d T},
&\qquad
\frac{\partial Q_e}{\partial (hT)} &= \frac{1}{h}\frac{d Q_e}{d T}, \\[.6em]
\text{capped:} &\quad
\frac{\partial Q_e}{\partial h} = -\frac{\rho_w L_v}{\Delta t},
&\qquad
\frac{\partial Q_e}{\partial (hT)} &= 0,
\end{aligned}
\tag{19}
$$

the capped branch being linear in $h$ alone, so the temperature derivative drops
out of it entirely. In both branches
$\partial Q_{\text{net}}/\partial h = -(T/h)\,dQ_{\text{nl}}/dT + \partial Q_e/\partial h$
and
$\partial Q_{\text{net}}/\partial (hT) = (1/h)\,dQ_{\text{nl}}/dT + \partial Q_e/\partial (hT)$,
with the component derivatives

$$
\frac{d Q_{\text{nl}}}{d T} =
-4\,\varepsilon\,\sigma\,T_K^3 - \rho_a\, c_{p,a}\, C_H,
\qquad
\frac{d Q_e}{d T} = -\rho_a\, L_v\, C_H \, \frac{d q_{\text{sat}}}{d T},
\tag{20}
$$

$$
\frac{d q_{\text{sat}}}{d T} =
\frac{\epsilon_v\, p_0}{\left[p_0 - (1-\epsilon_v)\, e_{\text{sat}}\right]^2}\;
\frac{d e_{\text{sat}}}{d T},
\qquad
\frac{d e_{\text{sat}}}{d T} = \frac{17.67 \cdot 243.5}{\left(T + 243.5\right)^2}\, e_{\text{sat}}.
\tag{21}
$$

Every term in $(20)$ is negative, so the surface exchange is unconditionally
damping in $T$ and the implicit solve is well conditioned. Because the Jacobian is
block diagonal, it is preallocated with a COO pattern of one dense block per cell
rather than the wider finite-volume stencil, which also lets the libCEED backend
compute the blocks on the device and hand them to `MatSetValuesCOO()` without a
host round trip.

The split is first-order accurate in $\Delta t$ irrespective of the accuracy of
either sub-step, since the transport and source vector fields do not commute in
general.

## Spatial discretization of the heat flux

The heat flux terms in $(4)$ are discretized exactly like any other passive
tracer. At a face between cells $i$ and $j$ the Roe eigen-system of the shallow
water equations is extended by one contact wave per tracer, travelling at the
Roe-averaged normal velocity $\hat{u}_\parallel$, with the Roe-averaged
temperature

$$
\hat{T} = \frac{\sqrt{h_i}\,T_i + \sqrt{h_j}\,T_j}{\sqrt{h_i} + \sqrt{h_j}}
\tag{22}
$$

and wave strength $\Delta(hT) - \hat{T}\,\Delta h$. The heat component of the
left and right physical fluxes is $h\,u_\parallel T$ evaluated on each side, and
the numerical flux is assembled as in the [shallow water equations](swe.md).
Setting `numerics.riemann: upwind_roe` selects a variant that keeps the Roe flux
for the flow components but takes the heat component from the upwind side, as
determined by the sign of the Roe mass flux. Faces where both sides are dry carry
zero flux in every component.

Boundary conditions for heat are prescribed as a Dirichlet water temperature on
the boundary face, from which the boundary value of $hT$ is formed using the
boundary depth. Second-order MUSCL reconstruction is not supported when heat (or
any other tracer) is active.

## Verification with the Method of Manufactured Solutions

The heat coupling is verified with the
[Method of Manufactured Solutions](../common/mms.md) using the `rdycore_mms`
driver. The manufactured solution supplies $h$, $u$, $v$, $z$, $n$ (in the `swe`
subsection of the input) and $T$ together with its partial derivatives (in the
`temperature` subsection).

### The manufactured heat source

Substituting the manufactured $h$, $u$, $v$, and $T$ into $(5)$ leaves a
residual that is installed as a prescribed per-cell surface flux,

$$
Q_{\text{mms}}(x, y, t) = \rho_w c_w \left[
\frac{\partial (hT)}{\partial t} + \frac{\partial (huT)}{\partial x}
+ \frac{\partial (hvT)}{\partial y} \right],
\tag{23}
$$

expanded with the product rule into the quantities the input file provides:

$$
\begin{aligned}
Q_{\text{mms}} = \rho_w c_w \Big[\;
& h\,T_t + T\,h_t \\
&+ h\,u\,T_x + T\,h\,u_x + T\,u\,h_x \\
&+ h\,v\,T_y + T\,h\,v_y + T\,v\,h_y \;\Big].
\end{aligned}
\tag{24}
$$

In this mode the heat solve takes $Q_{\text{mms}}$ in place of
$Q_{\text{net}}(T)$: the prescribed source **replaces** the atmospheric
parameterization rather than correcting it.

### How the correction is allocated across the split

The unsplit manufactured correction for the heat equation is the complete
conservative residual in $(23)$. Consistency requires the two sub-steps to
contribute that residual exactly once between them, so RDycore allocates it as

* **transport step**: the full manufactured $h$, $hu$, and $hv$ sources, and
  *nothing* for heat — so the numerical tracer flux performs the whole $hT$
  transport rather than having it cancelled analytically;
* **heat step**: the complete residual $Q_{\text{mms}}$, consumed by the
  prescribed-source branch.

To leading order the composite update is then

$$
(hT)^{n+1} = (hT)^{n} + \Delta t\,\frac{\partial (hT)}{\partial t}
+ \Delta t \left[ \mathcal{D}(hT) - \mathcal{D}_h(hT) \right] + O(\Delta t^2),
\tag{25}
$$

with $\mathcal{D}$ the analytic flux divergence and $\mathcal{D}_h$ its discrete
counterpart, so the measured error is exactly the spatial truncation error of the
tracer flux plus the temporal and splitting error. Installing a heat source in
the transport step as well would count $\partial (hT)/\partial t$ twice and halve
the observed order; the MMS driver asserts that the heat component of the
transport external source vector is identically zero.

The source is sampled with the quadrature matching the heat `TS`: the right
endpoint $Q_{\text{mms}}(t^{n+1})$ for backward Euler, and the endpoint average
$\tfrac{1}{2}\left[Q_{\text{mms}}(t^n) + Q_{\text{mms}}(t^{n+1})\right]$ for
Crank–Nicolson.

### Reported error norms

Heat-enabled MMS runs report two rows:

* `hT`, the error in the conservative variable the solver actually advances;
* `T`, the error in the derived temperature $hT/h$, guarded by $h_{\min}$
  exactly as the operators guard it, and compared against the manufactured $T$
  at the same cell centroids.

The derived temperature combines — and can partially cancel — the errors in $h$
and $hT$, so it generally converges at a slightly different rate and carries its
own `expected_rates` entry. The two coincide only when $h \equiv 1$.

The driver also reports `Max-|Q_mms|-inf`, the running maximum of
$\|Q_{\text{mms}}\|_\infty$, so that a manufactured construction intended to be
source-free can be checked directly rather than inferred from the final error.

### Test cases

The cases in `driver/tests/heat/` are built so that each isolates one source of
error.

| Case | Input | Flow | Temperature | What it verifies |
|------|-------|------|-------------|------------------|
| 0 | `heat_ts_mms.yaml` | at rest, $h \equiv 1$ | $T = R + A\sin(Kx)\sin(Ky)e^{t/\tau}$ | The source quadrature in isolation: no transport, so the spatial error is identically zero. Convergence in $\Delta t$ on a fixed mesh is $O(\Delta t)$ for backward Euler and $O(\Delta t^2)$ for Crank–Nicolson. Because $h \equiv 1$, the `hT` and `T` errors coincide. |
| 1 | `heat_mms_moving_transport.yaml` | uniform, flat bed, $n = 0$ | $T = R + A\sin(K(x - Ut))\sin(K(y - Vt))$ | Passive transport alone. The wave translates with the water, so $(hT)_t + \nabla\cdot(hT\vec{u}) = 0$ and $Q_{\text{mms}}$ is zero to roundoff; there is no source field and hence no splitting error. |
| 2 | `heat_mms_moving_source.yaml` | uniform, flat bed, $n = 0$ | $T = R + A e^{\Lambda t}\sin(K(x - Ut))\sin(K(y - Vt))$ | Transport plus a genuinely nonzero source. The phase-advection terms cancel analytically and $Q_{\text{mms}} = \rho_w c_w H \Lambda A e^{\Lambda t}\sin(K(x-Ut))\sin(K(y-Vt))$, which varies in space — so the source and transport fields no longer commute and this is the first case with a splitting error. |
| 3 | `heat_mms_conv_study.yaml` | spatially varying $h$, $u$, $v$, $z$, $n$ | $T = A(1 + \sin(Kx)\sin(Ky))e^{t/\tau}$ | The complete coupled system: manufactured SWE sources in the transport step and the full conservative heat residual in the heat step, on both backends. |

Case 1 requires $n = 0$ rather than merely benefiting from it: bed friction is
discretized semi-implicitly against the partially updated momentum, so it does
not cancel the analytic $C_D \vec{u}|\vec{u}|$ that the manufactured source
installs. Both are identically zero only when $n = 0$, which is what lets the
uniform flow be preserved to roundoff.

Two further tests run through the production driver rather than the MMS driver.
`heat_coupling_interval.yaml` holds a lake at rest under a constant prescribed
surface flux, for which $(16)$ integrates exactly to
$\Delta T = Q\,\Delta t / (\rho_w c_w h)$, and checks that increment while the
transport timestep is shorter than the coupling interval — so that the heat solve
is confirmed to advance over the full interval rather than over one transport
step. `heat_coupling_interval_atmospheric.yaml` drives the same configuration
with the five-parameter forcing of $(7)$, exercising $Q_{\text{net}}(T)$ and the
nonlinear Jacobian $(18)$–$(21)$ on both backends; because $Q_{\text{net}}(T)$ is
nonlinear there is no closed-form final temperature to assert against, so that
one is a smoke test.

`heat_evaporation.yaml` covers the mass and momentum sinks and the cap. It calls
`RDyHeatAdvance()` directly rather than `RDyAdvance()`, writing the state it wants
beforehand, so the transport solve never runs and every change in $h$, $hu$, and
$hv$ is attributable to the surface exchange alone. That isolation is what lets it
assert the two exact identities derived above rather than regression values: that
deep water evaporating into dry air leaves $u$ and $v$ unchanged to roundoff, and
that a film holding $10^{-6}$ m above $h_{\min}$ against a demand three orders of
magnitude larger lands on $(13)$ exactly and stays strictly wet. Removing the
momentum rows breaks the first; removing the cap drives $h$ negative and the
nonlinear solve diverges outright.

### Rates, timestep refinement, and self-convergence

Both the tracer discretization and the default forward Euler transport
integrator are first order, so with $\Delta t \propto \Delta x$ the error behaves
as $E \sim C_x \Delta x + C_t \Delta t = (C_x + K C_t)\Delta x$. What the studies
report is therefore a **joint space-time rate** near 1, not an isolated spatial
rate; the `timestep_refinement_exponent` input controls this coupling. Case 1
also recovers an isolated spatial rate by raising the transport integrator to
RK4, which is legitimate there only because that case has neither a splitting
error nor a manufactured SWE source.

Refining $\Delta t$ on a fixed mesh does **not** drive the error against the
exact solution to zero once transport is active: the solution converges to the
semi-discrete solution, whose distance from the exact solution is the $O(\Delta x)$
spatial truncation error, so such a study plateaus on that floor and reports a
rate near zero. Case 2 therefore measures the temporal-plus-splitting order by
self-convergence against a same-mesh, fine-timestep reference solution
(`-mms_save_final_state` and `-mms_reference_solution`), which cancels the
spatial floor exactly. The measured result is first order for *both* backward
Euler and Crank–Nicolson: Lie splitting is first order however accurately the
source step is integrated, so Crank–Nicolson lowers the error constant without
raising the order.

### Scope

The MMS path drives the heat solve through the prescribed-source branch, which
replaces $Q_{\text{net}}(T)$. These cases therefore verify passive transport of
$hT$, the manufactured source quadrature, and the Lie composition of the two
solves — but they do **not** verify the atmospheric parameterization $(6)$–$(9)$
or its analytic Jacobian $(18)$–$(21)$. A consequence of the same substitution is
that in this path the heat residual has no state dependence, so every consistent
one-step method produces the same update and the heat `TS` type selects only
which manufactured quadrature is sampled.

The evaporative mass and momentum sinks are outside the MMS path for the same
reason: a prescribed net flux has no latent component to separate, so
$\mathbf{S}_e$ is identically zero there and the manufactured solution never has
to account for it. The manufactured source $(23)$ is consistent as it stands, and
the sinks are covered by the separate tests described below rather than by a
convergence study.

## References

* [Bradford, S. F., & Sanders, B. F. (2002). Finite-volume model for shallow-water flooding of arbitrary topography. Journal of hydraulic engineering, 128(3), 289-298.](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%282002%29128%3A3%28289%29)

* [Roache, P. J. (2002). Code verification by the method of manufactured solutions. Journal of Fluids Engineering, 124(1), 4-10.](https://doi.org/10.1115/1.1436090)
