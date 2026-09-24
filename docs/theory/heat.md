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

### Wetting and drying

The surface exchange acts only on cells holding water. Writing $h_{\min}$ for
the wet/dry threshold `physics.flow.tiny_h`, the source term in $(5)$ is applied
as

$$
\frac{\partial (hT)}{\partial t}\bigg|_{\text{source}} =
\begin{cases}
\dfrac{Q_{\text{net}}(hT/h)}{\rho_w c_w}, & h \ge h_{\min} \\[1em]
0, & h < h_{\min}.
\end{cases}
\tag{10}
$$

Below the threshold the heat content of the cell is frozen, so no temperature is
derived from a vanishing depth.

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
\tag{11}
$$

The shallow water source terms $\mathbf{S}_r$, $\mathbf{S}_b$, and
$\mathbf{S}_f$ are those defined in the [shallow water equations](swe.md); the
heat component of the external source vector is zero in this step. The flow
`TS` may take several internal timesteps within the coupling interval.

**Step 2 — surface exchange.** The flow state is held fixed and only $hT$
evolves, according to the cell-local ordinary differential equation

$$
\frac{d (hT)}{d t} = \frac{Q_{\text{net}}\!\left(hT/h\right)}{\rho_w c_w},
\qquad (hT)(t^n) = (hT)^{*}
\;\;\longrightarrow\;\; (hT)^{n+1}.
\tag{12}
$$

This step runs on its own `TS` (options prefix `heat_`), which by default takes
a single backward Euler step across the whole coupling interval. The `TS`
operates on the complete state vector, with the implicit residual

$$
\mathbf{F}(\mathbf{U}, \dot{\mathbf{U}}) =
\begin{cases}
\dot{U}_c - \dfrac{Q_{\text{net}}\!\left(U_{hT}/h\right)}{\rho_w c_w},
  & c = hT \text{ and } h \ge h_{\min} \\[1em]
\dot{U}_c, & \text{otherwise,}
\end{cases}
\tag{13}
$$

so that every component other than $hT$ carries the trivial residual
$\dot{U}_c = 0$ and is left unchanged by the solve.

Because $Q_{\text{net}}$ depends on $T$ alone, and $T$ is cell-local, the
residual $(13)$ is pointwise and its Jacobian is exactly diagonal. With $\varsigma$
the `TS` shift $\partial\dot{\mathbf{U}}/\partial\mathbf{U}$, the diagonal entry
of the heat DOF is

$$
J_{hT,hT} = \varsigma - \frac{1}{\rho_w c_w\, h}\,\frac{d Q_{\text{net}}}{d T},
\tag{14}
$$

where the factor $1/h$ comes from $\partial T/\partial (hT) = 1/h$, and

$$
\frac{d Q_{\text{net}}}{d T} =
-4\,\varepsilon\,\sigma\,T_K^3
- \rho_a\, c_{p,a}\, C_H
- \rho_a\, L_v\, C_H \, \frac{d q_{\text{sat}}}{d T},
\tag{15}
$$

$$
\frac{d q_{\text{sat}}}{d T} =
\frac{\epsilon_v\, p_0}{\left[p_0 - (1-\epsilon_v)\, e_{\text{sat}}\right]^2}\;
\frac{d e_{\text{sat}}}{d T},
\qquad
\frac{d e_{\text{sat}}}{d T} = \frac{17.67 \cdot 243.5}{\left(T + 243.5\right)^2}\, e_{\text{sat}}.
\tag{16}
$$

Every term in $(15)$ is negative, so the surface exchange is unconditionally
damping in $T$ and the implicit solve is well conditioned. All remaining
diagonal entries are $\varsigma$. Because the Jacobian is diagonal, it is
preallocated with a one-entry-per-row COO pattern rather than the wider
finite-volume stencil, which also lets the libCEED backend assemble it on the
device with `MatSetValuesCOO()`.

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
\tag{17}
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
\tag{18}
$$

expanded with the product rule into the quantities the input file provides:

$$
\begin{aligned}
Q_{\text{mms}} = \rho_w c_w \Big[\;
& h\,T_t + T\,h_t \\
&+ h\,u\,T_x + T\,h\,u_x + T\,u\,h_x \\
&+ h\,v\,T_y + T\,h\,v_y + T\,v\,h_y \;\Big].
\end{aligned}
\tag{19}
$$

In this mode the heat solve takes $Q_{\text{mms}}$ in place of
$Q_{\text{net}}(T)$: the prescribed source **replaces** the atmospheric
parameterization rather than correcting it.

### How the correction is allocated across the split

The unsplit manufactured correction for the heat equation is the complete
conservative residual in $(18)$. Consistency requires the two sub-steps to
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
\tag{20}
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
surface flux, for which $(12)$ integrates exactly to
$\Delta T = Q\,\Delta t / (\rho_w c_w h)$, and checks that increment while the
transport timestep is shorter than the coupling interval — so that the heat solve
is confirmed to advance over the full interval rather than over one transport
step. `heat_coupling_interval_atmospheric.yaml` drives the same configuration
with the five-parameter forcing of $(7)$, exercising $Q_{\text{net}}(T)$ and the
nonlinear Jacobian $(14)$–$(16)$ on both backends; because $Q_{\text{net}}(T)$ is
nonlinear there is no closed-form final temperature to assert against, so that
one is a smoke test.

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
or its analytic Jacobian $(14)$–$(16)$. A consequence of the same substitution is
that in this path the heat residual has no state dependence, so every consistent
one-step method produces the same update and the heat `TS` type selects only
which manufactured quadrature is sampled.

## References

* [Bradford, S. F., & Sanders, B. F. (2002). Finite-volume model for shallow-water flooding of arbitrary topography. Journal of hydraulic engineering, 128(3), 289-298.](https://ascelibrary.org/doi/10.1061/%28ASCE%290733-9429%282002%29128%3A3%28289%29)

* [Roache, P. J. (2002). Code verification by the method of manufactured solutions. Journal of Fluids Engineering, 124(1), 4-10.](https://doi.org/10.1115/1.1436090)
