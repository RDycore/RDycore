# Augmented Roe Solver

The `augmented_roe` well-balancing method (`physics.flow.well_balancing:
augmented_roe`) replaces the cell bed-slope source term with an edge-based
bed source that is folded into an augmented Roe Riemann solver, following the
solver used by SERGHEI (`Solver::roe` in SERGHEI's `src/Solvers.h`). It
preserves a lake at rest exactly, including at stationary shorelines, on
unstructured meshes of triangles, quadrilaterals, or both.

The pointwise solver lives in `src/swe/swe_augmented_roe_flux_petsc.h` and
`src/swe/swe_augmented_roe_flux_ceed.h` (kept identical by
`src/tests/test_augmented_roe.c`).

## Fluctuations across an edge

Consider an edge with unit normal $\vec{n} = (n_x, n_y)$ pointing from its
left cell $L$ to its right cell $R$, with cell states
$\mathbf{U} = [h, hu, hv]^T$ and cell bed elevations $z_L$, $z_R$ (the mean of
each cell's vertex elevations, or the cell elevation read from a file).

**Roe averages.** With $\tilde{h} = (h_L + h_R)/2$, $\tilde{c} = \sqrt{g\tilde{h}}$
and $\sqrt{h}$-weighted velocities $\tilde{u}, \tilde{v}$, the eigenvalues are
$\lambda_{1,2,3} = \tilde{u}_n - \tilde{c},\ \tilde{u}_n,\ \tilde{u}_n + \tilde{c}$ with
eigenvectors

$$
\mathbf{e}_1 = \begin{bmatrix}1\\ \tilde{u} - \tilde{c}n_x\\ \tilde{v} - \tilde{c}n_y\end{bmatrix},\quad
\mathbf{e}_2 = \begin{bmatrix}0\\ -\tilde{c}n_y\\ \tilde{c}n_x\end{bmatrix},\quad
\mathbf{e}_3 = \begin{bmatrix}1\\ \tilde{u} + \tilde{c}n_x\\ \tilde{v} + \tilde{c}n_y\end{bmatrix},
$$

and the wave strengths $\alpha_k$ are computed from the jumps $\delta h$,
$\delta(hu)$, $\delta(hv)$ of the conserved variables.

**Bed source.** With $\delta z = z_R - z_L$, the bed-source strength is

$$
\beta_B = \frac{g}{2\tilde{c}}\left(h_p - \frac{|\delta z_p|}{2}\right)\delta z_p,
$$

where $h_p = h_L$ and $\delta z_p = \delta z$ if $\delta z \ge 0$ (but
$\delta z_p = h_L$ if the left water surface lies below $z_R$), and
symmetrically otherwise. The source enters the outer waves as
$\boldsymbol{\beta} = (\beta_B, 0, -\beta_B)$. For a lake at rest,
$\lambda_k\alpha_k - \beta_k = 0$ for every wave, and at a dry step that the
water does not reach, the edge acts as a wall.

**Entropy fix.** A transonic rarefaction in wave 1 or 3 (for wave 1:
$u_{n,L} - c_L < 0 < u_{n,R} - c_R$) is split into left- and right-going parts
(Harten–Hyman). Unlike RDycore's critical-flow fix, this split never fires at
rest, which keeps the method well-balanced for large depth ratios.

**Fluctuations.** The left- and right-going fluctuations are

$$
\mathbf{D}^\mp = \sum_{\lambda_k \lessgtr 0} (\lambda_k\alpha_k - \beta_k)\,\mathbf{e}_k
+ \text{(entropy-fix parts)},
$$

except at wet/dry fronts where an intermediate depth would be negative, in
which case the whole mass fluctuation goes to the wet side and the momentum
fluctuations vanish. Fluctuations below $10^{-14}$ are zeroed, and depths at or
below `tiny_h` are treated as dry.

## Flux form

RDycore's update is in flux form: the left cell receives
$-(L_e/A_L)\,\mathbf{F}^{\text{left}}$ and the right cell
$+(L_e/A_R)\,\mathbf{F}^{\text{right}}$, where $L_e$ is the edge length and
$A$ the cell areas. The augmented solver supplies different fluxes to the two
cells,

$$
\mathbf{F}^{\text{left}} = \mathbf{F}_n(\mathbf{U}_L) + \mathbf{D}^-,\qquad
\mathbf{F}^{\text{right}} = \mathbf{F}_n(\mathbf{U}_R) - \mathbf{D}^+ ,
$$

with $\mathbf{F}_n(\mathbf{U}) = [h u_n,\ hu\,u_n + \tfrac{1}{2}gh^2 n_x,\ hv\,u_n + \tfrac{1}{2}gh^2 n_y]^T$.
Since $\sum_e L_e\vec{n}_e = 0$ around a closed cell, the
$\mathbf{F}_n(\mathbf{U}_i)$ terms cancel and the update of cell $i$ reduces to
the fluctuation form $\frac{d\mathbf{U}_i}{dt} = -\frac{1}{A_i}\sum_e L_e\mathbf{D}_{i,e}$.
The mass components of $\mathbf{F}^{\text{left}}$ and $\mathbf{F}^{\text{right}}$
are equal, so mass is conserved exactly. Edge lengths and cell areas are
projected onto the $x$-$y$ plane (as for hydrostatic reconstruction), which
makes $\sum_e L_e\vec{n}_e = 0$ hold exactly.

Boundary edges use the standard Roe boundary fluxes (the bed source vanishes
there), and friction remains a cell source term, integrated as configured by
`physics.flow.source.method`.

## Differences from SERGHEI

* Friction is not folded into $\boldsymbol{\beta}$; it stays in the cell
  source.
* SERGHEI's time-step reduction (which caps the time step so that no cell
  loses more water than it holds) is not used: RDycore runs with a time step
  chosen by the user.
* SERGHEI's wet-wet subcritical clamp on $\beta_B$, which limits how much
  water an edge can drain from a cell in one step, is not used.
* Divisions by $\lambda_1$ and $\lambda_3$ in the intermediate depths are
  guarded.

Without the time-step reduction and the subcritical clamp, positivity at moving
shorelines relies on the wet/dry treatment alone, and that treatment applies
only when the cell receiving a negative intermediate depth is dry
($h \le$ `tiny_h`). Consider a thin film just above a step whose lower
neighbour's water surface lies below the film's bed. The clipped $\beta_B$ is
then the deep water's hydrostatic push $g h_{\text{deep}}^2/(4\tilde{c})$,
and $\tilde{c}$ comes from the mean depth, so the film-side intermediate depth
$h^*_3$ can be negative and hundreds of times larger in magnitude than the
film. Since the film is not dry, the regular split still sends the wave into
it. Its depth stays non-negative only for an edge Courant number below about
$h_{\text{film}}/|h^*_3|$, which tends to zero as the shoreline recedes. A
frictionless Thacker parabolic bowl at a fixed Courant number of about 0.09
therefore loses positivity (a $7.7\times10^{-6}$ m film lost 117% of its water
in one step) and diverges. With friction the same case remains stable, but
shows spurious velocities in thin films at the shoreline. Cases with moving
shorelines should be checked for negative depths.

## Limitations

The method is currently first order and supports flow without tracers. It is
rejected with `-second_order`, with sediment, salinity or heat, with the
ARK-IMEX temporal method, and with Riemann solvers other than `roe`.
