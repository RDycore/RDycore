# Second-Order Well-Balanced Hydrostatic Reconstruction

**Status: design / plan (PETSc backend).** This note specifies how to combine
[MUSCL second-order reconstruction](muscl.md) with
[hydrostatic-reconstruction (HR) well-balancing](swe.md) into a single scheme that
is *simultaneously* second-order accurate and exactly well-balanced (lake-at-rest
preserving) over arbitrary bed topography. The construction follows Audusse,
Bouchut, Bristeau, Klein & Perthame, *SIAM J. Sci. Comput.* 25 (2004).

## Why the two cannot simply be stacked

The current code gates these features apart (`CreateOperator` in `src/operator.c`)
because they use *different* well-balancing mechanisms:

* **HR** uses piecewise-constant (first-order) cell states, performs hydrostatic
  reconstruction at each interface, and adds a per-cell hydrostatic *pressure
  correction*. The bed-slope source is then identically zero
  (`SWESourcesHydroRecon`).
* **MUSCL** reconstructs $(h, hu, hv)$ to the faces and relies on the
  vertex-based $\Delta h_v$ correction for well-balancing — *not* HR.

The key fact that drives the combined design:

> Well-balanced HR at second order requires reconstructing the **free surface**
> $\eta = h + z_b$ (not the depth $h$), **plus** a centered source term. Either
> piece alone is insufficient.

## Continuous target

In the SWE momentum equation the bed enters as a source $-g\,h\,\nabla z_b$, while
the hydrostatic pressure $\tfrac{1}{2}gh^2$ lives in the flux. Combining them, the
exact momentum forcing is

$$
-\nabla\!\left(\tfrac{1}{2}g h^2\right) \;-\; g\,h\,\nabla z_b
\;=\; -\,g\,h\,\nabla\eta ,
\qquad \eta = h + z_b .
$$

A scheme is **well-balanced** iff this discrete forcing vanishes exactly for the
lake at rest, $\eta \equiv \text{const}$, $\mathbf{u} = 0$. It is **second-order**
iff the discrete forcing matches $-g h\nabla\eta$ to $O(\Delta x^2)$ for smooth
non-equilibrium states.

## The combined scheme

For an interior edge $e$ shared by cells $i$ (left) and $j$ (right), with edge
length $L_e$ and unit normal $\hat{\mathbf{n}}$ pointing out of cell $i$:

### 1. Reconstruct the free surface, velocities, and bed

Using limited least-squares gradients (the existing MUSCL machinery applied to
$\eta = h + z_b$ rather than $h$), reconstruct from each side $s \in \{i, j\}$ to
the face midpoint:

$$
\eta_{s,e}, \quad u_{s,e}, \quad v_{s,e}, \quad z_{s,e},
\qquad
h_{s,e} = \max\!\left(0,\; \eta_{s,e} - z_{s,e}\right).
$$

The bed $z_{s,e}$ is reconstructed from a **time-invariant** bed gradient
$\nabla z_b$ (precomputed once at setup). Because the limiter acts on $\eta$, a
constant free surface reconstructs exactly ($\eta_{s,e} = \text{const}$), which is
what preserves lake-at-rest regardless of limiter activity.

### 2. Hydrostatic reconstruction at the interface

$$
z^\ast_e = \max\!\left(z_{i,e},\, z_{j,e}\right),
\qquad
h^\ast_{s,e} = \max\!\left(0,\; \eta_{s,e} - z^\ast_e\right),
$$

with starred states $\mathbf{U}^\ast_{s,e} = \big(h^\ast_{s,e},\,
h^\ast_{s,e} u_{s,e},\, h^\ast_{s,e} v_{s,e}\big)$.

### 3. Riemann flux on starred states

$$
\hat{\mathbf{F}}_e = \mathrm{Roe}\!\left(\mathbf{U}^\ast_{i,e},\,
\mathbf{U}^\ast_{j,e};\, \hat{\mathbf{n}}\right).
$$

### 4. Per-cell corrections

The semi-discrete update for cell $i$ is

$$
A_i \frac{d\mathbf{U}_i}{dt}
= -\sum_{e \in \partial i} L_e
\left[\hat{\mathbf{F}}_e + \mathbf{S}^{\text{int}}_{i,e}\right]
\;+\; \sum_{e \in \partial i} L_e\, \mathbf{S}^{\text{cen}}_{i,e},
$$

with two momentum-only correction terms (zero in the mass component):

$$
\mathbf{S}^{\text{int}}_{i,e} =
\begin{pmatrix} 0 \\ P_{i,e}\,\hat n_x \\ P_{i,e}\,\hat n_y \end{pmatrix},
\quad
P_{i,e} = \tfrac{1}{2} g \left(h_{i,e}^2 - (h^\ast_{i,e})^2\right)
\qquad\text{(interface pressure correction)},
$$

$$
\mathbf{S}^{\text{cen}}_{i,e} =
\begin{pmatrix} 0 \\ C_{i,e}\,\hat n_x \\ C_{i,e}\,\hat n_y \end{pmatrix},
\quad
C_{i,e} = \tfrac{1}{2} g \left(h_{i,e} + h_i\right)\!\left(z_i - z_{i,e}\right)
\qquad\text{(centered source)} .
$$

Here $h_i, z_i$ are cell-centroid values and $h_{i,e}, z_{i,e}$ are the reconstructed
face values **on cell $i$'s own side** of the edge. Note the **sign placement**: the
interface correction sits inside the flux bracket (scaled $-L_e/A_i$ like the flux),
whereas the centered term enters as a $+$ source (scaled $+L_e/A_i$, i.e. the
*opposite* sign of the flux scaling). Getting this sign wrong silently breaks
well-balancing.

## Lake-at-rest derivation

Set $\eta \equiv H$ (constant) and $\mathbf{u} = 0$. Then $\eta_{s,e} = H$ for all
sides, $h_{s,e} = H - z_{s,e}$, and $h_i = H - z_i$.

**Starred states are equal across the edge.** Since $h^\ast_{i,e} = H - z^\ast_e
= h^\ast_{j,e} =: h^\ast_e$ and velocities vanish, the Roe flux reduces to pure
pressure (the dissipation $\propto \Delta\mathbf{U}^\ast = 0$):

$$
\hat{\mathbf{F}}_e \cdot \hat{\mathbf{n}}\big|_{\text{mom}}
= \tfrac{1}{2} g\, (h^\ast_e)^2 .
$$

**Flux + interface correction telescopes to the cell's own face depth.** Along
$\hat{\mathbf{n}}$ the magnitude is

$$
\tfrac{1}{2} g (h^\ast_e)^2 + P_{i,e}
= \tfrac{1}{2} g (h^\ast_e)^2 + \tfrac{1}{2} g\!\left(h_{i,e}^2 - (h^\ast_e)^2\right)
= \tfrac{1}{2} g\, h_{i,e}^2 .
$$

The neighbor's state has dropped out entirely — this decoupling is the heart of HR.
Cell $i$'s contribution from edge $e$ is therefore $-L_e\,\tfrac{1}{2}g h_{i,e}^2\,
\hat{\mathbf{n}}$.

**The centered term converts the face depth to the cell depth.** At rest,

$$
z_i - z_{i,e} = (H - h_i) - (H - h_{i,e}) = h_{i,e} - h_i,
$$

so

$$
C_{i,e} = \tfrac{1}{2} g\,(h_{i,e} + h_i)(h_{i,e} - h_i)
= \tfrac{1}{2} g\!\left(h_{i,e}^2 - h_i^2\right).
$$

Adding the centered contribution $+L_e\, C_{i,e}\,\hat{\mathbf{n}}$, the **net** edge
contribution along $\hat{\mathbf{n}}$ becomes

$$
-L_e\,\tfrac{1}{2}g\, h_{i,e}^2 + L_e\,\tfrac{1}{2}g\!\left(h_{i,e}^2 - h_i^2\right)
= -L_e\,\tfrac{1}{2} g\, h_i^2 .
$$

The face-dependent $h_{i,e}^2$ cancels, leaving a constant $h_i^2$ per cell. Summing
over the cell's edges,

$$
A_i \frac{d\mathbf{U}_i}{dt}\bigg|_{\text{mom}}
= -\tfrac{1}{2} g\, h_i^2 \sum_{e \in \partial i} L_e\,\hat{\mathbf{n}}_e
= \mathbf{0},
$$

because $\sum_e L_e\,\hat{\mathbf{n}}_e = 0$ for any closed polygon. The lake at rest
is preserved **exactly**. $\blacksquare$

### What each piece buys

| Variant | Pressure correction $P_{i,e}$ | Centered $C_{i,e}$ | Well-balanced | Source accuracy |
|---|---|---|---|---|
| A | face depth $h_{i,e}$ | none | **no** — $-\sum L_e h_{i,e}^2 \hat n \ne 0$ | 2nd (but useless) |
| B | **cell** depth $h_i$ | none | yes (constant $h_i^2$ telescopes directly) | not guaranteed 2nd |
| **C** | face depth $h_{i,e}$ | **yes** | **yes** (proof above) | **2nd (Audusse)** |

* **A** keeps the consistent face-depth pressure but, without the centered term, the
  surviving $h_{i,e}^2$ varies edge to edge and does not cancel ⇒ spurious currents.
* **B** forces a constant $h_i^2$ into the correction, so it cancels directly without
  a centered term. It is exactly well-balanced and is the smallest delta from the
  current first-order HR code, but it pairs face-reconstructed Roe states with a
  cell-depth bed correction — a non-standard combination with no second-order
  guarantee on the bed source. Acceptable on gentle topography; can degrade where
  depth and bed slope co-vary strongly (steep/curved beds, transients, wet–dry).
* **C** is the published scheme: second-order accurate *and* exactly well-balanced.
  The centered term is one isolated accumulation block, so building C and toggling
  the block off recovers B for free as an A/B baseline.

**Recommendation: implement C**, with the $\mathbf{S}^{\text{cen}}$ block guarded so
it can be disabled to fall back to B.

### Positivity and wet/dry

The derivation assumes wet cells (no $\max(0,\cdot)$ clamping active). The clamps in
steps 1–2 preserve $h \ge 0$ and the scheme degrades to the standard robust HR
behavior at wet/dry fronts, where exact well-balancing is necessarily relaxed.

## Implementation plan (PETSc backend)

MUSCL and HR are both implemented in the PETSc backend, so this is entirely a
PETSc-side change. CEED is out of scope for now.

### 1. `src/operator.c`

* Remove the `PetscCheck(... well_balancing != WELL_BALANCING_HR ...)` guard in
  `CreateOperator` that forbids the combination.
* In the PETSc branch of `CreateOperatorSubOperators`, route the
  `WELL_BALANCING_HR` case to a new `CreatePetscSWEInteriorFluxHR2ROperator` when
  `config->numerics.second_order` is set; otherwise keep
  `CreatePetscSWEInteriorFluxHROperator`. The source operator
  (`CreatePetscSWESourceHROperator`) is unchanged — it already zeroes the bed
  slope, which remains correct because both corrections live in the flux operator.

### 2. `src/swe/swe_petsc.c` — operator struct

Extend `InteriorFluxHROperator` with the reconstruction machinery (mirroring
`InteriorFluxOperator`):

```c
PetscBool    use_slope_reconstruction, use_limiter;
PetscReal   *ls_grad_coeffs;                  // [num_internal_edges * 4]
PetscScalar *grad_eta, *grad_hu, *grad_hv;    // [num_cells * 2]  — reconstruct eta, not h
PetscScalar *grad_zb;                         // [num_cells * 2]  — precomputed, time-invariant
PetscScalar *eta_cell;                        // [num_cells]      — scratch: h + zc each step
PetscScalar *q_reconstructed;                 // [num_owned_internal_edges * 6] = eta,hu,hv per side
PetscScalar *z_reconstructed;                 // [num_owned_internal_edges * 2] = z_L, z_R at face
```

### 3. `src/swe/swe_petsc.c` — new `ApplyInteriorFluxHR2R`

Clone `ApplyInteriorFlux2R` (the owned-edge / `DMLocalToGlobal(ADD)` parallel
pattern) and inject HR. Each step:

1. Build `eta_cell[c] = u_ptr[3c] + zc[c]` for all local cells (including ghosts).
2. `ComputeLeastSquaresGradients` on $(\eta, hu, hv)$, then `CommunicateCellGradients`.
   (`grad_zb` is precomputed and communicated once at setup; it does not change.)
3. Reconstruct to owned faces: $\eta_{s,e}, hu_{s,e}, hv_{s,e}$ into `q_reconstructed`
   and $z_{s,e}$ into `z_reconstructed`. A dedicated `ReconstructFaceValuesHR` is
   preferred over overloading `ReconstructFaceValues`, to keep the $\eta$-vs-$h$
   semantics explicit. Reconstruct $z_b$ **unlimited** (fixed input data).
4. Per owned edge: $h_{s,e} = \max(0, \eta_{s,e} - z_{s,e})$;
   $z^\ast = \max(z_{i,e}, z_{j,e})$; $h^\ast_{s,e} = \max(0, \eta_{s,e} - z^\ast)$.
   Fill `datal/datar` with the starred states; velocities from the **face** depth
   $h_{s,e}$ via `ComputeRiemannVelocities` (ANUGA regularization). Keep the
   first-order ghost-edge fallback, routed through HR with cell $zc$.
5. `ComputeSWERoeFlux` on starred states.
6. Accumulate into the local RHS, per owned edge with cells $l, r$
   (`flux_scale_l = -L_e/A_l`, `flux_scale_r = +L_e/A_r`):
   * **Roe flux**: `flux_vec * flux_scale_s`.
   * **Interface correction**: `P_s = 0.5*g*(h_face_s^2 - h_star_s^2)`, added to
     momentum via `P_s * {cn,sn} * flux_scale_s`.
   * **Centered term** (guarded `if (include_Sc)` → toggles C↔B):
     `C_s = 0.5*g*(h_face_s + h_cell_s)*(zc_s - z_face_s)`, added via
     `C_s * {cn,sn} * (-flux_scale_s)` — **opposite sign** of the flux scaling.
   * Courant diagnostic unchanged.

### 4. `src/swe/swe_petsc.c` — `CreatePetscSWEInteriorFluxHR2ROperator`

Clone `CreatePetscSWEInteriorFluxHROperator` plus the MUSCL allocations from
`CreatePetscSWEInteriorFluxOperator`: `PrecomputeLSGradCoeffs`, allocate gradients
and `q/z_reconstructed`, read the limiter flags. **Precompute `grad_zb` once here**
(LS gradient of the per-cell `zc` field + `CommunicateCellGradients`) and store it.
Reuse the existing vertex-averaged `zc` logic. Extend `DestroyInteriorFluxHR` to
free the new arrays.

### 5. Header

Declare `CreatePetscSWEInteriorFluxHR2ROperator` alongside
`CreatePetscSWEInteriorFluxHROperator` in `include/private/rdysweimpl.h`.

### 6. Configuration

Reuses the existing `numerics.second_order` and `numerics.no_limiter` flags under
`physics.flow.well_balancing: HR`. No new config keys.

## Verification

* **Lake at rest over non-flat bed** (sloped + bump, $\mathbf{u} = 0$): assert
  $\max\|\mathbf{u}\|$ stays at round-off across many steps. This is the test that
  distinguishes B/C (pass) from A (fail) and catches $\mathbf{S}^{\text{cen}}$ sign
  errors.
* **Smooth MMS / convergence study**: confirm the 2nd-order spatial rate, and that C
  beats B on the source-containing error when the bed varies.
* **Regression**: existing standalone HR and MUSCL tests must be unaffected when only
  one feature is active.
