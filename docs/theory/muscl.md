# Second-Order MUSCL Reconstruction

The first-order finite volume method described in [Shallow Water Equations](swe.md)
reconstructs left and right states at each face directly from cell-averaged values,
introducing $O(\Delta x)$ numerical diffusion. The MUSCL (Monotone Upstream-centered
Schemes for Conservation Laws) scheme reduces this to $O(\Delta x^2)$ by replacing the
piecewise-constant reconstruction with a piecewise-linear one.

## Overview

For each interior face shared by cells $i$ (left) and $j$ (right), the first-order
scheme passes cell-averaged states $\mathbf{U}_i$, $\mathbf{U}_j$ directly to the Roe
solver. MUSCL replaces these with reconstructed face states:

$$
\mathbf{U}_{i \to f} = \mathbf{U}_i + \nabla\mathbf{U}_i \cdot \Delta\vec{x}_i,
\qquad
\mathbf{U}_{j \to f} = \mathbf{U}_j + \nabla\mathbf{U}_j \cdot \Delta\vec{x}_j,
$$

where $\Delta\vec{x}_i = \vec{x}_f - \vec{x}_i$ is the displacement from cell $i$'s
centroid to the face midpoint $\vec{x}_f$. These reconstructed states are then passed
to the Roe solver in place of the cell-averaged values.

## Step 1 — Weighted Least-Squares Gradient

For each cell $i$, the gradient $\nabla q_i$ (computed independently for each of
$h$, $hu$, $hv$) is found by minimizing over the set of neighbors $\mathcal{N}(i)$:

$$
\min_{\nabla q_i} \sum_{j \in \mathcal{N}(i)} w_{ij}
\left[ q_j - q_i - \nabla q_i \cdot (\vec{x}_j - \vec{x}_i) \right]^2,
\qquad w_{ij} = \frac{1}{d_{ij}},
$$

where $d_{ij} = \|\vec{x}_j - \vec{x}_i\|$. The inverse-distance weighting gives
more influence to nearby neighbors. The normal equations reduce to the $2\times 2$
symmetric system $\mathbf{M}_i \,\nabla q_i = \mathbf{b}_i$, which is inverted
analytically. The per-edge coefficients of $\mathbf{M}_i^{-1}$ are precomputed once
at startup and reused every timestep.

**Code:** `PrecomputeLSGradCoeffs` + `ComputeLeastSquaresGradients`
in `src/operator_fluxes_ceed.c`.

## Step 2 — Linear Reconstruction to Face Midpoints

Given cell-centered gradients, the state at face midpoint $\vec{x}_f$ is approximated
by linear extrapolation from each neighboring centroid:

$$
q_{i \to f} = q_i + \nabla q_i \cdot (\vec{x}_f - \vec{x}_i),
\qquad
q_{j \to f} = q_j + \nabla q_j \cdot (\vec{x}_f - \vec{x}_j).
$$

Ghost cells (owned by another MPI rank) have incomplete gradients because only
edges local to this process contribute to their least-squares sum. To avoid
error, reconstruction falls back to first-order (zero extrapolation) on the ghost
side.

**Code:** `ReconstructFaceValues` in `src/operator_fluxes_ceed.c`.

## Step 3 — Minmod Slope Limiter

Pure linear reconstruction can produce values outside the range of surrounding
cell averages, causing spurious oscillations near discontinuities. The minmod
limiter clips the extrapolated increment so the face value cannot overshoot the
adjacent cell-average difference:

$$
q_{i \to f} = q_i + \operatorname{minmod}\!\left(\nabla q_i \cdot \Delta\vec{x}_i,\; \tfrac{1}{2}(q_j - q_i)\right),
$$

$$
q_{j \to f} = q_j + \operatorname{minmod}\!\left(\nabla q_j \cdot \Delta\vec{x}_j,\; -\tfrac{1}{2}(q_j - q_i)\right),
$$

where

$$
\operatorname{minmod}(a,\, b) =
\begin{cases}
0 & \text{if } a\,b \leq 0, \\
a & \text{if } |a| \leq |b|, \\
b & \text{otherwise.}
\end{cases}
$$

The limiter reduces the scheme to first-order accuracy locally wherever the solution
is non-smooth, preventing oscillations while preserving second-order convergence in
smooth regions. It is enabled by default when `second_order` is active. Disable via
`no_limiter: true` in the YAML configuration or the `-no_limiter` command-line flag.

**Code:** `Minmod` + `ReconstructFaceValues` in `src/operator_fluxes_ceed.c`.

## Step 4 — Roe Solver on Reconstructed States

The reconstructed states $\mathbf{U}_{i \to f}$ and $\mathbf{U}_{j \to f}$ replace
the cell-averaged states in the Roe flux computation described in
[Shallow Water Equations](swe.md#evaluating-normal-fluxes). Water depth is clamped
to zero from below after reconstruction to prevent unphysical negative values that
can arise from linear extrapolation.

**Code:** `SWEFlux_Roe_MUSCL` Q-function in `src/swe/swe_muscl_fluxes_ceed.h`,
applied via the CEED operator created by `CreateCeedFluxOperatorReconstructed`
in `src/operator_fluxes_ceed.c`.

## MPI Parallelism

Partition-boundary edges are treated at first-order accuracy because ghost-cell
gradients are incomplete. This does not affect the global convergence rate but can
introduce small MPI-count-dependent variations in $L_\infty$ norms.

## Enabling Second-Order Reconstruction

MUSCL reconstruction requires the CEED backend. Enable via YAML:

```yaml
numerics:
  spatial      : fv
  temporal     : euler
  riemann      : roe
  second_order : true   # enable MUSCL reconstruction (CEED backend required)
  no_limiter   : true   # optional: disable the minmod limiter
```

Or via command-line flags: `-second_order` and optionally `-no_limiter`.
If the CEED backend is not active, a warning is printed and the solver falls
back to first-order.
