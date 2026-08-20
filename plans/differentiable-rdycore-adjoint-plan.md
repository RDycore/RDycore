# Plan: Differentiable RDycore — Adjoint Sensitivities, Gradient-Based Calibration, and PyTorch Coupling

**Prepared for:** Gautam Bisht (PI)
**Prepared by:** Mark Adams
**Date:** 2026-08-16
**Status:** PROPOSED — not started

---

## 1. Executive Summary

RDycore today is a forward model: explicit finite-volume shallow water
(+ sediment) with no Jacobian, no adjoint, and no gradient path to its
physical parameters. This plan adds a **discrete adjoint** to RDycore using
PETSc's `TSAdjoint`, and connects it to PyTorch through an existing,
measured **PETSc↔PyTorch zero-copy sparse bridge**, giving three new
capabilities in increasing order of ambition:

1. **Gradient-based calibration** of spatially distributed Manning
   roughness (and later sediment coefficients and forcing) against
   stream-gauge observations — adjoint gradients at the cost of ~one extra
   forward solve, independent of the number of parameters.
2. **A training substrate for the AI-enhanced LETKF program**: ensemble
   state blocks exposed zero-copy to PyTorch on GPU, so learned
   localization/inflation/model-error operators train against live
   ensembles with gradients flowing back — end-to-end differentiable DA.
3. **Solver-in-the-loop learned physics**: neural source terms (model-error
   correction; surface–groundwater exchange flux) trained *through* the
   SWE solve rather than as offline surrogates.

Everything in Phases 1–3 uses PETSc machinery that exists today
(`TSAdjoint`, `TSSetRHSJacobian`, `TSSetRHSJacobianP`, `TSTrajectory`) plus
new RDycore code that is mechanical to write (a bs=3 block Jacobian on the
cell-adjacency pattern). Phases 4–5 pull in the PyTorch bridge and planned
differentiable-implicit-solve work.

### Why this complements (not competes with) the LETKF line

LETKF is ensemble/derivative-free; this plan is adjoint/gradient-based.
They meet in two places: (a) hybrid ensemble–variational DA (EnVar), where
adjoint gradients correct the ensemble analysis, and (b) the AI-LETKF
operators from the Genesis FOA plan, which need exactly the
gradients-through-physics this plan provides in order to be trained
end-to-end instead of on offline ensemble dumps.

---

## 2. Background: What Exists

### In RDycore (current state)

| Item | Status | Location |
|------|--------|----------|
| SWE RHS (Roe/HLLC fluxes, sources) | Explicit only, `TSSetRHSFunction` | `src/rdysetup.c:OperatorRHSFunction`, `src/swe/` |
| Time integration | TSEULER / TSRK4 (explicit); semi-implicit friction option; heat branch adds implicit TSBEULER/TSCN on a split heat TS | `src/rdysetup.c:InitSolver`, `src/heat/heat_petsc.c` |
| Jacobian | **None for SWE.** Heat branch (`bishtgautam/heat`) has the first `TSSetIJacobian` + implicit SNES/KSP solves — but per-cell *diagonal* only (source terms, incl. a CEED Q-function variant); no coupled flux Jacobian anywhere | `src/heat/heat_petsc.c`, `src/heat/heat_ceed.c` |
| Adjoint / sensitivity / TAO | **None** | — |
| Manning n | Per-region/domain, runtime-settable | `RDySetRegionalManningsN`, `RDySetDomainManningsN`, YAML `materials.*.manning` |
| LETKF DA | Phase 1 complete (twin experiment) | `driver/letkf_test.c` |
| GPU | CEED kernels (CUDA/HIP/SYCL) + PETSc CPU fallback | `src/ceed.c`, `src/operator_*_ceed.c` |
| State layout | 3 DOF/cell (h, hu, hv) on DMPlex FV | `include/private/rdycoreimpl.h` |

### In PETSc

- **`TSAdjoint`** (Zhang, Constantinescu, Smith; SISC 2022): discrete
  adjoints of PETSc time integrators, including explicit RK, with optimal
  trajectory checkpointing (`TSTrajectory`). Requires the RHS Jacobian
  (transpose actions) and, for parameter gradients, `TSSetRHSJacobianP`.
- **TAO**: gradient-based optimization, ready to consume adjoint gradients.

### The PETSc↔PyTorch bridge (Adams, 2026 — public paper + code)

A measured, A100-validated layer that exposes a PETSc device `Mat` as a
differentiable PyTorch operator with **zero copies**: operands cross as
DLPack pointers in both directions and both autograd passes. Extensions
already built and gated:

- **Trainable nonzero values over a fixed sparsity pattern** — gradients
  flow into the `Mat`'s values array (AIJ and blocked BAIJ, CUDA).
- **Batched right-hand sides** (dense multi-RHS via MatMatMult; ~2× over
  looped SpMV at k=64) — maps directly onto ensemble propagation.
- **MPI**: pooled, pointer-swapped distributed wrappers; validated to
  8 GPUs / 2 nodes.

Planned follow-on (relevant to Phase 5): `KSPSolve` and `TSAdjoint` as
PyTorch autograd nodes, with the operator-value gradient of an implicit
solve as a pattern-restricted outer product.

---

## 3. The Mathematical Core (small, and mostly local)

The SWE RHS per cell i is
`du_i/dt = -(1/|Ω_i|) Σ_edges F(u_i, u_j, n_edge) L_edge + S(u_i; n_i, Q_i)`.

**Flux Jacobian.** Each interior edge contributes two 3×3 blocks,
`∂F/∂u_L` and `∂F/∂u_R`, to the two adjacent cell rows. The global RHS
Jacobian is therefore **block-sparse with bs=3 on the cell-adjacency
pattern** — diagonal block + one off-diagonal block per neighbor
(≤3 for triangles, ≤4 for quads). Roe flux Jacobians in 2D SWE are
closed-form and short; HLLC likewise. Boundary conditions modify diagonal
blocks only.

**Source Jacobians.** The Manning drag term
`S_f = -C_D ||u|| u`, `C_D = g n² h^{-1/3}`, is *per-cell local*:
`∂S/∂u` adds to the diagonal block; the parameter Jacobian `∂S/∂n` has two
nonzeros per cell (the momentum rows). So `TSSetRHSJacobianP` for Manning
is a trivially sparse, embarrassingly local matrix — the ideal first
parameter. Rain/runoff forcing `Q(x,y,t)` is even simpler (identity-like
in the h row).

**Cost functional.** Gauge misfit
`J = Σ_gauges Σ_obs-times (h_model − h_obs)² / 2σ²`, implemented either as
a quadrature TS (`TSCreateQuadratureTS`) or as per-window terminal costs
at observation times. TSAdjoint then returns `dJ/du_0` and `dJ/dp` in one
backward sweep.

**Known nondifferentiability (design around it, don't discover it late):**
Roe entropy fixes, slope/flux limiters, and **wetting/drying fronts** make
the discrete map only piecewise differentiable; adjoint SWE is known to be
noisy at moving wet/dry boundaries. Mitigations, in order: validate on
fully wet cases first (dam break over wet bed, MMS); prefer HLLC or
smoothed limiter variants on the adjoint path; accept subgradients at
switching sets (standard practice); regularize h^{-1/3} with the same
h_min floor the forward model already uses.

---

## 4. Phased Implementation Plan

### Phase 1 — Assembled SWE RHS Jacobian (CPU/PETSc backend)

**Goal:** `MatFDColoring`-verified analytic Jacobian of the SWE RHS.

**Tasks:**
1. New file `src/swe/swe_jacobian_petsc.c`: closed-form 3×3 Roe (and
   HLLC) flux Jacobian blocks; drag/bed-slope source diagonal blocks.
   Mirror the loop structure of `src/operator_fluxes_petsc.c` /
   `src/operator_sources_petsc.c`.
2. Preallocate the Jacobian from DMPlex cell adjacency. Use blocked
   insertion (`MatSetValuesBlocked`, bs=3) into `MATBAIJ` on CPU; keep
   `MATAIJ` insertion path so the same code runs with GPU AIJ types later
   (release PETSc has no BAIJ-CUDA; blocked-on-AIJ is fine).
3. Register via `TSSetRHSJacobian` in `src/rdysetup.c:InitSolver`, guarded
   by a new YAML/config flag (e.g. `numerics.jacobian: analytic`), default
   off — zero impact on existing runs.
4. **Verification (CTest):** new unit/integration test comparing analytic
   J against `SNESComputeJacobianDefaultColor` finite differences on
   (a) the MMS convergence-study config (fully wet, smooth), and
   (b) the `planar_dam_10x5.msh` dam-break at an early wet time.
   Pass criterion: relative Frobenius error < 1e-6 (FD-limited).

**Estimated size:** ~800–1200 lines C + tests. No new dependencies.

### Phase 2 — TSAdjoint: gradients w.r.t. initial conditions and Manning n

**Goal:** `dJ/du_0` and `dJ/dn` for a gauge-misfit J, validated against
finite differences.

**Tasks:**
1. `TSSetSaveTrajectory` + checkpointing options in `InitSolver` (memory
   checkpointing is fine at test scale; disk for Harvey-scale).
2. Manning parameter Jacobian: `TSSetRHSJacobianP` with the 2-nonzeros-
   per-cell `∂S/∂n` matrix. Parameter vector p = per-region n (a handful
   of entries) first, then per-cell n.
3. Cost integrand: quadrature TS evaluating gauge misfit at observation
   cells (reuse the observation-operator plumbing from the LETKF driver).
4. New driver `driver/adjoint_test.c` (pattern: `driver/letkf_test.c`):
   twin experiment — run truth with n = n_true, perturb n, recover the
   gradient, check `dJ/dn` against central FD over each region parameter.
5. **Verification (CTest):** `adjoint_dam_break_np_1`: FD-vs-adjoint
   gradient agreement to ~1e-5 relative on the wet-bed dam break;
   a second test on the MMS config where dJ/dn has a manufactured value.

**Deliverable:** the first adjoint sensitivity map of a RDycore
simulation — e.g. "which cells' roughness controls the hydrograph at this
gauge" — publishable on its own as a capability note.

### Phase 3 — Gradient-based calibration demo (Houston / Hurricane Harvey)

**Goal:** calibrate spatially distributed Manning n on the Houston 1 km
Harvey case (`driver/tests/swe_roe/Houston1km.DirichletBC.yaml`, 2746
cells) against synthetic-then-real gauge data.

**Tasks:**
1. Wrap Phase-2 gradient in TAO (`TAOBLMVM` or `TAOBNCG`, with bound
   constraints n ∈ [0.01, 0.2]); per-region first, then per-cell with
   Tikhonov/total-variation regularization.
2. Twin experiment: recover a known two-zone roughness field from
   synthetic gauges + noise. Then real USGS Harvey gauges (data already
   in hand from the Harvey simulation work).
3. Report: iterations to convergence, cost-per-gradient vs forward solve
   (~1× expected), sensitivity of recovered field to gauge density.

**This is the go/no-go demo for proposal purposes**: an end-to-end
"observations → calibrated model" loop on a named U.S. flood event, on a
mesh the team already runs.

### Phase 4 — PyTorch coupling via the bridge (AI-DA substrate)

**Goal:** RDycore states, ensembles, and gradients visible to PyTorch on
GPU with zero copies; the FOA's learned-DA operators trainable end-to-end.

**Tasks:**
1. Expose `rdy->u_global` (and the LETKF ensemble block, an n×k dense
   Mat) to torch via DLPack through the bridge — no staging, no copies;
   the batched-RHS machinery handles the k-member dimension.
2. Reimplement the Phase-3 optimizer loop in torch (Adam/L-BFGS) with
   J and dJ/dn supplied by TSAdjoint through the bridge: RDycore becomes
   a `torch.autograd.Function`. This is the template every learned
   component reuses.
3. First learned component: a small NN **model-error correction** term
   added to the RHS (inputs: local state + bed slope; output: source
   correction), trained through TSAdjoint against gauges on cases where
   the coarse mesh visibly biases the hydrograph. This directly upgrades
   the Genesis-plan "model error correction" item from post-hoc DA to
   in-solver training.
4. Hybrid DA experiment: use adjoint gradients to correct/augment the
   LETKF analysis (EnVar-style) on the dam-break twin experiment; compare
   RMSE vs pure LETKF at equal ensemble size.

### Phase 5 — Stretch: implicit stepping and surface–groundwater coupling

1. **IMEX/implicit friction**: promote the semi-implicit friction option
   to a true `TSARKIMEX`/`TSBEULER` path using the Phase-1 Jacobian
   (SNES+KSP). Larger stable dt for stiff-drag and thin-film regimes;
   also the entry point for the planned differentiable-implicit-solve
   work (KSPSolve as an autograd node with pattern-restricted operator
   gradients), which would then run on a production DOE application.
2. **Surface–groundwater exchange**: a neural-operator exchange-flux
   source term (the FOA Focus-Area-B requirement) trained through the
   differentiable SWE solve — no 3D subsurface solver needed at Phase-I
   scale; when a real subsurface solver is coupled, it is implicit and
   lands on the same adjoint machinery.
3. **Sediment**: Hairsine-Rose coefficients (detachability, critical
   stream power, settling velocities) via the same `RHSJacobianP` route —
   modeled coefficients fit against flume/field data by gradient descent.

---

## 5. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Wet/dry fronts break adjoint smoothness | High (known in adjoint SWE) | Validate wet-only first; HLLC/smoothed limiters on adjoint path; subgradients; regularized h floor |
| CEED kernels are forward-only | Medium | Phase 1–3 use the PETSc (CPU) backend for J; assembled bs=3 Jacobian is cheap at current mesh sizes; CEED/matrix-free JVP is a later optimization, not a blocker |
| Checkpointing memory at Harvey scale | Low | `TSTrajectory` disk checkpointing is built in; 140 steps × 2746 cells is tiny anyway |
| Roe entropy-fix kinks pollute FD validation | Low | Validate on smooth MMS flows; compare against complex-step or tangent-linear where needed |
| Ill-posedness of per-cell n inversion | Medium | Start per-region (few parameters); TV/Tikhonov regularization; gauge-density study in Phase 3 |

## 6. Roles

- **Mark Adams**: PETSc/TSAdjoint integration, Jacobian assembly, bridge
  coupling, GPU path (author of the PETSc↔PyTorch bridge; PETSc
  developer). Direct line to Hong Zhang (ANL, TSAdjoint author) for
  adjoint internals.
- **Gautam Bisht / RDycore team**: physics review of flux/source
  Jacobians, Harvey case and gauge data, YAML/config and CI conventions,
  E3SM-facing requirements.
- Phases 1–3 are one-person-scale work items with clear interfaces; Phase
  4 onward is where a student/postdoc on the ML-DA side fits naturally.

## 7. Proposal Hooks

- **Genesis Mission follow-on / Phase II**: this supplies the missing
  training substrate for the AI-LETKF pitch (learned localization,
  inflation, model error) — end-to-end gradients through RDycore physics,
  plus the required surface–groundwater coupling path (Phase 5.2).
- **BER–ASCR SciDAC partnership shape**: ASCR-side differentiable-solver
  infrastructure (PETSc TSAdjoint + bridge), BER-side flood/hydrology
  deliverables (calibration, hybrid DA, learned coupling) — with the
  Harvey calibration demo (Phase 3) as the concrete use case.
- **Standalone capability papers**: (i) "Adjoint sensitivities and
  gradient-based Manning calibration in RDycore" after Phase 3;
  (ii) "End-to-end differentiable data assimilation for compound
  flooding" after Phase 4.

## 8. Verification Test Summary (CTest names, cumulative)

| Test | Phase | Checks |
|------|-------|--------|
| `swe_jacobian_fd_mms_np_1` | 1 | analytic J vs FD coloring, smooth wet flow |
| `swe_jacobian_fd_dambreak_np_1` | 1 | analytic J vs FD, dam break (wet) |
| `adjoint_grad_fd_mms_np_1` | 2 | dJ/dn adjoint vs central FD, manufactured |
| `adjoint_dam_break_np_1` | 2 | twin-experiment gradient recovery |
| `calibrate_manning_twin_np_1` | 3 | two-zone n recovered from synthetic gauges |
| `bridge_autograd_gradcheck` | 4 | torch gradcheck through the RDycore autograd node |
