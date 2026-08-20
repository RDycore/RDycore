# Title

Discrete adjoint + gradient-based Manning calibration: status and open questions (Wed meeting)

# Body

Branch `adams/manning-draft` (two commits on top of `f2d4da89`: code and
paper/notes) adds a
discrete-adjoint capability to RDycore and uses it for gradient-based
calibration of spatially distributed Manning roughness. A draft paper for
internal review is in `papers/manning-calibration/`. This issue summarizes the
state and collects the open questions for Wednesday's meeting.

## What's on the branch

- **Exact analytic RHS Jacobian** (`numerics.jacobian: analytic|fd|none`):
  closed-form Roe-flux blocks by hand-written forward-mode differentiation
  (entropy-fix branches included), source and boundary-condition blocks.
  FD-verified at three levels (edge harness, assembled matrix vs coloring FD,
  end-to-end gradients); all gates in CI (12 new tests, seconds each).
- **Adjoint sensitivities** via PETSc TSAdjoint: dJ/du0 and dJ/dn for
  multi-time water-height gauge misfits, at ~1 forward solve per gradient
  independent of parameter count. FD-validated to 1e-5 or better.
- **Calibration** with TAO/BLMVM (bounds [0.01, 0.2]): per-region, per-cell
  (Tikhonov), and NLCD land-cover-class modes
  (`-adjoint_calibrate{,_percell,_classes,_gauges}` in the adjoint driver).
- **Differentiable friction paths**: `source.method: explicit` (plain dt-free
  source; required by BEULER) and `source.method: ark_imex` (Manning drag
  alone implicit, per-cell 3x3 blocks; completes the PR #359 prep). The
  legacy semi-implicit/XQ2018 treatments embed dt in the RHS and are not
  differentiable — unchanged for production runs.
- **Both integrator adjoints pass the same FD gates** (backward Euler and
  ARK-IMEX). CEED parity for the ARK-IMEX explicit side: 5.1e-16 after 100
  steps.

## Key results (twin experiments)

- Two-zone region recovery to 4e-7 relative; per-cell dam-break twin to 8%
  with 20 observation times + Tikhonov.
- Houston 1 km Harvey twin (2,746 cells, implicit stepping): 19.4% relative
  L2 recovery with proper regularization (beta 1e-4, 686 gauges); the
  gauge-sparse setting cleanly exhibits semiconvergence (26.3% at 300 its ->
  38.0% at 2000 its with weak regularization) — the overfitting mode the
  friction-estimation literature warns about, reproduced and dissected.
- Explicit friction is genuinely stiff at Harvey scale, not a bug: median
  depth ~6 mm gives friction rates up to 420/s => dt < ~2.4 ms, vs CFL 0.25 s
  at 30 m. Matches the h^{4/3}/(g n^2 |v|) analysis. Implicit/IMEX carries
  dt = 30 s (1 km) and dt = 0.25 s (30 m).
- Turning 30 m mesh (2.93M cells): ARK-IMEX at 6.1 s/step after re-allocating
  the IJacobian as AIJ bs=3 (3 nnz/row) with preonly+bjacobi (was 56 s/step
  through DMCreateMatrix's full flux stencil).

## Data in hand for real-gauge calibration (v2)

- 746 USGS Harvey stage/discharge gauges (HydroShare
  10.4211/hs.c037167e497546a1bc1508dfb32a9cff, CC-BY); 20 fall in the Houston
  mesh, 17 with Harvey-window stage; stage confirmed as WSE ft NAVD88.
- Mesh CRS resolved: EPSG:32610. Flag for the team: that projection carries
  ~9% length / ~19% area inflation and 14.5 deg rotation at Houston ("1 km"
  cells are ~917 m true).
- NLCD 2021 land-cover prior mapped to the mesh (`data/nlcd/`), driving the
  class-calibration mode.

## Upstream PETSc findings (affect anyone using the adjoint path)

1. ARKIMEX adjoint dereferences the explicit-part RHSJacobianP even when the
   parameter is implicit-only — filed as petsc#1925 with a fix branch.
2. TAOBLMVM/LMVM segfault on current petsc main (reproducer ready, to be
   filed). **Recommendation: keep RDycore adjoint work on v3.25.3 + the
   #1925 fix; do not move to petsc main yet.**
3. PCPBJACOBI can reuse stale cached block inverses after value-only matrix
   updates (reproducer ready, to be filed).

## Open questions for Wednesday

Engineering (Jeff):
1. ARK-IMEX split as completed = Manning drag alone in the IFunction, bed
   slope + external sources explicit. Match your PR #359 intent? Scheme/order
   preference? Device-resident (CEED Q-function) implicit part as follow-up?
2. Code placement/naming before this hardens into a PR:
   `swe_jacobian_petsc.c`, `swe_roe_flux_jacobian_petsc.h`,
   `numerics.jacobian`, `source.method` values.
3. Which adjoint-driver calls deserve promotion to the public `rdycore.h`
   API (RDySetObservations / RDyAdjointSolve / RDyGetSensitivities)? Fortran
   bindings?
4. Jacobian preallocation: DMCreateMatrix closure adjacency (superset of FV
   edge adjacency) — accept the stored zeros or tighten?
5. CI matrix: new tests need TSAdjoint-capable PETSc; CEED-only lanes skip
   `numerics.jacobian` configs. Concerns?

Science (Gautam and team):
6. Drag formulation on the adjoint path: tau_b = g n^2 h^{-7/3} q|q| at
   current state, standard h_min cutoff, plain velocity reconstruction
   (ANUGA regularization off — it changed the 30 m trajectory not at all).
   Physics sign-off?
7. Real-gauge choices: which of the 17-20 in-mesh USGS gauges do you trust
   for Harvey (backwater/tidal contamination)? Stage vs discharge as misfit
   variable? Realistic per-gauge sigma?
8. Is the NLCD-derived prior the right regularization center, and is the
   two-zone synthetic truth adequate for the paper's twins, or do you want a
   channel-vs-overbank synthetic field?
9. Calibration windows and wet/dry: which Harvey phases to calibrate on,
   given adjoint noise at drying fronts? A 36 h window at 30 m is ~518k
   steps => adjoint checkpointing (revolve) needed — priority?
10. Bounds n in [0.01, 0.2] sensible for Houston land cover?
11. Venue (WRR / JAMES / GMD) and co-author list for the draft.

Paper PDF: `papers/manning-calibration/manning-calibration.pdf` on the
branch. Full run log: `plans/RESULTS-manning-draft.md`.
