# Incremental Development Plan: Adjoint-Based Manning Calibration in RDycore

**Endpoint (first paper):** a working end-to-end system that reads surface
observations (stream gauges first), runs RDycore forward + adjoint solves in
an iterative TAO optimization loop, and **outputs a calibrated Manning
roughness map** on the Houston 1 km Harvey mesh.

**Parent plan:** [differentiable-rdycore-adjoint-plan.md](differentiable-rdycore-adjoint-plan.md)
(this document scopes its Phases 1–2 plus the thin per-region/per-cell TAO
loop from Phase 3 into fine-grained increments; everything beyond the Manning
map moves to Future Work below).

**Paper target:** "Adjoint sensitivities and gradient-based Manning
calibration in RDycore" — methods + twin experiment + (if data ready) real
USGS Harvey gauges.

**Date:** 2026-08-18. **Status:** proposed, nothing started.

---

## Increment table

Model column = recommended Claude model for the coding sessions:
**S** = Sonnet (mechanical, well-specified), **O** = Opus (standard
development), **F** = Fable (novel math, design decisions, paper writing).

| # | Increment | Deliverable / exit test | Status | Model | RDycore-team input needed |
|---|-----------|------------------------|--------|-------|---------------------------|
| 0 | Config + FD-Jacobian baseline | `numerics.jacobian` YAML flag (default off); `TSSetRHSJacobian` wired in `InitSolver` using PETSc FD-coloring as the Jacobian; all existing tests unchanged | not started | S | Approve YAML schema addition and config naming convention |
| 1 | Jacobian preallocation + analytic source blocks | bs=3 Mat preallocated from DMPlex cell adjacency; analytic drag + bed-slope diagonal blocks in new `src/swe/swe_jacobian_petsc.c` (structural twin of the heat branch's diagonal IJacobian — reference example, 3×3 blocks instead of scalars); verified vs FD on MMS config | not started | O | Physics review of source terms: `h_min` floor semantics, interaction with the semi-implicit friction option |
| 1b | **IMEX friction** (committed; required before increment 6) | `TEMPORAL_ARK_IMEX` case in `InitSolver`: `TSARKIMEX` with friction moved into `TSSetIFunction`/`TSSetIJacobian` using the increment-1 source blocks (block-diagonal, per-cell Newton — no KSP); PETSc-backend twin of the existing CEED `SOURCE_ARK_IMEX` kernels; stability gain measured vs table in briefing §8. Rationale: the existing semi-implicit splittings embed `dt` inside the RHS (Euler-only, no well-defined ∂f/∂u), so IMEX is the only friction treatment that is implicit *and* adjoint-differentiable | scaffolding exists (config + CEED kernels; no TS wiring) | O | Who designed the `SOURCE_ARK_IMEX` path and what semantics were intended; sign-off on completing it |
| 2a | Edge-level derivative harness | Unit test that FDs/complex-steps `ComputeSWERoeFlux` itself on single-edge states (each entropy-fix branch, near-`h_min`, oblique normals) — no mesh/TS/assembly; verifies analytic blocks at ~1e-7 | not started | O | Sign off on the sampled state space (which regimes matter) |
| 2b | Physical-flux + frozen-dissipation Jacobian, assembly | Textbook `∂F/∂u` of physical fluxes + primitive→conservative chain (h-floor as coded) + dissipation frozen (`\|Ã\|∂(Δu)` only); exact as Δu→0, so `swe_jacobian_fd_mms_np_1` (smooth) passes; validates global assembly end-to-end | not started | O | — |
| 2c | Exact dissipation derivatives | Derivatives of Roe averages, eigenvectors, `dW`, and the piecewise critical-flow fix (one-sided, matching the code's branches); `swe_jacobian_fd_dambreak_np_1` at 1e-6. De-risk: sympy-generated candidates cross-checked in the 2a harness; literature check. **Offramp**: if late, 2b's frozen Jacobian gives usable approximate gradients so 3–5 proceed in parallel | not started | F | Review entropy-fix handling. **Roe is the committed path; HLLC is the backup**, triggered only if 2c/increment-6 evidence shows entropy-fix kinks poisoning gradients (`RIEMANN_HLLC` is enum-only today — the backup means implementing the forward flux first, then its Jacobian) |
| 2d | Boundary-condition blocks | Diagonal contributions per BC type; one CTest per BC config | not started | S | Which BC types the Harvey/validation cases actually exercise |
| 3 | Adjoint of the state (dJ/du₀) | `TSSetSaveTrajectory` + checkpointing options; quadrature-TS gauge-misfit cost; new `driver/adjoint_test.c`; dJ/du₀ matches FD | not started | O | Gauge locations via the existing `observations.sites` YAML convention (`RDyObservationSites`); decide the observed-values file format |
| 4 | Parameter gradient (dJ/dn) | `TSSetRHSJacobianP` with the 2-nonzeros-per-cell ∂S/∂n; per-region gradient vs central FD (~1e-5 rel.); first sensitivity-map XDMF output; CTest `adjoint_grad_fd_mms_np_1` | not started | O | **API review**: `RDySetObservations` / `RDyAdjointSolve` / `RDyGetSensitivities` shape in `rdycore.h` (E3SM/Fortran-facing); need Fortran bindings? |
| 5 | TAO calibration loop (per-region) | TAOBLMVM with bounds n ∈ [0.01, 0.2]; twin experiment recovers a two-zone n from synthetic noisy gauges; CTest `calibrate_manning_twin_np_1` | not started | S | Confirm physical bounds and the twin-experiment test case (dam break vs Houston1km) |
| 6 | Per-cell n + regularization → **Manning map** | Per-cell parameter vector; Tikhonov or TV regularization; calibrated-n XDMF field output; Houston1km twin experiment; real USGS Harvey gauges (**data acquired**: 746 stage/discharge gauges + locations in `data/harvey_gauges/`, HydroShare DOI 10.4211/hs.c037167e497546a1bc1508dfb32a9cff, CC-BY; see its README for the datum-conversion QC) | not started | O | QC review of the gauge datum conversion (stage-above-local-datum → WSE); regularization choice + prior (e.g. land-cover-based n₀); acceptance criteria for "calibrated" |
| 7 | Paper draft | Methods, twin + Harvey results, comparison/positioning vs related work | not started | F | Co-author list, venue choice (WRR / JAMES / GMD), synthetic-vs-real scope decision |

Increments 0–2 are pure library work with zero user-visible change (flag
default off). Increment 3 is the first driver work. From increment 4 on,
each increment produces a citable artifact (sensitivity map → recovered
zones → Manning map).

### Existing assets (why this is incremental, not greenfield)

- **Worked Jacobian examples exist in-tree** (on `bishtgautam/heat` — a
  branch this effort does **not** build on, but a useful reference): an
  implicit TS (TSBEULER/TSCN), a Jacobian Mat from `DMCreateMatrix`,
  `TSSetIJacobian` with analytic per-cell derivatives
  (`src/heat/heat_petsc.c`), and notably a **libCEED Q-function
  Jacobian** (`src/heat/heat_ceed.c`) — the pattern to copy if/when we
  want a CEED variant of the (equally diagonal) drag-source Jacobian on
  GPU. All the heat Jacobians are per-cell **diagonal**; the coupled
  off-diagonal flux Jacobian and the adjoint machinery — this plan's
  contribution — exist nowhere.
- Manning n already lives as a per-cell material property
  (`MATERIAL_PROPERTY_MANNINGS`, `src/swe/swe_petsc.c:769` on main), with
  per-region and per-domain setters in the public API
  (`RDySetRegionalManningsN`, `RDySetDomainManningsN`).
- **ARK-IMEX is half-built on main**: `TEMPORAL_ARK_IMEX` +
  `SOURCE_ARK_IMEX` ("bed friction moved to LHS") exist in the config
  schema with validation, and the CEED source kernels already branch on
  `SOURCE_ARK_IMEX` — but no `TSARKIMEX`/`TSSetIFunction` is ever
  created (`rdysetup.c` has no `TEMPORAL_ARK_IMEX` case). Increment 1b
  finishes this.
- **Gauge-site plumbing is native on main**:
  `output.time_series.observations.sites.cells` in YAML
  (`RDyObservationSites`, natural-ID scatter in `src/time_series.c`;
  ex2b observes cells `[0,1,2,43]`). Increment 3 reuses this as the
  gauge-location convention — only observed-*values* input and the
  misfit are new.
- CFL-adaptive stepping exists (`rdyadvance.c`, `target_courant_number`
  in `Houston1km.DirichletBC.adaptive_timestep.yaml`) — it targets CFL
  only, which is exactly why IMEX friction (1b) is what makes aggressive
  adaptivity safe in thin-water regimes.
- Driver pattern established: `driver/main.c`, `driver/mms.c`, and
  `driver/letkf_test.c` on the unmerged `adams/da-test` branch — the
  adjoint driver copies this shape (read it via
  `git show adams/da-test:driver/letkf_test.c`).
- Houston1km Harvey case + USGS gauge data already in hand from the Harvey
  simulation work.
- `TSAdjoint`, `TSTrajectory`, TAO, FD-coloring: all stock PETSc; no new
  dependencies. Direct line to Hong Zhang (ANL) for TSAdjoint internals.

### Constraints

- **Friction treatment on the adjoint path**: increments 0–5 run plain
  *explicit* friction (wet validation cases; clean RHS and Jacobian).
  The semi-implicit splittings (`SOURCE_SEMI_IMPLICIT`, `IMPLICIT_XQ2018`)
  are never used on the adjoint path — they embed `dt` inside the RHS,
  which is consistent only with forward Euler and admits no well-defined
  `∂f/∂u`. Increment 1b (friction in `TSSetIFunction` under `TSARKIMEX`)
  must land before the increment-6 Harvey calibration, where thin-water
  phases make explicit friction unaffordable (10–100× step count and
  trajectory storage).
- Base the work on `main`. The heat branch is not in this line of
  development; consult it read-only as the worked example for
  IJacobian registration and the CEED Q-function Jacobian pattern.
- CPU/PETSc operator backend for the SWE Jacobian and adjoint. The heat
  example shows CEED can express *pointwise/diagonal* Jacobians as
  Q-functions — a viable later route for a GPU drag-source Jacobian —
  but there is no CEED path to the coupled off-diagonal flux-Jacobian
  assembly the adjoint needs (GPU adjoint is future work).
- Wet/dry fronts limit adjoint smoothness: validate on fully wet cases
  first (MMS, wet-bed dam break); accept subgradients at switching sets;
  keep the regularized h floor consistent between forward and adjoint.

---

## Future work (post-paper; feeds the recompete proposal)

Collected from the parent plan and the related-work survey; none of this
blocks the Manning-map endpoint.

1. **PyTorch coupling via the PETSc↔PyTorch bridge** — states, ensembles,
   and TSAdjoint gradients exposed zero-copy via DLPack; RDycore as a
   `torch.autograd.Function`; torch optimizers (Adam/L-BFGS) replacing TAO.
   The training substrate for the AI-LETKF program.
2. **Learned physics in the solver** — NN model-error correction and
   surface–groundwater exchange-flux source terms trained *through* the
   SWE solve (cf. Hydrograd's universal-SWE approach, in a production code).
3. **Hybrid ensemble–variational DA** — adjoint gradients correcting the
   LETKF analysis (EnVar) on the twin experiment; RMSE vs pure LETKF.
4. **Satellite observations** — SWOT water-surface elevation in the same
   gauge-misfit machinery (proven by the DassFlow/HiVDI line); flood-extent
   (wet/dry indicator) data later, with smoothed misfits.
5. **Velocity observations** — drone-video surface velocimetry; literature
   says velocity is more sensitive to n than stage — better conditioning,
   harder convergence.
6. **Implicit/IMEX stepping** — promote semi-implicit friction to
   `TSARKIMEX`/`TSBEULER` using the increment-2 Jacobian (SNES+KSP); entry
   point for differentiable-implicit-solve (KSPSolve as autograd node).
7. **Second-order adjoints / UQ** — TSAdjoint Hessian-vector products for
   posterior uncertainty on the calibrated n field.
8. **Optimal experiment design** — gauge-placement studies (which
   observations constrain n where), following the Thetis/Warder line.
9. **Sediment parameters** — Hairsine-Rose coefficients via the same
   `RHSJacobianP` route.
10. **GPU adjoint via a Kokkos operator backend** — migrate the GPU
    operator path from libCEED to the PETSc FEM-Kokkos template-kernel
    model (`petscfekokkos.h` in petsc-claude: compile-time
    `KOKKOS_INLINE_FUNCTION` pointwise callbacks, device COO assembly),
    adapted from FE cell integrals to FV edge/cell loops. Venue choice:
    RDycore-local kernels (fast, unconstrained) vs upstreaming a
    `PETSCFVKOKKOS` on PETSc's existing `PetscFV` class — which RDycore
    doesn't use today and which has no Jacobian machinery for any
    backend, so (b) would add the first FV Jacobian path to PETSc.
    Pragmatic middle: local kernels with PetscDS-compatible callback
    signatures, promotable later. Gives
    single-source physics (retires the CEED/PETSc duplication), a
    device-assembled Jacobian (GPU adjoint + implicit stepping), and one
    programming model shared with PETSc's incoming Kokkos `DMSwarm`
    backend for the particle-tracking work. Run beside CEED behind the
    existing backend switch until parity. (Explored and deprioritized
    2026-08-18: no device work in paper-1 scope; revisit at the
    GPU-adjoint stage.)
11. **Other calibration targets** — bathymetry and inflow/boundary
    discharge via the same machinery (the DassFlow feature set).
