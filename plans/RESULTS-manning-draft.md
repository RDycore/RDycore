# RESULTS — Manning-calibration draft run

Branch: `adams/manning-draft` (off main, local only). Build:
`build_manning/` (Ninja). Started 2026-08-18.

## Environment log

- RDycore hard-requires PETSc with libceed+muparser (pkg-config) and
  hdf5/netcdf/exodusii support. No existing PETSc arch qualified:
  `petsc_gem/arch-macosx-gnu-g` once had the full set (externalpackages/
  remnants) but a later minimal reconfigure dropped every HAVE_ flag.
- Resolution in progress: Mark reconfiguring
  `petsc_gem/arch-macosx-gnu-O` (opt, no-fortran) — flagged that it must
  keep MPI on and add --download-{libceed,muparser,hdf5,netcdf,pnetcdf,
  exodusii}. RDycore will build with -DENABLE_FORTRAN=OFF.
- RDycore CI reference PETSc: v3.24.1; petsc_gem is v3.25.3-dev-361.
  build_gem (old working build) proves petsc_gem works with RDycore.

## Increment status

| Inc | Status | Notes |
|-----|--------|-------|
| 0 | **PASS** (committed 76f474f9) | config flag + FD-coloring Jacobian + **new `source.method: explicit`** + smoke test (`ex2b_jacobian_fd.yaml`); awaiting PETSc arch to compile |
| 1 | **PASS** (source block vs FD < 1e-6) | `SWESourceJacobian` (drag+bed-slope diagonal blocks) in `swe_roe_flux_jacobian_petsc.h`; FD-verified in unit harness (`TestSourceJacobian`) |
| 2a | **PASS** (near-equal 1e-5 gate; distinct-state frozen err 8.9e-2 recorded for 2c) | `src/tests/test_swe_jacobian.c`: edge-level FD harness (Richardson self-check, near-equal gate 1e-5, distinct-state 2c gate, dry/dry) |
| 2b | **PASS** (global rel err 1.4e-8, np=1 and 2) | frozen-dissipation flux blocks (`SWERoeFluxJacobian`) + full analytic assembly (`SWERHSJacobianAnalytic`, MatSetValuesBlockedLocal, mirrors RHS ownership/wet-dry branches); global FD-vs-analytic test on uniform flowing lake with boundary-row masking (`test_swe_jacobian_global`, gate 1e-6) |
| 3 | **PASS** (dJ/du0 vs FD 1.03e-8; commit 3a1e15db) | `driver/adjoint_test.c` (rdycore_adjoint): terminal-time gauge misfit v1 (multi-time obs deferred), truth+perturbed-IC twin, `TSSetSaveTrajectory` + `TSSetCostGradients` + `TSAdjointSolve`; per-component dJ/du0 FD gate 1e-5; CTest `adjoint_dam_break_np_1` (uses `jacobian: fd` so the gate is independent of frozen-dissipation accuracy) |
| 4 | **PASS** (dJ/dn aggregate vs FD 2.8e-6) | `SWERHSJacobianP` (∂f/∂n, 2 nnz/cell) + `rhs_jac_p` auto-created at registration; driver computes mu=dJ/dn in the same sweep; domain-aggregate FD gate 1e-5 |
| 5 | **PASS** | TAO/BLMVM two-zone twin: n_true (0.03, 0.06) recovered from uniform 0.02 to 4.1e-7 rel (gate 2e-2); CTest `calibrate_manning_twin_np_1`; `-adjoint_calibrate` mode in the driver |
| 6-twin | pending | per-cell Houston Manning map |
| paper | pending | draft |
| 2c | **PASS** | exact flux Jacobian (hand-written forward-mode differential, entropy-fix branches included): distinct-state 7.6e-9, transcritical 2.4e-10, dam-break global 1.5e-8 |
| 2d | **PASS** | boundary blocks (reflecting/Dirichlet/critical-outflow): full-matrix (unmasked) FD-vs-analytic 1.5-1.9e-8 on 3 state/BC combos; analytic-Jacobian adjoint passes both FD gates (1.05e-8 / 2.9e-6) |
| 6-percell | **PASS** (dam-break twin) | multi-time observations (segmented TSAdjointSetSteps sweeps, ex20td pattern): recovery 39.5% (terminal-only) -> 19.9% (8 obs times) -> **8.0%** (20 obs times, beta 1e-5, converged 1864 its, 2 min). Analytic Jacobian cut iteration cost ~6x (76 s -> 12.5 s per 200 its). CI test `calibrate_manning_percell_np_1` (gate 0.25, measures 0.199) |
| implicit (BEULER) | **PASS** | fully-implicit backward Euler enabled (validation now allows beuler with jacobian != none): SNES driven by the exact coupled Jacobian; theta-method adjoint passes both FD gates (1.4e-8 / 5.3e-6) with tight inner tolerances. Finding: implicit-adjoint gradient accuracy tracks SNES/KSP tolerance. CTest `adjoint_beuler_np_1` |
| 6-houston | **semiconvergence found** | implicit BEULER carries the Harvey window at dt=30 s (explicit friction NaN'd immediately, confirming briefing Sec. 8). Per-cell twin (2746 params, ALL observable -- whole domain wet+moving; 343 gauges x 14 obs times): 300-it capped run = 26.3% recovery, ~22 min laptop, `houston_manning.vtu` written. 2000-it run DEGRADED recovery to 38.0% (weak beta=1e-5, 8:1 param:gauge ratio -> textbook semiconvergence; parameters pinned at bounds) -- early stopping was the effective regularizer; densely-observed dam-break twin converges monotonically to 8%. Properly regularized run (beta=1e-4, 686 gauges) in background (`houston_reg.log`). Two-panel drift figure in the paper |
| paper | **drafted** | `papers/manning-calibration/manning-calibration.tex` (6 pp, builds clean): methods, full verification table, observability study, implicit result, nondifferentiability catalog; Houston numbers preliminary pending converged run |
| 1b (IMEX) | **PASS** | `temporal: arkimex` + `source.method: ark_imex` wired end-to-end: friction in TSSetIFunction/IJacobian (per-cell block-diagonal), Manning dependence via TSSetIJacobianP plus a zero explicit-part RHSJacobianP (the ARKIMEX adjoint dereferences BOTH dF/dp and dG/dp -- PETSc gotcha, SEGV otherwise). Adjoint gates 2.0e-8 / 2.5e-6 with `-ts_adapt_type none` (the embedded controller otherwise injects ~1e-4 FD noise). Houston at dt=30 s: 6.7 s/forward+adjoint, gradients match BEULER to 3 digits. CTest `adjoint_arkimex_np_1`. **CEED backend completed too** (finishing Jeff Johnson's PR #359 prep): his friction-free explicit Q-functions + the (backend-agnostic, host-array) implicit friction part; PETSc-vs-CEED parity after 100 ARK-IMEX steps: 5.1e-16. CTests `swe_roe_ex2b_arkimex_c_np_1_{basic,ceed}`. Device-resident CEED IFunction remains the follow-up |
| stretch: real gauges | **UNBLOCKED -- CRS solved, pipeline built** | Mesh CRS = **EPSG:32610** (WGS84/UTM 10N -- the California zone used 27 deg off-meridian; found in donghuix/OFMmesh ex1 `projcrs(32610)`, verified by bbox inverse-projection onto Addicks/Barker/Turning-Basin geography AND stage-vs-elevation consistency). `map_gauges_to_cells.py`: 20/90 gauges in mesh, 17 with Harvey stage series; `houston_gauges_cells.csv` + `observations_sites_snippet.yaml` (natural cell IDs). Datum trivial: in-mesh gauges publish stage as WSE ft NAVD88 (alt_va=0). **Team flag: projection carries ~9% length / ~19% area inflation + 14.5 deg rotation** -- "1 km" cells are ~917 m true; slopes/areas/rain volumes inherit this; calibrated n will absorb part of it (paper structural-error angle) |

## PETSc sandbox endgame (2026-08-18 late)

- petsc_gem branches: `adams/fix-arkimex-adjoint-jacprhs` (main + the
  #1925 fix, MR-ready for Hong; fix verified -- reproducer correct with
  NO workaround; ex20adj suite green) and `adams/rdycore-stable-base`
  (= 8caedce v3.25.3-361 + cherry-picked fix) which is what
  arch-macosx-gnu-rdycore-O is now built from and what RDycore links.
- SECOND petsc-main regression found: TAOBLMVM/BQNLS SEGV in
  MatLMVMUpdate -> LMBasisGetNextVec -> MatDenseGetColumnVecWrite at the
  2nd LMVM update (40-line pure-TAO reproducer
  `plans/ex_tao_blmvm_segv.c`; works on v3.25.3-361). FILED as petsc/petsc#1926. This is why RDycore
  stays on the stable base rather than main.
- Final suite vs stable-base+fix: 129/136; all 14 new tests green; same
  7 environmental failures (6 cgns, amr_np3 pre-existing).
- Houston regularized run: beta=1e-4, 686 gauges, 1000 its -> 19.4%
  recovery, no drift, sharp zone boundary (`houston_manning_reg.vtu`).
  Paper finalized: abstract, three-panel regularization figure, 19.4%
  headline; only the co-author TODO remains.

## Second-order FV (MUSCL) review + guards (2026-08-19)

- Reviewed RDycore PRs #382 (CEED MUSCL) and #393 (PETSc-backend 2R);
  both already merged into our base. Main has since added a van Leer
  limiter (PETSc-only) -- the adjoint-friendly limiter choice (no
  |a|=|b| kink; `-no_limiter` is NOT wet-dry-safe: unlimited
  extrapolation can reconstruct h>tiny_h on dry-dry edges).
- HAZARD found and closed (commit 88689246): no guard existed between
  `numerics.jacobian` and `numerics.second_order`. analytic+2R would
  silently corrupt TSAdjoint gradients (adjoint needs the exact
  linearization; forward Newton merely slows). Now rejected in
  `RegisterSWERHSJacobian`; fd+2R restricted to triangular meshes
  (closure-adjacency pattern covers the 2-hop MUSCL stencil only on
  simplicial cells -- planar_dam_10x5.msh is quads and trips it).
  Both branches verified to fire; 12-test jacobian/adjoint suite green.
- fd+2R on TRI meshes (Houston) is correct TODAY at FD cost. Analytic
  2R Jacobian is tractable: J = A(u*)·R with A = existing edge blocks
  at reconstructed states, R linear (precomputed ls_grad_coeffs) when
  unlimited; stage no_limiter -> van Leer (one-sided derivs). v2 idea:
  truth with 2R, calibrate 1st-order -> quantifies scheme-induced n
  bias (calibrated n absorbs O(dx) diffusion at 1 km).

## Suite status (build_manning, arch-macosx-gnu-rdycore-O)

129 CTests: 122 pass. 7 failures, all judged environmental/pre-existing:
6x cgns output tests ("Unknown PetscViewer type: cgns" -- arch built
without --download-cgns) and amr_c_np_3_basic (SEGV; VERIFIED identical on pristine main via worktree -- pre-existing; my code paths are
inert in that config -- no jacobian flag, default source method; verify
against pristine main in a worktree when convenient). All 7 new tests
pass. TSEULER has no PETSc adjoint -- adjoint-path configs use rk4.

## Findings (2c/2d round)

- Critical-outflow boundary is genuinely nondifferentiable at stagnation
  (uperp = 0): the ghost map (q^2/g)^{1/3} has an unbounded one-sided
  derivative and the inflow branch switches there. Consistent one-sided
  derivatives are used; test states sit off the kink (real outflows flow).
  Worth a sentence in the paper's nondifferentiability discussion.
- The exact flux Jacobian is a hand-written forward-mode differential of
  the flux computation (mirrors ComputeSWERoeEigenspectrum line by line);
  landed correct on first build against the 2a harness.

## Decisions taken during the run

- **Paper v1 scope (Mark, 2026-08-18): frozen at twin-experiments-only**
  for review speed; real-gauge calibration is follow-on work (v2 /
  revision), not a v1 blocker. The draft already states this in Sec. 7.
- Mesh-CRS question sent to the team by Mark; when answered, the
  real-gauge pipeline (houston_gauges.csv -> cells -> WSE misfit) is
  mechanical.

- **Discovery: RDycore had NO explicit friction option** — only
  `semi_implicit` (dt and flux-divergence inside the source!) and
  `implicit_xq2018`. Added `SOURCE_EXPLICIT` / `source.method: explicit`
  (`ApplySourceExplicit` in `swe_petsc.c`): tb = Cd|v|q/h, dt-free —
  prerequisite for a well-defined ∂f/∂u. PETSc backend only; CEED
  switches hit their guarded defaults.
- Jacobian pattern from `DMCreateMatrix(rdy->dm)` under the DM's
  closure adjacency (superset of edge adjacency — extra zeros accepted;
  can tighten later).
- FD baseline: `MatFDColoring` over `TSComputeRHSFunction` (SL coloring,
  distance 2), lazily built on first Jacobian request; kept permanently
  as `numerics.jacobian: fd`.
- New struct fields: `rdy->rhs_jac`, `rhs_jac_fd_coloring`,
  `rhs_jac_time`; cleanup via `DestroySWERHSJacobian` in `RDyDestroy`.
- Environment: PETSc arch with {libceed,muparser,hdf5,netcdf,pnetcdf,
  exodusii}+MPI still being rebuilt by Mark (`arch-macosx-gnu-O` in
  petsc_gem); two failed configures so far (--with-mpi=0 both times —
  pnetcdf needs mpi.h). Exact corrected option set handed over.

## Dam-representation finding (2026-08-19, evidence for team question #1)

Elevation transects across the Addicks and Barker dam alignments
(nearest-cell sampling in EPSG:32610) show NO embankments in
Houston1km_with_z.exo: Barker E-W line is a smooth 24->19 m slope (crest
would be ~34 m), Addicks N-S is a smooth 23->35 m regional gradient (no
~37 m ridge). Consistent with `add_dam = 0` in OFMmesh
Script_01_Generate_Meshes.m ("burn in dam geometries doesn't improve
the simulation") and 1-km cells vs ~100-m-wide embankments. Harvey
water therefore flows freely through both reservoir sites: downstream
Buffalo Bayou gauges (Piney Point, W Belt, nr Addicks) saw
USACE-operated impound-then-release hydrographs the model cannot
reproduce with ANY Manning field. Clean (rain-driven) observation set:
the Whiteoak Bayou network + upstream-of-reservoir creeks (Katy,
Fulshear, S Mayde, Bear, Langham) -- roughly 10-11 of the 17
stage-series gauges.

Follow-up with the OFMmesh dam polylines (ex1/dams/dam{1,2}.shp,
crest-profile sampling of nearest cells at 200-m spacing): the 1-km
mesh DID intend dams-as-elevation (Mark: no operations modeled, terrain
only) but coarsening smoothed them away. Crest-line cell z: Addicks min
25.7 / med 28.5 / max 33.9 m (real crest 36.9 m); Barker min 25.9 / med
27.7 / max 30.8 m (real 34.0 m). Harvey peak pools: 33.2 / 31.0 m. So
the model's effective sill (~26 m) is 5-7 m below Harvey pools -- both
reservoirs spill broadly once pools pass ~26 m (~late Aug 27), while
reality had closed gates (releases Aug 28, Addicks emergency spillway
32.9 m late Aug 29). Options for the team: (a) rain-driven gauges /
pre-Aug-27 window only, (b) raise the ~40 crest cells to true crest z
(restores closed-gate impoundment; releases still unmodeled).

## Real-gauge observation pipeline (2026-08-19, local work during PM day)

- `rdycore_adjoint -adjoint_calibrate_gauges`: per-cell Manning
  calibration from a gauge WSE table (plain text: cells, times, WSE m;
  nan = missing, zero-weighted via per-obs 0/1 weights threaded through
  ForwardObserve). H rows built from NATURAL cell IDs; y_k = WSE - zb
  (cell centroid z). `-adjoint_gauges_twin` synthesizes the table from
  a two-zone truth through the same file path. CTests
  `calibrate_manning_gauges_np_{1,2}` (J reduction 10.48x, np1/np6
  identical to 6 digits).
- `data/harvey_gauges/make_obs_table.py`: stage CSVs -> table
  (--subset rain-driven excludes the 8 dam-affected sites); Harvey
  Aug 26 + 36 h hourly: 12 gauges, 422/432 obs present.
- Fixes en route: (1) observation-matrix column layout must match
  u_global's uneven DMPlex distribution (PETSC_DECIDE broke np>1 --
  latent in the stride H too, fixed both); (2) driver now consumes
  -ts_trajectory_* options (RDycore's TSSetFromOptions predates the
  trajectory); (3) DMSetUseNatural(dm, FALSE) in the driver: the basic
  disk trajectory VecLoads through the natural-order path, unsupported
  for non-HDF5 at np>1. FINDING: -ts_trajectory_type memory SEGVs on
  the segmented multi-obs adjoint (NULL stack element in
  TSTrajectoryGet_Memory/UpdateTS, trajmemory.c:526) -- candidate THIRD
  petsc issue; disk trajectory is the supported path for now.
- QC finding (1-km representativeness): downtown Whiteoak gauges' WSE
  never exceeds their cell-mean bed (peak 13.5 m vs zb 14.8 m) --
  incised channels are subgrid at 1 km, so those gauges are degenerate
  (h_obs clamps to 0) on Houston1km. Resolved naturally at 30 m --
  further motivation for the Turning_30m move.

## Turning_30m mesh acquired + verified (2026-08-19 afternoon)

- NERSC DTNs (dtn01) stayed up through the PM day: mesh (70 MB),
  solution_219.int32.dat IC, and spatially-distributed-rainfall pulled
  via scp. Also on disk there: Turning_3m (273 GB!), _3m_quad, _6m.
- Same domain/CRS as Houston1km (EPSG:32610). 2,926,532 tri cells,
  z in [-1.3, 63.0] m (dredged ship channel resolved).
- DAM CRESTS AT 30 m (barrier = max cell z within 120 m of the OFMmesh
  crest polylines): Addicks min 32.2 / med 33.5 / max 36.5 m, NO
  sub-31 m gaps -- impounds to ~the real emergency spillway (32.9 m).
  Barker: part of the alignment IS the domain boundary (reflecting wall
  = perfect impoundment); in-mesh sections min 26.8 / med 31.1 m with a
  few 300-1900 m runs below 31 m -- leaks only near peak pool (31.0 m).
  VERDICT: dams-as-elevation is largely restored at 30 m (vs a uniform
  ~26 m sill at 1 km).
- Gauge remap (turning30m_gauges_cells.csv): 21 in-mesh (18 with stage
  series; new: 08074000 Buffalo Bayou at Houston). Turning Basin cell
  z = -0.1 m.
- STAGE-DATUM QC OPEN ITEM: NWIS says alt_va=0 NAVD88 for all 21 (and
  the HydroShare CSVs match official NWIS records exactly, e.g. peak
  44.31 ft at 08074500), but stage-as-NAVD88-elevation is inconsistent
  with mesh channel elevations at some gauges (08074500 baseline
  10.4 ft vs ~35 ft channel in-mesh) while consistent at others
  (Katy, Piney Point). Per-gauge datum verification needed (HCFCD
  conversion history) -- next team question.
- Production config: dt=0.25 s explicit Euler (Frontier: 345,600 steps
  = 1 simulated day in 4810 s on 112 cores). Local np=6 explicit is
  ~real-time; implicit dt=30 s probe pending (smoke tests running).

## Turning_30m smoke tests (2026-08-19, np=6 laptop, 2.93M cells)

Setup: mesh + solution_219.int32.dat IC + critical-outflow BC, uniform
n=0.015, dt=0.25 s (production). Config schema in the docs example is
STALE (final_time/output.step_interval rejected); base config rebuilt
from driver/tests/swe_roe/Houston1km.DirichletBC.yaml.

| config | dt | result |
|---|---|---|
| euler + `source: semi_implicit` (production) | 0.25 s | STABLE, Courant 0.27 steady, **0.634 s/step** (measured: 100 steps 82.8 s, 400 steps 273.1 s; setup 19 s) |
| euler + `source: explicit` (differentiable) | 0.25 s | **UNSTABLE** -- Courant 0.27, 0.27, then 1.8e4, 4.9e8, 8.9e11 ... NaN by step 4 |
| arkimex + `source: ark_imex` (differentiable) | 30 s / 5 s | DIVERGED_NONLINEAR_SOLVE (50 SNES its) -- Courant ~1e3 at 30 m |
| arkimex + `source: ark_imex` | 0.25 s | **STABLE**, Courant 0.27 steady, but **56 s/step** (90x explicit) |
| rdycore_adjoint, 20 rk4 steps + adjoint sweep | 0.25 s | RUNS at 2.9M cells (586 s) but J = NaN (explicit-friction blowup above) |

**Root cause of the implicit cost (from -log_view, 2 steps = 113 s):**
MatLUFactorNum 23.0 s + SNESJacobianEval 19.0 s + MatZeroEntries 9.4 s
+ MatSetValues 12.2M calls 7.7 s. Linear solves converge in **1 KSP
iteration** -- the Krylov solve is not the problem. The IMEX friction
IJacobian is allocated with `DMCreateMatrix(rdy->dm)` (full
closure-adjacency flux stencil) although friction is per-cell 3x3
BLOCK-DIAGONAL, so every Newton step zeroes, assembles, and ILU-
factorizes a ~10x oversized matrix. FIX (engineering, not research):
allocate imex_ijac as MATBAIJ bs=3 with 1 block/row and use
`-ksp_type preonly -pc_type pbjacobi` (an exact solve for a
block-diagonal system). Expect ~20-40x on the implicit path. (A first
try of pbjacobi on the DMCreateMatrix-backed AIJ diverged -- needs the
BAIJ allocation, not just the PC flag.)

**Cost projection (36 h Harvey window = 518,400 steps at dt=0.25 s):**
forward alone is ~91 h locally at np=6 explicit; trajectory storage for
an adjoint would be 70 MB/step x 518k = ~36 PB, so revolve-style
checkpointing with recomputation is MANDATORY at 30 m, not optional.
Cost is dominated by STEP COUNT (CFL at 30 m), not by parameter count.

**Minor RDycore bug found:** `TimeUnitString` (src/yaml_input.c:1735)
indexes a 6-entry table with RDyTimeUnit, whose enum starts at
RDY_TIME_UNSET=0 -- so every config's unit is logged one off ("seconds"
prints as "minutes"), and `years` reads out of bounds. Display only;
ConvertTimeToSeconds switches on the enum and is correct.

## Explicit-friction stability: pushback investigated (2026-08-19)

Mark pushed back on "explicit friction fails at 30 m", citing Donghui
Xu's recent work. Investigated; the claim survives, but the REASON is
source stiffness, not an implementation defect:

- Tested the hypothesis that the drag needed the ANUGA velocity
  regularization used everywhere else in the code (rewriting
  g n^2 h^{-7/3} q|q| as g n^2 h^{-1/3}|v|u). Rebuilt and reran:
  **trajectory identical to 6 digits** (Courant 17603.9 at step 2 both
  ways). Reverted -- the unstable cells have h >> h_anuga.
- Measured the actual stiffness from the spun-up Harvey IC
  (solution_219): **99.9% of the 2.93M cells are wet, median depth
  6.25 mm** (overland sheet flow). Friction rate tb = g n^2 h^{-4/3}|v|
  reaches **420/s**, so explicit friction needs dt < 2.4 ms -- ~100x
  below the CFL-limited production dt of 0.25 s. At dt = 0.25 s,
  **90,293 cells** have dt*tb > 1; the run diverges at step 2.
  dt < 0.002 s is needed for zero violating cells.
- So the friction source has its OWN stability limit, independent of
  CFL, that binds hard on thin films. `semi_implicit` escapes it by
  capping the impulse at the available momentum
  (factor = tb/(1+dt*tb)); this is what Donghui's published RDycore
  configuration uses ("explicit" in the literature = explicit
  ADVECTION + semi-implicit friction). The differentiable equivalent is
  `ark_imex`, which integrates the same drag implicitly. Documented in
  ApplySourceExplicit.

## IMEX friction Jacobian: 9.3x faster + PETSc bug #3 (2026-08-19)

- `imex_ijac` now allocated as **AIJ with block size 3**, 3 nnz/row
  (Mark: prefer AIJ over BAIJ, which is less well maintained;
  performance is close), replacing `DMCreateMatrix` whose flux-stencil
  sparsity was ~10x oversized and entirely zero off the diagonal
  blocks. Assembly switched to global blocked indices; MatZeroEntries
  dropped (every block is INSERTed). Default solver set to
  `preonly` + `bjacobi` (exact on a block-diagonal system) before
  TSSetFromOptions, so -ksp_type/-pc_type still override.
- Turning_30m, 2 steps: TSStep **56.5 -> 6.1 s/step (9.3x)**;
  KSPSolve 50.5 -> 0.71 s, MatLUFactorNum 23.0 -> 1.02 s,
  SNESJacobianEval 19.0 -> 3.19 s. Remaining cost is TSFunctionEval
  (explicit flux stages, 6.3 s/2 steps) -- irreducible physics.
- **PETSc bug #3 found and reproduced**: PCPBJACOBI silently uses STALE
  block inverses. MatInvertBlockDiagonal caches into Mat_Seq{AIJ,BAIJ}
  and the `idiagvalid`/`ibdiagvalid` flag is cleared only at the END of
  MatAssemblyEnd_Seq{AIJ,BAIJ}, AFTER an early return taken whenever
  `A->was_assembled && A->ass_nonzerostate == A->nonzerostate` -- i.e.
  on every re-assembly with unchanged sparsity. Symptom seen here:
  first linear solve converged in 1 KSP iteration, later ones in 78,
  44, 24... and with -ksp_type preonly the Newton steps were wrong
  (DIVERGED_NONLINEAR_SOLVE). 60-line reproducer
  `plans/ex_baij_invertblockdiag_stale.c` prints "second inverse: 1
  (expected 0.5)". TO FILE with petsc.

## Flow-regime / identifiability analysis of the 30 m Harvey state (2026-08-19)

From the spun-up IC (solution_219), Re = u*h/nu over 2.90M wet cells:

| regime | % of wet cells | % of water volume |
|---|---|---|
| laminar, Re < 500 | 40.6 | 2.9 |
| transitional, 500-2000 | 21.2 | 4.8 |
| turbulent, Re > 2000 | 38.2 | 92.3 |
| **Re > 2000 AND h > 5 cm** (n physically meaningful) | **10.5** | **76.0** |

Median depth 6.4 mm, median speed 0.154 m/s, median Re 914. So ~62% of
wet cells are in laminar/transitional sheet flow where Manning's n is
an effective parameter, not a channel roughness -- but those cells hold
under 8% of the water. Conversely 10.5% of cells carry 76% of the
volume. (Reassurance: the laminar-equivalent n where Re<500 has median
0.0182, close to the 0.015 the config uses -- Manning is not wildly
wrong in magnitude there, just not identifiable.)

Implication for 30 m calibration: per-cell (2.93M parameters) is
ill-posed twice over -- against ~18 gauges, and because most parameters
sit where n is neither physical nor observable. Combined with cost
(~10 node-hours per gradient even for a 6 h window) and the fact that
BLMVM iteration count scales with parameter dimension, the 30 m target
should be a LOW-DIMENSIONAL parameterization. Donghui's OFMmesh
ex2/code/Step03_Process_Manning.m already maps 18 NLCD classes to
n in [0.027, 0.160] -- a ready-made parameterization and Tikhonov prior.

FLAG FOR THE TEAM: the Harvey Turning_30m example config uses a UNIFORM
n = 0.015, below every NLCD class value (0.027-0.160). Worth asking what
that uniform value is standing in for.

## Observation-time sensitivity: which hours constrain n (2026-08-19)

New driver mode `-adjoint_sensitivity`: for each observation time t_k,
solve forward to t_k, seed the adjoint with a UNIT residual on every
gauge (J_k = sum_g h_g(t_k)), sweep back to 0, and report |dJ_k/dn|.
This measures how strongly gauge readings at t_k respond to roughness --
i.e. which observation hours carry information about n.

Houston1km, ARK-IMEX, dt = 30 s, 12 h window, 2746 observation cells.
Two experiments from the same IC: **rise** (uniform 72 mm/hr runoff) and
**recession** (no rain).

| hour | RISE depth | RISE \|dJ/dn\|_1 | per-hr | REC depth | REC \|dJ/dn\|_1 | per-hr |
|---|---|---|---|---|---|---|
| 1 | 0.193 | 1.36e3 | 1.36e3 | 0.122 | 1.30e3 | 1.30e3 |
| 4 | 0.382 | 8.22e3 | 3.10e3 | 0.106 | 4.91e3 | 1.20e3 |
| 8 | 0.572 | 3.27e4 | 8.49e3 | 0.089 | 8.20e3 | 6.26e2 |
| 9 | 0.600 | 4.19e4 | **9.24e3 (peak)** | 0.086 | 8.73e3 | 5.30e2 |
| 12 | 0.641 | 6.30e4 | 5.41e3 | 0.076 | 9.88e3 | 3.18e2 |

Findings:
1. **The rising limb carries ~6x more information about roughness than
   the recession** (6.30e4 vs 9.88e3 over the same 12 h).
2. **On the rise, information ACCELERATES** -- the per-hour increment
   grows through hour 9 and the first 6 h accrue only **28%** of the
   12 h total. Truncating the window early on the rise is expensive.
3. **In recession, information SATURATES** -- per-hour increments decay
   monotonically after hour 1 and the first 6 h already hold **69%**.
   Extending a drain-down window is poor value for money.

Window design implication: place the assimilation window on the rise and
through the peak; do not pay for long recessions. Caveats: 1 km mesh,
synthetic uniform rain, one seed functional; confirm the pattern at 30 m.

## RDycore BUG FOUND AND FIXED: YAML `sources:` runoff was a silent no-op

While setting the above up, uniform rain had NO effect -- 1e-2 m/s
(36 m/hr) produced bit-identical results to no rain, in BOTH the adjoint
driver and the main `rdycore` driver. Cause: `InitSourceConditions`
(src/rdysetup.c) loops over regions and calls
`RDySetHomogeneousRegionalWaterSource(rdy, r, ...)` passing the loop
INDEX, but that routine resolves a region **ID** via
`GetRegionIndexFromID`, which returns -1 for no match and is then
silently skipped. For the usual 1-based `grid_region_id`, index 0 never
matches ID 1, so every YAML-configured runoff source was discarded
without warning. Fixed by passing `rdy->regions[r].id`. Verified: rain
now changes the solution (and 36 m/hr correctly blows up the solver).
Existing tests unaffected -- the one test using `sources:`
(quad_tri_mesh) overrides via the driver's file-based
`-homogeneous_rain_region_ids` path, which passes proper IDs, so the
YAML path was never exercised. **Report upstream.**

## First land-cover class calibration twin, 1 km (2026-08-19 late)

`-adjoint_calibrate_classes`: truth = the NLCD-derived per-cell field, start =
uniform n = 0.015 (the current config value), observations = water level at the
**12 real rain-driven gauge cells** x 12 hourly times, 12 h rising window
(uniform 72 mm/hr runoff), ARK-IMEX dt = 30 s, beta = 1e-6 (regularization
effectively off, deliberately), 40 TAO iterations (hit the cap), np=4.

| NLCD | class | cells | truth n | recovered | rel err |
|---|---|---|---|---|---|
| 23 | Developed, Medium | 1197 (44%) | 0.1131 | 0.1098 | **0.029** |
| 24 | Developed, High | 491 (18%) | 0.1349 | 0.1446 | **0.072** |
| 81 | Pasture/Hay | 454 (17%) | 0.0484 | 0.0729 | 0.506 |
| 22 | Developed, Low | 224 (8%) | 0.0960 | 0.0901 | **0.061** |
| 90 | Woody Wetlands | 147 (5%) | 0.0954 | 0.0885 | **0.072** |
| 82 | Cultivated Crops | 67 (2%) | 0.0461 | 0.0095 | 0.794 |
| 21 | Developed, Open | 52 (2%) | 0.0755 | 0.1175 | 0.556 |
| 42 | Evergreen Forest | 44 (1.6%) | 0.1051 | 0.2439 | 1.321 |
| (9 classes < 1% each) | | 70 total | | | 0.20 - 0.77 |

Area-weighted mean relative error: **18.4%** over all 15 classes, **12.9%**
over the five classes with >= 100 cells (92% of the domain), and **4.6%** if
pasture is excluded.

Findings:
1. **The dominant classes are identifiable from 12 gauges.** The four
   developed/wetland classes covering 75% of the domain recover to 3-7% from a
   start 6x away, without converging (40-iteration cap).
2. **Small-area classes are not identifiable** and drift: everything under ~2%
   of the domain lands 20-130% off, with evergreen forest (1.6%) running to
   0.244. They should be merged, or pinned at their prior with a per-class
   regularization weight, rather than left free.
3. **Pasture (17% of area) recovers poorly (+51%) and crops (2%) collapse to
   the lower bound (-79%)** -- both agricultural, both low-n. They may be
   trading off against each other, or simply be poorly observed by this gauge
   set. Worth checking against the per-class adjoint sensitivity.
4. Cost: ~45 min wall for 40 iterations, np=4, dominated by disk-trajectory
   I/O (461 MB written and re-read per objective evaluation).

Next: re-run with per-class regularization (strong for small classes) and a
reduced parameter set (the 5-8 dominant classes), and report converged numbers
rather than iteration-capped ones.

## Team input received (2026-08-20)

**Donghui Xu:**
1. **Rainfall: MRMS** -- "according to our previous experiments, MRMS gives the
   best performance for this domain and event." Our provisional choice is
   confirmed; MRMS is locked in.
2. **At 1 km, real-gauge calibration is not meaningful** -- "we cannot match the
   observed water levels by calibrating Manning coefficients. This is because
   the dam effects are totally ignored at 1km resolution." This independently
   confirms, from his experience, what we found from the mesh geometry: the 1 km
   mesh smooths the Addicks/Barker embankments to a ~26 m effective sill against
   34-37 m real crests and 31-33 m Harvey pools, so both reservoirs leak. Two
   independent routes to the same conclusion.
3. **He suggests a synthetic objective for testing at 1 km** -- which is exactly
   the twin design we are already running (truth = NLCD-derived field, synthetic
   observations at the real gauge cells).

**Scoping consequence:** 1 km work is twin-only, for method development and
identifiability. Real-gauge calibration requires the 30 m mesh, where the dams
are resolved -- and that is exactly what Gautam wants discussed at Wednesday's
ASCR meeting. The two constraints agree.

**Gautam Bisht:** use 1 km for testing; nothing on 2.9M until the Wednesday
ASCR meeting; no single best rainfall product (deferred to Donghui, now
answered).

## CORRECTION (2026-08-20): PETSc #1926 was NOT a PETSc bug

Issue #1926 (TAOBLMVM/BQNLS SEGV in MatLMVMUpdate) is **closed as invalid**.
A clean configure+build at the same commit (v3.25.4-518-g40779859693) runs the
reproducer fine: BLMVM and BQNLS, n=2-500, np=1 and 2, 25/25 clean.

Root cause was a **contaminated PETSC_ARCH**, not a regression:
`petsc_gem/arch-macosx-gnu-rdycore-O` had a configure.log from a concurrent
configure of a *different* arch, and had been `make`d for both v3.25.4 and
v3.25.3 across branch switches without reconfiguring, leaving two
`libpetsc.3.025.*.dylib` sharing one install name.

**Manning results are unaffected**: the adjoint driver links the clean
v3.25.3-362 rebuild, and headers/library are consistent. Nothing to re-run.
The 14 adjoint/calibration/Jacobian tests were re-verified against the clean
`arch-macosx-gnu-rdycore-O-rev` (which is also the revolve-enabled arch):
14/14 pass. `build_rev` is now the working build; the contaminated arch and
`build_manning` are being deleted.

**Process lesson: reproduce against a freshly configured arch before filing a
PETSc issue.** A stale or cross-contaminated PETSC_ARCH is indistinguishable
from a real regression from the backtrace alone. This applies to the two other
PETSc findings recorded above -- the PCPBJACOBI stale block-inverse
(reproducer `plans/ex_baij_invertblockdiag_stale.c`) and the memory-trajectory
SEGV -- the memory-trajectory SEGV has NOT been
re-verified on a fresh arch and stays unconfirmed. The PCPBJACOBI one HAS now
been re-verified (below).

Note on the v3.25.3 pin: the "TAO regression" rationale is gone, but the pin
also carries the cherry-picked fix for **#1925 (ARK-IMEX adjoint dereferences
both dF/dp and dG/dp)**, which IS real and independently reproduced. Revisiting
the pin means checking whether that fix has landed upstream.

## PCPBJACOBI stale block-inverse: RE-VERIFIED on a clean arch (2026-08-20)

Re-tested per the new rule, on the freshly configured
`arch-macosx-gnu-rdycore-O-rev` (v3.25.3-362), with a reproducer that now
covers BOTH matrix types (`plans/ex_invertblockdiag_stale.c`, replaces the
BAIJ-only one):

```
  aij      pass 1: inverse[0] = 1 (expected 0.5)  <-- STALE
  baij     pass 1: inverse[0] = 1 (expected 0.5)  <-- STALE
```

So it is real, and **not BAIJ-specific: MATAIJ with a block size is equally
affected** -- which is the case that matters, since the IMEX friction Jacobian
uses AIJ with bs=3. After a matrix is re-assembled with new values and an
unchanged nonzero structure, `MatInvertBlockDiagonal()` returns the previous
values' inverse.

Looks like a genuine defect rather than intent: `MatAssemblyEnd_SeqBAIJ`
contains `a->idiagvalid = PETSC_FALSE` under the comment "diagonals may have
moved, so kill the diagonal pointers", but that line is unreachable in the
common path because the routine returns early when
`A->was_assembled && A->ass_nonzerostate == A->nonzerostate`.
`MatAssemblyEnd_SeqAIJ` has the same early return and never clears
`ibdiagvalid` at all.

Consequence for RDycore is unchanged: PCPBJACOBI would precondition every
Newton step after the first with the first step's blocks, so the ARK-IMEX
solver defaults to `preonly` + `bjacobi` (exact on a block-diagonal system).

**CONFIRMED on current main with a fresh minimal build** (pristine origin/main
worktree at dd2b456823a; `--with-debugging=0 --with-fc=0 --with-mpi=0
--download-f2cblaslapack`, no other packages). The reproducer now also shows the
user-visible symptom: `KSP(preonly) + PCPBJACOBI` on a block-diagonal matrix
returns a **silently wrong answer** after re-assembly (x = 1 where the exact
answer is 0.5), for both AIJ and BAIJ. Issue draft ready:
`plans/petsc-issue-pbjacobi-stale.md`.

## Revolve/memory-trajectory checkpointing WORKING (2026-08-20)

The memory-trajectory SEGV is root-caused and fixed driver-side (commit
7da381d4). Root cause (real PETSc design limitation, re-verified on the
clean arch): `TSTrajectoryMemorySet_N` treats `ts->reason != 0` as
"end of the forward run", but TSSolve sets reason before the final
TSTrajectorySet of EVERY call -- so a segmented forward (one TSSolve per
observation window) clobbers the scheduler's total_steps, pops the
second-last checkpoint, and skips the boundary step at every interior
observation time; the backward sweep then walks off the corrupted stack
(NULL StackElement in UpdateTS -> SEGV). Candidate PETSc issue #4
(usage-limitation report; disk trajectory tolerates segmentation, memory
cannot).

Fix: ForwardObserve now runs ONE TSSolve over the whole window with a
TSMonitor recording gauge residuals at observation steps, gated by an
`active` flag (the trajectory's ReCompute replays also fire monitors).
The segmented BACKWARD sweep is unchanged and compatible (no trajectory
writes; Get sequence is monotone across TSAdjointSolve segments).

Validation (dam break, all bit-identical to disk trajectory):
- memory default + revolve (5/40 cps): calibration 1.986e-01, FD gates
  1.0e-8 / 2.9e-6 (RK), np 1 and 2.
- BEULER 9.75e-9 / 4.18e-6 and ARKIMEX 2.77e-8 / 5.17e-6 with tight
  inner tolerances -- identical digits disk vs revolve (recompute with
  tight SNES/KSP lands on the same iterates). With DEFAULT tolerances
  implicit gradients degrade to ~1e-4-2e-4 (known inner-tolerance
  effect, same as disk).
- Full adjoint suite 14/14.

FIRST 30 m ADJOINT (Turning 2.93M cells, ARK-IMEX, np=6, dt=0.25 s,
10-step window, obs every step, per-cell twin, 1 TAO it):
- disk:    J0 5753.65 |g0| 111367 -> J1 3772.38 |g1| 64904.5,
           wall 1013.6 s, peak RSS 4.31 GB, traj ~150 MB/step on disk
- revolve (max_cps_ram 3): IDENTICAL to all printed digits,
           wall 951.9 s (revolve recompute is CHEAPER than disk I/O),
           peak RSS 3.32 GB
Config: build_rev/harvey_run/ark10.yaml (ark40.yaml minus stop/coupling).

36 h window projection (518,400 steps): solution-only checkpoint =
70 MB aggregate; c=200 cps (14 GB aggregate RAM) gives revolve
recompute factor <= ~4x (C(203,3) = 1.37M >= 518k). Naive trajectory
would be ~36 PB -- revolve makes the full-window adjoint feasible; at
6.1 s/step the wall time is a Perlmutter job, not a laptop job.
Gradient-critical production runs should use tight inner tolerances
(or -ts_trajectory_solution_only 0 to store stages instead of
recomputing them).
