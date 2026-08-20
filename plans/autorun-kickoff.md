# Autonomous run kickoff: Manning-calibration draft implementation + paper draft

**Mode:** semi-autonomous. Mark is reachable — stop and ask when a
genuine fork arises (physics judgment, scope, anything destructive);
never ask for read-only operations. Stay in this repo; commit locally;
never push. Decisions below are pre-made; for gaps, choose the simplest
option that keeps tests honest, record it in RESULTS.md, and continue.

**Specs (read first):**
- `plans/manning-map-incremental-plan.md` — the increment table (0–7 with
  1b, 2a–2d), exit tests, and constraints. This is the contract.
- `plans/pi-briefing-manning-calibration.tex` — §7 appendix maps every
  equation to its PETSc API; §8 is the IMEX rationale.

## Setup

- Branch off `main`: `adams/manning-draft`. Commit locally per
  sub-increment (test output summarized in the commit message). Never push.
- Fresh out-of-source build: `build_manning/` with
  `PETSC_ARCH=arch-macosx-gnu-g` (debug, for development) and a second
  `build_manning_opt/` only if timing numbers are wanted for the draft.
  Do NOT reuse the stale in-source CMake files in `driver/`.
- Maintain `plans/RESULTS-manning-draft.md`: per increment — status, test
  names + measured errors, decisions taken, anything punted.

## Priority order (must-have chain first)

1. **0** config flag + FD-coloring Jacobian baseline
2. **1** preallocated bs=3 Mat + analytic source blocks (drag, bed slope)
3. **2a** edge-level Roe-Jacobian FD harness (unit test, no mesh)
4. **2b** physical-flux + frozen-dissipation Jacobian + global assembly;
   MMS FD test passes
5. **3** trajectory + gauge-misfit cost + `driver/adjoint_test.c`;
   dJ/du₀ vs FD
6. **4** `TSSetRHSJacobianP` (∂S/∂n, 2 nonzeros/cell); per-region dJ/dn
   vs central FD; sensitivity-map XDMF output
7. **5** TAO BLMVM per-region twin on ex2b: recover two-zone n
8. **6-twin** per-cell n + Tikhonov on Houston1km, synthetic gauges →
   **the Manning map** (XDMF)
9. **Paper draft**: `papers/manning-calibration/` LaTeX. Structure and
   citations from the PI briefing; results = twin-experiment numbers
   only. Build must be clean.

**Stretch (only after 1–9 complete), in order:** 2c exact dissipation
derivatives → dam-break FD test at 1e-6; 1b IMEX friction
(TSARKIMEX + IFunction/IJacobian) with parabolic-bowl step-size
measurement; 2d BC blocks; real-gauge preprocessing
(`data/harvey_gauges/`, datum conversion via USGS NWIS site metadata) and
a real-data calibration clearly marked PRELIMINARY in the draft.

## Fixed decisions (do not relitigate)

- Roe flux only; entropy-fix branches get consistent one-sided
  derivatives. HLLC does not exist in this run.
- Explicit friction everywhere in increments 0–6 (wet cases; Houston at
  this scale affords it). Semi-implicit source modes are never enabled on
  the adjoint path.
- CPU/PETSc backend only. No CEED edits, no Kokkos, no GPU.
- New public API: prototype through `private/rdycoreimpl.h` like
  `letkf_test.c` (read it: `git show adams/da-test:driver/letkf_test.c`);
  promote to `rdycore.h` only if time remains after the chain.
- YAML: `numerics.jacobian: analytic|fd|none` (default `none`); gauge
  cells via existing `observations.sites`; observed values as a simple
  CSV `site,cell,time,value` the driver reads (twin mode generates it).
- TAO: BLMVM, bounds n ∈ [0.01, 0.2]; Tikhonov to constant n₀ = 0.03 for
  per-cell; per-region first.

## Guardrails

- **Tolerances are frozen**: 2a harness ~1e-7 (central FD), global FD
  tests 1e-6 relative Frobenius (MMS/2b), gradient tests ~1e-5 relative.
  If a test cannot meet its tolerance, the increment is NOT done — record
  the measured error and the diagnosis in RESULTS.md and either fix it or
  take the documented offramp (2c's offramp: ship frozen-dissipation
  Jacobian; gradient tests then validated on smooth MMS flow only, and
  RESULTS.md says so). Never weaken a tolerance or swap a test config to
  make red turn green.
- All existing tests must stay green (flag default off = zero impact);
  run the existing swe_roe suite after increments 0–2.
- Commit/checkpoint between build/test cycles, never mid-debug. If
  context is nearly exhausted, update RESULTS.md + memory with exact
  state and next step before doing anything else.
- No pushes, no force ops, no history rewrites, no edits outside this
  repo. Data downloads: USGS NWIS metadata only (for datum elevations),
  nothing else.
- Paper draft claims only what RESULTS.md substantiates. Preliminary
  things are labeled preliminary.

## Definition of done

The branch builds clean; CTests for increments 0–6 (twin) pass at frozen
tolerances; a Manning-map XDMF exists from the Houston twin; the paper
draft PDF builds with real twin-experiment numbers; RESULTS.md tells the
whole story, including what was punted.
