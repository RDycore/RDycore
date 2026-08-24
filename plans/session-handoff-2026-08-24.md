# Session Handoff — August 24, 2026

Branch `adams/gpu-implicit`, all work pushed through **`1a324e68`**.
Laptop and Perlmutter trees are at the same commit; nothing is running on
Perlmutter; working tree clean.

**Read first:** `plans/next-phase-strategy.md` (what to do and why), then
`plans/campaign-wednesday.md` (measured results + meeting agenda). The
engineering history is `plans/RESULTS-gpu-implicit.md`.

---

## 1. What landed this session

**ANUGA-regularized drag (`3110ac50`).** The Manning drag is now
evaluated with the ANUGA-regularized velocity when
`physics.flow.h_anuga_reg_parameter > 0`, so drag fades smoothly to zero
as depth does instead of switching on/off at the `tiny_h` gate. Shared
helper `ComputeSWEManningDrag` in `swe_roe_flux_petsc.h` serves the host
source loop, the device kernel, and the ARK-IMEX IFunction;
`SWEFrictionJacobian`/`SWESourceJacobian` carry the exact regularization
derivatives; new shared `SWEFrictionDN` serves both dF/dn paths.

Gates, both passed: at `h_anuga = 0` every path is **bitwise identical**
to the previous code (ctest 14/14; adjoint_beuler + adjoint_arkimex
kokkos trajectory dumps `cmp`-identical at np 1/2). At `h_anuga > 0` the
full FD suite passes — new unit twins, a new global FD-coloring twin pair
(`swe_jacobian_global_anuga_*.yaml`), and driver dJ/du0 + dJ/dn across
{beuler, arkimex} x {host, kokkos} x np {1,2}.

**Chosen recipe:** `h_anuga = 0.001` (smallest that clears the stall;
0.003 indistinguishable; 0.01 unstable) — in `beuler_dt1_anuga.yaml` and
`beuler_dt1_1hr_anuga.yaml` on PM.

**2-node smoke passed (o19):** 1-hr revolve gradient 13.5 -> 7.4 min on
2x4 A100s. The 12-hr paper window is viable as an overnight 2-node job.

**Both data clocks resolved.** USGS HydroShare CSVs are local CDT (Piney
Point 63.94 ft at 13:00 matches NWIS-with-timezone), and the MRMS rasters
are ALSO local CDT (rain vs d(stage)/dt cross-correlation peaks at offset
0, r = 0.65; the UTC hypothesis gives r = -0.09). So
`--t0` == `-raster_rain_start_date`, no shift. Documented in
`data/harvey_gauges/README.md`.

**Paper updated for Wednesday.** Nondifferentiability catalog corrected
(the critical-outflow bullet claimed the switch is "benign in practice" —
it is not), a new implicit-section paragraph on Newton robustness, and the
science open-questions section rewritten concisely with assumptions stated
and nomenclature defined. PDF rebuilds clean (14 pp).

## 2. The two blockers, and the decided response

### A. Critical-outflow BC pin — OURS TO FIX, do it first

Rain-forced NLCD runs are impossible until this is fixed. Evidence is
complete (`o22`, three 600-step controls, one change each):

| variant | steps | result |
|---|---|---|
| base (critical-outflow, h_anuga 0.001) | 498 | fails |
| **outlet -> reflecting wall** | **600** | **clean, zero failures** |
| h_anuga 0.005 (5x drag regularization) | 498 | fails at the identical step |

**The BC is solely responsible; the drag is exonerated.** It is a PIN,
not a divergence: the residual sticks at 7.8888e-5 against F0 = 7.502e-2
(= 1.05e-3 relative) and any step past lambda ~ 1e-5 raises it 24x.
Tolerance is a **treadmill** — the pin level is set by the local flow
state (2.7e-5 vs rtol 1e-5; 1.17e-4 vs 1e-4; 1.05e-3 vs 1e-3) and o23
proved it: `snes_rtol 3e-3` bought exactly 8 steps (failed at solve 506
vs 498).

Decision (Mark): a free-outflow BC is a numerical convenience, not a
physical claim about the outlet, so do not wait on the scientists.
Implement behind a flag, default to current behaviour, document, invite
correction.

**Before building anything**, check what RDycore's CEED path and the
literature actually use. The likely answer is the least clever one: a
**zero-gradient / transmissive ghost** (copy the interior state), which
is continuous by construction and is what most codes mean by "free
outflow" — simpler and more standard than a smooth blend or a
Froude-limited ghost. Whatever lands must be differentiated in the host
AND device Jacobians and pass the FD gates, exactly as the drag fix did.
Acceptance: the o22 base control completes 600/600 steps.

Touch points mirror the drag fix: `CONDITION_CRITICAL_OUTFLOW` in the
host boundary loop (`swe_petsc.c`), the device boundary kernel
(`swe_jacobian_kokkos.kokkos.cxx`), and the ghost-Jacobian block `G` in
`swe_jacobian_petsc.c` (~line 654) plus its device twin (~line 211).

### B. Gauges sit in channels the 30 m mesh cannot resolve — CHANGE THE OBSERVABLE

Model WSE is **1–11 m ABOVE** observed stage at all 13 rain-driven gauges
(o20). Only 1 of 13 has observed water above its cell bed on the rising
limb. The offset is **time-varying** (9.7 -> 1.7 m over 12 hr at Whiteoak
at Main St), so a fixed per-gauge datum correction or a plain anomaly
misfit will NOT remove it. And at that gauge the model stays essentially
DRY (2 cm at hour 12) — the gap closes because the river rose 8 m to meet
a static model surface, not because the model reproduced the flood.

Decision: stop fighting the datum. **Use high-water marks** (2,100+ USGS
surveyed peak WSE across SE Texas). They record the peak — which o21A
measured as exactly when model and observation agree best — they sit on
floodplains/structures rather than in channels, they need no hydrograph
timing, and they are the field's benchmark currency (Inunda reports
**0.67 m MAE against HWMs** on a Harris County Harvey hindcast, which is
the number to put ours beside). Full ranking and caveats in
`next-phase-strategy.md`.

## 3. Result worth carrying forward: observability (o25i)

Identical configuration, only the gauge set differs:

| gauge set | observations | J start -> final | rel L2 vs truth | worst class |
|---|---|---|---|---|
| dense, 418,076 strided cells | 8,361,520 | 1.45e7 -> 6.4e-7 | 0.0000 | 0.0000 |
| real 13-site network | 260 | 493 -> 3.85 | 0.66 | 0.81 |

Semiconvergence: the objective falls 128x while the class values are 66%
wrong. **But the claim is narrower than it looks** — see
`campaign-wednesday.md`. The 13 gauge cells span only 6 of 15 classes,
AND the window is **20 seconds**, which at sqrt(g h) = 0.24–3 m/s is under
two cells of dynamic reach. Distant classes (all three forests) finished
at exactly the 0.0300 start because nothing they do can reach a gauge in
20 s. So: *13 gauges with a 20-second window* cannot identify 15 classes.
It does NOT show that 13 gauges over a realistic 6–12 hr window would
fail. The right next experiment is the same gauges over LONGER windows,
not a gauge-count bisection — and with HWMs the question should simply be
re-asked against ~2,100 spatially distributed marks.

## 4. Suggested split for the next session(s)

Independent, can run in parallel:

- **Track 1 — free-outflow BC.** Real numerics: a new ghost state
  differentiated through host and device Jacobians, FD-gated. This is the
  piece worth escalating to Fable if it gets intricate (see the Model
  choice section in `~/.claude/CLAUDE.md`).
- **Track 2 — HWM data work.** No solver needed at all. Acquire the USGS
  Harvey HWM archive, map marks to Turning cells (reuse
  `data/harvey_gauges/map_gauges_to_cells.py`; the CRS question is
  settled, EPSG:32610). **QC FIRST**: measure how often a mark falls
  BELOW its cell's bed elevation before building any misfit on it — that
  is the exact check that saved 19 node-hours on the gauges. Also verify
  the archive actually covers the Turning domain (Buffalo Bayou /
  Whiteoak) densely rather than being concentrated elsewhere in SE Texas.

Then: peak-WSE misfit mode in the driver, re-ask identifiability against
the HWM set, and only then the long-window rain-forced calibration.

## 5. Traps that cost time this session — do not repeat

- **`-adjoint_calibrate_classes` ignores `-adjoint_gauge_cells_file`.**
  `-adjoint_classes_twin` reuses the obs table's gauge cells only if that
  file ALREADY EXISTS, otherwise it silently falls back to every 7th
  cell. This made a run reproduce the dense case digit-for-digit while
  claiming to be the 13-gauge test. Pre-seed a stub table with the real
  cell IDs and the time grid (`o25_observability.sh` does this) and
  confirm via the log line `classes twin: reusing N gauge cells`.
- **PM PETSc build.** petsc-claude was checked out to upstream
  `FETCH_HEAD`, built, then returned to fork `main` — leaving an
  upstream-built library AND a `conf/files` list regenerated WITHOUT the
  fork's GPU directories, so `baijkok.kokkos.cxx` / `baijcuda.cu` were
  silently never compiled ("Unknown Mat type: baijkokkos"). Fix:
  `rm $PETSC_ARCH/lib/petsc/conf/files`, rerun `config/gmakegen.py`,
  `make all`. Verify with
  `nm -D libpetsc.so | grep -c BAIJKokkos` (want ~15, not 0).
- **Size batch jobs to the wall clock.** A 20-TAO-iteration calibration
  on a 6-hr window needs ~19 hr; submitted against a 10-hr limit it got
  4 minutes of useful work. ~1 hr per TAO iteration per simulated hour of
  window, at n4.
- **Yaml `coupling_interval` must not exceed `stop`** — shortening a
  window means editing both.
- Never compare J across binaries with different partitioners; the
  observation set is partition-dependent.

## 6. Environment notes

- Laptop repo is now **standalone**: the old `~/Codes/RDycore` main
  checkout was deleted, its git database promoted into `RDycore-gpu` (all
  branches and stashes intact, submodule pointers repaired), and its
  untracked data rescued into `data/harvey_gauges/` and
  `rescued-from-main-checkout/`.
- Laptop builds need `PETSC_DIR`/`PETSC_ARCH` set explicitly for cmake
  regeneration (`petsc-claude`, `arch-macosx-gnu-rdycore-kokkos-O`).
- PM: `~/Codes/rdycore-manning`, `build-claude-gpu`, built by
  `cmake-claude-gpu.sh`; run dir `$SCRATCH/gpu-implicit` (all o*/b* logs
  and protocol scripts).
- NERSC sshproxy certificates last ~24 h; refresh before PM work.
- Pre-existing laptop test failures, not ours: 6 `swe_roe` cgns tests (no
  CGNS in this arch) and `amr_c_np_3_basic` (verified to SEGV identically
  on pre-change sources).
