# Wednesday campaign plan (meeting 2026-08-26)

Supersedes `campaign-tonight.md` (2026-08-19), whose central premise —
adjoint unaffordable at 30 m, revolve unavailable — is no longer true.
The GPU-implicit stack (sessions 2–3, `plans/RESULTS-gpu-implicit.md`)
makes the calibration itself affordable at Turning 30 m.

**Budget**: ≤100 GPU-node-hours (Mark). **Everything below fits in ~40
with margin for reruns.**

## Where we stand (all validated, all committed at `2ced86cf` + fork)

- Device-resident forward + adjoint + calibration at Turning 30 m
  (2.93M cells, 4 A100s = 1 node): RHS, assembled Jacobian, pbjacobi
  setup, parameter Jacobian, observation operator, revolve-checkpointed
  trajectory. Gradients identical to every printed digit vs the
  pre-optimization stack; laptop pairs bitwise.
- Real MRMS hourly rain wired into the adjoint driver (preloaded
  schedule + TSPreStage; replay-safe through revolve recomputes).
  Forced 1-hr gradient validated in session 2; rerun on the optimized
  stack in progress (o8) — gate: values match `b5_1hr_rain_n4.log` to
  printed digits.
- Measured cost basis (n4 = one GPU node):
  | quantity | measured |
  |---|---|
  | 1-hr-window forced gradient | ~9.4 min |
  | 6-hr-window gradient (extrapolated) | ~56 min |
  | 20-iteration calibration, 6-hr window | **~19 node-hr** |
  | 20-iteration calibration, 9-hr window | ~28 node-hr |
- NOT yet done: any run against **real gauge observations**. All
  Turning calibrations so far are twins. That is the campaign.

## Solver decision (from the 2026-08-22 sweep with Mark)

`-ksp_type gmres -ksp_pc_side right` replaces fgmres (bit-equivalent
at equal rtol, ~5% faster). **rtol 1e-2** is validated at the
convergence level: laptop twin and Turning 20-it calibration converge
indistinguishably from 1e-3 (55.47x vs 55.43x, same J to 5–6 digits),
TaoSolve 34.6 → 31.5 s. Caveat on the record: at rtol 1e-2 the
adjoint-vs-FD mismatch is inner-solve-limited at ~6e-4–4e-3 (the 1e-5
FD gate is a tight-tolerance instrument; gradient error does not affect
BLMVM convergence in any test). Recommendation: run the campaign at
**gmres+right rtol 1e-2**, with one 1e-3 spot-check calibration for the
paper's reviewer-proofing. rtol 1e-1 diverges — do not go looser.

## Run matrix (in order)

**R0 — dt-verification (first, cheap).** BEULER dt=1 vs ARK dt=0.25
gauge WSE hydrographs on a forced window, main `rdycore` driver (rain
native via RDyAdvance — no new wiring). Scientists' read: implicit
dt=1 safe, dt=5 likely too big (dt≥5 blocked anyway — not in this
campaign). Acceptance: hydrograph differences at the 21 gauge cells
small vs the WSE accuracy target (number pending from scientists).
Cost: one forced window each ≈ a few node-hr for the ARK reference.

**R1 — window placement.** Rising limb, per the observation-time
sensitivity study; confirm at 30 m with 1–2 short forced gradients at
candidate windows (~10 min each). Pick the 6-hr window for R3.

**R2 — real observation table.** `data/harvey_gauges/make_obs_table.py
--subset rain-driven` on the Turning gauge set
(`turning30m_gauges_cells.csv`, 21 sites, Buffalo Bayou system).
MANDATORY QC: `--t0` clock check against a documented crest time
before trusting timing (tool README). Scientists' answers folded in:
trusted-gauge list (their read: gauges likely avoid dam influence) and
per-gauge σ.

**R3 — the calibration.** 6-hr MRMS-forced window on the rising limb,
real-gauge observations, 20 TAO iterations, n4, revolve
(`-ts_trajectory_max_cps_ram 400`). ~19 node-hr. Parameterization
decision (meeting item): the Houston lesson says match parameters to
the network — 21 gauges × K obs times vs 2.93M per-cell parameters is
severely underdetermined, so the defensible primary run is
**land-cover-class (NLCD) parameters** with the per-cell map as a
Tikhonov-regularized secondary/appendix run.

**R4 — contingency/extension** (remaining budget): 9-hr window, second
rain product (mswep/nldas for forcing sensitivity), or a σ-sensitivity
rerun.

## Protocol (R3 command skeleton, from the validated runs)

```
srun -N1 -n4 -c32 --cpu-bind=cores -G4 --gpu-bind=none rdycore_adjoint \
  beuler_dt1_<window>.yaml \
  -ts_adapt_type none -snes_max_it 50 -ksp_max_it 300 \
  -ksp_type gmres -ksp_pc_side right -ksp_rtol 1e-2 -pc_type pbjacobi \
  -dm_mat_type baijkokkos -dm_vec_type kokkos \
  -ts_trajectory_type memory -ts_trajectory_max_cps_ram 400 \
  -raster_rain_dir .../mm-per-hr/mrms/bin \
  -raster_rain_start_date 2017,8,26,0,0 \
  -adjoint_calibrate_gauges -adjoint_obs_file <real_obs>.txt \
  -tao_max_it 20 -tao_monitor -log_view
```
Known limitation: RDyApplyForcing hardcodes region 1 = whole domain
(fine for Turning). Never compare J across binaries with different
partitioners (observation set is partition-dependent).

## Questions to settle at the meeting

1. WSE accuracy target (drives R0 acceptance and σ in the misfit).
2. Trusted-gauge list — confirm the rain-driven subset / dam question.
3. Window choice sign-off (rising limb, 6 vs 9 hr).
4. Parameterization for the primary run: NLCD classes vs per-cell.
5. Solver config sign-off: gmres+right rtol 1e-2 (evidence above).
6. Paper: venue (WRR / JAMES / GMD) and author order (roster now in
   the draft from the RDycore EMS 2026 paper).
