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
BLMVM convergence in any test). DECIDED (Mark, 2026-08-22): campaign runs at
**gmres+right rtol 1e-2**, with one 1e-3 spot-check calibration for the
paper's reviewer-proofing. rtol 1e-1 diverges — do not go looser.
WATCH ITEM: rtol 1e-2 costs ~13% more Newton iterations (~4.4 -> ~5.0
per implicit step at Turning, net 2.5x cheaper). If forced-window
Newton counts creep toward -snes_max_it 50 on hard steps, tighten back
to 1e-3 rather than raising the cap.

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

**Pre-campaign status (2026-08-22 evening)**: Turning NLCD per-cell
class/manning maps GENERATED ($SCRATCH/gpu-implicit/turning30m_class.bin
and _manning.bin, 0 uncovered cells, area-weighted mean n = 0.1005 ==
the committed summary). Class-mode driver VALIDATED on the laptop
dam-break twin (np1/2, device types: recovers both class values
exactly, J -> 3e-14; observation-count report fixed to reduce across
ranks).

**NLCD-prior blocker RESOLVED (2026-08-22 night, commit 3110ac50 +
o14–o18 logs)**: the o9 class-twin failures were TWO residual
discontinuities. (1) The Manning drag's tiny_h gate — FIXED by the
ANUGA-regularized drag (scientist item, Mark pre-approved; bitwise
identical at h_anuga = 0, full FD-gate suite passed). (2) The
critical-outflow BC's uperp < 0 switch (flux jumps by the wet-onto-dry
Roe flux at the crossing) — root-caused via a reflecting-outlet
diagnostic; smoothing it is a NEW Wednesday agenda item (scheme
decision). Working recipe until then, VALIDATED END-TO-END: 
`h_anuga_reg_parameter: 0.001` (beuler_dt1_anuga.yaml /
beuler_dt1_1hr_anuga.yaml) + `-snes_rtol 1e-3` + gmres+right rtol 1e-2.
**Turning 20-it class-mode twin calibration COMPLETES: J 1.45e7 →
6.4e-7, EXACT class recovery, TaoSolve 20.7 s on one GPU node**
(o18d_cal20_n4.log). h_anuga sweep: 0.001 and 0.003 clean and nearly
identical J-traces; 0.01 NaNs (weaker drag damping) — do not go bigger.

**R3 — the calibration.** 6-hr MRMS-forced window on the rising limb,
real-gauge observations, 20 TAO iterations, n4, revolve
(`-ts_trajectory_max_cps_ram 400`). ~19 node-hr. Parameterization
decision (meeting item): the Houston lesson says match parameters to
the network — 21 gauges × K obs times vs 2.93M per-cell parameters is
severely underdetermined, so the defensible primary run is
**land-cover-class (NLCD) parameters** with the per-cell map as a
Tikhonov-regularized secondary/appendix run.

**R3' — the 12-hr paper window** (decided direction, Mark
2026-08-22): the observability study says information accumulates
superlinearly (1.4e3/hr at 1 h -> 5.3e3/hr at 12 h), so after the 6-hr
shakeout the paper run is a 12-hr window. Cost at n4: 43,200 steps,
gradient ≈ 2–2.2 hr (revolve with 400 cps is still in the
two-recompute regime, capacity ~80,600 steps), 20 TAO its ≈ 40–44
node-hr — fits the budget, but ~40 hr wall on one node. PREREQ:
multi-node smoke (2 nodes x 4 GPUs, 1 rank/GPU, 366k cells/rank) — a
15-min test that would make the paper run an overnight 2-node job.
Everything to date is single-node.

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

## BLOCKER FOUND (2026-08-23): gauges sit in channels the 30 m mesh does not resolve

**Real-gauge calibration against ABSOLUTE water-surface elevation cannot
work on the Turning 30 m mesh as it stands.** Found for ~15 min of GPU
time (o20) rather than in the 19-node-hour 6-hr shakeout.

Ran a 1-hr rain-forced forward from 2017-08-26 18:00 CDT and dumped the
MODEL WSE at the 13 canonical rain-driven gauge cells (twin mode over
the real gauge geometry), then compared against the USGS stage:

| gauge | cell bed (m) | model WSE | observed WSE | model − obs |
|---|---|---|---|---|
| 08072300 Buffalo Bayou nr Katy | 33.27 | 33.97 | 32.05 | **+1.92** |
| 08072730 Bear Ck nr Barker | 34.73 | 34.76 | 31.99 | **+2.77** |
| 08074150 Cole Ck at Deihl Rd | 25.30 | 25.32 | 19.56 | **+5.76** |
| 08074250 Brickhouse Gully | 20.40 | 20.43 | 14.85 | **+5.57** |
| 08074500 Whiteoak Bayou at Houston | 15.03 | 15.04 | 4.10 | **+10.93** |
| 08074598 Whiteoak Bayou at Main St | 11.80 | 11.80 | 2.06 | **+9.74** |

All 13 gauges differ by more than 1 m; the Whiteoak sites by 8–11 m.
The structure is unambiguous: **the model WSE sits essentially AT the
cell bed** (a thin film, bed + 0.01–0.7 m) **while the observed stage
is 1–11 m BELOW the cell bed.** The gauges measure stage in incised
bayou channels; a 30 m cell's mean elevation is the surrounding
floodplain/bank. The model's ground surface is above the water level
the gauge reports, so the misfit carries a structural bias no Manning
field can remove — a calibration would drive n to its bounds and
"converge" to a meaningless field. Confirmed independently from the
data alone: only **1 of 13** gauges has observed WSE above its cell bed
through the whole rising limb, rising to only 3 of 13 near the peak.

Options (Wednesday decision, new agenda item 4b):
1. **Anomaly misfit** — calibrate on Δ(WSE) from the window start, not
   absolute WSE. Removes the per-gauge offset; standard for coarse
   meshes. Modest driver change. Caveat: where the model cell is a dry
   floodplain, its dynamics are not the channel's, so the anomaly is
   still not the observed quantity.
2. **Per-gauge datum/bias correction** from a documented channel invert
   — same effect as (1), better justified where inverts exist.
3. **Hydro-condition the DEM** (burn channels) — the real fix, but a
   mesh-generation change, out of scope for this paper's timeline.
4. **Restrict to floodplain-inundation periods/gauges** — physically
   clean but leaves ~3 usable gauges; too few for 15 classes.
5. **Discharge instead of stage** — needs a rating curve/cross-section
   we do not have.

Overnight job 57480313 (submitted 2026-08-23) addresses the empirical
half: (A) a 12-hr forward through the flood peak dumping model WSE
hourly at the gauge cells, to see whether the floodplain ever inundates
enough for model and observation to become comparable, and (B) a 6-hr
TWIN class calibration at the real 13-site geometry — unaffected by the
bathymetry problem and a direct answer to the observability question
(can a real sparse network constrain 15 NLCD classes, where the dense
418k-gauge twin recovers all 15 exactly?).

## Questions to settle at the meeting

1. WSE accuracy target (drives R0 acceptance and σ in the misfit).
2. Trusted-gauge list — confirm the rain-driven subset / dam question.
3. Window choice sign-off (rising limb, 6 vs 9 hr).
4. Parameterization for the primary run: NLCD classes vs per-cell.
4b. **NEW AND URGENT — unresolved channels at the gauges.** The 30 m
   mesh puts the gauge cells' bed 1–11 m ABOVE the observed stage
   (evidence above), so absolute-WSE calibration is structurally
   impossible. Which remedy: anomaly misfit, per-gauge invert
   correction, DEM hydro-conditioning, or a restricted gauge/time
   subset? This decides whether the paper's real-gauge run happens on
   this mesh at all.
5. Solver config sign-off: gmres+right rtol 1e-2 (evidence above),
   plus the NLCD recipe h_anuga 0.001 + snes_rtol 1e-3.
6. ANUGA-regularized drag: retroactive sign-off (h_anuga = 0.001;
   0.001-vs-0.003 J-traces indistinguishable — physics impact nil at
   this size) — the paper's open-questions section asked exactly this.
7. NEW: critical-outflow BC discontinuity (uperp < 0 wall switch) —
   the remaining Newton hazard; smoothing it (blend or Froude-limited
   ghost) would retire the snes_rtol 1e-3 crutch. Scheme decision.
8. Paper: venue (WRR / JAMES / GMD) and author order (roster now in
   the draft from the RDycore EMS 2026 paper).
