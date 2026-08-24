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
the real gauge geometry), then compared against the USGS stage.

NOTATION: "WSE" = water-surface elevation (bed elevation + water
depth), in metres on the NAVD88 datum — the same datum the USGS stage
is converted to, so the two are directly subtractable. The last column
is **the model's WSE minus the gauge's observed WSE**; a POSITIVE value
means the model's water surface sits that many metres ABOVE where the
gauge says the real water was. "cell bed" = the single mean ground
elevation the 30 m mesh stores for that cell.

| gauge | cell bed (m) | model WSE | observed WSE | model minus obs (m) |
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

### (A) RESULT: the bias COLLAPSES as the floodplain inundates — calibrate at the PEAK, not the rising limb

12-hr forward from 2017-08-26 18:00 CDT, model-minus-observed WSE (m) at the
13 canonical rain-driven gauges:

| gauge | bed | +1 h | +4 h | +7 h | +10 h | +12 h |
|---|---|---|---|---|---|---|
| 08074500 Whiteoak at Houston | 15.0 | +10.9 | +7.9 | +4.3 | +3.5 | — |
| 08074598 Whiteoak at Main St | 11.8 | +9.7 | +8.5 | +4.0 | +2.8 | **+1.7** |
| 08074540 Little Whiteoak | 14.5 | +7.9 | +5.9 | +1.9 | +1.7 | **+1.4** |
| 08074250 Brickhouse Gully | 20.4 | +5.6 | +1.4 | +1.8 | +1.8 | **+1.6** |
| 08074020 Whiteoak at Alabonson | 23.6 | +4.4 | +2.6 | +1.4 | +0.7 | **+0.6** |
| 08072800 Langham Ck nr Addicks | 28.2 | −1.0 | −0.7 | −0.5 | −0.2 | **+0.0** |

The structural bias is **not static** — it collapses from 5–11 m early
to 0.6–3 m by hours 10–12. Three gauges come within 1 m at some point
(08072800 to 0.05 m, 08074020 to 0.64 m, 08072760 to 0.75 m); the
median closest approach across all 13 is ~1.4 m.

**CAUTION on reading that as model skill.** The convergence is only
partly the model wetting up. Per-gauge model depth at hour 12 shows 11
of 13 cells genuinely wet (Buffalo Bayou nr Katy 2.78 m, Buffalo Bayou
at Houston 1.99 m, Langham Ck 1.70 m) — but the two Whiteoak sites that
show the largest *apparent* improvement are the ones where the model
stays essentially DRY:

| Whiteoak at Main St | h+1 | h+6 | h+12 |
|---|---|---|---|
| model WSE (m) | 11.80 | 11.81 | 11.82 |
| model depth (m) | **0.00** | **0.01** | **0.02** |
| observed WSE (m) | 2.06 | 6.93 | 10.15 |
| model minus obs (m) | +9.74 | +4.88 | +1.67 |

There the gap closes because the *real river rose 8 m to meet a static
model surface*, not because the model reproduced the flood. So the
selection criterion cannot be the window alone — it must be per gauge
AND per time: require the model cell genuinely wet AND the observation
above the cell bed. And because the offset varies in time (9.7 → 1.7 m
at this gauge), a FIXED per-gauge datum correction or a plain anomaly
misfit will not remove it — remedies 1 and 2 above are weaker than they
first appear.

**This overturns the R1 window assumption.** The observability study
said "rising limb" (information accumulates fastest there), but the
datum reality says the rising limb is exactly where the model and the
gauges are least comparable. The defensible window is at or after peak
inundation. R1 should be re-decided on this basis — and it makes the
12-hr window (R3') attractive for a second reason beyond information
content: it spans the interval where the observations become usable.

A residual 1–3 m offset remains at most gauges even at peak, so an
anomaly/bias-corrected misfit (option 1 or 2 above) is still likely
required — but the problem is far from hopeless at the right window.

## o21B FAILURE ROOT-CAUSED (o22, 2026-08-23): the outflow BC alone, and the pin sits at 1.05e-3

o21 part B (NLCD prior + rain + 6-hr window — the first run combining
all three, and exactly the R3 configuration) died at forward step 498:
Newton ran 2 its/step, jumped to 25, then DIVERGED_MAX_IT at 50.

Three 600-step controls, identical except for one change each
(o22_diag.sh; base reproduces o21B exactly at 498 solves / 1 failure):

| variant | steps | failures | verdict |
|---|---|---|---|
| base: critical-outflow, h_anuga 0.001 | 498 | 1 | fails |
| **outlet swapped to reflecting** | **600** | **0** | **clean** |
| h_anuga 0.005 (5x drag regularization) | 498 | 1 | fails identically |

**The critical-outflow BC is solely responsible; the drag is
exonerated** — raising h_anuga 5x changes nothing, not even the step at
which it dies, while swapping the outlet fixes it completely.

The failure signature is a textbook discontinuity pin, not divergence:
the residual sits at 7.8888e-5 and refuses to move (7.888900e-5 ->
7.888832e-5 over its 46-50) while ANY step past lambda ~ 1e-5 raises it
~24x (to 1.885e-3). The step's initial residual is 7.502e-2, so the pin
is at **1.05e-3 relative — it misses snes_rtol 1e-3 by 5%.**

This is the same state-dependent pin seen before (o9d missed at 2.7e-5
with rtol 1e-5; o18 missed at 1.17e-4 with rtol 1e-4; now 1.05e-3 with
rtol 1e-3). The pin level tracks the tolerance because it is set by the
local flow state, so **chasing it with tolerance is a treadmill** — each
new configuration can pin just above whatever we pick. It is harmless
physically (a 1e-3 relative residual is a converged solve for practical
purposes), which is why the crutch works, but the BC smoothing
(question 3 for the scientists) is the actual fix.

Interim: snes_rtol 3e-3 for NLCD+rain configurations.

## OBSERVABILITY ANSWERED (o25, 2026-08-23): 13 real gauges CANNOT constrain 15 NLCD classes

The experiment the campaign rests on. Identical configuration in every
respect (unforced recession IC, h_anuga 0.001, snes_rtol 1e-3, 20-step
window, 15 NLCD classes, uniform n = 0.03 start, 20 TAO its) — the ONLY
change is the gauge set:

| gauge set | observations | J start → final | rel L2 vs truth | max class err |
|---|---|---|---|---|
| dense, 418,076 strided cells (o18d) | 8,361,520 | 1.45e7 → **6.4e-7** | **0.0000** | **0.0000** |
| **real 13-site network (o25i)** | **260** | 493.3 → **3.85** | **0.66** | **0.81** |

**Textbook semiconvergence.** With the real network the objective still
falls 128x — the model fits its 13 gauges well — while the recovered
class values are 66% wrong in L2 and the worst class is off by 81%. A
good fit is NOT evidence of a good parameter field here.

**CLAIM NARROWED (2026-08-24) — the window, not just the network.**
The mechanism is not "260 observations is too few for 15 unknowns"
(that is over-determined 17:1). Two things actually cause it:

1. The 13 gauge cells occupy only **6 of the 15 NLCD classes**
   (23x5, 24x3, 22x2, 81, 52, 90). The nine unrepresented classes
   include all three forest classes (41/42/43), which finished at
   EXACTLY the 0.0300 start — zero gradient, never moved.
2. **The window is 20 SECONDS** (`stop: 20.0`, dt = 1 s). Information
   travels at the gravity-wave speed sqrt(g h): ~3 m/s at 1 m depth and
   only ~0.24 m/s at the ~6 mm depths of this spun-up state. In 20 s
   that is a fraction of a cell to about two cells. So no distant cell
   is DYNAMICALLY CONNECTED to any gauge: perturbing forest roughness
   cannot move a gauge reading within the window, and the optimizer
   correctly leaves it alone. It is not that water never crosses
   forest — it is that the window is far too short for it to matter.

So the honest claim is: **13 gauges with a 20-second window cannot
identify 15 classes.** It does NOT establish that 13 gauges over a
realistic 6–12 hr window would fail — which is what the campaign needs
to know. The dense case is immune to this locality (418k observation
cells put observations on top of every class), which is why the two
differ so starkly. Note also that gauge count per class does not
explain the pattern: class 52 has ONE gauge and recovers to 3%, while
class 23 has FIVE and lands 35% off and class 24 has three and is the
worst at 81% — depth/flow at those cells matters more than count.

RIGHT NEXT EXPERIMENT (supersedes the gauge-count bisection): the SAME
13 gauges over progressively LONGER windows. That is the axis that
decides whether distant classes ever become observable. Obstacle: long
windows want rain forcing, which is blocked by the outflow BC; a long
UNFORCED window is the intermediate test, and o25(ii) at 600 steps
already failed undiagnosed — worth diagnosing rather than shelving.

This is the Turning/NLCD analogue of the Houston result already in the
paper's abstract (17-gauge twin: fits 238 observations essentially
perfectly, recovers the field to only 45%). It is a stronger statement,
because reducing the parameter count to 15 was supposed to be the fix:
**even matched to the network scale, 13 gauges x 20 times is not enough
to identify 15 land-cover classes.** Directly relevant to meeting
question 4 (classes vs per-cell) — the honest answer is that neither is
identified by this network without regularization or more information.

Follow-ups this opens (none run yet):
- How many gauges/times WOULD suffice? Bisect the strided gauge set
  (e.g. 100 / 1000 / 10000) to find where recovery degrades — that curve
  is a genuine paper figure and cheap to produce.
- Fewer classes: are 15 classes over-parameterized for 13 gauges? Try
  merging to the 4-5 dominant NLCD classes by area.
- Prior weighting: beta = 1e-4 here. A stronger prior would trade fit
  for parameter sanity; the beta-sweep is the standard L-curve.
- NB o25(ii) (600-step unforced window, 10 obs times) FAILED with a
  nonlinear solve failure — the longer unforced window hits a solver
  problem of its own, not yet diagnosed. Only the 20-step result stands.

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
