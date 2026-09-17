# Number-provenance trace C — manning-calibration.tex lines 997–1365

Scope: Sec 5 twin experiments (incl. `fig:map` caption), Sec 5.1 identifiability +
`tab:snr`, Sec 6 opening, Sec 6.1 high-water marks and baseline (`tab:gauges`,
`tab:baseline`, `tab:bands`).

Source shorthand:
- `RMD` = `plans/RESULTS-manning-draft.md`
- `RGI` = `plans/RESULTS-gpu-implicit.md`
- `CW`  = `plans/campaign-wednesday.md`
- `PS26` = `plans/PROJECT-STATE-2026-08-26.md`
- `PS`  = `plans/PROJECT-STATE.md`
- `FB`  = `plans/fable-brief-2026-08-25.md`
- `MS`  = `plans/meeting-summary-2026-08-26.md`
- `SH24pm` = `plans/session-handoff-2026-08-24-pm.md`
- `HH`  = `plans/hurricane-harvey-simulation-plan.md`
- `GH`  = `plans/github-issue-manning-adjoint.md`
- `OBS` = `logs/o63/obs_turning_h29_41.txt` (13 gauges × 48 records, 900 s spacing)

Where a status says "recomputed", I re-derived the value directly from `OBS`
with a script in the scratchpad (see the block after `tab:gauges`).

---

## Sec 5 opening paragraph (lines 997–1022)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1004 | 2{,}746 cells | Houston 1 km mesh size | `HH:44` "1km resolution Houston area (2746 cells)"; `RMD:36` "2746 params" | EXACT |
| 1004 | 4200\,s window | twin window length | `HH:45` "**Final time**: 4200 seconds (70 minutes)" | EXACT |
| 1004 | $\Delta t = 30$\,s | twin step size | `RMD:36` "implicit BEULER carries the Harvey window at dt=30 s" | EXACT |
| 1005 | 343 synthetic gauges | observation count | `RMD:36` "343 gauges x 14 obs times" | EXACT |
| 1006 | 14 observation times | observation times | `RMD:36` "343 gauges x 14 obs times" | EXACT |
| 1007 | $\beta = 10^{-5}$ | weak Tikhonov weight | `RMD:36` "weak beta=1e-5" | EXACT |
| 1008 | 2{,}746 parameters | all cells observable | `RMD:36` "2746 params, ALL observable -- whole domain wet+moving" | EXACT |
| 1009 | 343 gauges | gauge-sparse ratio | `RMD:36` (same) | EXACT |
| 1010 | 2{,}746 parameters | ratio denominator | `RMD:36`; `RMD:36` also states the "8:1 param:gauge ratio" (2746/343 = 8.0) | EXACT |
| 1013 | 300 BLMVM iterations | early-stopped run | `RMD:36` "300-it capped run" | EXACT |
| 1015 | $26.3\%$ | rel. $L^2$ error at 300 it | `RMD:36` "300-it capped run = 26.3% recovery" | EXACT |
| 1016 | 2000 iterations | continued run | `RMD:36` "2000-it run" | EXACT |
| 1016 | $38.0\%$ | degraded error | `RMD:36` "DEGRADED recovery to 38.0%" | EXACT |
| 1019 | $\beta = 10^{-4}$ | regularized run | `RMD:55` "beta=1e-4, 686 gauges, 1000 its" | EXACT |
| 1019 | 686 gauges | regularized run | `RMD:55` "686 gauges" (= 2 × 343) | EXACT |
| 1020 | $19.4\%$ | regularized recovery | `RMD:55` "1000 its -> 19.4% recovery, no drift"; `GH:40` "19.4% relative" | EXACT |
| 1020 | 1000 iterations | regularized run | `RMD:55` "1000 its" | EXACT |

## fig:map caption (lines 1030–1044)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1031 | 2{,}746 parameters | panel setup | `RMD:36` "2746 params" | EXACT |
| 1031 | $n = 0.03$ west | two-zone truth | `RMD:29` "n_true (0.03, 0.06)" | EXACT |
| 1032 | $n = 0.06$ east | two-zone truth | `RMD:29` "n_true (0.03, 0.06)" | EXACT |
| 1032 | 14 observation times | panel setup | `RMD:36` | EXACT |
| 1033 | $\beta = 10^{-5}$, 343 gauges | far-left panel | `RMD:36` | EXACT |
| 1034 | 300 iterations | far-left panel | `RMD:36` | EXACT |
| 1034 | $26.3\%$ error | far-left panel | `RMD:36` | EXACT |
| 1035 | 2000 iterations | center-left panel | `RMD:36` | EXACT |
| 1036 | $38.0\%$ error | center-left panel | `RMD:36` | EXACT |
| 1037 | bounds $[0.01, 0.2]$ | saturated bounds | `GH:24` "TAO/BLMVM (bounds n ∈ [0.01, 0.2])"; `plans/manning-map-incremental-plan.md:38` same | EXACT |
| 1038 | $\beta = 10^{-4}$, 686 gauges | center-right panel | `RMD:55` | EXACT |
| 1038 | 1000 iterations | center-right panel | `RMD:55` | EXACT |
| 1038 | $19.4\%$ error | center-right panel | `RMD:55` | EXACT |
| 1040 | 17 in-mesh USGS gauges | far-right panel | `RMD:39` "20/90 gauges in mesh, 17 with Harvey stage series"; `RMD:613` "17 USGS gauge cells" | EXACT |
| 1041 | $\beta = 10^{-4}$, 1000 iterations | far-right panel | `RMD:613-614` "beta 1e-4, 1000 its" | EXACT |
| 1042 | $2\times10^{7}$-fold | misfit reduction | `RMD:614` "J falls 2.1e7x" | ROUNDED (source 2.1e7) |
| 1042 | $44.9\%$ | far-right recovery | `RMD:615` "recovery only 44.9% (np=1; np=6 gave 45.4%)" | EXACT |

## "The real gauge network" paragraph (lines 1048–1069)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1050 | Twenty USGS gauges | gauges inside mesh | `RMD:39` "20/90 gauges in mesh" | EXACT |
| 1051 | seventeen (with stage records) | Harvey stage series | `RMD:39` "17 with Harvey stage series" | EXACT |
| 1052–53 | HydroShare DOI 10.4211/hs.c037167e… | data source | citation only | N/A-citation |
| 1054 | seventeen gauge cells | observation operator | `RMD:613` "17 USGS gauge cells" | EXACT |
| 1055 | 238 observations | 17 × 14 | `CW:343` "fits 238 observations essentially perfectly" | EXACT (also 17 × 14 = 238, DERIVED) |
| 1055 | 17 gauges $\times$ 14 times | factorization | `RMD:39`, `RMD:36` | DERIVED (17 × 14 = 238) |
| 1056 | 2{,}746 parameters | denominator | `RMD:36` | EXACT |
| 1057 | $2\times10^{7}$ | misfit factor | `RMD:614` "J falls 2.1e7x" | ROUNDED (2.1e7) |
| 1058 | $44.9\%$ | recovery | `RMD:615` | EXACT |
| 1064 | 238 observations | repeat | `CW:343` | EXACT |

## Sec 5.1 identifiability, paragraph 1 (lines 1071–1091)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1072, 1076, 1080 | 30\,m | Turning mesh resolution | `RMD:189-194` "Turning_30m mesh … 2,926,532 tri cells" | EXACT |
| 1076 | 15 classes | NLCD parameterization | `RGI:986` "15 classes, 418076 obs cells x 20 times"; `CW:293` "15 NLCD classes" | EXACT |
| 1077 | 2.93M cells | Turning mesh | `RMD:194` "2,926,532 tri cells" | ROUNDED (2,926,532) |
| 1078–79 | "an order of magnitude more observations than parameters" | counting argument | `CW:310` "that is over-determined 17:1" (260/15 = 17.3) | DERIVED (260/15 = 17.3) |
| 1081 | 13-gauge geometry | real gauge network | `CW:301` "**real 13-site network (o25i)**" | EXACT |
| 1081 | 260 observations | o25i observation count | `CW:301` "**260**" | EXACT |
| 1081 | "a short early-transient window" | window | `CW:316` "**The window is 20 SECONDS** (`stop: 20.0`, dt = 1 s)" | EXACT (no number printed) |
| 1082 | $128\times$ | objective reduction | `CW:304` "the objective still falls 128x"; `PS26:55` "falls 128x" | EXACT |
| 1083 | $66\%$ wrong | rel. $L^2$ | `CW:301` "**0.66**"; `CW:305` "66% wrong in L2" | EXACT |
| 1083 | worst class $81\%$ | max class error | `CW:301` "**0.81**"; `CW:305` "worst class is off by 81%" | EXACT |
| 1085 | the three forest classes | never-moving classes | `CW:313-315` "all three forest classes (41/42/43), which finished at EXACTLY the 0.0300 start — zero gradient, never moved" | EXACT |
| 1088 | 418{,}076 strided cells | dense control (o18d) | `CW:300` "dense, 418,076 strided cells (o18d)"; `RGI:568`, `RGI:986` | EXACT |
| 1088 | all fifteen classes to machine precision | dense recovery | `CW:300` "1.45e7 → **6.4e-7** \| **0.0000** \| **0.0000**" | EXACT (rel $L^2$ and max class err both 0.0000) |
| 1090 | Fifteen parameters and thirteen gauges | restatement | `CW:293`, `CW:301` | EXACT |

## Sec 5.1 identifiability, paragraph 2 — the HWM re-ask (lines 1093–1117)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1094 | 108 QC-passed HWM cells | observation set | `RGI:1104` "the 108 QC-passed real mark cells' peaks" | EXACT |
| 1095 | one-hour rain-forced window | window | `RGI:1105` "12 samples over a 1-hr rain-forced window, free-outflow outlet" | EXACT |
| 1097 | fifteen classes | parameters | `RGI:1106` "15 TAO its"; class count 15 per `CW:293` | EXACT |
| 1097 | uniform start | start value | `RGI:1106` "uniform 0.03 start" | EXACT |
| 1098 | $81\times$ | objective reduction | `RGI:1108` "J 7.48e4 -> 926 (81x)" | EXACT |
| 1099 | $0.103$\,m | initial peak-WSE MAE | `RGI:1108` "peak-WSE MAE 0.103 m -> 0.017 m" | EXACT |
| 1099 | $0.017$\,m | final peak-WSE MAE | `RGI:1108` | EXACT |
| 1100 | three developed-intensity classes, pasture, crops | identified classes | `RGI:1112-1113` "22 dev-low 0.7%, 23 dev-med 10.5%, 24 dev-high 8.6%, 81 pasture 1.9%, 82 crops 8.6%" | EXACT |
| 1101 | $0.7$--$10.5\%$ | per-class error range | `RGI:1112-1114` (min 0.7%, max 10.5%) | EXACT |
| 1101 | $83\%$ of the domain's cells | area share | `RGI:1113-1114` "2.43M of 2.93M cells (83% of the domain…)"; `PS26:53`; `SH24pm:36` | EXACT |
| 1102 | $0.20$ | area-weighted rel. $L^2$ | `RGI:1118` "AREA-WEIGHTED rel L2 0.20" | EXACT |
| 1103 | $0.58$ | unweighted rel. $L^2$ | `RGI:1118` "Unweighted rel L2 0.58" | EXACT |
| 1104 | forests, shrub, emergent wetland, $\sim$5\% of cells | residual-error classes | classes named at `RGI:1115-1117` (41 deciduous, 52 shrub, 95 emergent, 42 evergreen, 21 dev-open). Areas: `RGI:1543` "42 evergreen 43k", `RGI:1548` "others … 12k-86k" of 2,926,532 | DERIVED, weak — no artifact sums these classes. Five classes in the 12k–86k band ≈ 60k–430k, i.e. 2–15%; "~5%" is plausible but is not a measured number anywhere in `plans/` or `logs/`. Note 21 dev-open (178k = 6.1%) is *also* on the failed list at `RGI:1117` but is excluded from the paper's "~5%" group |
| 1110 | $6$--$12$ hours | real calibration windows | `CW:327` "a realistic 6–12 hr window"; `PS:96` "12-hour window" | EXACT |
| 1111 | 15 quasi-Newton iterations | iteration cap | `RGI:1106` "15 TAO its"; `RGI:1132` "(2) 15 its is not converged (J still falling)" | EXACT |

## Noise paragraph (lines 1119–1152)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1128 | a few microns | toy-twin signal | `SH24pm:43` "planar twins have ~5 µm parameter signal"; commit `55cc2b9e` msg "the 5-micrometer parameter signal of the 1.8-s twins" | EXACT |
| 1129 | a tenth of a millimetre | destroying noise | `SH24pm:43-44` "0.1 mm noise destroys per-cell" | EXACT |
| 1130 | $\beta = 1$ | Tikhonov sweep ceiling | commit `55cc2b9e` msg "beta swept to 1 without rescue" | **NOT FOUND** in `logs/` or `plans/` — the only record is the paper's own commit message. Grepped `plans/*.md` and `logs/` for `beta sweep`, `obs_noise`, `sweep to 1`; the sweep itself has no logged artifact |
| 1132 | $\sim$$0.1$\,m | basin-scale signal | `RGI:1108` "peak-WSE MAE 0.103 m" (initial misfit at the uniform start); `FB:60` "~0.1 m signal vs 0.15 m noise" | EXACT |
| 1132 | one simulated hour | window | `RGI:1105` "1-hr rain-forced window" | EXACT |
| 1133 | five orders of magnitude larger | 0.1 m vs 5 µm | 0.1 / 5e-6 = 2×10⁴ | **CONTRADICTED (arithmetic)** — 2×10⁴ is 4.3 orders, not five. The same overstatement is in commit `55cc2b9e` ("five orders of magnitude above the toy twins'"). Either "four" or "more than four" is right |
| 1140 | $0.15$\,m observation noise | HWM-grade noise | `FB:57` "σ = 0.15 m HWM-grade noise"; `PS26:57` "sigma = 0.15 m"; `RGI:1168` "noisy (0.15 m) HWM twin" | EXACT |
| 1142 | $\sigma\sqrt{2/\pi} \approx 0.12$\,m | noise floor on MAE | 0.15 × 0.7979 = 0.1197; `PS26:58-59` "the ~0.12 m even the true field could achieve" | DERIVED (0.15·√(2/π) = 0.1197) + EXACT vs `PS26:59` |
| 1143 | five of the fifteen classes | bound-pinned | `FB:59` "5 of 15 classes pin to bounds"; `PS26:60-61`; `MS:49` | EXACT |
| 1144 | pasture, $428$k cells | pinned to the ceiling | `RGI:1547` "e_11 \| 81 pasture \| 428k"; `MS:49-50` "pasture, 428k cells, to the 0.30 ceiling" | EXACT |

## tab:snr (lines 1154–1175)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1160 | $5\,\mu$m | verification-twin signal | `SH24pm:43` "~5 µm parameter signal" | EXACT |
| 1161 | 40 steps | planar twin window | `driver/tests/adjoint/adjoint_dam_break_analytic.yaml:1` "Wet uniform flowing lake, 40 explicit-Euler steps"; yaml `stop_n: 40` | EXACT |
| 1161 | $1.8$\,s | planar twin window | same yaml `stop: 0.0005`, `unit: hours` → 0.0005 × 3600 = 1.8 s | DERIVED (0.0005 h = 1.8 s) |
| 1161 | (planar) | mesh | same yaml `grid.file: planar_dam_10x5.msh` | EXACT |
| 1160 | per-region recovery $4\times10^{-7}$ | noise-free two-zone | `RMD:29` "recovered from uniform 0.02 to 4.1e-7 rel"; `GH:38` "Two-zone region recovery to 4e-7 relative" | ROUNDED (4.1e-7) |
| 1161 | $0.1$\,mm noise | per-cell destroyed | `SH24pm:43-44` "0.1 mm noise destroys per-cell" | EXACT |
| 1161 | per-cell $0.20 \to 1.8$ | per-cell collapse | start value 0.20: `RMD:34` "recovery … **8.0%** (20 obs times)" and the ctest "gate 0.25, measures 0.199"; end value **1.8**: only commit `55cc2b9e` msg "per-cell 0.20 -> 1.8" | 0.20 = ROUNDED (0.199). **1.8 NOT FOUND** in `logs/` or `plans/` — paper commit message only |
| 1162 | $1$\,mm noise | per-region destroyed | `SH24pm:44` "1 mm destroys per-region recovery" | EXACT |
| 1162 | per-region $4\times10^{-7} \to 0.83$ | per-region collapse | start: `RMD:29`. End value **0.83**: only commit `55cc2b9e` msg "per-region 4e-7 -> 0.83" | 4e-7 ROUNDED (4.1e-7). **0.83 NOT FOUND** in `logs/` or `plans/` |
| 1162 | "both at bounds" | pinning | commit `55cc2b9e` prose "the optimizer driving both values to a bound" | NOT FOUND outside the paper's own history |
| 1164 | 30\,m basin, 1\,h | configuration | `RGI:1105` "1-hr rain-forced window" | EXACT |
| 1165 | 108 marks | observation set | `RGI:1104` | EXACT |
| 1164 | $\sim$$0.1$\,m peak WSE | signal | `RGI:1108` "MAE 0.103 m" | EXACT |
| 1164 | area-weighted rel. $L^2$ $0.20$ | noise-free outcome | `RGI:1118` | EXACT |
| 1164 | MAE $0.103 \to 0.017$\,m | noise-free outcome | `RGI:1108` | EXACT |
| 1165 | $0.15$\,m (HWM grade) | noisy row | `RGI:1168`; `FB:57` | EXACT |
| 1165 | misfit falls $9.4\times$ | noisy row | `FB:58` "J falls 9.4×"; `MS:49` "9.4×" | EXACT |
| 1165 | MAE $0.23 \to 0.12$\,m | noisy row | `FB:58` "MAE 0.234 → 0.117 m"; `RGI:1169` "J 361.8, MAE 0.2336 m (clean twin: 0.1034)"; `MS:49` "MAE 0.23→0.12 m" | ROUNDED (0.2336 → 0.117) |
| 1165 | 5 of 15 classes pin to bounds | noisy row | `FB:59`; `PS26:60-61` | EXACT |
| 1169–70 | seven orders of magnitude (caption) | spread between rows | 5 µm vs 0.1 m = 2×10⁴ (4.3 orders); recovery spread 4×10⁻⁷ vs 0.20 = 5.7 orders | **CONTRADICTED** — no reading of the table gives seven. The signal column spans 4.3 orders; the noise-free *recovery* column spans 5.7. Neither is seven. (The body text at line 1133 says "five", so the paper is also internally inconsistent here) |
| 1171 | $0.1$--$0.3$\,m for high-water marks | survey-grade error | grepped `plans/*.md` for `0.1-0.3`, `0.1--0.3`, `survey-grade`, `quality codes`: only `session-handoff-2026-08-24-evening.md:153` "sigma 0.15 m (USGS HWM quality codes)" | **NOT FOUND** — 0.15 m is sourced; the 0.1–0.3 m band is not, and carries no citation in the paper |

## Sec 6 opening (lines 1177–1192)

No numeric values in this range (section cross-references only).

## Sec 6.1 first paragraph (lines 1194–1211)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1198 | 30\,m cell | mesh resolution | `RMD:189-194` | EXACT |
| 1204 | seven of the twelve gauges with data | bed above peak stage | recomputed from `OBS`: 12 gauges have ≥1 record; 7 have max(WSE) < cell bed (Langham W Little York, S Mayde, Whiteoak Alabonson, Brickhouse, Cole Ck, Little Whiteoak, Whiteoak Main St) | EXACT (recomputed) |
| 1204 | $71\%$ of all gauge records | records below bed | recomputed from `OBS`: 465 non-`nan` records, 134 above bed → 331/465 = **71.2%** below. The 134 above-bed count matches `plans/o63-gauge-weight-audit.md:28` "The 134 kept observations, reconstructed from `obs_turning_h29_41.txt` and the cell beds in the paper's Table `tab:gauges` (exact match, 134)" | EXACT (recomputed; 71.2%) |
| 1205 | two Buffalo Bayou main-stem gauges | 100% above bed | recomputed: Katy and Buffalo Bayou at Houston are 48/48 above bed | EXACT (recomputed) |
| 1206 | two gauges in the Addicks and Barker pools | 100% above bed | recomputed: Langham nr Addicks 20/20, Bear Ck nr Barker 4/4. **Caveat already flagged in-project**: `o63-gauge-weight-audit.md:39-41` "Five gauges, not four; … Langham has 20 and Bear Ck 4" — i.e. 100% is over records *present*, not over the 48-slot window | EXACT under the caption's own definition ("fraction of the window's 15-minute records") |
| 1208 | a fifth, Fulshear, holds it for $29\%$ | partial | recomputed from `OBS`: 14 of 48 records above bed 31.47 → **29.2%**; `o63-gauge-weight-audit.md:34` "Buffalo Bayou nr Fulshear \| 48 \| 14" | EXACT (recomputed) |
| 1211 | nine of the thirteen gauges | in tributaries | no artifact states this. 13 − 4 (the two main-stem + two reservoir gauges) = 9 | DERIVED. Grepped `plans/*.md` for `tributar` — zero hits. Note the arithmetic drops Fulshear from the main stem even though it is a Buffalo Bayou gauge; under a "main-stem = Buffalo Bayou" reading it would be ten of thirteen in the tributaries or three on the main stem |

## tab:gauges (lines 1213–1245)

All thirteen rows recomputed from `logs/o63/obs_turning_h29_41.txt` (header
`13 48`; cell IDs on line 2; 48 records at 900 s = 15 min). Column mapping by
cell ID: 803709 Katy, 877521 Fulshear, 1078603 S Mayde, 1825690 Bear Ck,
1957338 Langham W Little York, 1858212 Langham nr Addicks, 2858868 BB at
Houston, 555139 Whiteoak Alabonson, 575905 Cole Ck, 2342408 Brickhouse,
2870906 Whiteoak at Houston (all `nan`), 2417009 Little Whiteoak, 718908
Whiteoak Main St. Cell-bed values independently corroborated for six gauges at
`CW:158-163` and two at `RGI:2637-2638`.

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1219 | $33.27$ | Katy cell bed | `CW:158` "08072300 Buffalo Bayou nr Katy \| 33.27"; `RGI:2638` "(803709, bed 33.27)" | EXACT |
| 1219 | $35.58$--$36.05$ | Katy window WSE | recomputed: min 35.5824, max 36.0517 | ROUNDED (35.5824 / 36.0517) |
| 1219 | $100\%$ | Katy above bed | recomputed: 48/48 | EXACT |
| 1220 | $6.93$ | BB at Houston cell bed | `RGI:2637` "(cell 2858868, bed 6.93)" | EXACT |
| 1220 | $10.71$--$12.77$ | BB at Houston WSE | recomputed: min 10.7107, max 12.7711 | ROUNDED |
| 1220 | $100\%$ | BB at Houston above bed | recomputed: 48/48 | EXACT |
| 1221 | $28.23$ | Langham nr Addicks bed | no independent artifact; consistent with `o63-gauge-weight-audit.md:32` "Langham Ck nr Addicks (reservoir) \| 20 \| 20 \| 2.51-3.02 m" → bed = 30.736 − 2.51 = 28.226 and 31.2511 − 3.02 = 28.231 | DERIVED (audit's depth column reproduces 28.23 to 2 dp) |
| 1221 | $30.74$--$31.25$ | Langham nr Addicks WSE | recomputed: min 30.7360, max 31.2511 | ROUNDED |
| 1221 | $100\%$ | Langham above bed | recomputed: 20/20 records present | EXACT (of records present; only 20 of 48 slots have data) |
| 1222 | $34.73$ | Bear Ck cell bed | `CW:159` "08072730 Bear Ck nr Barker \| 34.73" | EXACT |
| 1222 | $34.87$--$34.94$ | Bear Ck WSE | recomputed: min 34.8742, max 34.9413 | ROUNDED |
| 1222 | $100\%$ | Bear Ck above bed | recomputed: 4/4 records present | EXACT (of records present; only 4 of 48 slots) |
| 1223 | $31.47$ | Fulshear cell bed | no direct artifact; `o63-gauge-weight-audit.md:35` "Buffalo Bayou nr Fulshear \| 48 \| 14 \| 0.00-0.09 m" → 31.5590 − 0.09 = 31.469 | DERIVED (audit depth column reproduces 31.47) |
| 1223 | $31.23$--$31.56$ | Fulshear WSE | recomputed: min 31.2298, max 31.5590 | ROUNDED |
| 1223 | $29\%$ | Fulshear above bed | recomputed: 14/48 = 29.2% | ROUNDED (29.2%) |
| 1224 | $34.10$ | Langham W Little York bed | not independently sourced; consistent with the row's own 0% (max WSE 34.0919 < 34.10) | NOT FOUND (bed value itself) — grepped `plans/` and `logs/` for `34.10`; only the paper carries it. The regeneration note `PAPER-SESSION-HANDOFF.md:66` says "tab:gauges regenerated from the logged obs table (six endpoints moved 2-9 cm)", i.e. the WSE columns were regenerated but the bed column was carried forward from an unlogged source |
| 1224 | $32.76$--$34.09$ | Langham W Little York WSE | recomputed: min 32.7569, max 34.0919 | ROUNDED |
| 1224 | $0\%$ | above bed | recomputed: 0/48 | EXACT |
| 1225 | $34.60$ | S Mayde cell bed | same situation as 34.10 | NOT FOUND (bed value); grepped `34.60`/`34.6 ` across `plans/`,`logs/` |
| 1225 | $34.07$--$34.29$ | S Mayde WSE | recomputed: min 34.0675, max 34.2870 | ROUNDED |
| 1225 | $0\%$ | above bed | recomputed: 0/48 | EXACT |
| 1226 | $23.60$ | Whiteoak Alabonson bed | NOT FOUND (bed value) — no hit in `plans/`/`logs/` | NOT FOUND |
| 1226 | $22.26$--$23.48$ | Whiteoak Alabonson WSE | recomputed: min 22.2595, max 23.4787 | ROUNDED |
| 1226 | $0\%$ | above bed | recomputed: 0/48 | EXACT |
| 1227 | $20.40$ | Brickhouse Gully bed | `CW:161` "08074250 Brickhouse Gully \| 20.40" | EXACT |
| 1227 | $16.24$--$19.64$ | Brickhouse WSE | recomputed: min 16.2397, max 19.6444 | ROUNDED |
| 1227 | $0\%$ | above bed | recomputed: 0/48 | EXACT |
| 1228 | $25.30$ | Cole Ck bed | `CW:160` "08074150 Cole Ck at Deihl Rd \| 25.30" | EXACT |
| 1228 | $20.95$--$22.47$ | Cole Ck WSE | recomputed: min 20.9459, max 22.4729 | ROUNDED |
| 1228 | $0\%$ | above bed | recomputed: 0/48 | EXACT |
| 1229 | $14.50$ | Little Whiteoak bed | NOT FOUND (bed value) — no hit in `plans/`/`logs/` | NOT FOUND |
| 1229 | $10.65$--$13.37$ | Little Whiteoak WSE | recomputed: min 10.6528, max 13.3685 | ROUNDED |
| 1229 | $0\%$ | above bed | recomputed: 0/48 | EXACT |
| 1230 | $11.80$ | Whiteoak at Main St bed | `CW:163` "08074598 Whiteoak Bayou at Main St \| 11.80"; `CW:229` "model WSE (m) \| 11.80" | EXACT |
| 1230 | $9.71$--$11.33$ | Whiteoak Main St WSE | recomputed: min 9.7140, max 11.3294 (only 9 records present) | ROUNDED |
| 1230 | $0\%$ | above bed | recomputed: 0/9 | EXACT |
| 1231 | $15.03$ | Whiteoak at Houston bed | `CW:162` "08074500 Whiteoak Bayou at Houston \| 15.03" | EXACT |
| 1231 | "no record in window" | col 2870906 | recomputed: all 48 entries `nan` | EXACT |
| 1235 | thirteen rain-driven USGS gauges | table scope | `OBS` line 1 "13 48"; `RMD:172` "--subset rain-driven excludes the 8 dam-affected sites" | EXACT |
| 1235–36 | event hours 29--41 | calibration window | `PS:58-59` "12-hour window, event hours 29–41"; filename `obs_turning_h29_41.txt` | EXACT |
| 1239 | 15-minute records | record cadence | `OBS` time column: 900, 1800, 2700, … (900 s = 15 min), 48 rows over 12 h | EXACT (recomputed) |
| 1240 | At seven gauges the bed sits above even the window's peak stage | restatement | recomputed: 7 | EXACT |

### Recomputation, for reproducibility

```
per-gauge, from logs/o63/obs_turning_h29_41.txt with tab:gauges bed values
Katy               bed 33.27  n=48 above=48  100.0%  min 35.5824 max 36.0517
BB at Houston      bed  6.93  n=48 above=48  100.0%  min 10.7107 max 12.7711
Langham nr Addicks bed 28.23  n=20 above=20  100.0%  min 30.7360 max 31.2511
Bear Ck nr Barker  bed 34.73  n= 4 above= 4  100.0%  min 34.8742 max 34.9413
Fulshear           bed 31.47  n=48 above=14   29.2%  min 31.2298 max 31.5590
Langham W L York   bed 34.10  n=48 above= 0    0.0%  min 32.7569 max 34.0919
S Mayde            bed 34.60  n=48 above= 0    0.0%  min 34.0675 max 34.2870
Whiteoak Alabonson bed 23.60  n=48 above= 0    0.0%  min 22.2595 max 23.4787
Brickhouse         bed 20.40  n=48 above= 0    0.0%  min 16.2397 max 19.6444
Cole Ck            bed 25.30  n=48 above= 0    0.0%  min 20.9459 max 22.4729
Little Whiteoak    bed 14.50  n=48 above= 0    0.0%  min 10.6528 max 13.3685
Whiteoak Main St   bed 11.80  n= 9 above= 0    0.0%  min  9.7140 max 11.3294
Whiteoak Houston   bed 15.03  no record
total records 465, above bed 134 (28.8%), below bed 331 (71.2%)
gauges with data 12; gauges with bed above peak 7
```

## HWM-archive paragraph (lines 1246–1267)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1251 | $0.67$\,m MAE (Inunda) | literature benchmark | citation `li2026inunda`; also quoted at `RGI:1155` "beside Inunda's calibrated 0.67 m" | N/A-citation |
| 1252 | 2{,}364 Harvey marks | STN archive | `CW:405` "2,364 marks"; `SH24pm:28` "2,364 → 324 in-domain → 108 usable" | EXACT |
| 1252 | 324 | in Turning domain | `CW:405` "324 in the Turning domain"; `RGI:2018` "324 \| 5 \| every Harvey mark in-domain, BEFORE QC" | EXACT |
| 1255 | $62\%$ rejected | below-bed QC | `CW:405` "62.3% below cell bed (the gauge trap, now measured)"; `SH24pm:29` "62% below-bed trap measured" | ROUNDED (62.3%) |
| 1255 | 108 marks | QC-passed | `CW:406` "108 usable at quality <= fair"; `RGI:1091` "(108 QC-passed marks)" | EXACT |
| 1256 | quality fair or better | QC threshold | `CW:406` "quality <= fair" | EXACT |
| 1257 | 30\,m simulation (Xu et al.) | literature | citation `xu2025harvey` | N/A-citation |
| 1258 | 48 marks | Xu et al. validation set | citation `xu2025harvey` | N/A-citation |
| 1258 | $0.9$\,m RMSE | Xu et al. | citation `xu2025harvey` | N/A-citation |
| 1259 | 25 of 48 | Xu et al. | citation `xu2025harvey` | N/A-citation |
| 1259 | within $25\%$ | Xu et al. | citation `xu2025harvey` | N/A-citation |

## Uncalibrated-baseline paragraph (lines 1269–1286)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1271 | 72-hour rain-forced forward | o31/o37 baseline | `RGI:1144-1146` "72-hr rain-forced forward"; `PS:99` "o37 ran the full 72 hours" | EXACT |
| 1272 | 2017-08-26 18:00 | initial state | `RGI:1145` "from the only IC (2017-08-26 18:00)"; `PS:99` | EXACT |
| 1274 | $\Delta t = 1$\,s | step size | `RGI:1146` "beuler dt 1"; `PS:100` "259,200 steps at Δt = 1 s" | EXACT |
| 1274 | 2.93M-cell mesh | mesh | `RMD:194` "2,926,532 tri cells" | ROUNDED |
| 1274 | 259{,}200 implicit steps | step count | `RGI:1147` "**259,200/259,200 implicit solves converged, ZERO failures**"; `PS:99-100` | EXACT |
| 1275 | zero Newton failures | robustness | `RGI:1147`; `FB:37` "259,200 solves, 0 failures" | EXACT |
| 1276 | two hours on one GPU node | wall clock | `RGI:1147` "**2h05m wall**" at n4; `PAPER-SESSION-HANDOFF.md:78` "'Four A100 nodes' -> four GPUs on one node" | ROUNDED (2h05m) |
| 1276 | all 108 marks (wet) | coverage | `RGI:1148` "Every one of the 108 QC-passed marks goes wet" | EXACT |
| 1278 | "confirmed mark for mark at fully converged inner tolerances" | o36/o37 check | `FB:39-40` "Confirmed against the earlier loose-tolerance run mark for mark: max per-mark peak difference 0.0124 m, all 108 argmax steps identical" | EXACT (no number printed) |
| 1279 | The 71 marks that crest | admissible population | `RGI:1363` "genuine-crest 71 (control)"; `FB:47` | EXACT |
| 1280 | The other 37 | censored population | `RGI:1363` "censored 37"; `FB:48` | EXACT |
| 1281–82 | every one peaking at the final hour with zero recession | drainage signature | `RGI:1365-1366` "peak location \| ALL at hour 72 (window end)"; "recession from peak \| **0.000 -- every single mark**" | EXACT |
| 1283 | some ten metres of standing water | ponding depth | `RGI:1368` "0.52 m -> **10.7 m and still rising**"; `RGI:1381` "a mean 10.7 m of standing water" | ROUNDED (10.7 m) |
| 1283–84 | arrives laterally after the rain has ended | mechanism | `RGI:1371-1373`; `PS26:100-102` | EXACT (no number) |

## "The exit is not missing" paragraph (lines 1288–1311)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1290 | $35$ of its $6{,}198$ boundary nodes | bounding-box nodes | `RGI:2200` "Only **35 of 6,198 boundary nodes (0.6%)** lie on the bounding box" | EXACT |
| 1293 | thirteen element edges | outlet side set | `RGI:2188` "**ss1 with 13 edges**" | EXACT |
| 1295 | $z \approx 0$ | outlet node elevation | `RGI:2205-2206` "Outlet nodes span z = -0.10 to 5.50 m and the twelve lowest boundary nodes … (z = -0.10) ARE the outlet nodes" | EXACT |
| 1295 | $29.9$\,m boundary median | perimeter median z | `RGI:2207` "Boundary median z is 29.85 m" | ROUNDED (29.85) |
| 1300–1303 | (red TODO note) | citation request | n/a | N/A |
| 1305 | every one of the 37 marks is model-high | bias sign | `FB:48` "never crest … **7.061** \| **+7.061**"; `RGI:1366` | EXACT |
| 1305 | by $7$\,m on average | mean bias | `FB:48` "+7.061"; `PS26:97` "+7.06 m" | ROUNDED (7.06) |
| 1306 | the 71 marks that crest | calibration target | `RGI:1363`; `FB:47` | EXACT |

## tab:baseline (lines 1313–1329)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1319 | 71 | crest inside window, n | `FB:47` "crest inside the 72-h window \| 71"; `MS:16` | EXACT |
| 1319 | $1.51$\,m | MAE, crest group | `FB:47` "1.506"; `RGI:1385` "1.51 m on the admissible 71"; `MS:16` "**1.51 m**" | ROUNDED (1.506) |
| 1319 | $+1.20$\,m | bias, crest group | `FB:47` "+1.198"; `RGI:1385` "(bias +1.20 m)"; `MS:16` | ROUNDED (1.198) |
| 1320 | 37 | never-crest, n | `FB:48`; `MS:17` | EXACT |
| 1320 | $7.06$\,m | MAE, never-crest | `FB:48` "**7.061**"; `MS:17` "7.06 m" | ROUNDED (7.061) |
| 1320 | $+7.06$\,m | bias, never-crest | `FB:48` "**+7.061**"; `MS:17` | ROUNDED (7.061) |
| 1321 | 108 | all marks | `FB:49`; `RGI:1148` | EXACT |
| 1321 | $3.41$\,m | MAE, all | `FB:49` "3.409"; `RGI:1149-1150` "**3.41 m** (J 1.24e7)"; `MS:18` | ROUNDED (3.409) |
| 1321 | $+3.21$\,m | bias, all | `FB:49` "+3.207"; `MS:18` "+3.21 m" | ROUNDED (3.207) |
| 1324 | 108 surveyed marks (caption) | scope | `RGI:1148` | EXACT |

## "The defect is a gradient, not a switch" paragraph (lines 1331–1345)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1332 | three groups / three spatial bands | crest-time sort | `PS26:93-97` (three-row table); `PS:91` | EXACT |
| 1336 | $+2.85$\,m | intermediate-band bias | `PS26:96` "+2.85 m"; `PS:91` "(+0.30 / +2.85 / +7.06 m by band)"; `RGI:2231` | EXACT |
| 1341 | event hours 29--41 | upstream crest window | `PS26:95` "crest h29–41 (upstream)"; `PS:58-59` | EXACT |
| 1342 | the 46 marks in that band | calibration target | `PS26:95` "46"; `PS:107` "The 46 marks are those whose modelled crest falls inside the window" | EXACT |

## tab:bands (lines 1347–1364)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 1353 | crest h29--41 (upstream) | band label | `PS26:95` | EXACT |
| 1353 | 46 | n | `PS26:95` "46" | EXACT |
| 1353 | $-95.695$ | mean longitude | `PS26:95` "−95.695" | EXACT |
| 1353 | $+0.30$\,m | bias | `PS26:95` "+0.30 m"; `PS:91` | EXACT |
| 1353 | $0.72$\,m | MAE | `PS26:95` "0.72"; `PS:62-63` "NLCD lookup (uncalibrated) \| 0.7188" (46 cluster-A marks) | EXACT (0.7188 rounded to 0.72 in both sources) |
| 1354 | crest h48--72 (middle) | band label | `PS26:96` | EXACT |
| 1354 | 25 | n | `PS26:96` "25" | EXACT |
| 1354 | $-95.568$ | mean longitude | `PS26:96` "−95.568" | EXACT |
| 1354 | $+2.85$\,m | bias | `PS26:96` | EXACT |
| 1354 | $2.96$\,m | MAE | `PS26:96` "2.96" | EXACT |
| 1355 | never crest (downstream) | band label | `PS26:97` | EXACT |
| 1355 | 37 | n | `PS26:97` | EXACT |
| 1355 | $-95.440$ | mean longitude | `PS26:97` "−95.440"; `FB:48` "centroid lon −95.44" | EXACT |
| 1355 | $+7.06$\,m | bias | `PS26:97`; `FB:48` "+7.061" | EXACT / ROUNDED (7.061) |
| 1355 | $7.06$\,m | MAE | `PS26:97`; `FB:48` "7.061" | EXACT / ROUNDED (7.061) |

Consistency note: 46 + 25 + 37 = 108 ✓ and 46 + 25 = 71 ✓, so `tab:bands` and
`tab:baseline` partition the same population.

---

## Summary of NOT FOUND and CONTRADICTED

### NOT FOUND

1. **line 1130 — "a sweep up to $\beta = 1$"** (tab:snr body text). Only record is
   the paper's own commit `55cc2b9e` message ("beta swept to 1 without rescue").
   Grepped `plans/*.md` and `logs/` for `beta sweep`, `obs_noise`, `swept`,
   `sweep to 1` — no experiment artifact.
2. **line 1161 — per-cell "$\to 1.8$"** (tab:snr, 0.1 mm-noise outcome). The start
   value 0.20 traces (`RMD:34`, ctest "measures 0.199"); the post-noise 1.8 exists
   only in commit `55cc2b9e`'s message.
3. **line 1162 — per-region "$\to 0.83$"** and "both at bounds" (tab:snr, 1 mm-noise
   outcome). Same: start 4×10⁻⁷ traces (`RMD:29`), the 0.83 is commit-message only.
4. **line 1171 — "$0.1$--$0.3$\,m for high-water marks"** (tab:snr caption). Only
   0.15 m is sourced (`session-handoff-2026-08-24-evening.md:153`, "USGS HWM quality
   codes"). The band is uncited in the paper and unsourced in the repo.
5. **tab:gauges bed elevations for four gauges — $34.10$ (Langham at W Little York),
   $34.60$ (S Mayde), $23.60$ (Whiteoak at Alabonson), $14.50$ (Little Whiteoak)**
   (lines 1224–1229). The other nine beds are corroborated (`CW:158-163`,
   `RGI:2637-2638`, or back-derived from `o63-gauge-weight-audit.md:31-36`). The WSE
   columns were regenerated from `OBS` this September
   (`PAPER-SESSION-HANDOFF.md:66`), but the bed column was carried forward from a
   source that is not in `logs/` or `plans/`. Grepped each value across both trees.
6. **line 1104 — "$\sim$5\% of cells"** for forests + shrub + emergent wetland. No
   artifact sums these class areas; the only area data is `RGI:1543,1548` ("42
   evergreen 43k"; "others … 12k-86k"). Consistent with ~5% but not measured
   anywhere. Also note `RGI:1117` lists 21 developed-open (178k = 6.1% of cells)
   among the same failing classes, and the paper's "~5%" group excludes it.

### CONTRADICTED

1. **line 1133 — "five orders of magnitude larger"** ($\sim$0.1 m vs 5 µm). The
   ratio is 2×10⁴, i.e. **4.3 orders**. (The error is inherited from commit
   `55cc2b9e`'s prose.)
2. **lines 1169–1170 — "differ by seven orders of magnitude in the signal they
   descend on"** (tab:snr caption). No column supports seven: the *signal* column
   spans 5 µm → 0.1 m = 4.3 orders; the noise-free *recovery* column spans
   4×10⁻⁷ → 0.20 = 5.7 orders. The caption also contradicts the body text's "five"
   at line 1133 for the same comparison.

### Sourced, but with a caveat the paper should carry

- **tab:gauges "$100\%$" for Langham nr Addicks and Bear Ck nr Barker**
  (lines 1221–1222). Correct under the caption's definition ("fraction of the
  window's 15-minute records"), but those gauges have only 20 and 4 of 48 records;
  `plans/o63-gauge-weight-audit.md:39-41` already flags that the prose reading
  ("hold their water for the whole window") overstates coverage.
- **line 1211 — "nine of the thirteen gauges"**. Arithmetic complement of the four
  100% gauges, not a measured classification; no `tributar*` hit anywhere in
  `plans/`. Under a literal main-stem reading (Katy, Fulshear, Houston are all
  Buffalo Bayou) the complement would be ten, not nine.
