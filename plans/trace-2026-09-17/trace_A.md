# Number provenance trace — manning-calibration.tex, lines 66–237 and 2248–2331

Paper: `/Users/markadams/Codes/RDycore-gpu/papers/manning-calibration/manning-calibration.tex`
Sources rooted at `/Users/markadams/Codes/RDycore-gpu/`.
Short names: `RESULTS` = `plans/RESULTS-gpu-implicit.md`; `HANDOFF` = `plans/PAPER-SESSION-HANDOFF.md`;
`AUDIT` = `plans/o63-gauge-weight-audit.md`; `DECIDE` = `plans/team-decision-list.md`.

---

## ABSTRACT

### ¶1 (lines 67–79) — the capability

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 72 | "one exact, assembled Jacobian" | count of Jacobians | design claim, not a measurement; `papers/.../manning-calibration.tex:308` `\label{sec:jacobian}` | N/A (definitional) |
| 75 | "about two forward solves" | objective+gradient cost | paper body `tex:651-656` Eq.(gradcost) "$\approx 2.1$ forward solves"; primary: `RESULTS:794-800` "(3600 steps, revolve max_cps_ram 400) ... TSStep (10399 incl. recomputes) 654 s (62.9 ms/step), TSAdjointStep 53.4 s (14.8 ms/step)"; `RESULTS:606` "3199 revolve recompute steps (~0.89 extra forwards, near optimal for 400 cps)" | ROUNDED (source 2.1) |
| 77 | "four NVIDIA A100s" | GPU count, n4 | `RESULTS:577` "Single-node apples-to-apples (4 A100s vs 64 cores of the same node)"; `RESULTS:768` "1.73 s device n4" | EXACT |
| 78 | "$19\times$" | device-vs-host TAO iteration | `RESULTS:767-768` "Single-node honest device-vs-host is now ~19x per TAO iteration (33.1 s host-types n64 vs 1.73 s device n4)" | EXACT (33.1/1.73 = 19.1). **Caveat**: the 33.1 s host figure is the *pre-optimization* run (`RESULTS:574-576`, TaoSolve 165.5 s / 5 its); the 1.73 s device figure is post-optimization (`RESULTS:761`). The same-session apples-to-apples ratio recorded at `RESULTS:577-578` was "~6.2x". |
| 78 | "64-core CPU node" | host comparison | `RESULTS:577` "4 A100s vs 64 cores of the same node"; `RESULTS:574` "Host types n64" | EXACT |

### ¶2 (lines 81–96) — the measurement

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 81 | "2.93M-cell" | Turning Harvey mesh | `RESULTS:278` "Turning 30 m (2.93M cells)"; exact count `RESULTS:568` "2,926,532 per-cell Manning parameters" | ROUNDED (2,926,532) |
| 81 | "30\,m" | mesh resolution | `RESULTS:278`, `RESULTS:1669` "2.93M cells, GPU types, the 46 real cluster-A marks" | EXACT |
| 82 | "46 surveyed high-water marks" | scoring mark set | `RESULTS:1573` "46 real marks"; `logs/o57/o57_ic0.7.log:80` "hwm observations: 46 marks (0 below cell bed, zero-weighted)"; `logs/o61/o61_spectrum_sigma_alpha0.30.txt:1` "46 marks (46 weighted), 15 classes" | EXACT |
| 84 | "$0.08$\,m" | uniform-scale authority | body `tex:1432` (tab:alpha caption) "worth $0.08$\,m against a $0.72$\,m error"; primary `RESULTS:1584` "The entire uniform-roughness knob is worth 0.08 m against a 0.72 m error (11%)"; arithmetic `RESULTS:1579,1582` 0.7188 − 0.6392 = 0.0796 | ROUNDED/DERIVED (0.0796) |
| 84 | "$0.72$\,m" | NLCD-prior mark MAE | body tab:alpha `tex:1420` "$1.00$ (NLCD) ... $0.7188$"; primary `RESULTS:1582` "1.0 (NLCD) \| 0.027-0.160 \| 8.0826e2 \| 0.7188 \| 0/46"; also `RESULTS:1763` "J 8.082566e+02, peak-WSE MAE 0.7188 m, 0 of 46" | ROUNDED (0.7188) |
| 86 | "below any entry in the land-cover table" | α=0.3 n range | `RESULTS:1582` "0.3 \| 0.008-0.048"; table range `RESULTS:2105-2119` (0.027 barren … 0.160 developed-high) | **Imprecise**: only the *low end* (0.008) is below every table entry; the α=0.3 field's upper end is 0.048, above barren (0.027) and developed-open (0.040). Body `tex:1429-1431` states it correctly ("where $n$ reaches $0.008$"). |
| 86 | "the fifteen land-cover classes" | NLCD class count | `RESULTS:2103-2119` (15-row class table); `logs/o61/o61_spectrum_sigma_alpha0.30.txt:1` "15 classes" | EXACT |
| 87 | "about $0.10$\,m" | 15-class error removal | body tab:threefifteen `tex:1890,1893` (0.7188 → 0.6116 / 0.6154); primary `RESULTS:2071` "calibrated, 15 classes, 9 its \| 0.6116 \| -0.107"; and `RESULTS:2440` "MAE 0.6154 m" (σ_α=0.30) | DERIVED: 0.7188−0.6116 = 0.1072; 0.7188−0.6154 = 0.1034. "About 0.10 m under either prior width" is the wording HANDOFF:79 records. |
| 87 | "$15\%$" | share of 0.72 m error | `RESULTS:2073` "Roughness accounts for **15%** of the 0.72 m discrepancy"; body `tex:1595` | DERIVED (0.1072/0.7188 = 14.9%) |
| 89 | "sixteen forward runs" | spectrum cost | `RESULTS:1983` "costs 16 FORWARDS rather than 16 forward+adjoints"; artifact count `ls logs/o61/o58_e0.05w43200_pk_*.txt` = 16 (base + 15 class columns) | EXACT |
| 90 | "$\pm 30\%$" | prior width | `logs/o61/o61_spectrum_sigma_alpha0.30.txt:8` "sigma_alpha = 0.3 uniform, sigma_obs = 0.15"; `RESULTS:2432` "The coauthors (2026-09-09) put the honest width at +/-30%" | EXACT |
| 90 | "about three combinations" | eigenvalues > 1 | `logs/o61/o61_spectrum_sigma_alpha0.30.txt` "eigenvalues > 1 : 3 of 15"; "degrees of freedom for signal (Rodgers): 3.39"; cross-check `RESULTS:2688` "3 supported, dofs 3.01" (14 classes, 24 dropped) | EXACT |
| 91 | "the fifteen classes" | denominator | as line 86 | EXACT |
| 91 | "$84\%$" | three-class share of fifteen | body `tex:1911-1913` "three parameters remove $126$ of the $151$ objective units fifteen remove, $84\%$"; primary `RESULTS:2151-2152` "removed 126.1 of the 150.9 units ... **83.6% of the reduction from 20% of the parameters**"; independent MAE route `RESULTS:2459` "sigma_alpha 0.30 (task 3) \| 0.6290 \| -0.090 \| **84%**" | ROUNDED (83.6% objective / 84% MAE share; 126.1/150.9 = 83.6%) |
| 93 | "developed-medium and woody wetland at $0.3$ times their table values" | where the three land | `RESULTS:2465-2466` "23 -> 0.036, 90 -> 0.0294 = 0.3x"; `RESULTS:2553` "0.036 (0.30x) \| 0.029 (0.30x)"; body tab:crossobs `tex:2062` | EXACT |
| 93 | "$0.3$ times" (lower bound) | α box floor | `RESULTS:1702` "The bounds become alpha in [0.3, 3.0]" | EXACT (it is the bound, not an interior optimum — `RESULTS:2476` "reach the 0.3x floor under every prior") |
| 94 | "below the table's own value for developed open land" | 0.040 | `RESULTS:2106` "21 \| developed open \| 0.040"; `DECIDE:51-52` "n = 0.036 and 0.029 -- below the lookup's own value for developed open land (0.040)" | EXACT/DERIVED |
| 92 | "the same under every prior we ran" | determinacy | `RESULTS:2483-2484` "the three-class field is DETERMINED (same answer for 23/90 from three priors, 1-3 iterations vs 9)"; three priors = σ_n 0.015, σ_α 0.30, σ_α 0.50 (`RESULTS:2458-2460`) | EXACT |

### ¶3 (lines 98–107) — the gauges

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 99 | "seven of twelve gauges" | bed above peak stage | body tab:gauges `tex:1216-1244` (seven rows at "above bed $0\%$"; thirteen rows, one "no record in window" → twelve with data); `RESULTS:2522` "the same defect that disqualified seven of twelve gauges outright" | EXACT. Note the underlying obs table (`logs/o63/obs_turning_h29_41.txt`) is the regeneration source per HANDOFF:66-67 ("tab:gauges regenerated from the logged obs table"). |
| 99 | "30\,m cell's bed" | mesh resolution | as line 81 | EXACT |
| 102 | "the five gauges that do hold water" | above-bed set | body `tex:2023-2027` "Five of them hold water ... 134 records in all"; primary `AUDIT`/`RESULTS:2600-2601` "The 134 = Katy 48 + Houston 48 + Langham 20 + Fulshear 14 + Bear Ck 4 (five gauges, reconstructed exactly)" | EXACT |
| 102 | "the same three classes" | {22, 23, 90} | `RESULTS:2542` "Calibrate the three classes {23, 90, 22}"; `RESULTS:2552-2553` | EXACT count. **Imprecise claim**: only *two* of the three move oppositely — 22 rises under both objectives (gauges 0.27, marks 0.210 against the lookup's 0.090). Body `tex:2034-2035` states it correctly ("raises all three classes where the mark calibration lowered two of them"). |
| 103 | "scores worse at the marks than no calibration at all" | 0.7609 vs 0.7188 | `RESULTS:2555-2556` "the gauge-calibrated field is WORSE on the marks than no calibration at all (0.7609 vs 0.7188 m, J 894.9 vs 808.3, +10.7%)"; body tab:crossobs `tex:2061` | EXACT (no numeral printed in the abstract) |
| 105 | "$99.5\%$" | offset share of gauge J | body `tex:2110` (tab:gaugeresid caption) "the offsets carry $99.5\%$ of $J$"; primary `RESULTS:2811-2812` "**J 31313 = 31145 constant per-gauge offset (99.5%) + 168 hydrograph shape (0.54%)**" | EXACT |
| 105 | "a constant level error of metres at each gauge" | per-gauge offsets | `RESULTS:2805-2809` +4.901, +1.944, +1.965, −0.613, +1.197 m | **Imprecise**: Langham Ck nr Addicks is −0.61 m, i.e. sub-metre. Body `tex:2084-2086` states it correctly ("$4.9$\,m ... and $1$--$2$\,m too high at three of the other four"). |
| 106 | "$0.24$\,m" | hydrograph shape RMSE | body `tex:2111` "RMSE $3.24$\,m in all, $0.24$\,m in shape alone"; primary `RESULTS:2812` "RMSE 3.243 m overall, **0.238 m in shape alone**"; reproduced per HANDOFF:58-61 from `logs/o63/obs_turning_h29_41.txt` + `logs/o65/o65_gauge_base.txt(.zb)` | ROUNDED (0.238) |
| 106 | "tracks the shape of **every** hydrograph to $0.24$\,m" | scope of the claim | per-gauge shape rms `RESULTS:2805-2809`: 0.382 (Houston), 0.067, 0.055, 0.123, 0.012 | **CONTRADICTED as to "every"**: 0.238 m is the aggregate; Buffalo Bayou at Houston's shape rms is 0.382 m. The body avoids the word "every". |

### ¶4 (lines 109–113) — the initial condition

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 110 | "$20\%$ reduction" | IC scan perturbation | body tab:ic `tex:2188` (a = 0.80 row); primary `RESULTS:1940-1943` "\| a \| 0.80 \| ... \| MAE \| 0.6434 \|"; `RESULTS:1947` "Like for like at a 20% perturbation" | EXACT |
| 111 | "$0.075$\,m" | IC authority at 20% | body `tex:2196-2197` "$0.075$\,m versus $0.020$\,m"; primary `RESULTS:1945-1947` "dMAE per unit fractional change is **0.392** ... 20% perturbation: 0.075 m versus 0.020 m" | EXACT (0.392 × 0.20 = 0.0784; the recorded like-for-like value is 0.075) |
| 112 | "$0.020$\,m" | roughness authority at 20% | same as above; slope 0.098 per unit α (`RESULTS:1946`) | EXACT (0.098 × 0.20 = 0.0196) |

---

## INTRODUCTION (lines 116–237)

### ¶1 (lines 118–158)

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 122 | "\emph{every} cell's roughness" | parameter count claim | `RESULTS:568` "2,926,532 per-cell Manning parameters"; body `tex:666` "one parameter or to 2.93 million" | N/A (qualitative) |
| 123 | "a few forward solves" | gradient cost | Eq.(gradcost) `tex:655` "2.1"; `RESULTS:794-800` | ROUNDED |
| 125–138 | `ding2004manning`, `yan2014adjoint`, `lacasta2016gpu`, `dassflow`, `karna2023baltic`, `warder2021storm` | literature | bibliography | N/A (bibliographic) |
| 139–140 | "Pujol et al. ... 2D urban flood model" | related work; "2D" | `\cite{pujol2024urban}` | N/A (bibliographic) |
| 151 | "$15\%$" | the paper's answer | `RESULTS:2073` "Roughness accounts for **15%** of the 0.72 m discrepancy"; body `tex:1595` | EXACT |
| 152 | "$\pm 30\%$" | prior width | `logs/o61/o61_spectrum_sigma_alpha0.30.txt:8` | EXACT |
| 153 | "about three combinations" | supported count | `logs/o61/o61_spectrum_sigma_alpha0.30.txt` "eigenvalues > 1 : 3 of 15" | EXACT |
| 153 | "its fifteen classes" | denominator | same file, "15 classes" | EXACT |
| 154 | "a calibration of three parameters recovers most of what fifteen achieve" | 84% | `RESULTS:2151-2152` (83.6%), `RESULTS:2459` (84%) | EXACT (no numeral printed) |
| 155 | "the twelve the marks cannot narrow" | 15 − 3 | `logs/o61/o61_spectrum_sigma_alpha0.30.txt` (rows 3–14 "comparable"/"prior-determined"); `RESULTS:2131` "the other twelve frozen at the prior" | DERIVED (15 − 3 = 12) |

### ¶2 (lines 160–172)

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 163 | Inunda, "Hurricane Harvey hindcast" | related work | `\cite{li2026inunda}` | N/A (bibliographic) |
| 164 | Hydrograd, AegirJAX, routing | related work | citations | N/A (bibliographic) |
| 170 | CaMa-Flood-GPU | related work | `\cite{kang2026cama}` | N/A (bibliographic) |

### ¶3 (lines 174–197)

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 179 | "two decades of ECCO" | MITgcm/ECCO history | `\cite{forget2015ecco}` | N/A (bibliographic) |
| 193 | "471M cells" | RDycore validation scale | `plans/session-handoff-2026-03-23.md:147` "RDycore validated at 471M cells, R²=0.99 on Malpasset dam break"; also `plans/pi-briefing-manning-calibration.tex:176` "validated (471M cells)" | **Weak provenance**: found only in project notes; no log, no RESULTS entry, and **no citation attached in the paper**. Grepped `471M\|471 M\|471e6\|471 million` across `logs/`, `plans/`, `*.bib`, `*.tex`. Needs a citation before submission. |

### ¶4 — contributions (lines 199–236)

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 202 | "one exact, assembled Jacobian" | capability | design claim; gates in `tex:398` tab:verification | N/A (definitional) |
| 207 | "about $2.1$ forward solves" | gradient cost | body Eq.(gradcost) `tex:651-656`; primary `RESULTS:794-800` (1.00 fwd; 62.9 ms/step fwd, 14.8 ms/step adjoint → 0.235) + `RESULTS:606` ("~0.89 extra forwards") | DERIVED: 1.00 + 0.89 + 14.8/62.9 (= 0.235) = 2.13 ≈ 2.1 |
| 209 | "30\,m Hurricane Harvey mesh" | explicit-friction failure | body `tex:894-900`; primary `plans/RESULTS-manning-draft.md:271-273` "99.9% of the 2.93M cells are wet, median depth ... reaches **420/s**, so explicit friction needs dt < 2.4 ms"; `RESULTS:1196-1197` "dt 0.125 and dt 0.05 ... are ALSO non-finite, J inf" | EXACT |
| 213 | "about $15\%$" | roughness ceiling | `RESULTS:2073` | EXACT |
| 213 | "$0.72$\,m survey error" | baseline MAE | `RESULTS:1582` "0.7188" | ROUNDED |
| 220 | "about three of fifteen at a $\pm 30\%$ prior" | supported count | `logs/o61/o61_spectrum_sigma_alpha0.30.txt` "eigenvalues > 1 : 3 of 15" | EXACT |
| 221 | "$84\%$" | three-class share | `RESULTS:2151-2152` (83.6% objective), `RESULTS:2459` (84% MAE) | ROUNDED |
| 222–223 | "lands below the lookup's own range for developed ground" | 0.036 / 0.029 vs 0.040 | `RESULTS:2477` "n = 0.036, 0.0294"; `RESULTS:2106` "developed open \| 0.040" | DERIVED |
| 226 | "$99.5\%$" | gauge offset share | `RESULTS:2811` "J 31313 = 31145 constant per-gauge offset (99.5%)" | EXACT |
| 228 | "scores worse at the marks than the uncalibrated lookup" | 0.7609 vs 0.7188 | `RESULTS:2556`; `logs/o63/o63_score_gauge_c3_sa0.30.log` per `AUDIT:24` | EXACT (no numeral) |
| 232 | "the closed catchment-divide perimeter" | mesh forensics | `RESULTS:2199-2208` "Only **35 of 6,198 boundary nodes (0.6%)** lie on the bounding box"; attribution HANDOFF:303-308 (Donghui, 09-16) | EXACT (no numeral printed here) |

---

## CONCLUSIONS AND OUTLOOK (lines 2248–2314)

### ¶2 (lines 2258–2271) — the measurement

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 2262 | "$4\%$" | roughness authority inside the prior | body tab:alpha `tex:1418` "$0.70$ ... $0.6894$" + caption `tex:1430-1432` "Within the prior it is worth about $3$\,cm ($\alpha = 0.70$)"; primary `RESULTS:1588` "Over a defensible range (alpha >~ 0.7) it is worth about 3 cm"; `RESULTS:1608` "~4% within it"; `RESULTS:2069` "uniform alpha 0.70 \| 0.6894 \| -0.029" | DERIVED (0.7188 − 0.6894 = 0.0294; 0.0294/0.7188 = 4.1%) |
| 2263 | "$11\%$ at the scan's low end" | α = 0.30 | `RESULTS:1584-1585` "worth 0.08 m against a 0.72 m error (11%) -- and only at alpha 0.3"; `RESULTS:1579` "0.3 \| 0.008-0.048 \| 6.7367e2 \| **0.6392**" | DERIVED (0.7188 − 0.6392 = 0.0796; /0.7188 = 11.1%) |
| 2264 | "$15\%$ in sample" | calibrated ceiling | `RESULTS:2071-2073`; body `tex:1594-1597` | EXACT |

### ¶3 (lines 2273–2286) — next steps

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 2274 | "failed in sign as well as in size" | o63 | `RESULTS:2555-2566` | N/A (no numeral) |
| 2276 | "the middle band carries the drainage defect" | bands | body tab:bands `tex:1363`; `RESULTS:1371-1380` (37 censored marks, ~10.7 m ponding) | N/A (no numeral) |
| 2281 | "$dJ/du_0$ already verified" | FD gate | `plans/RESULTS-manning-draft.md:27` "PASS (dJ/du0 vs FD 1.03e-8; commit 3a1e15db)"; body tab:verification `tex:398` | EXACT (no numeral printed) |

### ¶4 (lines 2288–2314) — cost

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 2289 | "30\,m mesh" | resolution | `RESULTS:278` | EXACT |
| 2290 | "$\Delta t = 0.25$\,s" | CFL limit | body `tex:979` "CFL-limited $\Delta t = 0.25$\,s (518{,}400 steps)"; primary `RESULTS:1208` "0.25 s; 90k cells violate dt*tb > 1 at 0.25 s"; `plans/session-handoff-2026-08-24-evening.md:93` "not by the 0.25 s CFL" | EXACT |
| 2290 | "36-hour" | Harvey gradient window | body `tex:978-980` "a full 36-hour Hurricane Harvey window at the 30\,m mesh's CFL-limited $\Delta t = 0.25$\,s (518{,}400 steps, 2.93M cells)" | EXACT (36 h × 3600 / 0.25 = 518,400) |
| 2291 | "several forward sweeps" | gradient cost | Eq.(gradcost) 2.1 | ROUNDED |
| 2291 | "half a million steps" | step count | body `tex:979` "518{,}400 steps" | DERIVED (36 × 3600 / 0.25 = 518,400) |
| 2294 | "30\,m, 2.93M-cell" | mesh | `RESULTS:278` "Turning 30 m (2.93M cells)"; exact 2,926,532 at `RESULTS:568` | ROUNDED |
| 2295 | "$\Delta t = 1$\,s" | backward-Euler step | `RESULTS:278` "dt=1 BEULER x20 steps"; `RESULTS:1147` "**259,200/259,200 implicit solves converged, ZERO failures, 2h05m**"; body `tex:900` | EXACT |
| 2296 | "cannot take a step at any tested size" | explicit path | `RESULTS:1196-1197` "dt 0.125 and dt 0.05 (advective CFL ~0.007) are ALSO non-finite, J inf"; body `tex:898-900` ("$\Delta t = 0.25$, $0.125$, $0.05$\,s") | EXACT |
| 2298 | "about two iterations per step" | Newton count | `RESULTS:1304` "259,200 steps converge in **2 Newton iterations** (2 steps at 3, 1 at 5)"; `RESULTS:1040-1041` "598 of 600 solves at 2 Newton its"; body tab:cure `tex:965` | EXACT |
| 2303 | "four A100 GPUs" | n4 | `RESULTS:577` "4 A100s vs 64 cores of the same node" | EXACT |
| 2304 | "one-simulated-hour window" | gradient window | `RESULTS:599` "3600 steps at dt=1"; `RESULTS:792` "1-hr revolve gradient re-verified" | EXACT |
| 2304 | "3{,}600 implicit steps" | step count | `RESULTS:794` "b4 protocol exactly (3600 steps, revolve max_cps_ram 400, unforced)" | EXACT |
| 2305 | "400 in-memory checkpoints" | revolve budget | `RESULTS:601` "-ts_trajectory_max_cps_ram 400 (revolve checkpointing)"; `RESULTS:794` | EXACT |
| 2305 | "about nine minutes" | gradient wall time | `RESULTS:800` "Per-gradient at a 1-hr window: ~19 -> ~9.4 min at n4" | ROUNDED (9.4 min) |
| 2307 | "418{,}076 gauges" | dense-gauge twin | `RESULTS:568` "418,076 twin gauges, 2,926,532 per-cell Manning parameters, BLMVM"; also `plans/session-handoff-2026-08-24.md:116` "dense, 418,076 strided cells" | EXACT |
| 2307 | "39-step window" | twin window | `RESULTS:567` "gauges twin: 39-step windows, 13 obs times" | EXACT |
| 2307 | "$1.7$\,s" | per-TAO-iteration device time | `RESULTS:761` "TaoSolve (20 its) \| 106.2 (5.3/it) \| 34.6 (**1.73/it**)"; `RESULTS:768` "1.73 s device n4" | ROUNDED (1.73) |
| 2307 | "$19\times$" | device vs host | `RESULTS:767-768` "~19x per TAO iteration (33.1 s host-types n64 vs 1.73 s device n4)" | EXACT (19.1). Same caveat as the abstract: the host term is the pre-optimization run. |
| 2308 | "64-core CPU node" | host baseline | `RESULTS:574` "Host types n64 (SAME binary and parmetis partition)"; `RESULTS:577` "64 cores of the same node" | EXACT |
| 2309–2310 | "bitwise-identical trajectories ... gradients identical to every printed digit" | device/host equivalence | `RESULTS:434-437` "with the SAME binary and partition ... host-types vs device-types ... gradients are IDENTICAL to all printed digits: J = 143.434, \|dJ/du0\| = 1574->1564.37 both, sum(dJ/dn) = -446848 both"; `RESULTS:739` "Trajectories o1-vs-o2 cmp-BITWISE IDENTICAL"; `RESULTS:575-576` "J-trace IDENTICAL to the device run to every printed digit" | EXACT for the gradients. **Partly unsupported**: the logged *bitwise* trajectory comparison at `RESULTS:739` is a device-vs-device A/B (`-pc_pbjacobi_invert_device` on/off), not device-vs-host. The device/host claim in the sources is "identical to all printed digits", not bitwise. `plans/PROJECT-STATE.md:36` does assert "device and host bitwise identical" but without a logged comparison. |
| 2312–2313 | "tens of TAO evaluations instead of hundreds" | future cost reduction | **NOT FOUND**. Grepped `tens of TAO`, `hundreds of TAO`, `TAO evaluations` across `plans/` and `logs/`; nothing. Nearest measured iteration counts: 1–9 for class calibrations (`RESULTS:2456-2460`, `RESULTS:2028`), 20 for the per-cell dense twin (`RESULTS:571`). The "hundreds" figure has no source. | NOT FOUND |
| 2313 | "calibration windows placed on the storm peak" | future work | no number | N/A |

### Code and data availability (lines 2316–2331)

| line | number as printed | context | source (file:line, quoted) | status |
|---|---|---|---|---|
| 2317 | github.com/RDycore/RDycore | repo URL | — | N/A (URL) |
| 2319 | `adams/gpu-implicit` | branch | current branch (git status) | N/A |
| 2325 | "\textbf{[DOI to be minted]}" | Zenodo DOI | HANDOFF:109-111 "Zenodo DOI and archive location: deferred" | N/A (explicit placeholder, deliberate) |
| 2327 | stn.wim.usgs.gov | STN API URL | body `tex:1253-1256` (2,364 marks / 324 in domain) | N/A (URL) |
| 2329–2330 | "\textbf{[Mesh, checkpoint, and rainfall archive: location to be stated.]}" | placeholder | HANDOFF:109-111 | N/A (explicit placeholder) |

---

## Summary of problems

### NOT FOUND (1)
1. **line 2312–2313, "tens of TAO evaluations instead of hundreds"** — no source anywhere in `plans/` or `logs/`. Grepped `tens of TAO`, `hundreds of TAO`, `TAO evaluations`. The only measured counts are 1–9 (class calibrations) and 20 (dense per-cell twin). The "hundreds" comparator appears to be invented.

### CONTRADICTED / materially imprecise (5)
1. **line 106, "the model tracks the shape of every hydrograph to $0.24$\,m"** — 0.238 m is the *aggregate* shape RMSE (`RESULTS:2812`). Per gauge (`RESULTS:2805-2809`) Buffalo Bayou at Houston has shape rms **0.382 m**. "Every" is not supported; the body (`tex:2086-2087`) correctly says "What remains in hydrograph shape is $0.24$\,m RMSE".
2. **line 105, "a constant level error of metres at each gauge"** — Langham Ck nr Addicks is **−0.61 m** (`RESULTS:2808`), sub-metre. The body states it correctly.
3. **line 102–103, "the same three classes move the opposite way"** — only **two of three** move oppositely. Developed-low (22) rises under both objectives (gauges 0.27, marks 0.210, lookup 0.090; `RESULTS:2552-2553`). The body (`tex:2034-2035`) says "raises all three classes where the mark calibration lowered two of them".
4. **line 84–86, "removes at most $0.08$\,m ... and only at values below any entry in the land-cover table"** — the α = 0.3 field spans n = 0.008–0.048 (`RESULTS:1582`), so its upper end sits above barren (0.027) and developed-open (0.040). Only the low end is below every entry. The body's tab:alpha caption is precise ("where $n$ reaches $0.008$").
5. **line 2309–2310, "Device and host runs produce bitwise-identical trajectories"** — the logged bitwise trajectory comparison (`RESULTS:739`) is device-vs-device. The device-vs-host evidence (`RESULTS:434-437`, `RESULTS:575-576`) is "identical to all printed digits", which is what the second half of the sentence already claims. Either downgrade "bitwise" or cite a device/host bitwise run.

### Weak provenance (2)
1. **line 193, "471M cells"** — traces only to `plans/session-handoff-2026-03-23.md:147` and `plans/pi-briefing-manning-calibration.tex:176`; no log, no RESULTS entry, and **no citation in the paper**. Needs a reference.
2. **lines 78 and 2307, "$19\times$"** — arithmetically exact (33.1/1.73) but the numerator is a pre-optimization host run (`RESULTS:574-576`) and the denominator a post-optimization device run (`RESULTS:761`). The contemporaneous apples-to-apples ratio recorded in the same section was "~6.2x" (`RESULTS:577-578`). The claim is as `RESULTS:767` states it ("Single-node honest device-vs-host is now ~19x"), so it is sourced — but a reviewer asking whether the CPU side was re-optimized has no answer in the record.
