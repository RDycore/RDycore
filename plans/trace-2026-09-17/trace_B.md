# Number-provenance audit — manning-calibration.tex lines 238–996

Paths below are relative to `/Users/markadams/Codes/RDycore-gpu/`.
Abbreviations: **RGI** = `plans/RESULTS-gpu-implicit.md`, **RMD** =
`plans/RESULTS-manning-draft.md`, **CW** = `plans/campaign-wednesday.md`,
**MS** = `plans/meeting-summary-2026-08-26.md`, **PS26** =
`plans/PROJECT-STATE-2026-08-26.md`, **PS** = `plans/PROJECT-STATE.md`.

---

## §2.1 Forward model (lines 244–290)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 245 | `u = (h, q_x, q_y)` (3 components) | conservative variables | `src/swe/…`; block size 3 throughout (`src/tests/test_swe_jacobian.c` uses `cons[3]`) | N/A — definition |
| 251 | `h^{-7/3}`, `gh^2/2` | Manning drag / pressure | standard SWE closure | N/A — definition |

## §2.2.1 Exact Jacobian (lines 310–361)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 310 | `3×3` blocks | block sparsity on cell graph | RMD:26 "frozen-dissipation flux blocks … MatSetValuesBlockedLocal"; `plans/pi-briefing…tex` "block-sparse (3×3 blocks on the cell-adjacency graph)" | EXACT |
| 323 | `-L_e/|Ω_i|`, `+L_e/|Ω_j|` | scatter weights | derivation from Eq. (2) | N/A — definition |
| 345 | `six` tangent evaluations per edge | seeding cost | 3 conservative seeds × 2 sides = 6; no explicit source statement found (grepped "six tangent", "tangent evaluations", "6 seeds" across `plans/`, `src/`, `driver/`) | DERIVED (3+3 seeds); not stated anywhere |
| 355 | `h_g = (q^2/g)^{1/3}` | critical-outflow ghost | code's `CONDITION_CRITICAL_OUTFLOW` branch (RGI:966) | N/A — definition |
| 360 | `-2 g n h^{-7/3} q‖q‖` | ∂S/∂n | differentiation of Eq. (2) | N/A — definition |
| 361 | `two` nonzeros per cell | parameter Jacobian | RMD:28 (inc 4) "`SWERHSJacobianP` (∂f/∂n, **2 nnz/cell**)" | EXACT |

## Table 1 `tab:verification` (lines 382–391)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 382 | wet jump `10 m / 5 m` | edge-flux config | `src/tests/test_swe_jacobian.c:119` `PetscReal consL[3] = {10.0, 0.0, 0.0}, consR[3] = {5.0, 0.0, 0.0};` | EXACT |
| 382 | `7.6×10⁻⁹` | edge flux blocks, wet jump | RMD:32 (inc 2c) "distinct-state **7.6e-9**"; also `plans/pi-briefing-manning-calibration.tex:55` `$7.6\times10^{-9}$` | EXACT |
| 382 | gate `10⁻⁶` | | `test_swe_jacobian.c:126-127` `printf("distinct-state exact-Jacobian rel error: %.3e (gate: 1e-6)")` / `assert_true(err < 1e-6)` | EXACT |
| 383 | `2.4×10⁻¹⁰` | transcritical (entropy-fix branch) | RMD:32 "transcritical **2.4e-10**"; pi-briefing:55 | EXACT |
| 383 | gate `10⁻⁵` | | `test_swe_jacobian.c:142-143` "(gate: 1e-5)" / `assert_true(err < 1e-5)` | EXACT |
| 384 | `<10⁻⁶` | source block, shallow flowing state | RMD:25 (inc 1) "**PASS (source block vs FD < 1e-6)**"; `test_swe_jacobian.c:203` `cons[3] = {0.4, 0.25, -0.1}  // shallow, flowing: friction matters`, `:206 assert_true(err < 1e-6)` | EXACT |
| 384 | gate `10⁻⁶` | | `test_swe_jacobian.c:206` | EXACT |
| 385 | `1.6×10⁻⁸` | assembled matrix, uniform flow, reflecting BCs | **No individual source value.** RMD:33 (inc 2d) gives only the range "full-matrix (unmasked) FD-vs-analytic **1.5-1.9e-8** on 3 state/BC combos"; pi-briefing:56 likewise "1.5–1.9×10⁻⁸". The only per-config number for the *uniform flowing lake* is RMD:26 (inc 2b) "global rel err **1.4e-8**", but that is the frozen-dissipation assembly, not 2c/2d's exact one. Grepped `1.6e-8`, `1.6e-08`, `1.6[0-9]*e-0*8` repo-wide over `*.md *.txt *.log *.tex *.c *.h *.out` — zero hits. | **NOT FOUND** (inside the recorded range, but no source states it) |
| 385 | gate `10⁻⁶` | | `src/tests/test_swe_jacobian_global.c:83-84` "(gate: 1e-6)" / `assert_true(rel_err < 1e-6)` | EXACT |
| 386 | `1.5×10⁻⁸` | assembled matrix, dam break, reflecting | RMD:32 "dam-break global **1.5e-8**" | EXACT |
| 386 | gate `10⁻⁶` | | `test_swe_jacobian_global.c:84` | EXACT |
| 387 | `1.9×10⁻⁸` | assembled matrix, dam break, Dirichlet + critical outflow | RMD:33 "FD-vs-analytic 1.5-**1.9e-8** on 3 state/BC combos" (upper end of the range) | EXACT (as range endpoint) |
| 387 | gate `10⁻⁶` | | `test_swe_jacobian_global.c:84` | EXACT |
| 388 | `8` sampled components | dJ/du₀ (explicit RK) | `driver/adjoint_test.c:36` "`-adjoint_fd_samples <int>` # of u0 components for the FD check (**default: 8**; 0 disables)"; `:1647` `fd_samples = 8`; `driver/tests/adjoint/CMakeLists.txt:36` runs `adjoint_dam_break.yaml` with no override | EXACT |
| 388 | `1.0×10⁻⁸` | dJ/du₀ (explicit RK) | RMD:558 "FD gates **1.0e-8** / 2.9e-6 (RK), np 1 and 2" (RMD:33 gives 1.05e-8; RMD:27 gives 1.03e-8 for the fd-Jacobian variant) | EXACT |
| 388 | gate `10⁻⁵` | | RMD:27 (inc 3) "per-component dJ/du0 FD gate **1e-5**" | EXACT |
| 389 | `2.9×10⁻⁶` | dJ/dn (explicit RK), domain aggregate | RMD:558 "1.0e-8 / **2.9e-6** (RK)"; RMD:33 "(1.05e-8 / **2.9e-6**)" | EXACT |
| 389 | gate `10⁻⁵` | | RMD:28 (inc 4) "domain-aggregate FD gate **1e-5**" | EXACT |
| 390 | `1.4×10⁻⁸` | dJ/du₀ (implicit BE), tight inner tolerances | RMD:35 "theta-method adjoint passes both FD gates (**1.4e-8** / 5.3e-6) with tight inner tolerances" | EXACT |
| 390 | gate `10⁻⁵` | | same line ("both FD gates" = the 1e-5 pair above) | EXACT |
| 391 | `5.3×10⁻⁶` | dJ/dn (implicit BE) | RMD:35 "(1.4e-8 / **5.3e-6**)" | EXACT |
| 391 | gate `10⁻⁵` | | as above | EXACT |
| 396 | "automated test in RDycore's ctest suite" | caption | `driver/tests/adjoint/CMakeLists.txt:36,44,47,54,58` and `src/tests/CMakeLists.txt` register all of these | EXACT |

## §2.2.2 Adjoint sensitivities (lines 404–434)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 406 | `σ²`, `H` one row per gauge | observation operator | definition | N/A |
| 425 | Manning gradient fails at default tolerances | | RMD:558-560 "With **DEFAULT tolerances** implicit gradients degrade to ~1e-4–2e-4"; RGI:1345 "(6.4e-4, failing)" | EXACT (qualitative) |
| 430 | Krylov `10⁻⁴`, nonlinear `10⁻⁵` | production setting | RGI:1351 "run at **ksp 1e-4 / snes 1e-5**"; RGI:1410; MS:92; `plans/session-handoff-2026-08-26.md:142` | EXACT |
| 431 | tighter Krylov makes the forward **faster** | | RGI:1327 "**Tightening 1e-2 -> 1e-3 makes the run FASTER** (203 s vs 231 s) on 21% less linear work" | EXACT |
| 432 | "reproduces the fully converged answer to every printed digit" | | RGI:1322 "**C reproduces the converged answer D exactly**" (rung C `1e-4/1e-5` J = 3.429274e6 = rung D `1e-6/1e-8`) | EXACT |
| 432 | about `25%` more wall time than the loosest usable setting | | RGI:1313-1316 ladder: A (`1e-2/1e-3`, the campaign/loosest setting) **231 s**, C (`1e-4/1e-5`) **291 s**. 291/231 − 1 = **0.26**. | DERIVED → 26%, printed 25%. (Note: against rung B, 203 s — the *fastest* rung — it would be **43%**, so the sentence only holds if "loosest usable" means rung A.) |

## The peak observable + Table 2 `tab:fdwindow` (lines 437–501)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 447 | "passes the same central-difference gates" (peak-misfit parameter gradient) | | RGI:1089 "ctests `adjoint_hwm_fd_np_{1,2}` at **8.9e-7** (gate **1e-5**)"; CW:409 "argmax-time adjoint injection, FD-gated at **9e-7**" | EXACT |
| 455 | deterministic to **nine** digits | | MS:66 "evaluations reproduce to **nine digits**" | EXACT |
| 456 | "stable across a decade of probe step" | | RGI:1473 "**The ones-direction FD is stable across a decade** (1.3607e6 at 1e-3, 1.3586e6 at 1e-4)" | EXACT |
| 459 | grows **three orders of magnitude** | 60-step gate → basin windows | 9e-5 → 2e-1 = 2 222×, i.e. 3.35 orders (RGI:1521-1525) | DERIVED |
| 466 | `10⁻⁵` gate passed outright by the largest single-class direction at the smallest probe | | RGI:1545 "coord A (class 24) … 1e-5 rung: **8.28e-6 — PASSES the 1e-5 gate**"; RGI:1462 "coordinates are the **3 largest-|g| classes**" | EXACT |
| 467 | the **fifteen** per-class gaps sum to the domain-wide gap | | RGI:1554-1557 "**Superposition** … the signed per-class gaps SUM to the domain-wide gap (**2.35e5 vs 2.33e5, ~1%**)"; 15 NLCD classes | EXACT |
| 473 | "several percent from any one branch" | | RGI:1473-1475 "FD has converged to a value **7.5%** from the adjoint" | ROUNDED (7.5%) |
| 486 | smooth verification twin, **40** steps | | `driver/tests/adjoint/adjoint_dam_break_analytic.yaml:23` `stop_n: 40` (the yaml `adjoint_hwm_fd_np_{1,2}` runs, `driver/tests/adjoint/CMakeLists.txt:74-75`) | EXACT |
| 486 | `9×10⁻⁷` (gate `10⁻⁵`: pass) | | RGI:1089 "**8.9e-7** (gate 1e-5)"; CW:409 "9e-7"; `plans/fable-brief-2026-08-25.md:77` "8.9e-7" | ROUNDED (8.9e-7) |
| 487 | basin, 1 min, **60** steps, `9×10⁻⁵`, inner-solve floor: pass | | RGI:1521 "\| 60 s \| **8.923e-5** \| inner-solve floor: the gate effectively PASSES \|" (dt = 1 s → 60 steps) | ROUNDED (8.923e-5) |
| 488 | basin, 5 min, **300**, `8×10⁻³` | | RGI:1522 "\| 300 s \| **8.082e-3** \|" | ROUNDED |
| 489 | basin, 15 min, **900**, `2×10⁻²` | | RGI:1523 "\| 900 s \| **2.375e-2** \|" | ROUNDED (2.375e-2 → 2e-2) |
| 490 | basin, 30 min, **1800**, `2×10⁻¹` | | RGI:1524 "\| 1800 s \| **2.071e-1** \| o40d (30-min config) \|" | ROUNDED |
| 491 | basin, 1 h, **3600**, `7×10⁻²` | | RGI:1525 "\| 3600 s \| **7.470e-2** \| o39/o38C (1-hr config) \|" | ROUNDED |
| 497-499 | caption: 30-min and 1-h rows sample different horizons; single-class directions one to three orders lower | | RGI:1527-1531 "the 3600-s point is a different J (peaks over a longer horizon), so strict monotonicity across configs is not expected. **Single-class directions stay 1-3 orders cleaner at every window**" | EXACT |

## §2.3 Calibration algorithm (lines 507–635)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 519 | about `2.1` forward solves | per-iteration cost | RGI:797-798 + :606-607 (see the gradient-cost block below); `plans/team-decision-list.md:18` "cost about **2.1 forward solves**" | DERIVED / EXACT (see Eq. (6) rows) |
| 557-635 | Algorithms 1–2 line/index numbers (`k = 1..K`, `s = K m … 1`) | pseudocode | structural | N/A |

## "What a gradient actually costs" + Eq. (6) (lines 637–677)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 645 | `2.93M`-cell mesh | Turning 30 m | RGI:568 "…**2,926,532** per-cell Manning parameters"; RGI:278 "Turning 30 m (**2.93M cells**)" | EXACT |
| 645 | `3,600` implicit steps at `Δt = 1 s` | one-simulated-hour window | RGI:600 "beuler_dt1_1hr.yaml … **3600 steps at dt=1**" | EXACT |
| 646 | `400` in-memory checkpoints | | RGI:601 "`-ts_trajectory_max_cps_ram 400` (revolve checkpointing)" | EXACT |
| 646 | `four` A100 GPUs on one node | | RGI:598 "(real-window check, **device n4**)"; RGI:576 "4 A100s" | EXACT |
| 648 | `62.9` ms forward step | | RGI:797 "TSStep (10399 incl. recomputes) 654 s (**62.9 ms/step**)" | EXACT |
| 648 | `14.8` ms adjoint step | | RGI:798 "TSAdjointStep 53.4 s (**14.8 ms/step**, was 101)" | EXACT |
| 648 | `3,199` revolve recomputation steps | | RGI:606 "the checkpointed backward works: **3199 revolve recompute steps**" | EXACT |
| 648 | `0.89` of an additional forward | | RGI:607 "(**~0.89 extra forwards**, near optimal for 400 cps)". Cross-check 3199/3600 = 0.888 | EXACT |
| 652 | `1.00` (forward) | | normalization | N/A |
| 653 | `0.89` (recomputation) | | RGI:607 | EXACT |
| 654 | `0.24` (adjoint sweep) | | 14.8/62.9 = **0.2353** | DERIVED → 0.24 |
| 655 | `≈ 2.1` forward solves | | 1.00 + 0.89 + 0.24 = **2.13** | DERIVED |
| 658 | about **nine minutes** on that node | | RGI:800 "Per-gradient at a 1-hr window: ~19 -> **~9.4 min** at n4". Cross-check from the per-step costs: (3600+3199)×62.9 ms + 3600×14.8 ms = **481 s = 8.0 min** | ROUNDED (source 9.4 min) |
| 662 | the `400` used here are near optimal for this step count | | RGI:607 "**near optimal for 400 cps**" | EXACT |
| 666 | `2.93` million parameters | | RGI:568 "2,926,532 per-cell Manning parameters" | EXACT |
| 675 | reproduces the previous run's last objective **to six digits** | resume exactness | `plans/session-handoff-2026-08-24-evening.md:72` "the warm run's TAO iteration 0 reports the previous run's final objective **to 6 digits (1466.42)**" | EXACT |
| 673 | matched by land-cover class rather than by position | | same file, :68 "**Matched by NLCD code, not position**; a file missing any class is an error" | EXACT |
| 676 | BLMVM's curvature pairs do not survive | | same file, :73 "BLMVM's quasi-Newton history does not carry over" | EXACT |

## "Scaling the calibration problem" (lines 679–727)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 683 | `‖g‖ ≈ 1.5×10³` | on the production window, physical units | RGI:1697 "With n as the variable and **|g| = 1554** where n ~ 0.05". **Caveat:** this is measured on the o48 600-step smoke, not on the full production window; the paper attributes it to "the production window". | ROUNDED (1554 → 1.5e3); attribution slightly off |
| 684 | `n ≈ 0.05` | | RGI:1697 "where **n ~ 0.05**" | EXACT |
| 686 | uniform `α = 0.2` diverges | | RGI:1577 "\| **0.2** \| 0.005-0.032 \| **DIVERGED_NONLINEAR_SOLVE** \|"; RGI:1698 "uniform alpha 0.2 is measured to diverge (o44)" | EXACT |
| 689 | "the finite-difference gates pass throughout" | | RGI:1723-1729 (all 21 adjoint/calibration ctests pass; alpha gradient at 2.3e-6) | EXACT |
| 697 | `α = 1.5` carries 1.5× lookup | illustration of the multiplier convention | definition | N/A |
| 700 | roughness change of order `10%` | unit quasi-Newton step | RGI:1701 "The first trial step is **0.10 in alpha (a 10% roughness change)**" | EXACT |
| 702 | "over a shortened window that reproduces the zero-step failure exactly" | | RGI:1670 "over a **600-step window instead of 43,200**"; RGI:1676-1680 "Zero iterations, J_final = J_init" | EXACT |
| 703 | `‖g‖ / J₀ = 0.10` | | RGI:1685 "first trial step **|g|/J0 = 0.1036** in alpha" | ROUNDED |
| 706 | `α ∈ [0.3, 3]` | bounds | RGI:1702 "The bounds become **alpha in [0.3, 3.0]** — a physical statement that also excludes the divergent region" | EXACT |
| 714 | `β = σ_n^{-2}` | | RGI:1704-1705 "`-adjoint_sigma_n <s>` sets **beta = 1/s^2**" | EXACT |
| 719 | β chosen independently, spanning **six orders of magnitude** | | Recorded β values across experiments: **1e-6, 1e-5, 1e-4, 1, 1e2, 11.1, 4444** (grep over RGI/RMD/CW). 1e-6 → 4.4e3 is ~**9.6 orders**; even the hand-chosen subset 1e-6 → 1e2 is **8 orders**. | **CONTRADICTED** (source spread is 8–10 orders, not six) |
| 720 | `σ_n = 0.015` | | RGI:1712 "\| **0.015** \| **4444** \| **alpha 0.70** \|"; RGI:1682 `-adjoint_sigma_n 0.015` | EXACT |
| 721 | minimum at `α ≈ 0.70` | | RGI:1712 "uniform-mode minimum **alpha 0.70**" | EXACT |
| 722 | `σ_n = 0.020` leaves no interior minimum | | RGI:1713 "\| 0.020 \| 2500 \| **alpha 0.33 (bound)** \|"; :1716 "at 0.020 the problem runs to the bound again" | EXACT |
| 724 | MAE `0.6894` m measured at `α = 0.70` | | RGI:1900 "\| MAE \| … \| **0.6894** \|"; RGI:1904 "the measured point at 0.70 is **0.6894**"; RGI:2264-2265 "(J 7.654816e2, MAE 0.6894)" | EXACT |
| 725 | `0.6895` m the balance implied | | RGI:1903 "The predicted optimum for sigma_n = 0.015 was alpha 0.697 at MAE **0.6895**" | EXACT |

## "Absolute or fractional prior" (lines 729–765)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 731 | `σ_n = 0.015` | | as above | EXACT |
| 732 | `n = 0.027` (barren) | NLCD lookup low end | `logs/o62/o62_p_c15_sa0.30_it1.txt` (the prior dump) `31 0.027`; RGI:2110 "\| 31 \| barren \| **0.027** \|" | EXACT |
| 733 | `n = 0.16` (developed high) | lookup high end | same dump `24 0.16`; RGI:2109 "\| 24 \| developed high \| **0.160** \|" | EXACT |
| 733 | `±56%` | prior on barren | RGI:2430 "**+/-56% on barren**". Cross-check 0.015/0.027 = **0.5556** | EXACT + DERIVED |
| 734 | `±9%` | prior on developed high | RGI:2430 "**+/-9% on developed-high**". Cross-check 0.015/0.16 = **0.09375** | EXACT + DERIVED |
| 734 | penalty scales as `n_prior²` | | RGI:1706-1707 "with **sum_k n_prior_k^2 = 0.1399** over the 15 classes" | EXACT (mechanism) |
| 738-742 | "at the **production configuration's first quasi-Newton step**, every class driven to the **upper** bound was small-n (pasture `0.038`, developed-open `0.040`, developed-low `0.090`) and every class driven to the **lower** bound was large-n (developed-high `0.160`, developed-medium `0.120`, shrub `0.115`, woody wetland `0.098`)" | claimed signature of the **absolute** prior | The exact seven-class set {21, 22, 81 up; 23, 24, 52, 90 down} is recorded **only** for the *fractional* σ_α = 0.30 run: RGI:2442-2444 "The first step put **SEVEN classes on a bound (21, 22, 81 at alpha = 3; 23, 24, 52, 90 at 0.3)**", confirmed cell-by-cell in `logs/o62/o62_p_c15_sa0.30_it2.txt` (`21 0.12`=3×0.04, `22 0.27`=3×0.09, `81 0.114`=3×0.038, `23 0.036`, `24 0.048`, `52 0.0345`, `90 0.0294` = 0.3×prior). The one recorded **absolute-prior** first step (o52, σ_n = 0.015, RGI:1789-1794) pins a *different* set: 23, 24, 90 at α = 0.300 and 22 at α = **2.620** (not at the bound); no pasture, no developed-open, no shrub. RGI:1839 (armijo, the production line search) is smaller still: "23, 90 = 35.9%", "near alpha 3: **none**". | **CONTRADICTED** — the numbers belong to the fractional σ_α = 0.30 run (next paragraph), not to the absolute prior. The individual n values (0.038/0.040/0.090/0.160/0.120/0.115/0.098) are all EXACT against `o62_p_c15_sa0.30_it1.txt`; the attribution is what fails. |
| 739 | pasture `0.038` | lookup value | `o62_p_c15_sa0.30_it1.txt` `81 0.038` | EXACT |
| 740 | developed-open `0.040` | | same, `21 0.04` | EXACT |
| 740 | developed-low `0.090` | | same, `22 0.09` | EXACT |
| 741 | developed-high `0.160` | | same, `24 0.16` | EXACT |
| 742 | developed-medium `0.120` | | same, `23 0.12` | EXACT |
| 742 | shrub `0.115` | | same, `52 0.115` | EXACT |
| 743 | woody wetland `0.098` | | same, `90 0.098` | EXACT |
| 748 | the **fifteen**-class calibration | | 15 NLCD classes, `o62_p_c15_sa0.30*.txt` (15 rows) | EXACT |
| 749 | `σ_α = 0.30` | width assigned to the lookup | RGI:2434 "The coauthors (2026-09-09) put the honest width at **+/-30%**"; RGI:2440 "15 classes, **sigma_alpha = 0.30**, 3 TAO iterations" | EXACT |
| 750 | first step drove **seven of the fifteen** to a bound | | RGI:2442-2444 "**SEVEN classes on a bound** (21, 22, 81 at alpha = 3; 23, 24, 52, 90 at 0.3)"; verified in `logs/o62/o62_p_c15_sa0.30_it2.txt` | EXACT |
| 751 | finished with **two** classes on the floor | | RGI:2444-2445 "Final table: **developed-medium 0.036 and shrub 0.0345 on the floor**"; `logs/o62/o62_p_c15_sa0.30.txt` `23 0.036`, `52 0.0345` | EXACT |
| 752 | below the lookup's value for developed open land | | developed-open = 0.040 (`21 0.04`); 0.036 < 0.040 and 0.0345 < 0.040 | DERIVED |
| 753 | "at the same misfit the absolute prior reached" | | RGI:2441-2442 "Scored: **MAE 0.6154 m, J_mis 615.44** — against the absolute-prior 15-class field's **0.6116 m / 615.40**" | EXACT |
| 753 | in **three times** the iterations | | RGI:2440 fractional run = **3** TAO iterations; RGI:2067 absolute run = **9** iterations. 9/3 = 3 | DERIVED |

## §3 Model configurations — intro (lines 770–777)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 771 | **three** domains | | planar dam (`planar_dam_10x5.msh`), Houston 1 km (`Houston1km_with_z.exo`), Turning 30 m — the three meshes in `tab:configs` | DERIVED |
| 771 | resolutions **two orders of magnitude** apart | | 1 km vs 30 m = **33×** ≈ 1.5 orders (RMD:44/hurricane-harvey-simulation-plan.md:44 "1km"; RGI:278 "Turning 30 m"). The planar twin's cell size is not stated anywhere I could find (grepped `planar_dam` in `plans/`, `share/meshes/`). | **NOT FOUND / overstated** — the only two resolutions with sources differ by 33×, not 100× |
| 772 | **four** different observables | | water height (verification twin), stage gauges (Houston + Turning class twin), dense strided cells (dense twin), peak WSE (mark twin/production) = 4 | DERIVED |

## Table 3 `tab:configs` (lines 785–824)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 786 | "planar, **single region**" | verification twin mesh | `driver/tests/adjoint/adjoint_dam_break.yaml` declares **two** regions (`upstream`, `downstream`), both material `smooth`; `driver/adjoint_test.c:3036` recovers per-region `n_true[r] = 0.03 + 0.03*r` over `n_regions` ≥ 2, and RMD:29 records "**two-zone** twin: n_true (0.03, 0.06)". | **CONTRADICTED** (the mesh carries two regions; the *field* is uniform in the FD-gate runs) |
| 787 | `1.8` s | verification twin window | `adjoint_dam_break.yaml` / `adjoint_dam_break_analytic.yaml` `time: stop: 0.0005 unit: hours` → 0.0005 × 3600 = **1.8 s** | DERIVED (EXACT) |
| 787 | `40` steps | | same yamls, `stop_n: 40` | EXACT |
| 789 | per-cell, per-region | parameters | `calibrate_manning_twin_np_1` (per-region) and `calibrate_manning_percell_np_1` (per-cell) in `driver/tests/adjoint/CMakeLists.txt:40,50` | EXACT |
| 791 | `1` km, `2,746` cells | Houston twin | `plans/hurricane-harvey-simulation-plan.md:44` "1km resolution Houston area (**2746 cells**)" | EXACT |
| 792 | `4200` s | | `plans/hurricane-harvey-simulation-plan.md:45` "**Final time**: **4200 seconds** (70 minutes)" | EXACT |
| 792 | `Δt = 30` s | | RMD:36 "implicit BEULER carries the Harvey window at **dt=30 s**"; RMD:352 | EXACT |
| 793 | `343` or `686` gauges, `14` times | | RMD:36 "(**343 gauges x 14 obs times**)" and "(beta=1e-4, **686 gauges**)"; RMD:55 | EXACT (the "14 times" is stated for the 343 case; not restated for 686) |
| 794 | `2,746` per-cell | parameters | RMD:36 "Per-cell twin (**2746 params**, ALL observable)" | EXACT |
| 796-797 | `1` km, `2,746` cells; `4200` s, `Δt = 30` s | Houston real network | as above | EXACT |
| 798 | `17` USGS gauges | | RMD:47 "**17** with Harvey stage series"; RMD:613 "**17 USGS gauge cells**, Houston 1 km" | EXACT |
| 798 | `238` obs | | CW:343 "(**17-gauge twin: fits 238 observations** essentially exactly)". Cross-check 17 × 14 = **238** | EXACT + DERIVED |
| 799 | `2,746` per-cell | | RMD:613 | EXACT |
| 801-802 | `30` m, `2.93M` cells | Turning class twin | RGI:278, RGI:568 (2,926,532) | EXACT |
| 802 | "early transient" | window | CW:317 "**The window is 20 SECONDS** (`stop: 20.0`, dt = 1 s)" | EXACT |
| 803 | `13` gauges, `260` obs | | CW:300-301 "\| **real 13-site network (o25i)** \| **260** \|" | EXACT |
| 804 | `15` NLCD classes | | CW:296 "15 NLCD classes"; `logs/o62/*_it1.txt` (15 rows) | EXACT |
| 806-807 | `30` m, `2.93M` cells | dense twin | as above | EXACT |
| 808 | `418,076` strided cells | | RGI:568 "**418,076** twin gauges"; CW:300 "dense, **418,076 strided cells** (o18d) \| 8,361,520 \|"; RGI:986 "**418076** obs cells x 20 times" | EXACT |
| 811-812 | `30` m, `2.93M` cells | mark twin | as above | EXACT |
| 813 | `1` h, rain-forced | | RGI:1102-1104 "NLCD-truth twin observed at the 108 QC-passed real mark cells' peaks (12 samples over a **1-hr rain-forced window**, free-outflow outlet)" | EXACT |
| 814 | `108` marks, peak WSE | | RGI:1091 "`data/harvey_hwm/turning30m_hwm_obs.txt` (**108 QC-passed marks**)"; RGI:1104 | EXACT |
| 816-817 | `30` m, `2.93M` cells | full event | as above | EXACT |
| 818 | `72` h, `Δt = 1` s | | RGI:1144-1147 "**72-hr** rain-forced forward … beuler **dt 1** … **259,200/259,200** implicit solves". 72 × 3600 = 259,200 | EXACT + DERIVED |
| 819 | `108` marks, peak WSE; forward only | | RGI:1148 "Every one of the **108 QC-passed marks** goes wet"; eval-only | EXACT |
| 821-822 | `30` m, `2.93M` cells | production | as above | EXACT |
| 823 | `12` h, event h29–41 | | RGI:1572 "(**12-hr cluster-A window h29-41**, 46 real marks, IC = o37 h29 checkpoint)"; PS:104 "— event hour 29 — and integrate **43,200 steps** to hour 41" | EXACT |
| 824 | `46` marks, peak WSE | | RGI:1572; RGI:2058 "(43,200 steps, **46 marks**, 144 obs times)"; PS:107 | EXACT |
| 824 | global scale, `15` classes | parameters | RGI:2440 (15-class), RGI:1596 (uniform α scan) | EXACT |
| 838 | the last **five** rows are the 30 m Harvey domain | caption | rows: class twin, dense twin, mark twin, full event, production = **5** | DERIVED (internally consistent) |

## §3 The twins / the 30 m domain / shared settings (lines 846–883)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 846 | `40`-step verification twin | | `adjoint_dam_break*.yaml` `stop_n: 40` | EXACT |
| 850 | `1` km Houston mesh | | `plans/hurricane-harvey-simulation-plan.md:44` | EXACT |
| 851 | `2,746` cells | | same | EXACT |
| 852 | all `2,746` per-cell parameters observable | | RMD:36 "**2746 params, ALL observable -- whole domain wet+moving**" | EXACT |
| 854 | `n = 0.03` west, `n = 0.06` east of the domain midline | truth field | `driver/adjoint_test.c:2605` `n_true[i] = (xc[i] < 0.5 * (x_min + x_max)) ? 0.03 : 0.06;` (also :2826, :2897); RMD:29 "n_true (**0.03, 0.06**)" | EXACT |
| 857 | the **seventeen** in-mesh USGS gauges | | RMD:47 "20/90 gauges in mesh, **17** with Harvey stage series" | EXACT |
| 859 | `30` m, `2.93M` cells | Turning mesh | RGI:278, :568 | EXACT |
| 860 | `15` classes | | `logs/o62/o62_p_c15_sa0.30_it1.txt` (15 rows) | EXACT |
| 861 | **Three** configurations share it | | The same paragraph then names **four** (class twin, dense twin, mark twin, production), and `tab:configs` carries **five** Turning rows (adding *full event*). | **CONTRADICTED** (internal count) |
| 866 | `108` QC-passed high-water-mark cells | | RGI:1091 | EXACT |
| 869 | `12`-hour window, event hours `29–41` | | RGI:1572; PS:104 | EXACT |
| 870 | hourly checkpoint of the `72`-hour forward | | RGI:1288-1290 (o34: "**72-hr** eval-only forward with HOURLY checkpoints (72 files, ~70 MB each)") | EXACT |
| 871 | rain re-aligned to the window start; both validated **bit-exactly** | | RGI:1257-1259 "hour 2 of a continuous run vs the same hour restarted from the 1-hr checkpoint is **bitwise identical**"; RGI:1271-1275 (o33 at 2.93M: leg B reproduces leg A to every printed digit) | EXACT |
| 872 | `46` surveyed marks whose modelled crest falls inside it | | PS:107 "The **46 marks** are those whose modelled crest falls inside the window" | EXACT |
| 880 | Krylov `10⁻⁴`, nonlinear `10⁻⁵` | | RGI:1351, :1410; MS:92 | EXACT |
| 882 | Manning bounds `[0.01, 0.2]` | | `plans/autorun-kickoff.md:65` "bounds n ∈ **[0.01, 0.2]**"; `plans/github-issue-manning-adjoint.md:24`; `plans/differentiable-rdycore-adjoint-plan.md:187` | EXACT |

## §4 Implicit stepping (lines 887–913)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 888 | `Δt ≲ 2 h^{4/3}/(g n² |v|)` | linearized stability limit | RGI:1205-1207 gives "**`dt < 1/tb`, `tb = g n^2 h^{-4/3} |v|`**" — i.e. **no factor 2**; and the quoted 2.4 ms below equals 1/tb (1/420 = 2.38 ms), not 2/tb (4.8 ms). | **CONTRADICTED** — the printed factor 2 is inconsistent both with the source formula and with the 2.4 ms value the paper derives from it |
| 890 | `30` s gravity-wave step of the 1 km mesh | | RMD:36 "at **dt=30 s**"; RMD:352 | EXACT |
| 891-893 | explicit source returns a non-finite state within the first objective evaluation at Δt = 30 s | | RMD:36 "(**explicit friction NaN'd immediately**, confirming briefing Sec. 8)" | EXACT |
| 895 | margin is a factor of **400** | basin scale | RGI:1219 "the implicit path wins by **~400x in step count**" | EXACT |
| 896 | `30` m, `2.93M`-cell Harvey mesh | | RGI:278, :568 | EXACT |
| 897 | `99.9%` wet, median depth `6` mm | spun-up state | RGI:1206-1207 "On the spun-up 30 m state (**99.9% wet, median depth 6 mm**)"; RMD:271 "**99.9% of the 2.93M cells are wet**"; `plans/note-to-team-manning-config.md:9` "median depth of **6.4 mm**" | EXACT (6.4 mm → 6 mm elsewhere) |
| 898 | `Δt ≲ 2.4` ms | friction-rate limit | RGI:1207 "tb reaches 420/s => **dt < ~2.4 ms**"; RMD:273; `plans/github-issue-manning-adjoint.md:46` | EXACT |
| 899 | tested explicit steps `Δt = 0.25, 0.125, 0.05` s | | RGI:1193 "o29f \| explicit euler \| **0.25** s \| J inf, 108/108 marks dry"; RGI:1196-1198 "o32 walked it down: **dt 0.125** and **dt 0.05** … are ALSO non-finite, J inf, 108/108 dry" | EXACT |
| 900 | backward Euler at `Δt = 1` s completes a `72`-hour window without a Newton failure | | RGI:1147 "**259,200/259,200** implicit solves converged, **ZERO failures**, 2h05m wall" | EXACT |
| 906 | theta method carries the Harvey window at `Δt = 30` s | | RMD:35-36 | EXACT |
| 911 | ARK-IMEX stage system is block-diagonal, `3×3` Newton per cell | | RMD:38 (inc 1b) "friction in TSSetIFunction/IJacobian (**per-cell block-diagonal**)" | EXACT |

## §4 "Newton robustness is set by residual discontinuities" (lines 916–956)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 921 | descends for **a dozen** iterations then stalls | | RGI:879 "Newton descends cleanly 903 -> 1.26e-3 in **~13 its**, then the residual FREEZES" | ROUNDED (13) |
| 922-923 | a line-search step of negligible length changes the residual norm **severalfold** | | RGI:880 "at the stall a **lambda = 1e-13** step jumps the residual **8x** (1.26e-3 -> 1.01e-2)"; RGI:955-958 "residual JUMPING 1.158e-3 -> 1.019e-2"; CW:275-277 "ANY step past lambda ~ 1e-5 raises it **~24x**" | EXACT (8×) — note CW records 24× in the other configuration |
| 925 | amplitude scales as `n²` | drag cutoff | RGI:886-887 "**NLCD-scale n^2 raised the jump amplitude ~100x**" | EXACT |
| 926 | critical-outflow **stagnation switch** | | RGI:965-969 "The CONDITION_CRITICAL_OUTFLOW branch zeroes BOTH states when **uperp < 0** … at the **uperp = 0** crossing the flux jumps by the full wet-onto-dry Roe flux, **O(g h²/2) ~ 1e-2** in residual norm" | EXACT |
| 929 | `g n² h^{-7/3} q‖q‖`, gated at the dry cutoff | | RGI:881-885 "gated by h >= tiny_h with **tiny_h = 1e-7** and a **PLAIN q/h velocity**" | EXACT |
| 932 | `O(h^{-7/3})` term | | same | N/A — definition |
| 933 | `u = q h/(h² + h_anuga²)` | ANUGA regularized velocity | `src/tests/test_swe_jacobian.c:164-175` (`SourceMap(..., h_anuga, ...)`); RGI:893-895 (the decided fix) | N/A — definition |
| 934 | `h_anuga = 0.001` m | | RGI:1002-1003 "**DECISION: NLCD/class-mode recipe = h_anuga_reg_parameter 0.001** (smallest clean value)"; CW:89 "`h_anuga_reg_parameter: 0.001`" | EXACT |
| 935 | drag vanishes as `h^{5/3}` | | algebra: `n² h^{-7/3} · (q h/(h²+h_a²))² · h²` → `h^{5/3}` as h→0 | N/A — definition |
| 936 | reduces to the plain form at `h_anuga = 0` | | RGI:909-911 "with h_anuga_reg_parameter = 0 the new code must be [bitwise identical] -- that is the A/B gate"; RGI:935 "h_anuga = 0 bitwise: ctest adjoint\|calibrate\|jacobian 14/14" | EXACT |
| 936 | the smallest that removes the stall | | RGI:1002 "(**smallest clean value**)" | EXACT |
| 937 | `0.003` m is indistinguishable | | RGI:989-990 "**h_anuga 0.003: CLEAN, J-trace nearly identical (7.46462e6)** -- the regularization at this size barely perturbs the physics" (vs 7.47493e6 at 0.001) | EXACT |
| 937 | `0.01` m is unstable | | RGI:991-994 "**h_anuga 0.01**: truth forward clean (2-6 its) but the calibration forward … hits **DIVERGED_FUNCTION_NANORINF** at Newton it 33 on step 1 -- bigger is NOT safer" | EXACT |
| 950 | zero nonlinear failures | free outflow | RGI:1040-1041 "completes **600/600 steps, zero failures**" | EXACT |
| 953 | a class-twin calibration under [the loosened tolerance] recovered all **fifteen** class values to machine precision | | RGI:995-999 "at **snes_rtol 1e-3** the full 20 its COMPLETE: J 1.45342e7 -> 6.42e-7, **EXACT class recovery (rel L2 vs prior 0.0000, max class rel err 0.0000)**" (15 NLCD classes) | EXACT |

## Table 4 `tab:cure` (lines 964–967)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 964 | `600`-step control, critical outflow, Newton pinned from step **498** | | CW:257-266 "died at forward step **498** … base reproduces o21B exactly at **498 solves / 1 failure**"; RGI:1042-1044 "still dies at solve **499** (DIVERGED_MAX_IT 50 after 5 solves grinding at 25 its)" | EXACT |
| 965 | `600`-step control, free outflow: zero failures; `598/600` solves in two iterations | | RGI:1040-1042 "completes **600/600 steps, zero failures, 598 of 600 solves at 2 Newton its** (1x3, 1x5)"; CW:396 "completes 600/600 with **598 solves at 2 Newton its**" | EXACT |
| 966 | `1`-h calibration window (forward + adjoint + line search): clean end to end | | RGI:1065-1070 "1-hr rain-forced NLCD classes twin, free-outflow … completed with **17,198 converged solves and ZERO failures**: the 3600-step truth forward, the adjoint sweeps, and a TAO iteration's line-search forwards all clean"; CW:398 | EXACT |
| 967 | `72`-h forward through the Harvey crest (`2.93M` cells), free outflow: `259,200` solves, zero failures | | RGI:1144-1148 "**259,200/259,200** implicit solves converged, **ZERO failures**, 2h05m wall"; `plans/fable-brief-2026-08-25.md:37`; MS:11 | EXACT |
| 971 | "tolerance relief bought steps only linearly" | caption | CW:279-285 "the pin level tracks the tolerance … **chasing it with tolerance is a treadmill**" (o9d 2.7e-5 @ rtol 1e-5; o18 1.17e-4 @ 1e-4; 1.05e-3 @ 1e-3) | EXACT (qualitative; "linearly" is the paper's gloss on the tolerance-tracking ladder) |

## §4 "Checkpointed adjoints for long windows" (lines 978–995)

| line | number as printed | context | source | status |
|---|---|---|---|---|
| 978 | full `36`-hour Harvey window | | RMD:577 "**36 h window projection (518,400 steps)**" | EXACT |
| 979 | CFL-limited `Δt = 0.25` s | | RGI:1207 "vs the CFL's **0.25 s**" | EXACT |
| 979 | `518,400` steps | | RMD:577 "(**518,400 steps**)". Cross-check 36 × 3600 / 0.25 = 518,400 | EXACT + DERIVED |
| 980 | `2.93M` cells | | RGI:568 | EXACT |
| 980 | would occupy **tens of petabytes** | | RMD:580-581 "Naive trajectory would be **~36 PB**" | EXACT |
| 985-987 | reproduces the disk-trajectory calibration and gradient gates **bit-for-bit** for explicit RK, backward Euler, and ARK-IMEX | dam-break twin | RMD:556-562 "Validation (dam break, **all bit-identical to disk trajectory**): memory default + revolve (5/40 cps): calibration 1.986e-01, FD gates 1.0e-8 / 2.9e-6 (RK) … **BEULER 9.75e-9 / 4.18e-6 and ARKIMEX 2.77e-8 / 5.17e-6 with tight inner tolerances -- identical digits disk vs revolve**" | EXACT |
| 989 | only **three** states checkpointed in memory | 30 m mesh | RMD:572 "**revolve (max_cps_ram 3): IDENTICAL to all printed digits**" | EXACT |
| 990 | matches the disk-trajectory gradient to all printed digits | | same line | EXACT |
| 990 | and runs **faster** | | RMD:572-573 "wall **951.9 s** (revolve recompute is CHEAPER than disk I/O)" vs disk "wall **1013.6 s**" (RMD:569) | EXACT |
| 992 | a few hundred checkpoints | | RMD:578-579 "**c=200 cps**" | EXACT |
| 992 | `~14` GB aggregate | | RMD:578-579 "c=200 cps (**14 GB aggregate RAM**)" | EXACT |
| 993 | bound the recompute overhead by a factor of about **four** | | RMD:579-580 "gives revolve recompute **factor <= ~4x** (C(203,3) = 1.37M >= 518k)" | EXACT |

---

## Summary — NOT FOUND

1. **Line 385, `1.6×10⁻⁸`** (assembled matrix, uniform flow + reflecting BCs). No source states this value. RMD:33 gives only the aggregate "1.5-1.9e-8 on 3 state/BC combos"; the only per-config number for a uniform flowing lake is RMD:26's **1.4e-8**, which is the *frozen-dissipation* assembly (increment 2b), not the exact one the table reports. Grepped `1.6e-8`, `1.6e-08`, `1\.6[0-9]*e-0*8` across `plans/`, `logs/`, `papers/`, `src/`, `driver/` — no hits.
2. **Line 771, "resolutions two orders of magnitude apart."** The two sourced resolutions are 1 km and 30 m = **33×** (≈1.5 orders). The planar verification mesh's cell size is not recorded anywhere I could find (`planar_dam_10x5.msh` has no stated dx; the only size fact is "the 50-cell free-outflow twin", RGI:1345).
3. **Line 345, "six tangent evaluations per edge."** Consistent with 3 seeds × 2 sides, but no source states the count.

## Summary — CONTRADICTED

1. **Lines 738–743, the first-step bound list attributed to the absolute prior.** The seven classes named (pasture 0.038, developed-open 0.040, developed-low 0.090 up; developed-high 0.160, developed-medium 0.120, shrub 0.115, woody wetland 0.098 down) are, exactly and only, the **σ_α = 0.30 fractional** run's first step — RGI:2442-2444 and `logs/o62/o62_p_c15_sa0.30_it2.txt`. The *absolute*-prior first step on record (o52, σ_n = 0.015) pins a different, smaller set: **23, 24, 90 at α = 0.300 and 22 at α = 2.620** (RGI:1789-1794), and under the production armijo line search only **23 and 90** (RGI:1839, "near alpha 3: none"). As written, the same measurement is cited in consecutive paragraphs as evidence for two opposed priors, and the "ordering by prior value, not by hydrology" reading is not supported by any absolute-prior iterate.
2. **Line 719, β "spanning six orders of magnitude."** The recorded β values are 1e-6, 1e-5, 1e-4, 1, 1e2, 11.1, 4444 — a spread of **~9.6 orders** (or 8 orders over the hand-chosen subset). "Six" understates the record.
3. **Line 888, `Δt ≲ 2 h^{4/3}/(g n²|v|)`.** The source formula carries **no factor 2**: RGI:1206 "`dt < 1/tb`, `tb = g n² h^{-4/3}|v|`". The paper's own 2.4 ms (line 898) equals 1/tb at tb = 420/s, so the printed 2 is inconsistent with the number it is used to derive (which would be 4.8 ms).
4. **Line 861, "Three configurations share it."** The same paragraph names four (class twin, dense twin, mark twin, production) and `tab:configs` lists five Turning rows.
5. **Line 786, "planar, single region."** `adjoint_dam_break.yaml` declares two grid regions (`upstream`, `downstream`), and the per-region calibration gate (`calibrate_manning_twin_np_1`, RMD:29) recovers a **two-zone** truth (0.03, 0.06) from it. The FD-gate runs use a uniform field, but the mesh/config is not single-region.

## Borderline attributions worth a second look (not counted above)

- **Line 683, `‖g‖ ≈ 1.5×10³` "on the production window."** Source 1554 (RGI:1697) is measured on the **600-step o48 smoke**, not the 43,200-step production window. The value is right; the window label is not.
- **Line 432, "about 25% more wall time."** Arithmetic gives 26% against rung A (the campaign setting) but 43% against rung B (the fastest rung). Only the rung-A reading supports the printed number.
- **Line 658, "about nine minutes."** RGI:800 says ~9.4 min; the paper's own per-step decomposition gives 8.0 min. Both round to "about nine", but the two do not agree with each other.
