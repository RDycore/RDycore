# RDycore adjoint calibration — project state

**Authoritative document.** Last substantive update 2026-08-27. This
supersedes `PROJECT-STATE-2026-08-26.md` and both
`session-handoff-2026-08-26*.md` files, which are kept for history and
should not be read for current status — the older PROJECT-STATE's
Section 4 in particular is now false.

Layers, so it is clear what to read for what:

| file | role |
| --- | --- |
| **this file** | what is true now, and what to do next |
| `RESULTS-gpu-implicit.md` | the run log — every experiment, its numbers, and its caveats |
| `campaigns/` | the scripts behind every cited result, and a README of the conventions |
| `papers/manning-calibration/` | the argument, 30 pp, builds clean |

---

## 1. What this project built

A transient discrete adjoint for RDycore — to our knowledge the first in
an E3SM component — obtained by differentiating the production solver in
place rather than reimplementing the model in a differentiable-programming
framework. An exact assembled Jacobian of the finite-volume SWE
right-hand side (closed-form Roe-flux blocks by hand-written forward-mode
differentiation, entropy-fix branches included) drives adjoint
sensitivities through PETSc `TSAdjoint`, bound-constrained calibration
through TAO, and fully implicit stepping — one artifact serving all
three. GPU-resident throughout; device and host bitwise identical.
23 adjoint/calibration ctests gate it.

## 2. The scientific result, in one paragraph

Surveyed peak water-surface elevations at survey accuracy carry about
**two degrees of freedom of information** about distributed Manning
roughness in this basin — one strongly. That is measured, not inferred:
the prior-preconditioned Gauss-Newton spectrum is
λ = 3.03, 0.64, 0.33, 0.32, 0.21, … , one eigenvalue above unity with a
gap of 4.8. Calibrated as well as we can currently calibrate it,
roughness accounts for **13% of a 0.72 m peak-elevation discrepancy**;
a uniform scale within defensible values accounts for 4%. The same
measurement pointed at the initial condition finds **four times the
authority** from a more defensible perturbation, and the residual is
therefore dominated by the water balance rather than the friction.
Along the way the machinery isolated a downstream reach the model
cannot drain.

## 3. The numbers, current

**Peak-WSE MAE against the 46 cluster-A marks** (12-hour window, event
hours 29–41, IC from the o37 hour-29 checkpoint):

| field | MAE | vs NLCD | defensible? |
| --- | --- | --- | --- |
| NLCD lookup (uncalibrated) | 0.7188 | — | yes, by construction |
| uniform α = 0.70 (the prior-consistent optimum) | 0.6894 | −0.029 | yes |
| uniform α = 0.30 (unregularized floor) | 0.6392 | −0.080 | no — n to 0.008 |
| **15-class calibration, 2 iterations** | **0.6242** | **−0.095** | one class on a bound |
| 15-class calibration, 9 iterations (J 6.573e2) | *pending o59 Part A* | | |
| **IC scaled 0.6** (antecedent water −40%) | **0.5818** | **−0.137** | plausible; no turnover found |

All in-sample. Cross-validation is the outstanding gap (Section 6).

**Identifiability** (o58, Gauss-Newton spectrum from 16 forwards):
one supported parameter; 2.02 degrees of freedom for signal; the marks
reduce prior uncertainty by 36% on developed-medium, 12–17% on four
more, and ≤1% on seven. More marks do not rescue it — all 71 admissible
marks still support one parameter, every QC-passed mark supports two,
and five would need every Harvey mark in the domain including the 62%
rejected as channel surveys. **The limitation is the observing system.**

**A finding the scans could not have produced:** the leading mode
overlaps the equal-per-*class* uniform direction the α-scan varies by
0.263 — random is 0.258 — and the per-*cell* area-weighted version by
0.768. The α-scan probes the informative direction obliquely, which is
why class redistribution beat uniform scaling. It is not that fifteen
parameters beat one; the scan's one was the wrong one.

**The model defect:** 37 of 108 marks sit in a downstream reach that
fills monotonically for 72 hours with zero recession, ending +7.1 m
above every surveyed value. The bias is a smooth gradient across the
domain (+0.30 / +2.85 / +7.06 m by band), so the middle band is
contaminated too and cannot serve as held-out data.

## 4. Why the calibration had never worked, and the fix

A scaling defect, not the adjoint — the FD gates always passed. BLMVM's
first trial is `x − g`, and with Manning *n* as the variable ‖g‖ = 1554
where n ≈ 0.05 put every trial on the lower bound, where the solver
provably fails. `-adjoint_classes_relative` optimizes α = n/n_prior with
the objective scaled by 1/J(start); both O(1), so the unit step is a
~10% roughness change and the bounds α ∈ [0.3, 3] exclude the divergent
region. `o48_smoke.sh` reproduces the original zero-step failure in five
minutes on full production wiring — any future "the optimizer will not
move" question costs five minutes, not twelve hours.

## 5. Practical facts that cost time to learn

- **Run production on 4 nodes / 16 ranks.** Measured on the production
  window: n4 36m21s, n8 21m55s, n16 15m01s, all reproducing J and MAE to
  every printed digit. n16 is 2.42× at 61% efficiency, which is worth
  paying because queue wait, not node-hours, is the binding constraint.
  (n64 is 2.8× *slower* than n4 — it does not scale to 64, but it scales
  usefully to 16.)
- **~80 min per TAO iteration** at n16 on the 12-hour window, line-search
  trials included. A 6-hour slot schedules far faster than a 12-hour one.
- **`-tao_ls_type armijo`.** More-Thuente expands past −g and overshoots:
  it bought 15.8 units of misfit for 50.5 units of prior violation and
  finished at a *higher* objective than the shorter step.
- **The prior is mis-specified as an absolute width.** σ_n = 0.015 over a
  table spanning n = 0.027–0.16 asserts ±56% for barren and ±9% for
  developed-high; use `-adjoint_sigma_alpha` (fractional) instead. The
  tension survives the fix, which is the point — see Section 6, question 1.
- **Never rebuild a build directory a queued job will launch from.**
  `build-claude-gpu2..8` exist for this reason; gpu8 is current.
- **The NERSC cert dies at ~24 h and fails *silently*** — ssh returns
  exit 0 with no output. An empty result means expired, not "nothing new".
- `-adjoint_hwm_twin` overwrites its table; `-adjoint_hwm_fd` and
  `-adjoint_hwm_eval_only` are read-only.
- Write job scripts to a file and `scp`; nested heredocs over ssh can
  execute locally.

## 6. What remains, and what it costs

**Budget: ~130 of 300 node-hours spent.** Each row below *adds* to the
running total; a defensible paper needs the first two.

| item | adds | running total | status |
| --- | --- | --- | --- |
| o59: score the converged field, then calibrate the 3 supported classes | 24 | 154 | **running** (`57649525`) |
| cross-validation within cluster A, 2 folds × 3 parameters | 40–70 | ~220 | **required** — fixes the in-sample caveat |
| ε-robustness of the spectrum (a second ε) | 3 | ~223 | cheap insurance |
| production-window spectrum (43,200 steps) | 18 | ~241 | upgrade — §7.6 is a 2-hour-window pilot |
| σ_α rerun | 24 | ~265 | only if the domain team answers; otherwise future work |

The cross-validation range is wide because it depends on how fast three
parameters converge, which nothing has measured yet. At the observed rate
(~80 min per TAO iteration plus a ~50 min start-point evaluation) a
five-iteration fold is 33 node-hours and two folds is 66; three
iterations would make it ~45. **o59 is the measurement that settles it.**

**Five questions for the domain team**, stated in full at the end of the
paper draft. The first is load-bearing:

1. **How far can the NLCD lookup be wrong, in percent?** Holding the
   calibration physical needs ±15%; allow ±30% and the data drives
   developed-medium urban to n = 0.036, smoother than concrete. If ±30%
   is the honest width, the headline becomes that this observable cannot
   support a defensible distributed roughness field at all — a stronger
   negative claim, worth making deliberately.
2. Is 13% of peak-WSE error being roughness-attributable expected here?
3. Does the downstream drainage failure match known mesh/outlet
   behaviour, and who owns it?
4. Is a 20% error in antecedent water storage at hour 29 plausible? The
   IC scan says the water balance carries more of this residual than
   friction does.
5. Is the calibrated field acceptable? It sits 9.2σ from the lookup and
   moved seven classes the marks cannot see.

## 7. Where the leverage is, if this continues

Our own measurements point away from roughness. The initial condition
and the forcing carry more of this residual; the drainage defect
dominates a third of the domain; and the adjoint machinery applies
unchanged to both. A 4D-Var over u₀ is the natural next capability, and
its hard parts are design rather than implementation — background
covariance, positivity of *h*, wet/dry front motion under an IC control,
and component scaling across *h* and *hu*. The authority scan
(`-adjoint_ic_scale`) was the cheap part and is already done.
