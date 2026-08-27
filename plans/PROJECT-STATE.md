# RDycore adjoint calibration — project state

**Goal: a complete draft to circulate to experts for feedback** — not a
submission. Completeness and legibility therefore rank above closing every
scientific gap; a caveat clearly stated is fine, a hole in a table is not.
**Node-hours are not the binding constraint** (the 300 figure is soft), so
prioritise by what makes the argument visible, not by what is cheap.

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
roughness accounts for **15% of a 0.72 m peak-elevation discrepancy**;
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
| 15-class calibration, 2 iterations | 0.6242 | −0.095 | one class on a bound |
| **15-class calibration, 9 iterations** (J 6.573e2) | **0.6116** | **−0.107** | one class on a bound |
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

## 3a. The experiment

The calibration experiment is a **12-hour window over the upstream crest
cluster**, event hours 29–41, run as a restart from a 72-hour forward:

- o37 ran the full 72 hours rain-forced from 2017-08-26 18:00 (259,200
  steps at Δt = 1 s) at converged inner tolerances with zero Newton
  failures, writing 73 hourly checkpoints. That run is the uncalibrated
  baseline and the source of every initial condition.
- Calibrations restart from `checkpoints_o37/o37.rdycore.r.104400.bin`
  — event hour 29 — and integrate 43,200 steps to hour 41, with the
  rainfall clock re-aligned by `-adjoint_rain_start_hour 29`. The
  restart path reproduces a continuous run bit-exactly.
- The 46 marks are those whose modelled crest falls inside the window.

**This is the right window for the question, not a compromise.** The
observable is a *peak*, so a mark only carries information if its crest
happens inside the window; crests here are bimodal (h29–42 and h60–72)
with an empty gap between, so a window covering both would spend most of
its length integrating marks that have already crested. The 46 in
cluster A are the population where peak WSE means what it is supposed to
mean, and they sit in the reach the model drains correctly — the
downstream 37 are unusable at any window length. Every result in this
document is internally consistent on that window, including the
uncalibrated baseline, the scans, the calibration, and the spectrum.

For scale, a full-event calibration would run ~8 hours per TAO iteration
at n16 and a converged one ~160 node-hours. That is a fact about what a
longer experiment would cost, not the reason this one was chosen.

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

**~130 node-hours spent.** The 300 figure is a soft target, not a hard
limit, so these are for planning rather than rationing. Each row *adds*
to the running total.

**Priorities, for a draft that experts can react to.** In order:

1. **Figures.** The paper has 12 tables and one figure, and that figure is
   from the old twin work — every headline result is currently a table.
   A reader skimming for the argument sees no picture of it. Three would
   carry the paper, and every number for them already exists:
   - **the authority figure**: MAE against relative parameter change,
     with the roughness α-scan and the IC scan *on the same axes*. This
     single plot is the thesis — the IC line is 4× steeper and does not
     turn over.
   - **the spectrum**: eigenvalues on a log axis with the λ = 1 line, one
     above it and a cliff below; optionally the per-class "learned" bars
     beside it.
   - **the ladder**: uncalibrated → uniform → calibrated → IC, as a
     single bar or dot plot against the 0.72 m error budget.
   No GPU time. This is the highest-value work remaining.
2. **o59 Part A**, which fills the one actual hole — the converged
   15-class MAE, currently written as *pending* in Section 3.
3. **A read-through.** The paper has been edited in pieces across a long
   session; nobody has read it end to end since.
4. Everything below, which is science rather than presentation and can
   be reported as caveats in a feedback draft.

**Cross-validation is deliberately NOT next.** It is required before
submission — the reported MAE is in-sample and §7.3 says so — but it
costs days of wall-clock, and whether readers consider that caveat
disqualifying is exactly the kind of thing to *ask* rather than guess.
Circulate the draft with the caveat stated, and run it knowing whether
it matters.

**The paper's headline calibration is o59** — three parameters over the
12-hour window, chained until converged. That is the row to protect.

| item | adds | running total | status |
| --- | --- | --- | --- |
| **o59: the production calibration**, 3 parameters, chained to convergence | **24–48** | 154–178 | **running** (`57649525`, link 1 of 1–2) |
| cross-validation within cluster A, 2 folds × 3 parameters | 40–70 | ~220–250 | **required before submission**, not before circulating — see below |
| production-window spectrum (43,200 steps) | 16 | ~235–265 | **required** — puts identifiability on the calibration's window |
| ε-robustness of the spectrum (a second ε) | 3 | | cheap insurance |
| σ_α rerun | 24–48 | | only if the domain team answers; otherwise future work |

**Calibration rows are ranges because chaining is the norm here, not the
exception**, and the plan should not pretend a slot equals a run. The
15-parameter calibration used a full 12-hour slot (48 node-hours) and
was still cut off at iteration 9 with J flattening and the gradient
down 8.5×. Three free parameters should converge in fewer iterations,
but nothing has measured that yet, so o59 is budgeted for a second
6-hour link. Cross-validation is two more such calibrations and inherits
the same uncertainty: at the observed ~80 min per TAO iteration a
five-iteration fold is 33 node-hours, a three-iteration fold ~22.
**o59 is the measurement that settles both rows**, and until it lands
every calibration estimate here is a range rather than a number.

Already spent on calibration for comparison: the 15-parameter run
(`57627058`) cost 48 node-hours for 9 iterations and is *not* wasted —
it is the control the spectrum interprets, and §7.4 uses it to show what
fitting fifteen parameters to a one-parameter observable does.

### What the production-window spectrum adds

§7.4's spectrum is measured on a 7,200-step pilot; the calibration runs
on 43,200. Re-running it at the production window costs 16 forwards
(~4 hours on 4 nodes, no adjoint) and does two things. It puts the
paper's identifiability number on the same window as its calibration
number, which it should be. And it measures how λ scales with window
length — the mark-count scaling in Section 3 holds sensitivity fixed,
whereas a longer window raises the sensitivity *per* mark because each
peak integrates more trajectory. That converts "how much would a longer
experiment buy?" from speculation into a measured quantity, which is
the same move this project makes everywhere else.

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

## 6a. Resuming: operational state as of 2026-08-27 04:30 PDT

**Tree clean, everything pushed** through `cda43e77`. 23 ctests pass on
the laptop (`build-claude`, `PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O`).
PM repo at the same commit; use **`build-claude-gpu8`** — it has every
option below. Do not rebuild gpu7 or gpu8 while a job may launch from them.

**In flight: `57649525` (o59), RUNNING** since ~05:23 PDT 2026-08-27.
6-hour slot, 4 nodes. Two parts:

- **Part A is DONE and in the paper.** The converged 15-class field
  (`o48_p_57627058.txt`) scores **J 6.153969e2, peak-WSE MAE 0.6116 m**,
  0 of 46 marks dry. Better than the 2-iteration 0.6242, so the feared
  J-vs-MAE divergence did *not* blow up. Roughness now accounts for 15%
  of the 0.72 m discrepancy and beats uniform α = 0.70 by 0.078 m. Full
  entry, the free J_total = misfit + prior verification, and the
  per-class α table are in `RESULTS-gpu-implicit.md` under "o59 Part A".
- **Part B is DONE** — Perlmutter died mid-run at ~11:01 PDT but TAO had
  flattened and written its dump; all o59 outputs are mirrored to
  `logs/o59/` in this repo. **Gate passed** (all twelve frozen classes
  bit-exact at prior). **Falsification verdict: the spectrum called it**
  — three parameters, converged in 4 iterations, reached J 682.1 and
  removed 83.6% of the converged 15-parameter reduction; classes 22 and
  23 reproduce their 15-class values to three digits. The 24.8-unit gap
  decomposes exactly (12 frozen classes carry ~5% of J in aggregate).
  Full entry in `RESULTS-gpu-implicit.md` under "o59 Part B". Not yet in
  the paper — §7.4 gains one paragraph when the numbers are reviewed.
  Missing and cheap when PM returns: eval-only MAE for the 3-param field.

**Monitoring does not survive a session.** The tracking job was a
session-only cron; re-establish it or poll by hand:

    ssh -o BatchMode=yes -i ~/.ssh/nersc madams@perlmutter-p1.nersc.gov \
      'cd /pscratch/sd/m/madams/gpu-implicit && squeue -u madams -h; \
       grep -h "hwm eval" o59_score15.log; \
       grep -hE "TAO,|class recovery" o59_57649525.log | tail -5'

**If ssh returns nothing at all, the NERSC cert has expired** — it dies
at ~24 h and fails silently with exit 0 and no output. An empty result
means expired, not "nothing new". This cost two monitoring cycles on
2026-08-26.

**Next actions, in order.**

1. ~~Read o59 Part A, put the MAE into §7.3 and the ladder in Section 3.~~
   **Done 2026-08-27.** The frozen-class gate for Part B has already
   passed its first half — the log reports `3 of 15 classes active, 12
   frozen at prior`; the `rel_err` = 0.000 check still needs doing when
   Part B finishes.
2. Read o59 Part B against the falsification test; adjust §7.4 if it
   fails. If Part B is unconverged at the wall, chain a second link:
   `sbatch o59_spectral_active_set.sh 12 23,90,22 o59_p_57649525.txt`
3. Cross-validation within cluster A. Needs a split of
   `turning30m_hwm_obs_clusterA.txt` into two folds, then one
   3-parameter calibration per fold and an eval-only score on the
   held-out half. This is the run that makes the reported MAE
   out-of-sample, which is the paper's weakest remaining claim.
4. Production-window spectrum: `WINDOW=43200 bash o58_gauss_newton_spectrum.sh`
   then `o58_gauss_newton.py 0.05 <dumps> --sigma-obs 0.15`. Outputs are
   tagged by ε and window so they cannot clobber the 7,200-step pilot.

**Things that are easy to get wrong**, all learned the hard way:

- `o58_gauss_newton.py` **requires** `--sigma-obs`; every eigenvalue
  scales as 1/σ², and 0.15 vs the driver default 0.01 is a factor of 225
  — enough to turn 7 supported directions into 0.
- Read the analysis's **self-checks before its numbers**: the argmax
  count (how many marks moved their peak time) and, for the o56-style
  full Hessian, that an unconstrained direction returns λ = 1. The second
  caught a real error that would otherwise have reached the paper.
- Peak dumps and class tables are matched by NLCD code, not position.
- `-adjoint_hwm_twin` **overwrites its table**; `-adjoint_hwm_fd` and
  `-adjoint_hwm_eval_only` are read-only.

## 7. Where the leverage is, if this continues

Our own measurements point away from roughness. The initial condition
and the forcing carry more of this residual; the drainage defect
dominates a third of the domain; and the adjoint machinery applies
unchanged to both. A 4D-Var over u₀ is the natural next capability, and
its hard parts are design rather than implementation — background
covariance, positivity of *h*, wet/dry front motion under an IC control,
and component scaling across *h* and *hu*. The authority scan
(`-adjoint_ic_scale`) was the cheap part and is already done.
