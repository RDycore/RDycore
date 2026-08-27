> **SUPERSEDED.** Read `plans/PROJECT-STATE.md` for current status.
> This file is kept for history. Claims here that are now known to be
> wrong are listed at the top of that document.

# RDycore adjoint calibration — project state, findings, and where to go next

Written 2026-08-26 for the team (and for future-us). Branch
`adams/gpu-implicit`, everything committed and pushed. This supersedes
the earlier session handoffs as the top-level document; run-by-run
detail lives in `plans/RESULTS-gpu-implicit.md`, and the paper draft in
`papers/manning-calibration/`.

---

## 1. What this project built

A transient discrete adjoint for RDycore — to our knowledge the first
in an E3SM component — obtained by differentiating the production
solver in place rather than reimplementing the model in a
differentiable-programming framework. Concretely:

- **Exact assembled Jacobian** of the finite-volume SWE right-hand
  side: closed-form Roe-flux blocks by hand-written forward-mode
  differentiation (entropy-fix branches included), plus source and
  boundary-condition blocks. FD-gated in `ctest` on every commit.
- **Adjoint sensitivities** `dJ/du0` and `dJ/dn` via PETSc `TSAdjoint`,
  one gradient per extra forward solve, independent of parameter count.
- **Fully implicit (backward Euler) stepping and its adjoint**, plus an
  ARK-IMEX splitting; revolve-checkpointed trajectories for long windows.
- **GPU-resident gradient pipeline** (kokkos vecs, blocked-COO BAIJ
  assembly), device and host bitwise-identical.
- **Calibration driver** (`driver/adjoint_test.c`): TAO/BLMVM over
  per-cell, per-region, or land-cover-class parameterizations; gauge-WSE
  and peak-WSE (high-water-mark) observables; deterministic observation
  noise; mid-event window restart; chainable across queue slots.

All of this is verified, gated, and reusable. It is the durable output
of the project regardless of what the Harvey science concludes.

## 2. The scientific question, and the honest answer

**Question.** Can spatially distributed Manning roughness be recovered
by calibrating against Hurricane Harvey observations, at 30 m over the
2.93M-cell Turning domain?

**Answer, as measured: essentially no — and we can now say precisely
how much roughness could ever have explained.**

Four independent measurements agree:

1. **Identifiability twin (o28).** With *perfect* synthetic
   observations at the 108 real mark locations, only the area-dominant
   classes are recovered (83% of the domain's cells, 0.7–10.5% error);
   small-area classes overshoot to bounds. With the real 13-gauge
   geometry instead, the objective falls 128x while class values stay
   66% wrong — three forest classes never move at all.
2. **Noise twin (o30).** At survey-grade noise (sigma = 0.15 m), the
   optimizer reaches the noise floor (MAE 0.117 m, vs the ~0.12 m even
   the true field could achieve) while ending 82% wrong in the
   parameters, with 5 of 15 classes pinned to bounds. It bought misfit
   with unphysical parameters.
3. **Uniform-scale scan (o44/o45).** Scaling all roughness by alpha on
   the production configuration: MAE 0.7188 (NLCD) -> 0.6614 (a=0.45)
   -> 0.6392 (a=0.3); a=0.2 diverges. **The entire uniform roughness
   knob is worth 0.08 m against a 0.72 m error (11%)** — and only at
   30% of published NLCD values (n as low as 0.008; smooth concrete is
   ~0.012). Within defensible roughness it is worth ~3 cm.
4. **Half-domain scans (o47).** Scaling only the developed classes
   (71.6% of cells) makes the fit *worse* (MAE 0.7306); scaling only
   the remainder also makes it worse (0.7268); scaling both together
   helps a lot (0.6614) — a 45x positive interaction. Peaks are set by
   *path-integrated* conveyance along flow paths crossing both groups,
   so only domain-coherent changes drain them.

**Synthesis.** The peak-WSE observable at these marks constrains
approximately **one roughness degree of freedom** — the domain-scale,
path-integrated conveyance — not fifteen. That single mode can explain
~10% of the model–survey discrepancy. The other ~85–90% is structural:
rainfall forcing, DEM, datum, 30 m representation error, and a genuine
model defect (below).

This unifies everything: o28 identifies only classes that move the
domain mean; o30 pins the rest because they carry no signal to fit;
o47 shows why the modes are not separable.

## 3. A model defect found along the way (independently valuable)

Evaluating the uncalibrated model over a 72-hour crest-covering window
against the 108 marks gives MAE 3.41 m — but it decomposes into three
spatial bands, sorted by crest timing, with monotonically increasing
bias downstream:

| group | n | mean lon | bias | MAE |
| --- | --- | --- | --- | --- |
| crest h29–41 (upstream) | 46 | −95.695 | +0.30 m | 0.72 |
| crest h48–72 (middle) | 25 | −95.568 | +2.85 m | 2.96 |
| never crest (downstream) | 37 | −95.440 | +7.06 m | 7.06 |

The 37 downstream marks **never crest**: every one peaks at the final
hour with exactly zero recession, still filling at 0.02–0.06 m/hr while
the upstream group is already draining, accumulating a mean 10.7 m of
standing water that arrives laterally *after the rain ends*. The model
cannot drain the downstream Buffalo Bayou reach. No roughness value
repairs a missing exit.

**This is a finding in its own right**, and it points where the leverage
actually is: conveyance/outlet representation, not friction.

## 4. What is NOT established — the hole to close

**We have never completed a successful calibration.** The first
real-data attempt (o43) did **zero** TAO iterations: two trial points
diverged, the line search failed, nothing moved. The cause is
diagnosed and quantitative:

- At beta = 1e2 the objective has **no interior minimum** — the
  Tikhonov term balances the misfit gradient only at a relative
  roughness change of 13.6, far outside the bounds. The alpha scan
  shows the same thing directly: J decreases monotonically toward lower
  roughness until the solver diverges. The optimizer was handed an
  effectively unbounded-below problem and headed for the floor.
- Separately, the **first quasi-Newton step is mis-scaled by ~5 orders
  of magnitude**: at the NLCD start the prior gradient is exactly zero,
  so BLMVM's first trial is n − g with ‖g‖ = 1554, which projects onto
  the lower bound everywhere — precisely where the model cannot solve.

Until a calibration runs to a sensible stopping point, a reader can ask
whether the negative result is a property of the problem or of an
optimizer we never got working. It is the former — the alpha scan is
optimizer-independent — but the argument is far stronger with a
converged run that lands where the sensitivity analysis predicts.

### Fixing it (est. 1 day)

1. **Regularization on principle.** beta is dimensional and was set ad
   hoc in every experiment (1e-6 to 1e2 across the project — the
   spread is the symptom). Use `J = (1/2 sigma^2) Σr² + (1/2 sigma_n^2)
   ‖n − n_prior‖²`, i.e. **beta = 1/sigma_n²** with sigma_n a prior
   uncertainty on n. sigma_n = 0.015–0.02 gives beta ≈ 2500–4400;
   the gradient-balance argument gives 5400–6800 for a 20–25% roughness
   change. **Use sigma_n ≈ 0.015 (beta ≈ 4400) and never quote a bare
   beta again.**
2. **Avoid the expensive line search.** Each objective+gradient is
   ~2 hr but a forward alone is ~36 min. Add a gradient dump, then run
   an explicit line search along −g with eval-only forwards at steps
   sized so max|Δn| ≈ 0.01–0.04. Robust, uses the adjoint, 36 min per
   trial. (Alternatively rescale the optimizer's objective/variables so
   the first unit step is sane.)
3. **Calibrate few parameters.** `-adjoint_classes_active <codes>`
   (already implemented and gated) freezes classes at their prior. The
   data supports ~1 mode; calibrating 1–3 parameters is the structural
   fix for the pinning, as opposed to suppressing 14 weak directions
   with regularization after the fact.

## 5. What can be written up now

**A methods + measurement paper**, which is what
`papers/manning-calibration/` already is (19 pp, builds clean). Its
claims, all supported:

- The infrastructure and its verification (Sections 2–6).
- Implicit stepping is *required* for a differentiable RHS, with the
  friction-stiffness analysis and the measured explicit divergence.
- The nondifferentiability catalog (5 of 6 entries resolved with gated
  fixes; the peak-WSE argmax remains formally open with a proposed
  remedy).
- **Identifiability is a property of the observable and geometry, not
  of observation counts** — the strongest science contribution.
- **How much of a model's error a parameter can explain**, measured at
  basin scale. This is the generalizable method: it applies to any
  parameter, observable, and model.
- The drainage defect, as a demonstration that the machinery finds
  model problems, not just parameter values.

What it cannot yet claim: a calibrated real-data number, or improved
predictive skill.

**Framing recommendation.** Lead with the capability and the
measurement method; report the Harvey application as a case study whose
answer is "this observable supports one roughness mode worth ~10% of
the error, and here is the model defect that dominates the rest."
That is an honest, useful, and unusual paper. Resist framing it as a
failed calibration — nothing failed except an unregularized optimizer
run we can fix.

## 6. Options for a next study shape (for the team to weigh in on)

Our own analysis suggests where to look:

1. **Calibrate what the data supports.** 1–3 parameters (global scale,
   or developed/non-developed), converged, with sigma_n stated. Modest
   positive result, closes the hole in Section 4, cheap (~1 day).
2. **Change the observable.** Peak WSE discards timing. Flood *extent*
   (satellite wet/dry) carries different information; gauge hydrographs
   carry timing but are unusable at 30 m because the channel is
   sub-grid. Extent may constrain different modes.
3. **Change the target parameter.** The evidence says conveyance and
   drainage dominate the residual. Calibrating or diagnosing channel
   conveyance / outlet representation is where the leverage is — and
   the adjoint machinery applies unchanged.
4. **Change the scale.** A sub-basin where the model drains correctly
   (the upstream cluster-A region), or higher resolution where channels
   are resolved. Smaller, cleaner, more likely to show recoverability.
5. **Longer windows.** Signal grows superlinearly with window length;
   12 hr was chosen for cost. A 24–48 hr window raises SNR, though the
   drainage defect contaminates the downstream half regardless.

## 7. Practical state

- **Budget:** ~35 of 300 GPU-node-hr used (Mark can extend).
- **Queued/available:** `o43_calibrateA_job.sh` (chainable calibration,
  needs the beta and scaling fixes above), `o46` alpha points 0.8/1.2,
  hourly checkpoints `checkpoints_o37/` (73 files) as ICs for any window
  start, `turning30m_hwm_obs_clusterA.txt` (46 crest-in-window marks).
- **Builds:** PM `~/Codes/rdycore-manning/build-claude-gpu4` (current);
  laptop `build-claude` (`PETSC_DIR=$HOME/Codes/petsc-claude`,
  `PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O`). 19 adjoint/calibration
  ctests pass; 6 cgns + amr_np3 fail pre-existing.
- **Traps** (each cost us time): `-adjoint_hwm_twin` overwrites its
  table; never rebuild a build dir a queued job will launch from;
  nested heredocs over ssh execute locally; don't run clang-format on
  `driver/adjoint_test.c`; cmake/ctest need PETSC_DIR/ARCH exported or
  they inherit the wrong arch.
