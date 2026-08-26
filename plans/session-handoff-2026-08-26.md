# Session Handoff — August 26, 2026

Branch `adams/gpu-implicit`, everything committed and pushed through
**`ae465398`**. Laptop tree clean; 19 adjoint/calibration ctests pass
(6 cgns + amr_np3 fail, pre-existing). PM tree at the same commit.
**No jobs running or queued.**

**Read first:** `plans/PROJECT-STATE-2026-08-26.md` — the standing
top-level document (what was built, what it measured, what remains,
next-study options). This file is the operational layer: what happened
in the last session, what is staged, and exactly how to resume.

---

## 1. The headline, in one paragraph

The machinery is finished and verified. The Harvey science came back a
**measured negative**: the peak-WSE high-water-mark observable at 30 m
constrains approximately **one** roughness degree of freedom, worth
**~10% of the model–survey discrepancy** (~3 cm within physically
defensible roughness). Four independent measurements agree — a
perfect-data twin, a noisy twin, a five-point uniform-scale scan, and a
half-domain interaction test — and the scan is optimizer-independent, so
the conclusion does not rest on any calibration run. Along the way the
same machinery isolated a **model drainage defect** in the downstream
Buffalo Bayou reach, which accounts for most of the remaining error and
is arguably where the real leverage is.

## 2. What landed in this session (all committed)

- **o37** — 72-hr baseline redone at converged tolerances
  (ksp 1e-4 / snes 1e-5): MAE 3.4092 m, 259,200 solves, zero failures,
  73 hourly checkpoints. Mark-for-mark identical to the loose-tolerance
  run (max peak diff 0.0124 m, all 108 argmax steps unchanged), which
  closed the inner-tolerance concern.
- **Drainage analysis** — the 37 marks that never crest are a
  structural defect, not late crests: every one peaks at hour 72 with
  zero recession, still filling at 0.02–0.06 m/hr while the other 71
  drain; mean 10.7 m of standing water arriving laterally after the
  rain ends. Spatially the three crest-timing groups are three bands
  with monotone downstream bias (+0.30 / +2.85 / +7.06 m).
- **o39–o42 + kink-probe instrumentation** — the basin-scale FD-gate
  failure is **closed**: the adjoint is exact, and the long-window gate
  measures the objective's dense wet/dry branch structure. Four
  confirmations: zero observable-level flips, eps-convergence per
  direction (a single-class direction passes the 1e-5 gate outright),
  per-class additivity summing to the domain-wide gap, and accumulation
  with trajectory length (9e-5 at 60 steps -> 1e-1 at 1800).
- **o44/o45 alpha scan** — the uniform roughness knob is worth 0.08 m
  of a 0.72 m error; alpha 0.2 diverges (numerical floor, not drying).
- **o47 half-domain scans** — each half alone makes the fit *worse*
  (developed +0.0118, remainder +0.0080 MAE), both together
  −0.0574: a 45x positive interaction. Peaks are set by
  path-integrated conveyance; only domain-coherent changes drain them.
- **o43 (first real-data calibration): 0 TAO iterations, diagnosed.**
  See section 3 — this is the one hole left.
- **Driver additions**, all gated: `-adjoint_rain_start_hour`,
  `-adjoint_forward_only`, `-adjoint_classes_dump/init` (chainable
  calibration), `-adjoint_classes_active` (freeze classes at prior),
  kink-probe instrumentation, FD lines now print direction index and
  Jp/Jm/eps.
- **Paper** (`papers/manning-calibration/`, 19 pp, builds clean) —
  restructured per Mark's review: S5 is now the 4D-Var algorithm alone,
  the nondifferentiability catalog is S6.1 with per-entry status, the
  red-team section is nine current-state items, open questions became
  "positions", numbers moved from prose into tables (`tab:cure`,
  `tab:fdwindow`, `tab:baseline`, `tab:snr`), the abstract is halved and
  in plain words, and "branch" is glossed where the Roe flux first
  appears.

## 3. THE ONE HOLE — a calibration that runs to completion

`o43` did **zero** TAO iterations: two trial points diverged, the line
search failed, nothing moved. Fully diagnosed, two stacked causes:

1. **No interior minimum at beta = 1e2.** The Tikhonov term balances
   the misfit gradient only at a relative roughness change of 13.6 —
   outside the bounds. The alpha scan shows the same thing directly:
   J falls monotonically toward lower roughness until the solver
   diverges. The optimizer was handed an effectively unbounded-below
   problem.
2. **First quasi-Newton step mis-scaled by ~5 orders of magnitude.** At
   the NLCD start the prior gradient is exactly zero, so BLMVM's first
   trial is `n − g` with ‖g‖ = 1554 — projects onto the lower bound
   everywhere, where the model provably cannot solve (alpha 0.2
   diverged). Fixing beta alone does NOT fix this.

### Resume recipe (est. 1 day)

- **Set regularization on principle.** beta is dimensional and was set
  ad hoc everywhere (1e-6 to 1e2 across the project — the spread is the
  symptom). Use `beta = 1/sigma_n^2` with sigma_n a prior uncertainty on
  Manning n. **sigma_n ≈ 0.015 -> beta ≈ 4400**, which agrees with the
  independent gradient-balance estimate (5400–6800 for a 20–25%
  roughness change). Report sigma_n, never a bare beta.
- **Do not pay 2 hr per line-search trial.** An objective+gradient is
  ~2 hr but a forward alone is ~36 min. Add a gradient dump to the
  classes block (mirror `-adjoint_classes_dump`), then line-search
  explicitly along −g with `-adjoint_hwm_eval_only` forwards at steps
  sized so max|Δn| ≈ 0.01–0.04. Robust, uses the adjoint, 36 min/trial.
- **Calibrate 1–3 parameters, not 15.** `-adjoint_classes_active
  <codes>` is implemented and verified (frozen classes bit-identical
  across 1 vs 6 iterations). The data supports ~one mode; this is the
  structural fix for bound-pinning.
- **Expected outcome, which is the point:** convergence to roughly the
  alpha-curve level (MAE ~0.64–0.68 on the 46 cluster-A marks). That
  turns "the optimizer never worked" into "the calibration converges to
  exactly where the sensitivity analysis predicted."

## 4. Staged and ready on PM (`$SCRATCH/gpu-implicit`)

| item | what it is |
| --- | --- |
| `checkpoints_o37/` | 73 hourly ICs (converged tolerances) for any window start |
| `turning30m_hwm_obs_clusterA.txt` | 46 marks cresting in h29–41 (from `data/harvey_hwm/filter_hwm_by_crest.py`) |
| `o43_calibrateA_job.sh` | chainable calibration, `<TAO_ITS> [INIT_FILE]`; **needs the beta + scaling fixes** |
| `o43_p_nlcd.txt` | the NLCD per-class prior = the calibration's start point |
| `o44_alphascan.sh`, `o45_alphascan_low.sh` | uniform-scale scans (eval-only, 36 min/point) |
| `o47_wherefrom.sh` | half-domain scans |
| `scale_manning.py`, `scale_manning_classes.py` | field scalers (in `data/nlcd/`) |
| `o37_drainage.py` | h(t) at mark cells from the checkpoint series |
| `o46_alphascan_rest.sh` | alpha 0.8 / 1.2, never ran — optional, completes the curve |

Builds: PM `build-claude-gpu4` (current); laptop `build-claude`.

## 5. Decisions taken (so they are not relitigated)

- **Calibration target is the 71 admissible marks**, or the 46
  cluster-A marks for a 12-hr window — not all 108. The 37 downstream
  marks sit in a reach the model cannot drain; scoring against them
  folds a model defect into the roughness field.
- **Window: 12 hr over crest cluster A (h29–41).** Crests are bimodal
  (h29–42, h60–72) with an empty gap, so no 12-hr window covers both;
  going to 24 hr buys 3 more marks for double the cost.
- **Cluster B is NOT clean held-out data** (+2.85 m bias — the same
  defect at intermediate strength). Out-of-sample testing should be
  cross-validation *within* cluster A.
- **Production inner tolerances: ksp 1e-4 / snes 1e-5.** Loose settings
  were a false economy — tightening 1e-2 -> 1e-3 is *faster* (fewer
  Newton iterations). The old "1e-3 passes the FD gates" claim is
  withdrawn.
- **Never compare our MAE to Inunda's 0.67 m** without stating that the
  mark sets differ; ours is a favourable upstream subset.

## 6. Open questions genuinely needing the science team

1. **Is ~10% of peak-WSE error being roughness-attributable expected**
   for a 30 m model of this event? If yes, that is the paper. If not,
   the forcing or DEM deserves a look first.
2. **Does the drainage defect match known behaviour** of this mesh /
   outlet configuration, and who owns fixing it?
3. **sigma_n** — how far can the NLCD lookup plausibly be wrong? This
   now sets the regularization directly.
4. **Next study shape** — see PROJECT-STATE section 6: calibrate what
   the data supports (1–3 params); change the observable (flood
   extent); change the target parameter (conveyance, where our own
   evidence points); change the scale (upstream sub-basin, or finer);
   or longer windows.

## 7. Traps (each of these cost time)

- `-adjoint_hwm_twin` **overwrites its table** with synthetic values —
  `turning30m_hwm_obs.txt` is the pristine real data. `-adjoint_hwm_fd`
  and `-adjoint_hwm_eval_only` are read-only.
- **Never rebuild a build dir a queued job will launch from.** Make a
  fresh one (`build-claude-gpu2/3/4` exist for this reason).
- Nested heredocs over ssh break and can execute **locally** — write
  job scripts to a file and `scp`.
- Do not run `clang-format` on `driver/adjoint_test.c` (rewrites ~324
  lines); hand-format new hunks.
- cmake/ctest need `PETSC_DIR`/`PETSC_ARCH` exported or they inherit the
  wrong arch from `settings.json`.
- `git add papers/...` from the repo root, not from inside the paper dir.
- sshproxy certs die at ~24 h; a dead cert looks like `exit=255` with no
  message.
- An orphaned `salloc` keeps charging — `scancel` it.
- Eval-only and calibration runs print **nothing** during stepping;
  use `sstat -j <id> --format=JobID,AveCPU` to confirm liveness.

## 8. Cost basis (measured, for planning)

- 12-hr window (43,200 steps), 2.93M cells, n4, converged tolerances:
  **forward ~36 min**, forward+adjoint **~2 hr**.
- 72-hr forward: ~3.4 hr. Hourly checkpoints: 4.8 GB.
- Budget: **~35 of 300 GPU-node-hr** used.
