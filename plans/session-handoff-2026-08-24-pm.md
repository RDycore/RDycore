# Session Handoff — August 24, 2026 (afternoon; supersedes the morning handoff)

Branch `adams/gpu-implicit`, all work pushed through **`783c3fa1`**.
Laptop and Perlmutter trees are at the same commit; **three jobs are
queued/running on PM** (section 3 — collect them first). Working tree
clean (the rebuilt paper PDF may show modified; commit or discard at
will).

**Read first:** this file, then `plans/RESULTS-gpu-implicit.md`
(entries o26–o31 are today's), then the paper's two temporary sections
(red-team review + ranked open questions) — they ARE the current work
plan in priority order. `plans/next-phase-strategy.md` remains the
strategy record; its step 1 (BC) and step 2–3 (HWM data + misfit) are
DONE, step 4 (identifiability) is measured (o28/o30), step 5 (the real
calibrated run) is what remains.

---

## 1. What landed today (all committed, all gated)

- **Free-outflow BC** (`type: free-outflow`, transmissive ghost qR=qL)
  across host/device RHS + Jacobians + CEED, FD-gated (twin pair
  1.8e-8; driver dJ/du0+dJ/dn 2e-8..8e-7 across integrators × backends
  × np). Acceptance: the o22 pin is CURED — 600/600 (o26), 1-hr
  rain-forced calibration clean (o27, 17,198 solves 0 fail), 72-hr
  forward clean (o31, 259,200 solves 0 fail). The snes_rtol 1e-3
  crutch is no longer needed on free-outflow runs.
- **HWM pipeline**: USGS STN archive QC'd (2,364 → 324 in-domain →
  108 usable; 62% below-bed trap measured), `data/harvey_hwm/`;
  peak-WSE misfit mode in rdycore_adjoint (`-adjoint_hwm_file`,
  argmax-time adjoint injection, FD-gated 9e-7, ctests np1/2);
  `-adjoint_hwm_twin` (NOTE: overwrites its table — always point it
  at a COPY), `-adjoint_hwm_eval_only` (+ `-adjoint_hwm_dump`),
  `-adjoint_obs_noise` (deterministic, np-independent).
- **o28 identifiability**: 108 real mark cells × 1-hr window identify
  the NLCD classes covering **83% of the domain** (area-weighted rel
  L2 0.20); small-area classes move but overshoot to bounds. Reverses
  the o25i gauges conclusion — the observable was the problem.
- **o31 — FIRST REAL-DATA NUMBER**: uncalibrated NLCD-prior model,
  72-hr crest-covering window: **peak-WSE MAE 3.41 m vs the 108 real
  marks, all marks wet**. The "before" number vs Inunda's calibrated
  0.67 m.
- **Noise findings**: planar twins have ~5 µm parameter signal → 0.1 mm
  noise destroys per-cell, 1 mm destroys per-region recovery; noise
  robustness ≡ window-length/SNR question. Paper carries this.
- **Robustness fix**: a trial field that breaks Newton no longer
  aborts calibration (TSSetErrorIfStepFails off; J=1e30 rejection;
  CheckTruthForward guards on data-producing forwards). Found by o30's
  crash at a noisy trial point.
- **Paper** (`papers/manning-calibration/`, 19 pp, builds clean):
  red/tiger-team section (10 ranked weaknesses, statuses tracked) +
  open questions RANKED by importance with owners (science / Matt+Jed
  numerics / Jeff engineering); novelty claim corrected (MALI does
  steady-adjoint basal-friction inversion — Perego 2014, Hoffman 2018
  cited; ours narrowed to first TRANSIENT discrete adjoint in an E3SM
  component); measured identifiability + HWM observable + real-data
  baseline + noise paragraph + production-tolerance disclosure all
  written in; Harvey section retitled honestly.
- **o29 wall-clocks at 30 m** (1-hr forward, n4): beuler dt1 = 221 s;
  explicit euler dt0.25 = 130 s; **arkimex dt0.25 ≥ 27× slower than
  beuler dt1** (did not finish in 100 min — 1-km-era arkimex numbers
  do not transfer; needs tuning before quoting).

## 2. The two blockers of the morning handoff: both RESOLVED

A (critical-outflow pin): fixed by the free-outflow BC, documented,
opt-in, scientists asked to veto/confirm (paper Q4). B (observable):
HWMs adopted, machinery complete, identifiability measured, real
baseline in hand (paper Q1).

## 3. IN FLIGHT on PM — collect these first (jobs submitted ~13:05 PT)

All in `$SCRATCH/gpu-implicit`; this session's wakeup timers die with
it, so POLL these yourself (`squeue -u madams`, then the `*_slurm.out`
files):

- **57540985 (o29f, debug q)**: explicit euler dt0.25 eval fingerprint
  vs the real mark table → `o29f.log` "hwm eval" line. If FINITE and
  close to o29g's (J 3.43e6, MAE 1.6479, 22 dry), the explicit path
  survives the 30 m CFL step → then write the paper's W4 update
  (implicit-vs-explicit ~1.7× wall measured, arkimex caveat) into
  Section sec:implicit / red-team item 4.
- **57540987 (o30 rerun, regular q)**: noisy (0.15 m, sigma 0.15) HWM
  identifiability with failure-tolerant trials → compare vs o28
  (clean: area-weighted 0.20, MAE 0.017): per-class table via
  `grep -A18 "NLCD  prior_n" o30.log`. Fills the paper's o30 \todo in
  the noise paragraph (Section sec:calibration) and red-team item 7.
- **57540988 (o31d, regular q, ~2 hr)**: o31 rerun with
  `-adjoint_hwm_dump o31_marks.txt` → per-mark weight, h_obs, model
  peak h, PEAK STEP. Analysis to run (laptop, scp the file):
  signed bias + quantiles (is the 3.41 m MAE a uniform high bias — the
  gauge work suggests model-high — or scattered?), and the peak-step
  histogram → **place the 12-hr calibration window over the crest
  times**. Update the paper's baseline paragraph with the bias
  decomposition.

## 4. The next real step: the calibrated real-data run

Everything is staged for it once o31d places the window:

1. **Peak-time IC**: the calibration window will not start at the only
   IC (Aug 26 18:00). Options: (a) RDycore checkpoint machinery
   (`checkpoint.c`, yaml `checkpoint:` section) — run the o31 forward
   once with a checkpoint written at the window start, then restart
   from it; (b) drive the calibration window inside a longer forward
   (expensive). Investigate (a) first — the driver reads its IC via
   `flow_conditions` binary file, so a checkpoint→IC-file conversion
   path may be needed (RDyWriteOneDOFGlobalVecToBinaryFile /
   checkpoint HDF5 → natural-order .dat).
2. **The run**: classes mode + `-adjoint_hwm_file turning30m_hwm_obs.txt`
   (REAL data — NO `-adjoint_hwm_twin`!), 12-hr window over the crest,
   ~15–20 TAO its ≈ 20–30 node-hr (o27 cost basis), sigma from HWM
   quality (~0.15 m), beta to taste (o28 suggests small classes need
   MORE regularization — consider raising beta or bounds tightening).
   Deliverable: calibrated MAE vs 3.41 m baseline vs Inunda 0.67 m —
   the paper's headline.
3. **Wednesday meeting (Aug 26)**: `plans/campaign-wednesday.md` has a
   day-before addendum; the paper's ranked questions are the agenda.

## 5. Traps learned today — do not repeat

- The rebuilt PETSc (2026-08-20) **rejects `-tao_max_it 0`** — use
  `-tao_max_it 1` or eval-only mode for forward-only protocols.
- **`-adjoint_hwm_twin` OVERWRITES its table file** with synthetic
  values — always `cp` the real table first. `turning30m_hwm_obs.txt`
  (PM + repo) is the pristine real data; eval-only mode never writes.
- `numerics.jacobian: fd` is refused with device matrix types (our own
  guard) — use `analytic` even for forward-only runs.
- **arkimex at 30 m is ≥27× slower than beuler dt1** — do not budget
  ARK runs from the 1-km numbers.
- sshproxy certs die at ~24 h and WILL expire mid-session — check
  `ssh -o BatchMode=yes` before relying on PM, and stage work so an
  expiry blocks nothing local.
- Don't rebuild the PM tree while a job is about to launch from it
  (execve "Permission denied" on the half-relinked binary — cost one
  o29e run).
- The parameter-FD gate probe step is 1e-3 RELATIVE (measured V-curve
  optimum); the u0-check default 1e-6 drowns in the forward's noise
  floor.
- Laptop pre-existing: `adjoint_beuler` + `-dm_mat_type aijkokkos`
  SEGVs even on the reflecting baseline (never reaches new code);
  6 cgns ctests + amr_np3 also pre-existing failures.
- `latexmk` must run FROM `papers/manning-calibration/`; and `git add
  papers/...` from repo root (relative paths double up otherwise).
  Don't `git add` the whole paper dir — `.fdb_latexmk`/`.fls` are
  gitignored now, but aux/bbl churn is noise.

## 6. Environment

- Laptop: `PETSC_DIR=$HOME/Codes/petsc-claude`,
  `PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O`, build dir
  `build-claude` (ninja). Unit tests: `ctest` from `build-claude`.
- PM: repo `~/Codes/rdycore-manning` (same branch), build
  `build-claude-gpu` via `bash cmake-claude-gpu.sh`; run dir
  `$SCRATCH/gpu-implicit` (all o*/b* logs + protocol scripts + the
  o29fg_job.sh / o30_job.sh / o31d_job.sh templates).
- Budget: ~15 GPU-node-hr spent today of Mark's ≤100; the calibrated
  run (~20–30) fits comfortably.
