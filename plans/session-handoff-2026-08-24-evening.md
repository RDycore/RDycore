# Session Handoff — August 24, 2026 (evening; supersedes the afternoon handoff)

Branch `adams/gpu-implicit`, all work pushed through **`4802e69f`**.
Laptop tree clean, full ctest suite green except the 7 documented
pre-existing failures (6 cgns + amr_np3). PM tree pulled to the same
commit. **Three jobs queued on PM, none had started as of ~14:05 PT** —
collect them first (section 3).

**Read first:** this file, then `plans/RESULTS-gpu-implicit.md` (the new
entries are o29f/o32, the mid-event window IC, and o33), then the paper's
two temporary sections. `plans/next-phase-strategy.md` step 5 (the real
calibrated run) is still what remains — but its two machinery blockers
are now gone.

---

## 1. What changed today (afternoon → evening)

### The mid-event window IC is solved, and it needed almost no machinery

The afternoon handoff feared a `checkpoint HDF5 → natural-order .dat`
conversion path. There isn't one to write:

- `-restart <file>` is already an `RDySetup` option, and
  `ReadCheckpointFile` overwrites `u_global` **after** the yaml IC; the
  adjoint driver stashes `u_ic` straight from `u_global` right after
  `RDySetup`. So a checkpoint **is** the window's IC.
- Checkpoints are written in **natural order**, so the window need not run
  on the rank count of the forward that produced it (verified np1→np2).
- The driver's `DMSetUseNatural(PETSC_FALSE)` does **not** block checkpoint
  write/read: the natural-SF entry points key off `dm->sfNatural`, which
  survives the flag.

The one real gap was the **rain clock** — every forward restarts the TS
clock at 0, so a mid-event window replayed rain from the event's first
hour. Two new driver options close it:

- **`-adjoint_rain_start_hour <h0>`** — aligns the window's rain to event
  hour h0. The raster dataset advances only one file per call past its
  current one, so skipped hours are streamed through and only hours ≥ h0
  are kept. Correct for absolute-time (homogeneous) datasets too.
- **`-adjoint_forward_only`** — exactly one forward, then exit. Without it
  the checkpoint on disk is from whichever *extra* forward ran last
  (perturbed IC, optimizer trial), not the window's answer. Also covers
  the forward-only protocols that lost `-tao_max_it 0` in the 2026-08-20
  PETSc.

Gated two ways:
- **ctests** `adjoint_restart_*` (5 tests, ~2 s, rain-forced Houston 1 km):
  hour 2 of a continuous run vs the same hour restarted from the 1-hour
  checkpoint is **bitwise identical** at matched decomposition, 2.4e-10
  max relative across np1→np2, and off by 2.3e-2 relative L2 if the rain
  offset is dropped — so the gate discriminates.
- **o33 on the production config** (30 m, 2.93M cells, GPU kokkos +
  baijkokkos, real MRMS rain, beuler dt 1, n4): the restarted window
  reproduces the continuous run's `|u|_2` **and both depth extremes to
  every printed digit** (7.8231705568e+02); the no-offset control is off
  by 7% and 40× in min depth.

### Class calibration is now resumable across queue slots

A 12-hour window needs ~15–20 TAO iterations at roughly 1.5 node-hr each —
more wall time than one job gets. Classes mode always started from a
uniform `n0` and only **printed** the result, so a job that ran out of
time lost every iteration.

- **`-adjoint_classes_dump <file>`** writes `<NLCD code> <n>` per line.
- **`-adjoint_classes_init <file>`** warm-starts TAO from it. Matched by
  NLCD code, not position; a file missing any class is an error.

Measured: the warm run's TAO iteration 0 reports the previous run's final
objective to 6 digits (1466.42). BLMVM's quasi-Newton history does not
carry over, and on the Houston twin that is **not** a penalty — chained
2+2 reaches J 115.0 where a single cold 4-iteration run reaches 277.2.
One twin, so treat that as an observation, not a rule. Gated by two
chained ctests (`adjoint_classes_*`).

### o29f/o32 — the explicit path, and a paper correction

o29f came back **negative**, and it corrected a claim in the paper.

| run | integrator | dt | result |
| --- | --- | --- | --- |
| o29g | beuler | 1.0 s | J 3.429e6, MAE 1.6479 m, 22/108 dry |
| o29f | euler | 0.25 s | J inf, MAE inf, 108/108 dry |
| o32 | euler | 0.125 s | J inf, 108/108 dry |
| o32 | euler | 0.05 s | J inf, 108/108 dry |

The configs differ in **exactly two lines** (integrator, step size). This
re-confirms a limit the project already measured and `swe_petsc.c` already
documents: with the Δt-**free** drag the adjoint requires, forward Euler is
bounded by the friction rate, `dt < 1/tb`, `tb = g n² h^{-4/3}|v|` — about
**2.4 ms** on the spun-up 30 m state — not by the 0.25 s CFL. (I ran the
ladder before grepping for the known result; the code comment predicted
all three outcomes. What the runs add is the end-to-end confirmation.)

What matters:
1. **The o29 wall-clock comparison is void as written.** "Explicit 130 s vs
   beuler 221 s, so implicit costs ~1.7×" timed a run that produces a
   non-finite state. Against explicit stepping of the *same differentiable
   RHS*, implicit wins by ~400× in step count.
2. **The paper is fixed.** It called dt = 1 s "4× the explicit CFL limit"
   and inferred a modest factor. Both the numerics item (W4) and red-team
   item Q6 now state the measurement and the distinction; Q6 records that
   its own proposed comparison was run and its premise did not hold.
3. The genuinely modest comparison — explicit at 0.25 s with the
   operator-split (`semi_implicit`) friction — is against a
   **non-differentiable** discretization (it buries dt in the RHS, which is
   exactly why the adjoint cannot use it). Quote it only as such.

## 2. IN FLIGHT on PM — collect these first

Queue was congested (a `series_*`/`ladder` sweep of Mark's was ahead of
everything). None of these had started at ~14:05 PT. Poll with
`squeue -u madams`, then the logs in `$SCRATCH/gpu-implicit`.

- **57542255 (o34, regular q, ~2 hr)** — *the one that matters most.*
  Repeats the o31 72-hr eval-only forward with **hourly checkpoints**
  (72 files, ~70 MB each, landing in `checkpoints_o34/`) plus the per-mark
  dump `o34_marks.txt`. This decouples window placement from IC
  production: once the crest histogram picks a window, **no further
  forward is needed**, and several placements can be tried for free
  (paper Q7). Its `hwm eval` line is also a regression check that the new
  build reproduces o31's uncalibrated MAE **3.41 m** — both new options
  are default-off, so it should match exactly. Runs from
  `build-claude-gpu2`.
- **57540988 (o31d)** — the original per-mark dump run, old binary,
  `build-claude-gpu`. o34 supersedes it; left queued deliberately as an
  independent cross-check of the new build. If queue pressure matters,
  this is the one to cancel.
- **57540987 (o30)** — noisy (0.15 m) HWM identifiability with
  failure-tolerant trials; compare per-class table
  (`grep -A18 "NLCD  prior_n" o30.log`) against o28's clean run
  (area-weighted 0.20, MAE 0.017). Fills the paper's o30 `\todo` in the
  noise paragraph and red-team item 7.

**Analysis to run on `o34_marks.txt` (laptop, scp it over):** signed bias
+ quantiles (is the 3.41 m MAE a uniform high bias — the gauge work
suggests model-high — or scattered?), and the **peak-step histogram** →
that places the 12-hr window. Update the paper's baseline paragraph with
the bias decomposition.

## 3. The calibrated run — staged and ready to fire

`o35_calibrate_job.sh` is written and staged on PM:

```
sbatch o35_calibrate_job.sh <START_HOUR> <WINDOW_HOURS> <TAO_ITS> [INIT_FILE]
```

It takes the IC from `checkpoints_o34/o34.rdycore.r.<START_HOUR*3600>.bin`,
sets `-adjoint_rain_start_hour START_HOUR`, uses the REAL mark table,
sigma 0.15 m (USGS HWM quality codes), beta 1e-4, and **always dumps** its
parameter vector so the next job resumes with `INIT_FILE`. Deliverable:
calibrated MAE vs the 3.41 m baseline vs Inunda's calibrated 0.67 m.

**Before the first o35 run:** build `build-claude-gpu3` on PM (the script
points at it) — the calibration needs the `classes_dump/init` commit, and
a fresh build dir avoids relinking a tree a queued job is about to launch
from (that trap cost an o29e run). `sed s/build-claude-gpu/build-claude-gpu3/
cmake-claude-gpu.sh` is how gpu2 was made.

Open choices for o35, all worth a scientist's opinion first:
- **beta.** o28 found small-area classes overshoot to the bounds; consider
  raising beta or tightening bounds rather than accepting bound-pinned
  classes.
- **Window length.** 12 hr is the plan; the cost scales with it, and the
  chain makes longer windows survivable but not cheap.
- **Warm-start bias (not yet written into the paper).** The window's IC
  comes from a forward run with the **prior** roughness, so calibration
  cannot correct state error inherited at the window start — it only
  controls dynamics inside the window. This is the usual background-state
  caveat in a 4D-Var-style setup, and it belongs beside the calibrated
  number when it is reported.

## 4. Traps (carried forward, plus new ones)

New today:
- **Don't rebuild a PM build dir a queued job is about to launch from.**
  Build into a fresh directory instead (`build-claude-gpu2`, and
  `gpu3` next). Reading `git pull` in the source tree is safe — the
  already-linked binary is untouched.
- **The local `clang-format` disagrees with the checked-in formatting** of
  `driver/adjoint_test.c` — running it rewrites 324 lines. Hand-format new
  hunks; do not run clang-format on the whole file.
- **Reconfiguring `build-claude` inherits the shell's `PETSC_ARCH`** (the
  project `settings.json` sets `arch-macosx-gnu-kokkos-g`, which is not
  this build's). Prefix cmake/ctest with
  `PETSC_DIR=$HOME/Codes/petsc-claude PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O`.
- **Nested heredocs over ssh break** and can execute locally instead —
  write job scripts to a file and `scp` them.
- Checkpoint files are named by *step*, zero-padded to the digit count of
  `stop_n`: o34's hourly files are `o34.rdycore.r.%06d.bin`.

Carried forward, still true: `-adjoint_hwm_twin` OVERWRITES its table (the
real one is `turning30m_hwm_obs.txt`); `numerics.jacobian: fd` is refused
with device matrix types; arkimex at 30 m is ≥27× slower than beuler dt 1;
sshproxy certs die at ~24 h; the parameter-FD probe step is 1e-3 relative;
laptop `adjoint_beuler` + `-dm_mat_type aijkokkos` SEGVs pre-existing;
`latexmk` must run from `papers/manning-calibration/`.

## 5. Environment

- Laptop: `PETSC_DIR=$HOME/Codes/petsc-claude`,
  `PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O`, build `build-claude`
  (ninja). Tests: `ctest` from `build-claude` with those vars set.
- PM: repo `~/Codes/rdycore-manning` (same branch, pulled to `4802e69f`),
  builds `build-claude-gpu` (old, used by o30/o31d) and
  `build-claude-gpu2` (current, used by o32/o33/o34); run dir
  `$SCRATCH/gpu-implicit`.
- Budget: roughly 16–17 GPU-node-hr of Mark's ≤100 spent (o32 ≈ 0.1,
  o33 ≈ 0.2, o34 ≈ 2.1 when it runs); the calibration chain (~20–30) still
  fits.
