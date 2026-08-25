# Fable brief — sharpen the Harvey calibration state before the Aug 26 meeting

Written 2026-08-25 (evening) by the Opus session that produced o36–o39.
Branch `adams/gpu-implicit`, everything pushed. Read this, then
`plans/RESULTS-gpu-implicit.md` (entries o36, o37 are today's), then the
paper's §6.1 nondifferentiability catalog.

## Why Fable, and where it is NOT needed

**Warranted (P1):** the basin-scale gradient discrepancy. The objective is
`J = ½σ⁻² Σ_m w_m (max_k H_m u(t_k) − y_m)²` — a max over sampled times,
with wet/dry branch switches underneath, differentiated through a discrete
adjoint of backward Euler. At basin scale it disagrees with central
differences by 1.6e-1 / 6.5e-2 / 7.6e-2 depending on solver tolerance, and
**the disagreement does not shrink as the solves tighten**. Deciding
whether that is a kink artifact or a real gradient defect — and what it
implies for a calibration we are about to publish — is subtle multi-part
numerics where being wrong is expensive. That is the Fable case.

**Also worth Fable (P2/P3):** designing the discriminating experiments and
the calibration protocol so the result is defensible rather than merely
lower. Structure matters more than effort there.

**NOT worth Fable time — these are closed, do not redo them:**
- Inner tolerances. o36 ladder + o37 settled it; `ksp 1e-4 / snes 1e-5` is
  converged and is the production setting. The old loose setting changed
  the 72-hr answer by 1.1e-4 relative in J and moved zero of 108 peak steps.
- The mid-event-window restart path. Bitwise-gated in ctests and validated
  on the production GPU config (o33). It works; use it.
- Paper restructuring. Done and committed (S5 → 4D-Var algorithm, catalog →
  §6.1, open questions → positions).

---

## State: what is established

**The uncalibrated baseline (o37, converged tolerances, 259,200 solves,
0 failures).** Peak-WSE MAE vs the 108 real marks = **3.409 m**. Confirmed
against the earlier loose-tolerance run mark for mark: max per-mark peak
difference 0.0124 m, all 108 argmax steps identical.

**It is two populations, not one** (this is the day's most consequential
finding, and it is ~4 hours old — scrutinize it):

| population | n | MAE | bias | note |
| --- | --- | --- | --- | --- |
| crest inside the 72-h window | 71 | 1.506 | +1.198 | centroid lon −95.65 |
| never crest (still rising at h72) | 37 | **7.061** | **+7.061** | centroid lon −95.44 |
| all | 108 | 3.409 | +3.207 | |

For the censored 37, MAE == bias exactly: every one is model-high. They sit
at the eastern/downstream end (19 Buffalo Bayou, 11 coastal-classified),
i.e. the reach controlled by Addicks/Barker gate releases the model does
not represent — the same reason open issue 7 already excludes eight
reservoir-influenced *gauges*.

**Noise (o30).** At σ = 0.15 m HWM-grade noise on a 1-hr window, 15 classes
from 108 marks is not identifiable: J falls 9.4×, MAE 0.234 → 0.117 m, but
5 of 15 classes pin to bounds (pasture → the 0.30 ceiling, 689% error).
Consistent with the paper's SNR table: ~0.1 m signal vs 0.15 m noise.

**Gradient (o38, 1-hr window, classes mode, real marks, 3 probes):**

| ksp / snes | max rel FD error |
| --- | --- |
| 1e-2 / 1e-3 | 1.595e-01 |
| 1e-3 / 1e-3 | 6.483e-02 |
| 1e-4 / 1e-5 | 7.635e-02 |
| 1e-6 / 1e-8 | *(job hit its wall limit — not measured)* |

Gate is 1e-5. The adjoint agrees with FD in **sign and rough magnitude** in
every probed direction (e.g. 3.101e6 vs 3.109e6), so it is directionally
right, not garbage. 32 of 108 marks are dry at this parameter point
(uniform n = 0.03).

**This is new ground, not a regression.** Verified today: all gated ctests
pass (`adjoint_hwm_fd_np_{1,2}` at 8.9e-7); **no PM run before o38 ever
passed `-adjoint_hwm_fd`** (checked every job script and log in
`$SCRATCH/gpu-implicit`); and the twin that gates it is explicitly
"smooth, fully wet dynamics keep the discrete map differentiable" — built
to exclude exactly the kinks the basin has.

## In flight

- **o39** (job 57594544, queued behind Mark's `weakscale_L3`): probe-step
  sweep at the converged rung, eps = 1e-2, 1e-4, 1e-5 (o38C already gives
  1e-3 → 7.635e-02). Discriminator: a **kink** makes the FD error *fall* as
  eps shrinks (fewer marks flip) until the forward noise floor takes over;
  a **wrong gradient** leaves it flat in eps. Collect this first.

---

## P1 — Settle the gradient question (highest value)

### P1.1 Collect o39 and read it against the two hypotheses
Falling with eps → kink. Flat → suspect the gradient itself. Note the
confound: at eps below ~1e-4 the FD reference drowns in the forward's own
noise floor (the driver's default probe selection documents a measured
V-curve optimum at 1e-3), so expect a V, and the *left* branch is noise,
not signal. Do not read a rising left branch as evidence of a bug.

### P1.2 The decisive measurement (small code change, recommended)
Neither o38 nor o39 observes the hypothesized mechanism directly. Instrument
`FDCheckParamGradient` (`driver/adjoint_test.c`, ~line 1169) to report, for
each probe direction, over the ± evaluations:
- how many marks changed their **argmax step**, and by how much;
- how many marks changed **wet/dry state** (peak h crossing `tiny_h`);
- the FD-vs-adjoint error recomputed **over only the marks that did not
  flip** in either sense.

If the non-flipping subset satisfies the 1e-5 gate, the adjoint is correct
and the discrepancy is entirely the known nonsmoothness — that is a
publishable, precise statement and it closes catalog entry 6 with evidence
rather than conjecture. If the non-flipping subset still fails, there is a
real defect and the calibration results need re-examination before the
paper claims anything.
The peak step per mark is already tracked (`-adjoint_hwm_dump` writes it),
so the bookkeeping exists; it needs exposing inside the probe loop.

### P1.3 If it is a kink — what follows
The paper's §6.1 lists the argmax as the one OPEN entry, with a proposed
remedy (freeze the argmax across an outer iteration, re-evaluate between
outer iterations; a fixed point on peak times). Two things then need
judgment: whether to implement it before the production run (it changes the
algorithm), and whether BLMVM's line search straddling kinks explains
o30's bound-pinning under noise. Those interact — do not treat them
separately.

## P2 — Make the admissibility question decidable (for the scientists)

The 37 censored marks decide whether the headline is 3.41 m or 1.51 m. We
should walk in with the evidence, not the question alone.

**Cheap decisive check, data already on disk:** `checkpoints_o37/` holds 73
hourly natural-order binary checkpoints (4.8 GB, `o37.rdycore.r.%06d.bin`,
blocksize 3, cell order = natural). Extract `h(t)` at the 37 censored cells
and at the 71 others across all 73 hours and characterize:
- Are the censored cells **monotonically filling** (never draining) — a
  structural drainage failure — or merely late-cresting?
- Does their fill rate track the rain, or continue after rain stops?

Monotone filling through the whole event with no recession is strong
evidence the model cannot drain that reach, which makes those marks
inadmissible for roughness calibration on physical grounds rather than by
assertion. Cell ids: join `data/harvey_hwm/turning30m_hwm_obs.txt` (mark
order = line order after the count) to
`data/harvey_hwm/turning30m_hwm_cells.csv` (lon/lat/waterbody). Mark→
censored mapping: `o37_marks.txt`, `peak_step >= 259200-300`.

Vec layout note: the checkpoint is `[PetscBag][natural-order Vec]`; the Vec
payload is the trailing `3*ncells` big-endian float64 (there is a working
reader in the scratch analysis from today, or scan for classid 1211214).

## P3 — Calibration protocol that is defensible

Only after P1/P2. The open design points, all interacting:
- **Window.** No 12-hr window covers the crests: best 12 h (h29–41) gets
  43/71 genuine crests. Excluding the censored 37 changes this materially.
- **Marks whose peak is censored** contribute a residual at the window edge,
  not a peak — decide explicitly whether they are zero-weighted.
- **Regularization.** o30 says β = 1e-4 is too weak under realistic noise.
  Discrepancy-principle stopping is the principled option and is already
  named as future work in the paper.
- **Stopping.** A lower MAE with five classes on their bounds is a worse
  result, not a better one. Decide what is reported: MAE alone, or MAE with
  the roughness field shown to remain physical.
- The run is chainable across queue slots (`-adjoint_classes_dump` /
  `-adjoint_classes_init`, gated by ctests); `o35_calibrate_job.sh` on PM is
  staged and parameterized `<START_HOUR> <WINDOW_HOURS> <TAO_ITS> [INIT]`
  but points at `build-claude-gpu3`, which does not exist yet — build it
  before first use (see traps).

---

## Environment and traps

- **PM:** repo `~/Codes/rdycore-manning` (branch `adams/gpu-implicit`),
  binary `build-claude-gpu2/driver/rdycore_adjoint` (current), run dir
  `$SCRATCH/gpu-implicit`. Laptop: `build-claude`, and cmake/ctest need
  `PETSC_DIR=$HOME/Codes/petsc-claude
  PETSC_ARCH=arch-macosx-gnu-rdycore-kokkos-O` exported or they inherit the
  wrong arch from `settings.json`.
- **Never rebuild a build dir a queued job will launch from** — relinking
  under a launching job cost an o29e run. Make a fresh dir
  (`sed s/build-claude-gpu/build-claude-gpu3/ cmake-claude-gpu.sh`).
- **`-adjoint_hwm_twin` OVERWRITES its table** with synthetic values.
  `turning30m_hwm_obs.txt` is the pristine real data. `-adjoint_hwm_fd` and
  `-adjoint_hwm_eval_only` are read-only.
- Nested heredocs over ssh break and can execute locally — write job scripts
  to a file and `scp`.
- Do not run `clang-format` on `driver/adjoint_test.c`; the local version
  disagrees with the checked-in formatting and rewrites ~324 lines.
- `numerics.jacobian: fd` is refused with device matrix types.
- Pre-existing laptop failures (not yours): 6 cgns tests + `amr_np3`.

## One correction to carry forward

`plans/campaign-wednesday.md` and the paper's open issue 3 both state that
"the same configuration at rtol 1e-3 passes the 1e-5 gates". On the laptop
free-outflow twin it does not (6.4e-4, failing), and o38 shows nothing
passes at basin scale. Do not repeat the claim until P1 explains it.
