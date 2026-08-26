# Campaign job scripts

Batch and interactive scripts for the Perlmutter runs whose numbers the
paper (`papers/manning-calibration/`) and the run log
(`plans/RESULTS-gpu-implicit.md`) cite.

**Why these are in the repo at all.** Every earlier campaign (o1--o47)
lived only in `$SCRATCH/gpu-implicit` on Perlmutter, which NERSC purges.
That was tolerable while the numbers were exploratory. It stopped being
tolerable once the paper started citing them, so the scripts behind
cited results are kept here from o48 onward. This is not a claim that
the older scripts were unimportant --- `RESULTS-gpu-implicit.md` records
what they did and with which options --- only that the ones a reader
might need to re-run are now durable.

They are not runnable from this directory: each does `cd
$SCRATCH/gpu-implicit` and expects the staged inputs listed below. Copy
to that directory to use.

| script | what it does | cost |
| --- | --- | --- |
| `o48_smoke.sh` | reproduces the o43 zero-step failure and its fix on a 600-step window, on the full production wiring | ~10 min, 1 node |
| `o48_calibrate_rel.sh` | the production calibration: 15 classes, relative variables, `sigma_n` 0.015, chainable | 12-hr slot, 4 nodes |
| `o49_alpha_bar.sh` | eval-only forwards at alpha 0.7 and 0.8, filling the gap in the uniform-scale scan | ~72 min, 1 node |
| `o50_calibrate_active.sh` | the few-parameter calibration; takes the active NLCD codes as an argument | 12-hr slot, 4 nodes |
| `o51_scaling.sh` | forward at 4, 8 and 16 ranks on the production window | ~75 min, 4 nodes |
| `o52_linesearch.sh` | More-Thuente vs armijo, one iteration each, 7200-step window | ~2.5 hr, 1 node |
| `o53_production_gradient.sh` | the production start-point gradient at n16, recorded from an interactive run | ~2.5 hr, 4 nodes |
| `o54_ic_authority.sh` | MAE at a dumped calibration iterate, and the initial-condition authority scan | ~90 min, 4 nodes |
| `o55_continue_calibration.sh` | continues a calibration from a dumped iterate, with the armijo line search | 4-hr session, 4 nodes |
| `o56_hessian_spectrum.sh` + `o56_hessian.py` | the 15x15 class Hessian by columns, and its prior-preconditioned spectrum | ~100 min, 4 nodes |

## Staged inputs these expect in `$SCRATCH/gpu-implicit`

- `checkpoints_o37/o37.rdycore.r.104400.bin` --- hour-29 restart from the
  converged-tolerance 72-hour forward
- `turning30m_hwm_obs_clusterA.txt` --- the 46 marks cresting inside the
  window, from `data/harvey_hwm/filter_hwm_by_crest.py`
- `turning30m_class.bin`, `turning30m_manning.bin` --- NLCD class map and
  the Manning prior derived from it
- `o31_72hr_freeflow.yaml` --- the 72-hour configuration the window YAML
  is `sed`-derived from
- `scale_manning.py`, `scale_manning_classes.py` --- from `data/nlcd/`
- a build with `-adjoint_classes_relative` (i.e. at or after commit
  `48485e24`); for `o54` one with `-adjoint_ic_scale` (`fda6f09e`); for
  `o56` one with `-adjoint_classes_grad_only` (`e3c9ecde`). Each script
  checks for the option it needs and refuses to run against a stale
  binary.
- `o43_p_nlcd.txt` --- the published NLCD lookup as a class table, which
  is both the calibration's start point and `o56`'s base point

## Conventions worth keeping

- **Never rebuild a build directory a queued job will launch from.** Make
  a fresh one; `build-claude-gpu2..6` exist for exactly this reason.
- **Write job scripts to a file and `scp` them.** Nested heredocs over
  ssh break and can execute locally.
- Calibration and eval-only runs print nothing while stepping. Use
  `sstat -j <id> --format=JobID,AveCPU` to confirm liveness, or watch the
  parameter dump, which is rewritten every TAO iteration.
