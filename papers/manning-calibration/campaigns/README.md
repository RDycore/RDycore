# Campaign job scripts

Batch, interactive, and post-processing scripts for the Perlmutter runs
whose numbers the paper (`../manning-calibration.tex`) cites; corresponding
output logs and dumps are archived in `../logs/`.

They are not runnable from this directory directly: the batch scripts do
`cd $SCRATCH/gpu-implicit` and expect the staged inputs listed below.

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
| `o58_gauss_newton_spectrum.sh` + `o58_gauss_newton.py` | prior-preconditioned Gauss-Newton Hessian spectrum and posterior widths from observation sensitivities (`../logs/o61/`) | ~40 min, 4 nodes |
| `o59_spectral_active_set.sh` | scores the converged 15-class field, then calibrates only the classes the spectrum says the data determines (`../logs/o59/`) | 12-hr slot, 4 nodes |
| `o60_ic_extend.sh` | extends the initial-condition authority scan down to `a = 0.6, 0.7` (`../logs/o57/`, `../logs/o60/`) | ~50 min, 4 nodes |
| `o61_spectrum_production.sh` | production 16-forward per-mark peak sensitivity campaign for the Gauss-Newton spectrum (`../logs/o61/`) | ~45 min, 4 nodes |
| `o62_sigma_alpha.sh` | fractional-prior (`sigma_alpha = 0.30, 0.50`) 3-class and 15-class calibrations (`../logs/o62/`) | 12-hr slot, 4 nodes |
| `o63_gauge_validation.sh` | cross-observable validation of the 3-class HWM-calibrated field against stream gauges (`../logs/o63/`) | ~2.5 hr, 4 nodes |
| `o64_gauge_score.sh` | forward gauge scoring and gradient at the prior, 3-class, and 15-class fields (`../logs/o64/`) | ~75 min, 4 nodes |
| `o65_gauge_spectrum.sh` + `o65_gauge_spectrum.py` | 16-forward gauge-series sensitivity campaign, per-gauge residual decomposition (Table 16), and gauge vs. HWM Gauss-Newton spectrum comparison (`../logs/o65/`) | ~40 min, 2 nodes |
| `o66_gauge_spectrum_central.sh` | negative-step (`-eps`) gauge sensitivity columns for central-difference nonlinearity verification (`../logs/o65/`) | ~40 min, 2 nodes |
| `stepA_perimeter_flux.py` | boundary/outlet flux integration and 72-hr mass-balance audit (`../logs/stepA/`) | post-processing |

## Staged inputs these expect in `$SCRATCH/gpu-implicit`

- `checkpoints_o37/o37.rdycore.r.104400.bin` --- hour-29 restart from the
  converged-tolerance 72-hour forward
- `turning30m_hwm_obs_clusterA.txt` --- the 46 marks cresting inside the
  window, from `../data/harvey_hwm/filter_hwm_by_crest.py`
- `turning30m_class.bin`, `turning30m_manning.bin` --- NLCD class map and
  the Manning prior derived from it
- `o31_72hr_freeflow.yaml` --- the 72-hour configuration the window YAML
  is `sed`-derived from
- `scale_manning.py`, `scale_manning_classes.py` --- from `../data/nlcd/`
- a build with `-adjoint_classes_relative`; for `o54`/`o60` one with
  `-adjoint_ic_scale`. Each script checks for the option it needs and
  refuses to run against a stale binary.
- `o43_p_nlcd.txt` --- the published NLCD lookup as a class table, which
  is both the calibration's start point and `o58`/`o61`'s base point

## Conventions worth keeping

- **Never rebuild a build directory a queued job will launch from.** Make
  a fresh one; `build-claude-gpu2..6` exist for exactly this reason.
- **Write job scripts to a file and `scp` them.** Nested heredocs over
  ssh break and can execute locally.
- Calibration and eval-only runs print nothing while stepping. Use
  `sstat -j <id> --format=JobID,AveCPU` to confirm liveness, or watch the
  parameter dump, which is rewritten every TAO iteration.
