# Artifact Description: Manning roughness calibration in RDycore

This is the Artifact Description (AD) for *Adjoint measurement and calibration
of Manning roughness in E3SM's river dynamical core*. It maps each figure and
table of the paper to the scripts, observation tables and archived output that
produced it.

The paper describes the tag `manning-calibration-arxiv-v1` of
[RDycore/RDycore](https://github.com/RDycore/RDycore), on the
`adams/gpu-implicit` branch. Check out the tag rather than the branch tip,
which keeps moving.

**All paths below are from the repository root.** The material lives where the
work was done rather than being gathered here, so nothing in this directory is
a copy that can drift from its source.

## What is where

| location | contents |
| --- | --- |
| `papers/manning-calibration/` | the manuscript (`manning-calibration.tex`, `.bib`), the TikZ figures (`fig_authority.tex`, `fig_spectrum.tex`), the Figure 1 panels (`houston_manning_*.png`) and `plot_manning_map.py`, which draws them |
| `data/harvey_hwm/` | USGS STN high-water marks: retrieval, cell mapping, QC (`QC_REPORT.md`), and the 108-mark observation table |
| `data/harvey_gauges/` | USGS NWIS stream gauges: cell mapping, clock cross-check, and the stage observation tables |
| `data/nlcd/` | NLCD 2021 land cover: fetch and sampling scripts, class summaries, and the 1 km test maps |
| `plans/campaigns/` | the Perlmutter job scripts and the Python analyses behind every campaign (`o48` to `o66`, `stepA`); see `plans/campaigns/README.md` for the staged inputs |
| `logs/` | archived output: parameter and peak dumps, scores, spectra (`o57` to `o65`, `stepA`), and the verification logs (`verification/`, `ksp_look_n4.log`) |
| `plans/RESULTS-gpu-implicit.md` | the Perlmutter campaign logbook (`o20` to `o66`) |
| `plans/RESULTS-manning-draft.md` | the local verification and 1 km twin logbook |

**Not archived here:** the inputs of the 30 m production runs -- the Turning
mesh, the hour-29 restart checkpoint, the 30 m NLCD class and Manning maps, the
72-hour configuration, and the MRMS rainfall forcing. `plans/campaigns/README.md`
lists what each campaign script expects to find staged. Rerunning a 30 m
campaign therefore needs those inputs and multi-GPU nodes; the verification
gates and everything under "Reproducing on a workstation" need neither.

## Building RDycore and running the verification gates (Table 1)

The analytic SWE Jacobian (`src/swe/swe_jacobian_petsc.c`,
`src/swe/swe_roe_flux_jacobian_petsc.h`), the Kokkos GPU assembly
(`src/swe/swe_jacobian_kokkos.kokkos.cxx`), implicit stepping (`BEULER`,
`ARKIMEX`) and the discrete-adjoint calibration driver (`driver/adjoint_test.c`,
which builds `rdycore_adjoint`) are in the main source tree.

```bash
cmake -S . -B build -DPETSC_DIR=$PETSC_DIR -DPETSC_ARCH=$PETSC_ARCH -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build -R "swe_jacobian|adjoint|calibrate" --output-on-failure
```

Every finite-difference gate in Table 1 (`tab:verification`) is one of these
`ctest` targets:

- `test_swe_jacobian`: the face-level 3x3 Roe flux Jacobian against central
  differences.
- `swe_jacobian_global_*`: the assembled right-hand-side Jacobian against
  finite-difference coloring, across interior, reflecting, Dirichlet and
  free-outflow boundaries.
- `adjoint_dam_break_analytic_np_1`, `adjoint_beuler_np_1`,
  `adjoint_arkimex_np_1`, `adjoint_beuler_freeflow_np_1`,
  `adjoint_arkimex_freeflow_np_1`: `TSAdjoint` gradients with respect to the
  initial condition and to Manning's n, against central differences.
- `adjoint_hwm_fd_np_1`, `adjoint_hwm_fd_np_2`: the peak water-surface
  (high-water-mark) gradient, on one and two ranks.
- `adjoint_classes_relative_fd_np_1`, `adjoint_classes_relprior_fd_np_1`,
  `adjoint_classes_active_relative_np_1`: relative land-cover-class scaling, the
  fractional prior, and the active-set projection.
- `adjoint_restart_ref_np_1`, `adjoint_restart_window_np_1`,
  `adjoint_restart_compare_np_1`, `adjoint_restart_compare_np_2`: bitwise and
  cross-decomposition checkpoint-restart consistency for a mid-event window.

## Observation data (`data/`)

Raw external downloads (the NLCD GeoTIFF, the raw STN JSON/CSV dump, the
HydroShare gauge archives) are not tracked; the scripts in `data/` fetch them.
Every processed table the paper and the driver read is tracked.

- **High-water marks**, `data/harvey_hwm/`: `fetch_usgs_hwm.py` retrieves event
  180 (Hurricane Harvey) from the STN API; `map_hwms_to_cells.py`,
  `make_hwm_obs.py` and `filter_hwm_by_crest.py` project the marks onto the
  mesh, apply the bed-elevation and quality-code QC (`QC_REPORT.md`), and write
  `turning30m_hwm_cells.csv` (324 in-domain marks) and `turning30m_hwm_obs.txt`
  (the 108 marks above the bed with quality 3 or better). See
  `data/harvey_hwm/README.md`.
- **Stream gauges**, `data/harvey_gauges/`: `map_gauges_to_cells.py` and
  `make_obs_table.py` map the gauges to cells and build the production-window
  table `obs_turning_h29_41.txt` (5 above-bed gauges, 134 records over event
  hours 29 to 41). The clock cross-check is in `data/harvey_gauges/README.md`.
- **Land cover**, `data/nlcd/`: `fetch_nlcd.sh` downloads the NLCD 2021 tile;
  `make_nlcd_manning.py` maps classes to cells (`houston1km_summary.csv`,
  `turning30m_summary.csv`); `scale_manning.py` and `scale_manning_classes.py`
  scale a Manning field uniformly or per class. The 1 km test maps
  (`houston1km_class.bin`, `houston1km_manning.bin`) are here too. See
  `data/nlcd/README.md`.

## Provenance of each figure and table

| item | label | scripts, data and archived output |
| --- | --- | --- |
| Figure 1 | `fig:map` | `papers/manning-calibration/plot_manning_map.py`, rendering the calibrated-field `.vtu` that `rdycore_adjoint` writes on the 1 km twins (the `.vtu` files are not archived); panel (d)'s 17-gauge network is `data/harvey_gauges/gauge_cells_real17.txt` |
| Figure 2 | `fig:authority` | `papers/manning-calibration/fig_authority.tex`; roughness scan `plans/campaigns/o49_alpha_bar.sh`; initial-storage scan `plans/campaigns/o54_ic_authority.sh`, `plans/campaigns/o60_ic_extend.sh`; output `logs/o57/`, `logs/o60/`, `plans/RESULTS-gpu-implicit.md` |
| Figure 3 | `fig:spectrum` | `papers/manning-calibration/fig_spectrum.tex`; eigenvalues from `plans/campaigns/o58_gauss_newton.py` on the per-mark peak dumps `logs/o61/o58_e0.05w43200_pk_*.txt`; results `logs/o61/o61_spectrum_sigma_n0.015.txt`, `logs/o61/o61_spectrum_sigma_alpha0.30.txt` |
| Table 1 | `tab:verification` | the `ctest` gates above; `logs/verification/ctest_jacobian_2026-09-17.txt`, `plans/RESULTS-manning-draft.md` |
| Table 2 | `tab:fdwindow` | `driver/adjoint_test.c` with `-adjoint_hwm_fd`; `plans/RESULTS-gpu-implicit.md` |
| Table 3 | `tab:configs` | `driver/tests/adjoint/*.yaml`, `plans/campaigns/o48_calibrate_rel.sh` |
| Table 4 | `tab:cure` | `plans/RESULTS-gpu-implicit.md`, `logs/ksp_look_n4.log` |
| Table 5 | `tab:snr` | `driver/adjoint_test.c` with `-adjoint_obs_noise`; `logs/verification/noisy_twins_2026-09-17.txt`, `plans/RESULTS-gpu-implicit.md` |
| Table 6 | `tab:gauges` | `data/harvey_gauges/turning30m_gauges_cells.csv`, `data/harvey_gauges/obs_turning_h29_41.txt`, `logs/o65/o65_gauge_base.txt.zb` |
| Tables 7, 8 | `tab:baseline`, `tab:bands` | `data/harvey_hwm/turning30m_hwm_obs.txt`, `data/harvey_hwm/filter_hwm_by_crest.py`, `plans/RESULTS-gpu-implicit.md` |
| Table 9 | `tab:alpha` | `plans/campaigns/o49_alpha_bar.sh`, `logs/o61/o58_e0.05w43200_pk_base.txt`, `plans/RESULTS-gpu-implicit.md` |
| Table 10 | `tab:halves` | `logs/o61/o58_e0.05w43200_pk_base.txt`, `plans/RESULTS-gpu-implicit.md` |
| Table 11 | `tab:ladder` | `logs/o59/`, `logs/o61/`, `logs/o62/`, `plans/RESULTS-gpu-implicit.md` |
| Table 12 | `tab:learned` | `plans/campaigns/o58_gauss_newton.py`, `logs/o61/o61_spectrum_sigma_n0.015.txt`, `logs/o61/o61_spectrum_sigma_alpha0.30.txt` |
| Table 13 | `tab:threefifteen` | `plans/campaigns/o48_calibrate_rel.sh`, `plans/campaigns/o59_spectral_active_set.sh`, `plans/campaigns/o62_sigma_alpha.sh`; `logs/o59/`, `logs/o61/o61_p3_score.log`, `logs/o62/`, `plans/RESULTS-gpu-implicit.md` |
| Table 14 | `tab:scaling` | `plans/campaigns/o58_gauss_newton.py`, `logs/o61/o61_spectrum_sigma_n0.015.txt`, `logs/o61/o61_spectrum_analysis.txt` |
| Table 15 | `tab:crossobs` | `plans/campaigns/o63_gauge_validation.sh`, `plans/campaigns/o64_gauge_score.sh`, `logs/o63/`, `logs/o64/` |
| Table 16 | `tab:gaugeresid` | `plans/campaigns/o65_gauge_spectrum.py`, `data/harvey_gauges/obs_turning_h29_41.txt`, `logs/o65/o65_gauge_base.txt` and `.zb` |
| Table 17 | `tab:ic` | `plans/campaigns/o54_ic_authority.sh`, `plans/campaigns/o60_ic_extend.sh`, `logs/o57/`, `logs/o60/`, `plans/RESULTS-gpu-implicit.md` |
| Sec. 6.1 outlet balance | `sec:hwm` | `plans/campaigns/stepA_perimeter_flux.py`, `logs/stepA/` |

## Reproducing on a workstation

The Gauss-Newton spectra (Sec. 6.4, Figure 3, Tables 12, 14 and 16) are
computed from the archived forward dumps, so they reproduce on a laptop with
NumPy and no RDycore build. Each command below reproduces the archived file it
names; the only difference to expect is the sign of an eigenvector's loadings,
which is arbitrary and depends on the LAPACK build. From the repository root:

```bash
# HWM spectrum, absolute prior -> logs/o61/o61_spectrum_sigma_n0.015.txt
python3 plans/campaigns/o58_gauss_newton.py 0.05 \
  logs/o61/o58_e0.05w43200_pk_base.txt logs/o61/o58_e0.05w43200_pk_col*.txt \
  --sigma-obs 0.15 --sigma-n 0.015

# HWM spectrum, fractional prior -> logs/o61/o61_spectrum_sigma_alpha0.30.txt
python3 plans/campaigns/o58_gauss_newton.py 0.05 \
  logs/o61/o58_e0.05w43200_pk_base.txt logs/o61/o58_e0.05w43200_pk_col*.txt \
  --sigma-obs 0.15 --sigma-alpha 0.30

# gauge residual decomposition and gauge-vs-mark spectra (Table 16)
#   -> logs/o65/o65_spectrum_iid_no24.txt
python3 plans/campaigns/o65_gauge_spectrum.py 0.05 \
  data/harvey_gauges/obs_turning_h29_41.txt logs/o65/o65_gauge_base.txt \
  logs/o65/o65_gauge_col{11,21,22,23,24,31,41,42,43,52,71,81,82,90,95}.txt \
  --sigma-obs 0.15 --sigma-alpha 0.30 --drop 24 --sigma-scan \
  --marks logs/o61/o58_e0.05w43200_pk_base.txt logs/o61/o58_e0.05w43200_pk_col*.txt

# the manuscript
cd papers/manning-calibration && latexmk -pdf manning-calibration.tex
```
