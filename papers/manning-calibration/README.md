# Artifact Description (AD): Manning Roughness Calibration and Parameter Identifiability in RDycore

This directory contains the manuscript source (`manning-calibration.tex`, `manning-calibration.bib`) and the self-contained **Artifact Description (AD)** — data preprocessing pipelines, observation tables, Perlmutter campaign scripts, post-processing tools, and archived output logs — for reproducing the verification gates, figures, and tables of the paper in the [`RDycore-4dvar`](https://github.com/markadams/RDycore-4dvar) repository.

---

## 1. Directory Layout

```text
papers/manning-calibration/
├── README.md                        # This Artifact Description & reproducibility guide
├── manning-calibration.tex          # Paper LaTeX source
├── manning-calibration.bib          # Bibliography
├── fig_authority.tex                # Figure 2 (IC vs. roughness authority scan, TikZ/pgfplots)
├── fig_spectrum.tex                 # Figure 3 (Gauss-Newton prior-preconditioned spectrum, TikZ/pgfplots)
├── houston_manning_300it.png        # Figure 1 panel (a): weak reg., 300 iterations
├── houston_manning_2000it.png       # Figure 1 panel (b): weak reg., 2000 iterations (semiconvergence)
├── houston_manning_reg.png          # Figure 1 panel (c): beta = 1e-4, 686 gauges
├── houston_manning_real17.png       # Figure 1 panel (d): real 17-gauge USGS network
├── plot_manning_map.py              # Script generating Figure 1 panels
├── data/                            # Processed observation/summary tables & data-fetch scripts
│   ├── harvey_gauges/               # USGS NWIS gauge mapping, clock cross-check, & obs tables
│   ├── harvey_hwm/                  # USGS STN High-Water Marks, QC report, & 108-mark obs table
│   └── nlcd/                        # NLCD 2021 fetch/sampling scripts & land-cover class summaries
├── campaigns/                       # Perlmutter job scripts & Python analysis scripts
│   ├── README.md                    # Staged Perlmutter inputs & campaign inventory
│   ├── o48..o66_*.sh                # Forward, adjoint, calibration, and authority-scan batch scripts
│   ├── o58_gauss_newton.py          # Gauss-Newton Hessian spectrum & posterior widths (Fig. 3, Tables 12 & 14)
│   ├── o65_gauge_spectrum.py        # Gauge observable GN spectrum & residual decomposition (Table 16)
│   └── stepA_perimeter_flux.py      # Boundary/outlet flux & mass-balance audit (Sec. 4.8)
└── logs/                            # Output logs & parameter/peak dumps backing paper tables/figures
    ├── RESULTS-gpu-implicit.md      # Complete Perlmutter campaign logbook (o20..o66, Tables 2, 4, 5, 7–11, 13, 17)
    ├── RESULTS-manning-draft.md     # Local verification & 1 km twin campaign logbook (Tables 1, 3, 5)
    ├── o57/, o60/                   # Initial-condition authority scan logs (Fig. 2, Table 17)
    ├── o59/, o62/                   # 3-class & 15-class calibration dumps/scores (Tables 11, 13)
    ├── o61/                         # 16 per-mark peak dumps & GN spectra (Fig. 3, Tables 12, 14)
    ├── o63/, o64/                   # Cross-observable gauge validation logs & weight audit (Sec. 4.7, Table 15)
    ├── o65/                         # Per-gauge WSE dumps, FD sensitivity columns, & gauge spectra (Table 16)
    ├── stepA/                       # Mass-balance & perimeter flux logs (Sec. 4.8)
    └── verification/                # CTest Jacobian, noisy-twin, & 4-GPU profiling logs (Tables 1, 4, 5)
```

---

## 2. Building RDycore & Running the Verification Suite (Table 1)

The analytic SWE Jacobian (`src/swe/swe_jacobian_petsc.c`, `src/swe/swe_roe_flux_jacobian_petsc.h`), Kokkos GPU assembly kernels (`src/swe/swe_jacobian_kokkos.kokkos.cxx`), implicit time-stepping support (`BEULER` and `ARKIMEX`), and discrete-adjoint calibration driver (`driver/adjoint_test.c`, building `rdycore_adjoint`) live in the root repository tree.

### Configure and Build
```bash
cmake -S . -B build \
  -DPETSC_DIR=$PETSC_DIR \
  -DPETSC_ARCH=$PETSC_ARCH \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run the Automated Verification Gates (`Table 1` / `tab:verification`)
Every finite-difference verification gate in **Table 1** (`tab:verification`) is an automated CTest target in `src/tests/` and `driver/tests/adjoint/`:
```bash
ctest --test-dir build -R "swe_jacobian|adjoint|calibrate" --output-on-failure
```
Key test targets include:
- `test_swe_jacobian`: Face-level exact $3 \times 3$ Roe numerical flux Jacobian vs. central finite differences.
- `swe_jacobian_global_*`: Global assembled analytic RHS Jacobian vs. finite-difference coloring across interior, reflecting, Dirichlet, and free-outflow boundaries.
- `adjoint_dam_break_analytic_np_1`, `adjoint_beuler_np_1`, `adjoint_arkimex_np_1`, `adjoint_beuler_freeflow_np_1`, `adjoint_arkimex_freeflow_np_1`: `TSAdjoint` initial-condition ($dJ/du_0$) and parameter ($dJ/dn$) gradient checks against central finite differences.
- `adjoint_hwm_fd_np_1`, `adjoint_hwm_fd_np_2`: Peak-WSE (high-water-mark) gradient verification on 1 and 2 MPI ranks.
- `adjoint_classes_relative_fd_np_1`, `adjoint_classes_relprior_fd_np_1`, `adjoint_classes_active_relative_np_1`: Relative land-cover-class scaling ($\alpha = n / n_{\rm prior}$), fractional prior ($\sigma_\alpha$), and active-set projection gates.
- `adjoint_restart_ref_np_1`, `adjoint_restart_window_np_1`, `adjoint_restart_compare_np_1`, `adjoint_restart_compare_np_2`: Bitwise and cross-decomposition checkpoint-restart consistency for mid-event windows.

---

## 3. Data Acquisition & Preprocessing (`data/`)

To keep the repository lightweight, multi-megabyte raw external archives (the 6.1 MB NLCD GeoTIFF, the 2.6 MB raw regional USGS STN JSON/CSV dump, and the 120 MB HydroShare gauge zip archives) are excluded via `.gitignore` and fetched on demand by the scripts in `data/`, while all processed tables consumed by the paper and driver are tracked in Git:

1. **NLCD 2021 Land-Cover Roughness (`data/nlcd/`):**
   - `fetch_nlcd.sh`: Downloads `nlcd_2021_houston_5070.tif` from the MRLC WCS service.
   - `make_nlcd_manning.py`: Maps NLCD classes to mesh cells (`houston1km_summary.csv`, `turning30m_summary.csv`). The 1 km test maps (`houston1km_class.bin`, `houston1km_manning.bin`) are stored in `share/materials/` for CTest.
   - `scale_manning.py`, `scale_manning_classes.py`: Utilities for uniform and per-class scaling of Manning fields.
   - Details in [`data/nlcd/README.md`](data/nlcd/README.md).
2. **USGS Short-Term Network High-Water Marks (`data/harvey_hwm/`):**
   - `fetch_usgs_hwm.py`: Retrieves Event 180 (Hurricane Harvey) marks from the USGS STN API.
   - `map_hwms_to_cells.py`, `make_hwm_obs.py`, `filter_hwm_by_crest.py`: Projects marks to EPSG:32610, performs bed-elevation and quality-code QC (`QC_REPORT.md`), and writes `turning30m_hwm_cells.csv` (324 in-domain marks) and `turning30m_hwm_obs.txt` (108 above-bed, quality $\le 3$ marks).
   - Details in [`data/harvey_hwm/README.md`](data/harvey_hwm/README.md) and [`data/harvey_hwm/QC_REPORT.md`](data/harvey_hwm/QC_REPORT.md).
3. **USGS NWIS Stream Gauges & MRMS Clock Alignment (`data/harvey_gauges/`):**
   - `map_gauges_to_cells.py`, `make_obs_table.py`: Maps Houston/Turning gauges to mesh cells (`houston_gauges_cells.csv`, `turning30m_gauges_cells.csv`) and constructs the production window stage observation table (`obs_turning_h29_41.txt`, 5 above-bed gauges, 134 records over event hours 29–41).
   - Details and CDT clock verification in [`data/harvey_gauges/README.md`](data/harvey_gauges/README.md).

---

## 4. Provenance of Figures and Tables

| Paper Item | Label | Source Script(s) & Archived Log(s) / Data |
|---|---|---|
| **Figure 1** | `fig:map` | Generated by `plot_manning_map.py` (`houston_manning_{300it,2000it,reg,real17}.png`) using `data/harvey_gauges/gauge_cells_real17.txt`. |
| **Figure 2** | `fig:authority` | TikZ/pgfplots in `fig_authority.tex`; roughness scan from `campaigns/o49_alpha_bar.sh`, initial-condition storage scan from `campaigns/o54_ic_authority.sh` & `campaigns/o60_ic_extend.sh` (`logs/o57/`, `logs/o60/`, `logs/RESULTS-gpu-implicit.md`). |
| **Figure 3** | `fig:spectrum` | TikZ/pgfplots in `fig_spectrum.tex`; eigenvalues computed by `campaigns/o58_gauss_newton.py` on the 16 per-mark peak dumps in `logs/o61/o58_e0.05w43200_pk_*.txt` (`logs/o61/o61_spectrum_sigma_n0.015.txt` and `logs/o61/o61_spectrum_sigma_alpha0.30.txt`). |
| **Table 1** | `tab:verification` | Automated `ctest` suite (`src/tests/`, `driver/tests/adjoint/`), `logs/verification/ctest_jacobian_2026-09-17.txt`, and `logs/RESULTS-manning-draft.md`. |
| **Table 2** | `tab:fdwindow` | Trajectory-length FD vs. adjoint comparison across window lengths (`driver/adjoint_test.c` with `-adjoint_hwm_fd`; `logs/RESULTS-gpu-implicit.md`). |
| **Table 3** | `tab:configs` | Summary of twin and Hurricane Harvey configurations (`driver/tests/adjoint/*.yaml`, `campaigns/o48_calibrate_rel.sh`). |
| **Table 4** | `tab:cure` | Critical-outflow vs. free-outflow (`free_outflow`) boundary stability on the 30 m mesh (`logs/RESULTS-gpu-implicit.md`, `logs/verification/ksp_look_n4.log`). |
| **Table 5** | `tab:snr` | Signal-to-noise twin experiments (`driver/adjoint_test.c -adjoint_obs_noise`); logged in `logs/verification/noisy_twins_2026-09-17.txt` and `logs/RESULTS-gpu-implicit.md`. |
| **Table 6** | `tab:gauges` | 13 rain-driven USGS gauges in the Turning 30 m mesh (`data/harvey_gauges/turning30m_gauges_cells.csv`, `data/harvey_gauges/obs_turning_h29_41.txt`, `logs/o65/o65_gauge_base.txt.zb`). |
| **Tables 7–8** | `tab:baseline`, `tab:bands` | Uncalibrated baseline and crest-timing spatial bands on the 108 surveyed marks (`data/harvey_hwm/turning30m_hwm_obs.txt`, `data/harvey_hwm/filter_hwm_by_crest.py`, `logs/RESULTS-gpu-implicit.md`). |
| **Table 9** | `tab:alpha` | Uniform Manning scaling scan $\alpha \in [0.20, 1.00]$ (`campaigns/o49_alpha_bar.sh`, `logs/o61/o58_e0.05w43200_pk_base.txt`, `logs/RESULTS-gpu-implicit.md`). |
| **Table 10** | `tab:halves` | Half-domain roughness scaling at $\alpha = 0.45$ on developed classes 21–24 vs. everything else vs. both (`logs/o61/o58_e0.05w43200_pk_base.txt`, `logs/RESULTS-gpu-implicit.md`). |
| **Table 11** | `tab:ladder` | Summary ladder of peak-WSE MAE and $\Delta$ across uniform and calibrated Manning fields (`logs/o59/`, `logs/o61/`, `logs/o62/`, `logs/RESULTS-gpu-implicit.md`). |
| **Table 12** | `tab:learned` | Prior-preconditioned Gauss–Newton posterior standard-deviation reductions per NLCD class (`campaigns/o58_gauss_newton.py`, `logs/o61/o61_spectrum_sigma_n0.015.txt`, `logs/o61/o61_spectrum_sigma_alpha0.30.txt`). |
| **Table 13** | `tab:threefifteen` | Three-class vs. fifteen-class calibration comparison under $\sigma_n = 0.015$ and $\sigma_\alpha \in \{0.30, 0.50\}$ (`campaigns/o48_calibrate_rel.sh`, `campaigns/o59_spectral_active_set.sh`, `campaigns/o62_sigma_alpha.sh`, `logs/o59/`, `logs/o61/o61_p3_score.log`, `logs/o62/`, `logs/RESULTS-gpu-implicit.md`). |
| **Table 14** | `tab:scaling` | Supported-parameter count vs. survey size and observation precision (`campaigns/o58_gauss_newton.py`, `logs/o61/o61_spectrum_sigma_n0.015.txt`, `logs/o61/o61_spectrum_analysis.txt`). |
| **Table 15** | `tab:crossobs` | Cross-observable validation (calibrating on HWMs vs. gauges and scoring on both; `campaigns/o63_gauge_validation.sh`, `campaigns/o64_gauge_score.sh`, `logs/o63/`, `logs/o64/`). |
| **Table 16** | `tab:gaugeresid` | Per-gauge offset vs. shape residual decomposition over the 134 above-bed stage records (`campaigns/o65_gauge_spectrum.py`, `data/harvey_gauges/obs_turning_h29_41.txt`, `logs/o65/o65_gauge_base.txt`, `.zb`). |
| **Table 17** | `tab:ic` | Antecedent water-storage scaling scan $a \in [0.80, 1.20]$ (and $a \in \{0.60, 0.70\}$ in Fig. 2; `campaigns/o54_ic_authority.sh`, `campaigns/o60_ic_extend.sh`, `logs/o57/`, `logs/o60/`, `logs/RESULTS-gpu-implicit.md`). |

---

## 5. Reproducing the Gauss–Newton Spectrum and Figures Locally

While the full 30 m Hurricane Harvey forward and adjoint runs require multi-GPU nodes (see [`campaigns/README.md`](campaigns/README.md)), the Gauss–Newton spectrum analyses (Section 4.6, **Figure 3**, **Tables 12, 14, & 16**) and the manuscript itself can be reproduced directly on a workstation from the archived forward dumps in `logs/o61/` and `logs/o65/`:

```bash
# 1. HWM spectrum under absolute prior (sigma_n = 0.015) -> matches logs/o61/o61_spectrum_sigma_n0.015.txt
python3 campaigns/o58_gauss_newton.py 0.05 \
  logs/o61/o58_e0.05w43200_pk_base.txt \
  logs/o61/o58_e0.05w43200_pk_col*.txt \
  --sigma-obs 0.15 --sigma-n 0.015

# 2. HWM spectrum under fractional prior (sigma_alpha = 0.30) -> matches logs/o61/o61_spectrum_sigma_alpha0.30.txt
python3 campaigns/o58_gauss_newton.py 0.05 \
  logs/o61/o58_e0.05w43200_pk_base.txt \
  logs/o61/o58_e0.05w43200_pk_col*.txt \
  --sigma-obs 0.15 --sigma-alpha 0.30

# 3. Gauge residual decomposition (Table 16) and gauge vs. HWM spectrum comparison -> matches logs/o65/
python3 campaigns/o65_gauge_spectrum.py 0.05 \
  data/harvey_gauges/obs_turning_h29_41.txt \
  logs/o65/o65_gauge_base.txt \
  logs/o65/o65_gauge_col11.txt logs/o65/o65_gauge_col21.txt logs/o65/o65_gauge_col22.txt \
  logs/o65/o65_gauge_col23.txt logs/o65/o65_gauge_col24.txt logs/o65/o65_gauge_col31.txt \
  logs/o65/o65_gauge_col41.txt logs/o65/o65_gauge_col42.txt logs/o65/o65_gauge_col43.txt \
  logs/o65/o65_gauge_col52.txt logs/o65/o65_gauge_col71.txt logs/o65/o65_gauge_col81.txt \
  logs/o65/o65_gauge_col82.txt logs/o65/o65_gauge_col90.txt logs/o65/o65_gauge_col95.txt \
  --sigma-obs 0.15 --sigma-alpha 0.30 --drop 24

# 4. Build the manuscript PDF (requires TeX Live / latexmk)
latexmk -pdf manning-calibration.tex
```
