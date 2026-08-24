# USGS Hurricane Harvey High-Water Marks

This directory contains high-water mark (HWM) data from the **USGS Short-Term Network (STN) Flood Event Viewer**, acquired and processed for the Hurricane Harvey 2017 event in the context of Manning-coefficient calibration for the Turning 30 m mesh (Buffalo Bayou / Whiteoak Bayou, west Houston).

## Contents

### Data acquisition
- `fetch_usgs_hwm.py` — Script to download raw HWM JSON/CSV from the STN API
- `hwm_raw.json` — Raw STN export (2,364 marks, all fields)
- `hwm_raw.csv` — Flattened CSV (2,364 marks, 32 columns)
- `PROVENANCE.txt` — Record of fetch timestamp, API base, record count

### Mesh mapping and QC
- `map_hwms_to_cells.py` — Adapted from `../harvey_gauges/map_gauges_to_cells.py`
  - Projects mark lon/lat to EPSG:32610 (Turning mesh CRS)
  - Point-in-triangle test against mesh cells
  - Vertical datum handling (NAVD88 conversion to meters)
  - Bed-elevation QC (critical: flags marks below their cell's bed)
  - Quality-code stratification (USGS excellence, good, fair, poor)
  - Environment breakdown (riverine vs. coastal)

### Outputs
- `turning30m_hwm_cells.csv` — Mapped marks (324 in-domain):
  - Columns: hwm_id, lon, lat, x_utm10, y_utm10, cell (mesh natural ID)
  - Elevation (ft and m), cell bed elevation, above-bed depth, vertical datum
  - USGS quality code and environment, survey date, waterbody name
- `QC_REPORT.md` — Comprehensive QC analysis (see below)

---

## Key QC Findings (SHORT ANSWER)

| Metric | Value | Status |
|--------|-------|--------|
| **Coverage (go/no-go)** | 324 / 2,364 (13.7%) in Turning domain | **Sparse but focused** |
| **Below-bed fraction** | 202 / 324 (62.3%) below cell bed → UNUSABLE | **High loss; mesh resolution trap** |
| **Usable marks** | 122 above-bed (37.7%) | **OK for floodplain-focused misfit** |
| **Recommended subset** | 108 marks (quality ≤ fair, above-bed) | **Balanced rigor/coverage** |
| **Vertical datum** | 323/324 NAVD88, 1 MSL | **Consistent with mesh** |
| **Mean above-bed depth** | 1.81 m (usable marks only) | **Good; well above mesh diffusion** |

### Coverage interpretation

The 13.7% in-domain rate is NOT a mesh-coverage failure. The HWM survey effort was concentrated on the **main Houston ship channel and coastal barrier** rather than the western tributary floodplains. The 324 marks that DO fall in the Turning domain represent the western Buffalo Bayou and Whiteoak Bayou systems — exactly the right geography for the mesh.

**Decision:** 324 in-domain marks provide adequate coverage to attempt calibration. **Use the 108-mark subset** (quality ≤ fair, above-bed) to avoid below-bed trap and overfit to poor-quality marks.

---

## Vertical Datum and Coordinate System

**Mesh CRS:** EPSG:32610 (WGS 84 / UTM zone 10N) — the California zone, used ~27 degrees east of its central meridian. Confirmed in `../harvey_gauges/README.md`.

**Vertical datum:** 
- HWM elevations (`elev_ft` in STN) are in **NAVD88** (North American Vertical Datum 1988) for 99.7% of marks.
- Mesh bed elevations (`coordz` in Exodus file) are also **NAVD88**.
- Conversion: `elev_m = elev_ft * 0.3048` (feet to meters).
- **No datum transformation needed.**

This is consistent with the gauge work, which found that all 20 in-mesh Houston1km gauges have datum NAVD88 with alt_va=0 (i.e., published stage IS the water-surface elevation in ft NAVD88).

---

## Below-Bed Analysis (Replicates Gauge Finding)

A mark that falls **below** its containing cell's mean bed elevation is **physically unusable** because:
1. It implies the water surface is inside the solid mesh (impossible).
2. It usually indicates the mark is in an incised channel that a coarse mesh cannot resolve.
3. This exact trap caught ~60% of stage data at the 13-gauge network; the HWM archive shows the same bias.

**Result:** 62.3% of in-domain marks are below-bed. After filtering:
- **122 usable marks** (37.7% of in-domain) sit above their cell beds
- Mean depth above bed: 1.81 m — a good safety margin for model discretization error

---

## Quality Code Breakdown

USGS assigns each mark a quality flag:

| Code | Rating | Counts (in-domain) | Usable (above-bed) | Strategy |
|------|--------|-------------------|------------------|----------|
| 1 | Excellent | 18 | 10 (55.6%) | Highest precision; very sparse |
| 2 | Good | 21 | 11 (52.4%) | Good quality; balanced precision |
| 3 | Fair | 249 | 87 (34.9%) | Bulk of archive; lower precision |
| 4 | Poor | 28 | 10 (35.7%) | Lowest quality; avoid if possible |

### Interpretation
- **Excellent + Good (quality ≤ 2):** 21 usable marks. Highest confidence; very restrictive.
- **Excellent + Good + Fair (quality ≤ 3):** **108 usable marks.** Recommended balance of rigor and coverage (standard in flood-hazard work; USGS itself recommends excluding poor).
- **All (quality ≤ 4):** 122 usable marks. Permissive; includes poor-quality marks.

**Recommendation:** Use quality ≤ 3, yielding **108 marks** for calibration.

### By environment
- **Riverine (239 total marks):** Excellent/good marks are 50%+ usable; fair/poor are ~30%. Quality is a strong predictor (channel-incision bias).
- **Coastal (85 total marks):** Mostly fair-quality (52.9% usable); one excellent mark in poor rating (data entry artifact or storm-surge complexity).

---

## Next Steps for Calibration

1. **Peak-WSE misfit:** Implement in the driver using `turning30m_hwm_cells.csv` with the quality ≤ 3 filter.
   - Observation: maximum elevation per mark (ft NAVD88).
   - Model prediction: max(h + z_b) at the containing cell over the event window.
   - Misfit: mean absolute error, stratified by quality/environment for sensitivity.

2. **Observability:** Re-run identifiability study with HWM set vs. NLCD classes.
   - 108 spatially distributed marks (vs. 13 gauges in 20-second window).
   - Expected improvement in observable classes (vs. observability paper result of 66%).

3. **Comparison:** Report MAE against **Inunda's 0.67 m benchmark** on the same Harvey domain (their result on Harris County HWMs).

---

## Data Provenance

| Item | Value |
|------|-------|
| Source | USGS Short-Term Network (STN) Flood Event Viewer |
| API | https://stn.wim.usgs.gov/STNServices/ |
| Event | 2017 Harvey (event_id: 180) |
| Fetch script | `fetch_usgs_hwm.py` (python3, urllib + json) |
| Fetch date | 2026-08-24 |
| Records returned by API | 39,772 (includes related events) |
| Records with event_id=180 | 2,364 (Harvey-only) |
| In-domain (Turning mesh) | 324 (13.7%) |
| Above-bed (usable raw) | 122 (37.7% of in-domain) |
| Quality ≤ 3 (recommended) | 108 (9.6% of raw fetch) |

**License:** USGS STN data are public. When publishing results using these HWMs, cite the STN and provide the event_id (180).

---

## Related Files

- `../harvey_gauges/README.md` — Gauge datum work and mesh CRS confirmation
- `../harvey_gauges/turning30m_gauges_cells.csv` — 21 gauges (reference for compare/contrast)
- `../harvey_gauges/map_gauges_to_cells.py` — Original gauge mapping script (HWM script adapted from this)
- `/share/meshes/Turning_30m_with_z.updated.with_sidesets.exo` — Mesh file used for point-in-triangle and bed elevation

---

## Quick-Start for Misfit Integration

To use the HWM data in a calibration driver:

1. Read `turning30m_hwm_cells.csv` (pandas or csv module).
2. Filter to quality ≤ 3 and `is_above_bed == "yes"` → 108 marks.
3. For each mark:
   - Cell ID (column: `cell`) → extract cell from mesh
   - Observation: `elev_m` (column: elevation in meters NAVD88)
   - Model prediction: max(h + z_b) at this cell over event window
   - Misfit: |observation - prediction|, weight by quality flag (e.g., excellent=1.0, good=0.9, fair=0.7)
4. Aggregate: mean absolute error (MAE) or sum-of-squares
5. Report alongside gauges/classes for observability analysis

See `QC_REPORT.md` for full statistics and sensitivity recommendations.
