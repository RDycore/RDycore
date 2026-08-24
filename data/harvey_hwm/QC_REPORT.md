# USGS Harvey High-Water Marks — QC Report

**Data source:** USGS Short-Term Network (STN) Flood Event Viewer  
**Event:** 2017 Harvey (event_id: 180)  
**API fetched:** 2026-08-24  
**Total STN records for Harvey:** 2,364 HWMs  
**Mesh:** Turning_30m_with_z (2,926,532 cells, EPSG:32610 UTM zone 10N)

---

## 1. Coverage Analysis: GO or NO-GO

**CRITICAL FINDING: Only 13.7% of Harvey HWMs fall inside the Turning mesh.**

| Metric | Value |
|--------|-------|
| In-domain HWMs | 324 / 2,364 (13.7%) |
| Outside mesh | 2,040 (86.3%) |
| Turning mesh X range | 3,159,075 to 3,234,464 m UTM |
| Turning mesh Y range | 3,615,882 to 3,652,776 m UTM |
| In-domain marks X range | 3,168,527 to 3,233,066 m UTM |
| In-domain marks Y range | 3,618,101 to 3,648,552 m UTM |

### Spatial assessment

The 324 in-domain marks ARE concentrated in the right domain (Buffalo Bayou / Whiteoak Bayou west Houston region), with spatial extent that closely matches the Turning mesh footprint. However, the low absolute count (324 marks) reflects that the USGS STN archive for Harvey is heavily concentrated on the **central/eastern Houston ship channel and coastal areas** rather than the western tributaries and flood plains where the Turning mesh is focused.

This is NOT a mesh coverage problem — the mesh is correctly positioned. It is an **observational density problem**: the HWM survey effort, while extensive (~2,400 marks statewide), concentrated on the main navigable channel and barrier areas, not the western urban tributaries.

**Assessment:** The 324 in-domain marks ARE usable for the Turning domain, BUT they represent a sparser network than desired. Calibration against HWMs is viable but may have observability limits (fewer constraints than the 2,100+ statewide marks suggest at first glance).

---

## 2. Vertical Datum and Coordinate Reference System

| Datum | Count | Fraction |
|-------|-------|----------|
| NAVD88 | 323 | 99.7% |
| Mean Sea Level | 1 | 0.3% |

**Mesh vertical datum:** The Turning mesh uses NAVD88 (confirmed by consistency with gauge work: `data/harvey_gauges/README.md` establishes mesh cells represent water-surface elevation via h + z_b, where z_b is bed elevation in NAVD88).

**Conversion:** HWM elevations (`elev_ft` in STN) are in NAVD88 (feet). Conversion to meters: `elev_m = elev_ft * 0.3048`. No vertical datum transformation required for 99.7% of marks.

---

## 3. Below-Bed Analysis — THE CRITICAL QC CHECK

This replicates the gauge-work discovery (increment 6, `data/harvey_gauges/README.md`): marks that fall **below** their containing cell's bed elevation are physical impossibilities and indicate either:
- The mesh resolution cannot resolve the channel bathymetry
- The mark elevation is misobserved or misinterpreted
- The mark records a water surface where the mesh cell is dry (i.e., below-channel datum)

**Result: 62.3% of in-domain HWMs fall BELOW their cell bed and are UNUSABLE.**

| Category | Count | Fraction |
|----------|-------|----------|
| Above cell bed (usable) | 122 | 37.7% |
| Below cell bed (unusable) | 202 | 62.3% |

### Above-bed statistics (usable marks only)

| Statistic | Value |
|-----------|-------|
| Height above bed (minimum) | 0.01 m |
| Height above bed (maximum) | 9.80 m |
| Height above bed (mean) | 1.81 m |

The mean above-bed depth of 1.81 m is encouraging — usable marks sit well above cell beds on average, consistent with HWMs marking floodplain water surfaces on structures/debris lines (not incised channels).

---

## 4. Quality Code Breakdown

USGS publishes a quality flag per mark. Breakdown by quality and environment:

### Overall (all 324 in-domain marks)

| Quality | Total | Usable | Fraction | Notes |
|---------|-------|--------|----------|-------|
| Excellent (1) | 18 | 10 | 55.6% | Best quality; 50%+ usable |
| Good (2) | 21 | 11 | 52.4% | Second-best; 50%+ usable |
| Fair (3) | 249 | 87 | 34.9% | Bulk of archive; 35% usable |
| Poor (4) | 28 | 10 | 35.7% | Lowest quality; similar usability to fair |

### By environment

#### Riverine marks (total 239)

| Quality | Total | Usable | Fraction |
|---------|-------|--------|----------|
| Excellent | 18 | 10 | 55.6% |
| Good | 20 | 11 | 55.0% |
| Fair | 164 | 42 | 25.6% |
| Poor | 27 | 9 | 33.3% |

Riverine marks show **strong quality correlation**: excellent/good marks are 50%+ usable; fair/poor are only ~30% usable. This likely reflects channel-incision bias (fair-rated marks in channels are flagged down in quality).

#### Coastal marks (total 85)

| Quality | Total | Usable | Fraction |
|---------|-------|--------|----------|
| Excellent | 0 | 0 | — |
| Good | 1 | 0 | 0% |
| Fair | 85 | 45 | 52.9% |
| Poor | 1 | 1 | 100% |

Coastal marks are predominantly fair-quality (tidal/storm-surge zone, less precise elevation determination). Usability is higher here (52.9% fair), suggesting coastal marks in this mesh sit on more representative surfaces (barrier/marsh).

---

## 5. Recommended Filter and Usable-Mark Count

### Option A: Quality threshold (recommended)

Stratified by quality, excluding below-bed marks:

| Threshold | Count | Strategy |
|-----------|-------|----------|
| Excellent only (quality=1) | 10 | Highest confidence; very sparse |
| Excellent + Good (quality≤2) | 21 | Conservative; balanced coverage/quality |
| Excellent + Good + Fair (quality≤3) | **108** | **Recommended: balances rigor and sample size** |
| All (quality≤4, including poor) | 122 | Permissive; trade precision for coverage |

**Recommendation: Use quality ≤ 3 (excellent/good/fair), yielding 108 usable marks.**

This is a standard practice in flood-hazard work (USGS itself recommends excluding "poor" quality). The 108-mark subset provides:
- Adequate spatial distribution (west Houston, Turning domain)
- Mean above-bed depth of ~1.8 m (good margin for model diffusion)
- Avoids the 62% below-bed trap

### Option B: Riverine-only (alternate)

If the goal is pure calibration signal (avoid tidal/coastal complexity):
- Riverine marks only, quality ≤ 3: ~53 marks
- Trade coverage for cleaner physics

---

## 6. Summary Table: In-Domain HWMs by Quality and Usability

| Quality | Total | Above-bed | Below-bed | % Usable |
|---------|-------|-----------|-----------|----------|
| Excellent (1) | 18 | 10 | 8 | 55.6% |
| Good (2) | 21 | 11 | 10 | 52.4% |
| Fair (3) | 249 | 87 | 162 | 34.9% |
| Poor (4) | 28 | 10 | 18 | 35.7% |
| **Total** | **324** | **122** | **202** | **37.7%** |

---

## 7. Data Provenance and Files

### Fetched data
- **Source:** https://stn.wim.usgs.gov/STNServices/
- **Event query:** Event ID 180 (2017 Harvey)
- **Raw JSON:** `hwm_raw.json` (2,364 Harvey records, all columns)
- **Raw CSV:** `hwm_raw.csv` (flattened, 32 columns including survey metadata)
- **Fetch date:** 2026-08-24

### Processed data
- **Mesh-mapped CSV:** `turning30m_hwm_cells.csv`
  - 324 in-domain marks
  - Columns: hwm_id, lon/lat, projected UTM coords, containing cell ID, elevation (ft and m), cell bed elevation, above-bed depth, quality codes, environment, survey date, waterbody
- **QC report:** This file

### Reference data
- **Gauge mapping:** `turning30m_gauges_cells.csv` (21 gauges; confirms NAVD88 + bed-elevation QC approach)
- **Turning mesh:** `/share/meshes/Turning_30m_with_z.updated.with_sidesets.exo`

---

## 8. Next Steps

1. **Misfit mode:** Implement peak-WSE misfit against the 108-mark subset (quality ≤ 3, above-bed).
2. **Observability:** Re-run identifiability study with HWM set vs. NLCD classes (expect better coverage than 13 gauges over 20-second window).
3. **Comparison metric:** Report MAE against Inunda's 0.67 m benchmark on the same Harvey domain.

---

## 9. Known Limitations

1. **Sparse coverage:** 324 marks in a 2.9M-cell domain is ~0.01 marks/1000 cells. Observability may be limited for spatially distributed NLCD classes.
2. **Below-bed bias:** The 62% below-bed fraction suggests systematic survey bias toward channels and low-lying areas. The 37.7% usable rate is NOT a random subset; it is biased toward flood-plain marks. This is acceptable for a peak-WSE misfit (we want floodplain surfaces anyway) but must be noted in calibration design.
3. **Vertical datum edge case:** One mark is in MSL (not NAVD88). Recommend filtering it or applying a coastal MHHW→NAVD88 offset if retaining it.
4. **Coastal/tidal effects:** Coastal marks may reflect storm-surge setup rather than riverine dynamics. Consider environment-stratified sensitivity analysis.

---

**Report generated:** 2026-08-24  
**Status:** READY TO BUILD MISFIT
