# USGS Hurricane Harvey gauge data

Downloaded 2026-08-18 from HydroShare resource
`4f439754503c4ff4959c7e3703121940` ("USGS - Harvey Gaged Streamflow
Timeseries", part of the "Hurricane Harvey Flood Data Collections"
collection, D. Arctur; DOI: 10.4211/hs.c037167e497546a1bc1508dfb32a9cff).
License: CC BY 4.0 — cite the collection in any publication using this
data.

## Contents

- `USGS_gage_height_timeseries.zip` → `height/USGS_gage_height_inst/`:
  746 CSVs, one per gauge (`<site_no>_gage_height.csv`), columns
  `site_no, dateTime, X_00065_00000`. USGS parameter 00065 = gage
  height in **feet above local gauge datum**, ~15–30 min cadence,
  starting 2017-08-16 (covers the Houston config's 2017-08-26 window).
- `USGS_gage_discharge_timeseries.zip`: same layout for discharge
  (param 00060, cfs).
- `USGS_Gages_TxLaMsAr.{shp,shx,dbf,prj,cpg}`: gauge locations, point
  shapefile, NAD83 geographic (lon/lat), TX/LA/MS/AR extent.

## Use in the Manning-calibration plan (increment 6)

Preprocessing needed before the misfit can consume this:
1. Subset gauges to the Houston1km mesh footprint (shapefile bbox
   filter, then point-in-mesh).
2. **Datum conversion**: stage is feet above each gauge's local datum.
   `houston_gauges.csv` (generated from the shapefile attributes, no
   network needed) lists the 90 Houston-area gauges with lon/lat, gauge
   datum altitude (alt_va, ft), datum code, and whether a Harvey stage
   series exists (75 do). WSE = alt_va + stage (ft -> m). The 23
   NGVD29-datum gauges need the NGVD29->NAVD88 shift (~-0.1..-0.4 ft in
   Houston; VERTCON) -- team QC item.
3. Map gauge lon/lat -> mesh cell indices for the
   `observations.sites.cells` YAML entry. **RESOLVED (2026-08-19): the
   Houston1km mesh CRS is EPSG:32610 (WGS 84 / UTM zone 10N)** -- the
   California zone, used ~27 deg east of its central meridian.
   Established from the mesh-generation pipeline (donghuix/OFMmesh, ex1
   "Houston Harvey Flooding": `proj = projcrs(32610)`), confirmed by
   inverse-projecting the mesh bbox onto west Houston (Addicks/Barker
   reservoirs -> Buffalo Bayou -> ship-channel Turning Basin) and by
   gauge-stage vs mesh-elevation consistency (three Harvey peaks land
   1-1.5 m below their 1-km cell's mean bed, and Piney Point's 63.94 ft
   crest matches the documented value). CAVEAT for the team: at this
   longitude offset the projection carries ~9% length inflation, ~19%
   area inflation, and ~14.5 deg grid rotation vs true north -- the
   "1 km" cells are ~917 m on the ground.

## Pipeline outputs (map_gauges_to_cells.py)

- `houston_gauges_cells.csv`: the 20 gauges inside the mesh (17 with
  Harvey stage series), each with its containing cell's natural ID
  (0-based mesh-file order, as `observations.sites.cells` expects),
  EPSG:32610 coords, and the cell's mean bed elevation for QC.
- `observations_sites_snippet.yaml`: ready-to-paste cells list for the
  17 stage-series gauges.
- Datum simplification: all 20 in-mesh gauges have alt_va = 0 with
  datum NAVD88, i.e. published stage IS the water-surface elevation in
  ft NAVD88. WSE_m = stage_ft * 0.3048, directly comparable to model
  h + z_b. The NGVD29/VERTCON item only affects east-side gauges
  outside this mesh.

## Turning 30 m mesh (the campaign mesh)

- `turning30m_gauges_cells.csv`: 21 in-mesh gauges (Buffalo Bayou /
  Whiteoak Bayou system) with Turning-mesh natural cell IDs;
  `turning30m_observations_sites_snippet.yaml` is the paste-ready
  sites list. `make_obs_table.py --cells-csv` takes it directly.
- Verified working 2026-08-23: a 12-hour hourly table from
  2017-08-26 00:00 yields 13 rain-driven gauges (after excluding the 8
  DAM_AFFECTED sites) with 156/156 observations present -- no gaps to
  work around.

## CSV CLOCK: RESOLVED (2026-08-23) -- the HydroShare export is LOCAL CDT

The HydroShare CSVs carry no timezone column, which left `--t0`
ambiguous by up to 5 hours. Settled by measurement, not assumption:
the Piney Point (08073700) series peaks at exactly 63.94 ft at
`2017-08-27 13:00:00` on the CSV clock; USGS NWIS instantaneous
values requested WITH an explicit timezone offset return the same
63.94 ft at `2017-08-27 13:00 CDT` (and the USGS annual peak file
independently confirms 63.94 ft on 2017-08-27 as the water-year
maximum gage height). So **CSV timestamps are local CDT = UTC-5**;
`--t0` must be given on that clock.

## RAIN CLOCK: RESOLVED (2026-08-23) -- MRMS rasters are ALSO local CDT

The MRMS hourly rasters are named `YYYY-MM-DD:HH-00.int32.bin`. Their
labels are on the SAME local CDT clock as the gauge CSVs, so **no
shift is needed**: `--t0` for `make_obs_table.py` equals the date
passed to `-raster_rain_start_date`.

Established empirically, at zero compute cost, by cross-correlating
the basin-mean MRMS rain rate against the gauge-average rate of stage
rise d(stage)/dt (the quantity rain actually drives), scanning the
offset between the two clocks over the Aug 25-28 storm:

| offset applied to rain clock | correlation |
|---|---|
| -1 h | +0.32 |
| **0 h (labels are CDT)** | **+0.65** |
| +1 h | +0.56 |
| +2 h | +0.27 |
| +5 h (labels would be UTC) | **-0.09** |

The peak is sharp and sits at 0-1 h, exactly the physical expectation
for these flashy urban catchments (Brickhouse Gully, Cole Creek etc.
respond in about an hour). The UTC hypothesis is not merely worse, it
is uncorrelated -- it would require a 5-6 hour catchment lag that
these basins do not have. Consistent with the series beginning at
`2017-08-24:19-00`, i.e. 00:00 UTC Aug 25 relabelled to local time.

Reproduce with the snippet in `plans/RESULTS-manning-draft.md` (reads
the rasters as PETSc binary Vecs: 5-value header ncols, nrows, xlc,
ylc, cellsize, then row-major data, big-endian float64).

NOTE this was worth checking rather than assuming: a wrong clock would
not have crashed anything -- the optimizer would have quietly biased
the calibrated Manning field to compensate for a timing error.
