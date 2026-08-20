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
