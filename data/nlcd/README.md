# NLCD 2021 land-cover roughness map for the Harvey domain

`fetch_nlcd.sh` downloads the NLCD 2021 land-cover subset covering the mesh
(MRLC WCS, native EPSG:5070 Albers, 30 m pixels, ~6 MB).
`make_nlcd_manning.py` samples it per mesh cell and writes a per-cell Manning
map, using Donghui Xu's 18-class table (OFMmesh `ex2/code/Step03_Process_Manning.m`).

    ./fetch_nlcd.sh
    python3 make_nlcd_manning.py ../../share/meshes/Houston1km_with_z.exo \
        nlcd_2021_houston_5070.tif "-2786,721154,76932,775420" houston1km

Outputs per mesh: `*_manning.bin` and `*_class.bin` (PETSc binary Vecs in
natural cell order) and `*_summary.csv`.

## What the domain is made of

Over half of west Houston is developed: medium intensity alone is 44% of the
1 km cells, high intensity 18%, low 8%, open space 2%; pasture/hay is 17% and
woody wetlands 5%.

## The headline number

| | Manning n |
|---|---|
| current Harvey configuration | **0.015** (uniform) |
| NLCD-derived, area-weighted mean | **0.1005** |
| NLCD range over the domain | 0.027 - 0.160 |

The land-cover map implies roughness ~6.7x the uniform value now in use, and
the two meshes agree closely (1 km mean 0.1006, 30 m mean 0.1005), so the
difference is not a resolution artifact. This is the prior and the Tikhonov
reference for calibration; it also means the correction we are looking for is
large and mostly one-signed, which is much easier to identify from 18 gauges
than small scattered adjustments.

NLCD is natively 30 m, so on `Turning_30m` this is a 1:1 mapping (1 sample per
cell); at 1 km each cell averages a 9x9 sample grid, matching how the
continental map was built.
