# Overview

The RDycore mesh, `Turning_30m_with_z.updated.with_sidesets.exo` for the
Houston Harvey flooding problem is in the Exodus II format and consist of
2,926,532 grid cells. The mesh also includes a single edge sideset that
is includes 13 (`=num_side_ss1`) edges, which are identified by `elem_ss1` and
`side_ss1` in the mesh file. The following two Hurricane Harvey examples
showcase different capabilties of the RDycore driver:

1. [`critical-outflow-bc`](critical-outflow-bc/harvey-critical-outflow-bc.md): 
    - Rainfall: Spatially-distributed and temporally-varying
    - Boundary condition: Critical outflow

2. [`ocean-bc`](ocean-bc/harvey-ocean-bc.md): 
    - Rainfall: Spatially-averaged and temporally-varying
    - Boundary condition: Spatially-homogeneous and temporally-varying
      ocean water height
