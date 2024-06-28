
# RDycore SWE physics with CEED

Here we describe the RDycore's implemetation of 2D shallow water equation (SWE) that uses [libCEED](https://github.com/CEED/libCEED). The combination of PETSc and libCEED provides ***algorithimic and hardware protability***. 

- RDycore uses PETSc's `TS` solvers that provide support for multiple time-integrators such as forward euler, RK4, etc., which can be selected at run time (e.g. `--ts_type euler`, `--ts_type rk4`, etc). 
- The use libCEED allows RDycore to compute the RHS-function for the explicit time-integrators on CPU or GPU, which can also be selected at runtime via
     - For CPUs: `-ceed /cpu/self`
     - For NVIDIA GPUs: `-ceed /gpu/cuda -dm_vec_type cuda`
     - For AMD GPUs: `-ceed /gpu/hip -dm_vec_type hip`

## Example Mesh

Let's consider an example 3x2 mesh that consists of

- 6 cells: `c00` to `c05`
- 17 edges: `e00` to `e11`
    - Internal edges: `e04`, `e05`, `e07`, `e08`, `e09`, `e11`, and `e12`.
    - Boundary edges: Remaining 10 edges.
- 12 vertices: `v00` to `v11`

```text
v08---e14---v09---e15---v10---e16---v11
 |           |           |           |
 |           |           |           |
e10   c03   e11   c04   e12   c05   e13
 |           |           |           |
 |           |           |           |
v04---e07---v05---e08---v06---e09---v07
 |           |           |           |
 |           |           |           |
e03   c00   e04   c01   e05   c02   e06
 |           |           |           |
 |           |           |           |
v00---e00---v01---e01---v02---e02---v03

```

## SWE physics with libCEED

The libCEED version of RDycore's explicit time-integrator of the SWE solver has two `CeedOperator`:

1. `rdy->ceed_rhs.op_edges` : Computes fluxes across edges and it includes two sub-operators that correspond to:
    - Internal edges,
    - Boundary edges

2. `rdy->ceed_rhs.src`: Opeator that computes the source terms including rainfall and terms associated with bed slope and bed friction.

### Computation of Fluxes across Internal Edges

For the mesh shown above, the prognostic variables of the 2D SWE are saved in a strided PETSc Vec (`X`).
The block size of `X` would be `3` corresponding to the following prognostic variables:

- Height (`h`),
- Momentum in x-dir (`hu`), and
- Momentum in y-dir (`hv`).
The size of `X` will `6 * 3` where `6` corresponds to number of cells in the mesh. The layout of `X` will be as follows:

```text
X = [[h0 hu0 hv0] [h1 hu1 hv1] ... [h5 hu5 hv5]]
```

RDycore uses first-order finite volume discretization to compute the flux across the edges, which requires values on the left and the right of the edge.

| Internal Edge | Left Cell | Right cell |
| ---- | ---- | ---- |
| e04  | c00  | c01  |
| e05  | c01  | c02  |
| e07  | c00  | c03  |
| e08  | c01  | c04  |
| e09  | c02  | c05  |
| e11  | c03  | c04  |
| e12  | c04  | c05  |

The steps involved in creating the `CeedOperator` associated with the internal edges are as follows:

- First, create a `CeedQFunction`, `qf`, and add input and output fields.
The input fields includes geometric attributes associated with the edges and the prognostic variables left and right of the edges.
The output fields include contribution of fluxes to the cells left and right of the edge and a diagnostic variable that saves fluxes through the edge.
The pointer to the user-defined function is specified at the time of creation.

| Field name | Size | In/Out | Notes |
| ---------- | ---- | ------ | ----- |
| geom       |  4   | In     | Geometric attrbutes [sn, cn, L_edge/Area_left, L_edge/right] |
| q_left     |  3   | In     | State left of the edge [h_left, hu_left, hv_left] |
| q_right    |  3   | In     | State right of the edge [h_right, hu_right, hv_right] |
| cell_left  |  3   | Out    | Flux contribution to the left cell [f_h f_hu f_hv] * L_edge/Area_left |
| cell_left  |  3   | Out    | Flux contribution to the right cell [f_h f_hu f_hv] * L_edge/Area_right |
| flux       |  3   | Out    | Flux through the edge [f_h f_hu f_hv] |

- Second, create `CeedElemRestriction` for all the input and output fields of previously created `CeedQFunction`.
A `CeedElemRistriction` tells libCEED the indices of a `CeedVec` from/to which the values are to extracted/written
for an input/output field. The `CeedElemRestriction` for the fields and the example mesh is given below.

| Variable name  | Size                            | Created via                        | Notes |
| -------------- | ------------------------------- | ---------------------------------- | ----- |
| `restrict_geom`  |  4 * num_owned_internal_edges   | `CeedElemRestrictionCreateStrided` |  [ 0,  4,  8, 12, 16, 20, 24] |
| `q_restrict_l`   |  3 * num_cells                  | `CeedElemRestrictionCreate`        |  [ 0,  1,  0,  1,  2,  3,  4] |
| `q_restrict_r`   |  3 * num_cells                  | `CeedElemRestrictionCreate`        |  [ 1,  2,  3,  4,  5,  4,  5] |
| `c_restrict_l`   |  3 * num_cells                  | `CeedElemRestrictionCreate`        |  [ 0,  1,  0,  1,  2,  3,  4] |
| `c_restrict_r`   |  3 * num_cells                  | `CeedElemRestrictionCreate`        |  [ 1,  2,  3,  4,  5,  4,  5] |
| `restrict_flux`  |  3 * num_owned_internal_edges   | `CeedElemRestrictionCreateStrided` |  [ 0,  3,  6,  9, 12, 15, 18] |

- Third,  create the `CeedOperator` using the previously created `CeedQFunction` and all `CeedElementRestriction`.
The multiple fields are added via `CeedOperatorSetField`.

| Field name | Size             | CeedVector          | Notes |
| ---------- | ---------------- | ------------------- | ----- |
| geom       |  restrict_geom   | `geom`              | This `CeedVector` has values `geom[:][0:3] = [sn cn L/A_l L/A_r]` |
| q_left     |  q_restrict_l    | CEED_VECTOR_ACTIVE  |  |
| q_right    |  q_restrict_r    | CEED_VECTOR_ACTIVE  |  |
| cell_left  |  c_restrict_l    | CEED_VECTOR_ACTIVE  |  |
| cell_left  |  c_restrict_r    | CEED_VECTOR_ACTIVE  |  |
| flux       |  restrict_flux   | `flux`              | This `CeedVector` is initalized to `0.0` |
