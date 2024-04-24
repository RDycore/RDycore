#include <private/rdycoreimpl.h>
#include <private/rdysweimpl.h>
#include <rdycore.h>

PetscErrorCode RDyGetNumGlobalCells(RDy rdy, PetscInt *num_cells_global) {
  PetscFunctionBegin;
  *num_cells_global = rdy->mesh.num_cells_global;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetNumLocalCells(RDy rdy, PetscInt *num_cells) {
  PetscFunctionBegin;
  *num_cells = rdy->mesh.num_cells_local;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetNumBoundaryConditions(RDy rdy, PetscInt *num_bnd_conds) {
  PetscFunctionBegin;
  *num_bnd_conds = rdy->num_boundaries;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckBoundaryConditionIndex(RDy rdy, const PetscInt boundary_index) {
  PetscFunctionBegin;
  PetscCheck(boundary_index < rdy->num_boundaries, rdy->comm, PETSC_ERR_USER,
             "Boundary condition index (%" PetscInt_FMT ") exceeds the max number of boundary conditions (%" PetscInt_FMT ")", boundary_index,
             rdy->num_boundaries);
  PetscCheck(boundary_index >= 0, rdy->comm, PETSC_ERR_USER, "Boundary condition index (%" PetscInt_FMT ") cannot be less than zero.",
             boundary_index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckBoundaryNumEdges(RDy rdy, const PetscInt boundary_index, const PetscInt num_edges) {
  PetscFunctionBegin;

  RDyBoundary boundary = rdy->boundaries[boundary_index];

  PetscCheck(boundary.num_edges == num_edges, rdy->comm, PETSC_ERR_USER,
             "The given number of edges (%" PetscInt_FMT ") for boundary with index %" PetscInt_FMT " is incorrect (should be %" PetscInt_FMT ")",
             num_edges, boundary_index, boundary.num_edges);

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckNumLocalCells(RDy rdy, const PetscInt size) {
  PetscFunctionBegin;
  PetscAssert(rdy->mesh.num_cells_local == size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of array is not equal to the number of local cells");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetNumBoundaryEdges(RDy rdy, const PetscInt boundary_index, PetscInt *num_edges) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));
  RDyBoundary *boundary = &rdy->boundaries[boundary_index];
  *num_edges            = boundary->num_edges;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryConditionFlowType(RDy rdy, const PetscInt boundary_index, PetscInt *bc_type) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));
  RDyCondition *boundary_cond = &rdy->boundary_conditions[boundary_index];
  *bc_type                    = boundary_cond->flow->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetDirichletBoundaryValues(RDy rdy, const PetscInt boundary_index, const PetscInt num_edges, const PetscInt ndof,
                                             PetscReal values[num_edges * ndof]) {
  PetscFunctionBegin;

  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));

  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "The number of DOFs (%" PetscInt_FMT ") for the boundary condition need to be three.", ndof);

  RDyBoundary boundary = rdy->boundaries[boundary_index];
  PetscCheck(boundary.num_edges == num_edges, rdy->comm, PETSC_ERR_USER,
             "The given number of edges (%" PetscInt_FMT ") for boundary with index %" PetscInt_FMT " is incorrect (should be %" PetscInt_FMT ")",
             num_edges, boundary_index, boundary.num_edges);

  RDyCondition boundary_cond = rdy->boundary_conditions[boundary_index];
  PetscCheck(boundary_cond.flow->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
             "Trying to set dirichlet values for boundary with index %" PetscInt_FMT ", but it has a different type (%u)", boundary_index,
             boundary_cond.flow->type);

  // dispatch this call to CEED or PETSc
  PetscReal tiny_h = rdy->config.physics.flow.tiny_h;
  if (rdy->ceed_resource[0]) {  // ceed
    PetscInt size = ndof * num_edges;
    PetscCall(SWEFluxOperatorSetDirichletBoundaryValues(rdy->ceed_rhs.op_edges, &rdy->mesh, rdy->boundaries[boundary_index], size, values));
  } else {  // petsc
    // fetch the boundary data
    RiemannDataSWE bdata;
    PetscCall(GetPetscSWEDirichletBoundaryValues(rdy->petsc_rhs, boundary_index, &bdata));

    // set the boundary values
    RDyCells *cells = &rdy->mesh.cells;
    RDyEdges *edges = &rdy->mesh.edges;
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      PetscInt iedge = boundary.edge_ids[e];
      PetscInt icell = edges->cell_ids[2 * iedge];
      if (cells->is_local[icell]) {
        bdata.h[e]  = values[3 * e];
        bdata.hu[e] = values[3 * e + 1];
        bdata.hv[e] = values[3 * e + 2];

        if (bdata.h[e] > tiny_h) {
          bdata.u[e] = values[3 * e + 1] / bdata.h[e];
          bdata.v[e] = values[3 * e + 2] / bdata.h[e];
        } else {
          bdata.u[e] = 0.0;
          bdata.v[e] = 0.0;
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyGetPrognosticVariableOfLocalCell(RDy rdy, PetscInt idof, PetscReal *values) {
  PetscFunctionBegin;

  PetscReal *x;
  PetscCall(VecGetArray(rdy->X, &x));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    values[i] = x[3 * i + idof];
  }
  PetscCall(VecRestoreArray(rdy->X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellHeights(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));
  PetscInt idof = 0;
  PetscCall(RDyGetPrognosticVariableOfLocalCell(rdy, idof, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellXMomentums(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));
  PetscInt idof = 1;
  PetscCall(RDyGetPrognosticVariableOfLocalCell(rdy, idof, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellYMomentums(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));
  PetscInt idof = 2;
  PetscCall(RDyGetPrognosticVariableOfLocalCell(rdy, idof, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetSourceVecForLocalCells(RDy rdy, Vec src_vec, PetscInt idof, PetscReal *values) {
  PetscFunctionBegin;

  PetscInt ndof;
  PetscCall(VecGetBlockSize(src_vec, &ndof));

  PetscCheck(idof < ndof, rdy->comm, PETSC_ERR_USER, "The block index (%" PetscInt_FMT ") exceeds the total number of blocks = %" PetscInt_FMT ")",
             idof, ndof);

  PetscReal *s;
  PetscCall(VecGetArray(src_vec, &s));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    s[i * ndof + idof] = values[i];
  }
  PetscCall(VecRestoreArray(src_vec, &s));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetWaterSourceForLocalCells(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  if (rdy->ceed_resource[0]) {  // ceed
    PetscCall(SWESourceOperatorSetWaterSource(rdy->ceed_rhs.op_src, values));
  } else {  // petsc
    PetscInt idof = 0;
    PetscCall(RDySetSourceVecForLocalCells(rdy, rdy->swe_src, idof, values));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetXMomentumSourceForLocalCells(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  if (rdy->ceed_resource[0]) {
    PetscCall(SWESourceOperatorSetXMomentumSource(rdy->ceed_rhs.op_src, values));
  } else {
    PetscInt idof = 1;
    PetscCall(RDySetSourceVecForLocalCells(rdy, rdy->swe_src, idof, values));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetYMomentumSourceForLocalCells(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  if (rdy->ceed_resource[0]) {
    PetscCall(SWESourceOperatorSetYMomentumSource(rdy->ceed_rhs.op_src, values));
  } else {
    PetscInt idof = 2;
    PetscCall(RDySetSourceVecForLocalCells(rdy, rdy->swe_src, idof, values));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyGetIDimCentroidOfLocalCell(RDy rdy, PetscInt idim, PetscInt size, PetscReal *x) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  RDyCells *cells = &rdy->mesh.cells;

  PetscInt count = 0;
  for (PetscInt icell = 0; icell < rdy->mesh.num_cells; ++icell) {
    if (cells->is_local[icell]) {
      x[count++] = cells->centroids[icell].X[idim];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellXCentroids(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscInt idim = 0;  // x-dim
  PetscCall(RDyGetIDimCentroidOfLocalCell(rdy, idim, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellYCentroids(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscInt idim = 1;  // y-dim
  PetscCall(RDyGetIDimCentroidOfLocalCell(rdy, idim, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellZCentroids(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscInt idim = 2;  // z-dim
  PetscCall(RDyGetIDimCentroidOfLocalCell(rdy, idim, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellAreas(RDy rdy, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));

  RDyCells *cells = &rdy->mesh.cells;

  PetscInt count = 0;
  for (PetscInt icell = 0; icell < rdy->mesh.num_cells; ++icell) {
    if (cells->is_local[icell]) {
      values[count++] = cells->areas[icell];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellNaturalIDs(RDy rdy, const PetscInt size, PetscInt values[size]) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  RDyCells *cells = &rdy->mesh.cells;

  PetscInt count = 0;
  for (PetscInt icell = 0; icell < rdy->mesh.num_cells; ++icell) {
    if (cells->is_local[icell]) {
      values[count++] = cells->natural_ids[icell];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyGetIDimCentroidOfBoundaryEdgeOrCell(RDy rdy, const PetscInt boundary_index, const PetscInt num_edges,
                                                             PetscBool data_for_edge, PetscInt idim, PetscReal *x) {
  PetscFunctionBegin;

  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));
  PetscCall(CheckBoundaryNumEdges(rdy, boundary_index, num_edges));

  RDyBoundary boundary = rdy->boundaries[boundary_index];
  RDyEdges   *edges    = &rdy->mesh.edges;

  if (data_for_edge) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      PetscInt iedge = boundary.edge_ids[e];
      x[e]           = edges->centroids[iedge].X[idim];
    }
  } else {
    RDyCells *cells = &rdy->mesh.cells;
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      PetscInt iedge = boundary.edge_ids[e];
      PetscInt icell = edges->cell_ids[2 * iedge];
      x[e]           = cells->centroids[icell].X[idim];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryEdgeXCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_TRUE;
  PetscInt  idim          = 0;  // x-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryEdgeYCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_TRUE;
  PetscInt  idim          = 1;  // y-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryEdgeZCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_TRUE;
  PetscInt  idim          = 2;  // z-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellXCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_FALSE;
  PetscInt  idim          = 0;  // x-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellYCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_FALSE;
  PetscInt  idim          = 1;  // y-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellZCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal values[size]) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_FALSE;
  PetscInt  idim          = 2;  // z-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellNaturalIDs(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscInt values[size]) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));
  PetscCall(CheckBoundaryNumEdges(rdy, boundary_index, size));

  RDyBoundary boundary = rdy->boundaries[boundary_index];
  RDyCells   *cells    = &rdy->mesh.cells;
  RDyEdges   *edges    = &rdy->mesh.edges;

  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt iedge = boundary.edge_ids[e];
    PetscInt icell = edges->cell_ids[2 * iedge];
    values[e]      = cells->natural_ids[icell];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellManningsNs(RDy rdy, const PetscInt size, PetscReal n_values[size]) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  if (rdy->ceed_resource[0]) {  // ceed
    PetscCall(SWESourceOperatorSetManningsN(rdy->ceed_rhs.op_src, n_values));
  } else {  // petsc
    for (PetscInt icell = 0; icell < rdy->mesh.num_cells_local; ++icell) {
      n_values[icell] = rdy->materials_by_cell[icell].manning;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetManningsNForLocalCells(RDy rdy, const PetscInt size, PetscReal n_values[size]) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  if (rdy->ceed_resource[0]) {  // ceed
    PetscCall(SWESourceOperatorSetManningsN(rdy->ceed_rhs.op_src, n_values));
  } else {  // petsc
    for (PetscInt icell = 0; icell < rdy->mesh.num_cells_local; ++icell) {
      rdy->materials_by_cell[icell].manning = n_values[icell];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetInitialConditions(RDy rdy, Vec ic) {
  PetscFunctionBegin;
  PetscCall(VecCopy(ic, rdy->X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyCreatePrognosticVec(RDy rdy, Vec *prog_vec) {
  PetscFunctionBegin;
  PetscCall(VecDuplicate(rdy->X, prog_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}
