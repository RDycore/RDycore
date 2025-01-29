#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysweimpl.h>
#include <rdycore.h>

PetscErrorCode RDyGetNumGlobalCells(RDy rdy, PetscInt *num_cells_global) {
  PetscFunctionBegin;
  *num_cells_global = rdy->mesh.num_cells_global;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetNumLocalCells(RDy rdy, PetscInt *num_cells) {
  PetscFunctionBegin;
  *num_cells = rdy->mesh.num_owned_cells;
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

static PetscErrorCode CheckRegionIndex(RDy rdy, const PetscInt region_index) {
  PetscFunctionBegin;
  PetscCheck(region_index < rdy->num_regions, rdy->comm, PETSC_ERR_USER,
             "Region index (%" PetscInt_FMT ") exceeds the max number of regions (%" PetscInt_FMT ")", region_index, rdy->num_regions);
  PetscCheck(region_index >= 0, rdy->comm, PETSC_ERR_USER, "Region index (%" PetscInt_FMT ") cannot be less than zero.", region_index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckNumLocalCells(RDy rdy, const PetscInt size) {
  PetscFunctionBegin;
  PetscAssert(rdy->mesh.num_owned_cells == size, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "The size of array is not equal to the number of local cells");
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

// sets Dirichlet boundary values using the data in the strided array
// values[num_edges * ndof]
PetscErrorCode RDySetFlowDirichletBoundaryValues(RDy rdy, const PetscInt boundary_index, const PetscInt num_edges, const PetscInt ndof,
                                                 PetscReal *values) {
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

  OperatorData dirichlet;
  PetscCall(GetOperatorBoundaryValues(rdy->operator, boundary, &dirichlet));
  for (PetscInt comp = 0; comp < ndof; ++comp) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      dirichlet.values[comp][e] = values[ndof * e + comp];
    }
  }
  PetscCall(RestoreOperatorBoundaryValues(rdy->operator, boundary, &dirichlet));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetSedimentDirichletBoundaryValues(RDy rdy, const PetscInt boundary_index, const PetscInt num_edges, const PetscInt ndof,
                                                     PetscReal *values) {
  PetscFunctionBegin;

  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));

  RDyBoundary boundary = rdy->boundaries[boundary_index];
  PetscCheck(boundary.num_edges == num_edges, rdy->comm, PETSC_ERR_USER,
             "The given number of edges (%" PetscInt_FMT ") for boundary with index %" PetscInt_FMT " is incorrect (should be %" PetscInt_FMT ")",
             num_edges, boundary_index, boundary.num_edges);

  RDyCondition boundary_cond = rdy->boundary_conditions[boundary_index];
  for (PetscInt i = 0; i < rdy->num_sediment_classes; ++i) {
    PetscCheck(boundary_cond.sediment[i]->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
               "Trying to set dirichlet values for boundary with index %" PetscInt_FMT ", but it has a different type (%u)", boundary_index,
               boundary_cond.sediment[i]->type);
  }

  OperatorData dirichlet;
  PetscCall(GetOperatorBoundaryValues(rdy->operator, boundary, &dirichlet));

  PetscInt comp_offset = 3;
  PetscCheck(rdy->config.physics.flow.mode == FLOW_SWE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend the code to set offset correctly");

  for (PetscInt comp = 0; comp < ndof; ++comp) {
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      dirichlet.values[comp + comp_offset][e] = values[ndof * e + comp];
    }
  }
  PetscCall(RestoreOperatorBoundaryValues(rdy->operator, boundary, &dirichlet));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyGetPrognosticVariableOfLocalCell(RDy rdy, PetscInt idof, PetscReal *values) {
  PetscFunctionBegin;

  PetscReal *u;
  PetscCall(VecGetArray(rdy->u_global, &u));
  for (PetscInt i = 0; i < rdy->mesh.num_owned_cells; ++i) {
    values[i] = u[3 * i + idof];
  }
  PetscCall(VecRestoreArray(rdy->u_global, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The following functions retrieve single-component solution data on local
// cells, placing them into the values array (of length size)

PetscErrorCode RDyGetLocalCellHeights(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));
  PetscCall(RDyGetPrognosticVariableOfLocalCell(rdy, 0, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellXMomenta(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));
  PetscCall(RDyGetPrognosticVariableOfLocalCell(rdy, 1, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellYMomenta(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));
  PetscCall(RDyGetPrognosticVariableOfLocalCell(rdy, 2, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetRegionalSourceComponent(RDy rdy, const PetscInt region_idx, PetscInt component, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(CheckRegionIndex(rdy, region_idx));
  RDyRegion region = rdy->regions[region_idx];
  PetscCheck(size == region.num_owned_cells, rdy->comm, PETSC_ERR_USER,
             "Wrong size (%" PetscInt_FMT ") for owned cells in region %" PetscInt_FMT " (%" PetscInt_FMT ")", size, region_idx,
             region.num_owned_cells);

  if (region.num_owned_cells) {
    OperatorData source_data;
    PetscCall(GetOperatorRegionalExternalSource(rdy->operator, region, &source_data));
    for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
      source_data.values[component][c] = values[c];
    }
    PetscCall(RestoreOperatorRegionalExternalSource(rdy->operator, region, &source_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external water source term on the region with the given index to
/// the values in the given array (whose length is equal to the number of owned
/// cells in that region).
PetscErrorCode RDySetRegionalWaterSource(RDy rdy, const PetscInt region_idx, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(SetRegionalSourceComponent(rdy, region_idx, 0, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external x-momentum term on the region with the given index to
/// the values in the given array (whose length is equal to the number of owned
/// cells in that region).
PetscErrorCode RDySetRegionalXMomentumSource(RDy rdy, const PetscInt region_idx, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(SetRegionalSourceComponent(rdy, region_idx, 1, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external y-momentum term on the region with the given index to
/// the values in the given array (whose length is equal to the number of owned
/// cells in that region).
PetscErrorCode RDySetRegionalYMomentumSource(RDy rdy, const PetscInt region_idx, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(SetRegionalSourceComponent(rdy, region_idx, 2, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetRegionalSedimentSource(RDy rdy, const PetscInt region_idx, PetscInt size_class, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(SetRegionalSourceComponent(rdy, region_idx, 3 + size_class, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetHomogeneousRegionalSourceComponent(RDy rdy, const PetscInt region_idx, PetscInt component, PetscReal value) {
  PetscFunctionBegin;
  PetscCall(CheckRegionIndex(rdy, region_idx));

  RDyRegion region = rdy->regions[region_idx];
  if (region.num_owned_cells) {
    OperatorData source_data;
    PetscCall(GetOperatorRegionalExternalSource(rdy->operator, region, &source_data));
    for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
      source_data.values[component][c] = value;
    }
    PetscCall(RestoreOperatorRegionalExternalSource(rdy->operator, region, &source_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external water source term on the region with the given index to
/// the single (homogeneous) value.
PetscErrorCode RDySetHomogeneousRegionalWaterSource(RDy rdy, const PetscInt region_idx, PetscReal value) {
  PetscFunctionBegin;
  PetscCall(SetHomogeneousRegionalSourceComponent(rdy, region_idx, 0, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external x-momentum source term on the region with the given index
/// to the single (homogeneous) value.
PetscErrorCode RDySetHomogeneousXMomentumSource(RDy rdy, const PetscInt region_idx, PetscReal value) {
  PetscFunctionBegin;
  PetscCall(SetHomogeneousRegionalSourceComponent(rdy, region_idx, 1, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external y-momentum source term on the region with the given index
/// to the single (homogeneous) value.
PetscErrorCode RDySetHomogeneousYMomentumSource(RDy rdy, const PetscInt region_idx, PetscReal value) {
  PetscFunctionBegin;
  PetscCall(SetHomogeneousRegionalSourceComponent(rdy, region_idx, 2, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetDomainSourceComponent(RDy rdy, PetscInt component, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCheck(size == rdy->mesh.num_owned_cells, rdy->comm, PETSC_ERR_USER,
             "Wrong size (%" PetscInt_FMT ") for owned cells in domain (%" PetscInt_FMT ")", size, rdy->mesh.num_owned_cells);
  OperatorData source_data;
  PetscCall(GetOperatorDomainExternalSource(rdy->operator, & source_data));
  for (PetscInt c = 0; c < rdy->mesh.num_owned_cells; ++c) {
    source_data.values[component][c] = values[c];
  }
  PetscCall(RestoreOperatorDomainExternalSource(rdy->operator, & source_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external water source term on the entire domain to the values in
/// the given array (whose length is equal to the number of owned cells in the
/// domain).
PetscErrorCode RDySetDomainWaterSource(RDy rdy, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(SetDomainSourceComponent(rdy, 0, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external x-momentum term on the entire domain to the values in the
/// given array (whose length is equal to the number of owned cells in the
/// domain).
PetscErrorCode RDySetDomainXMomentumSource(RDy rdy, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(SetDomainSourceComponent(rdy, 1, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the external y-momentum term on the entire domain to the values in the
/// given array (whose length is equal to the number of owned cells in the
/// domain).
PetscErrorCode RDySetDomainYMomentumSource(RDy rdy, PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(SetDomainSourceComponent(rdy, 2, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RDyGetIDimCentroidOfLocalCell(RDy rdy, PetscInt idim, PetscInt size, PetscReal *x) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  RDyCells *cells = &rdy->mesh.cells;

  PetscInt count = 0;
  for (PetscInt icell = 0; icell < rdy->mesh.num_cells; ++icell) {
    if (cells->is_owned[icell]) {
      x[count++] = cells->centroids[icell].X[idim];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellXCentroids(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscInt idim = 0;  // x-dim
  PetscCall(RDyGetIDimCentroidOfLocalCell(rdy, idim, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellYCentroids(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscInt idim = 1;  // y-dim
  PetscCall(RDyGetIDimCentroidOfLocalCell(rdy, idim, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellZCentroids(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscInt idim = 2;  // z-dim
  PetscCall(RDyGetIDimCentroidOfLocalCell(rdy, idim, size, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellAreas(RDy rdy, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscCall(CheckNumLocalCells(rdy, size));

  RDyCells *cells = &rdy->mesh.cells;

  PetscInt count = 0;
  for (PetscInt icell = 0; icell < rdy->mesh.num_cells; ++icell) {
    if (cells->is_owned[icell]) {
      values[count++] = cells->areas[icell];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetLocalCellNaturalIDs(RDy rdy, const PetscInt size, PetscInt *values) {
  PetscFunctionBegin;

  PetscCall(CheckNumLocalCells(rdy, size));

  RDyCells *cells = &rdy->mesh.cells;

  PetscInt count = 0;
  for (PetscInt icell = 0; icell < rdy->mesh.num_cells; ++icell) {
    if (cells->is_owned[icell]) {
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

PetscErrorCode RDyGetBoundaryID(RDy rdy, const PetscInt boundary_index, PetscInt *id) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));
  *id = rdy->boundaries[boundary_index].id;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryEdgeXCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_TRUE;
  PetscInt  idim          = 0;  // x-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryEdgeYCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_TRUE;
  PetscInt  idim          = 1;  // y-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryEdgeZCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_TRUE;
  PetscInt  idim          = 2;  // z-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellXCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_FALSE;
  PetscInt  idim          = 0;  // x-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellYCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_FALSE;
  PetscInt  idim          = 1;  // y-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellZCentroids(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscReal *values) {
  PetscFunctionBegin;
  PetscBool data_for_edge = PETSC_FALSE;
  PetscInt  idim          = 2;  // z-dim
  PetscCall(RDyGetIDimCentroidOfBoundaryEdgeOrCell(rdy, boundary_index, size, data_for_edge, idim, values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryCellNaturalIDs(RDy rdy, const PetscInt boundary_index, const PetscInt size, PetscInt *values) {
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

PetscErrorCode RDySetRegionalManningsN(RDy rdy, const PetscInt region_index, const PetscInt size, PetscReal *n_values) {
  PetscFunctionBegin;

  PetscCall(CheckRegionIndex(rdy, region_index));
  RDyRegion region = rdy->regions[region_index];
  PetscCheck(size == region.num_owned_cells, rdy->comm, PETSC_ERR_USER,
             "Wrong size (%" PetscInt_FMT ") for owned cells in region %" PetscInt_FMT " (%" PetscInt_FMT ")", size, region_index,
             region.num_owned_cells);

  OperatorData mannings_data;
  PetscCall(GetOperatorRegionalMaterialProperty(rdy->operator, region, OPERATOR_MANNINGS, &mannings_data));
  for (PetscInt c = 0; c < region.num_owned_cells; ++c) {
    mannings_data.values[0][c] = n_values[c];
  }
  PetscCall(RestoreOperatorRegionalMaterialProperty(rdy->operator, region, OPERATOR_MANNINGS, &mannings_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetDomainManningsN(RDy rdy, const PetscInt size, PetscReal *n_values) {
  PetscFunctionBegin;

  PetscCheck(size == rdy->mesh.num_owned_cells, rdy->comm, PETSC_ERR_USER,
             "Wrong size (%" PetscInt_FMT ") for owned cells in domain (%" PetscInt_FMT ")", size, rdy->mesh.num_owned_cells);

  OperatorData mannings_data;
  PetscCall(GetOperatorDomainMaterialProperty(rdy->operator, OPERATOR_MANNINGS, &mannings_data));
  for (PetscInt c = 0; c < rdy->mesh.num_owned_cells; ++c) {
    mannings_data.values[0][c] = n_values[c];
  }
  PetscCall(RestoreOperatorDomainMaterialProperty(rdy->operator, OPERATOR_MANNINGS, &mannings_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetInitialConditions(RDy rdy, Vec ic) {
  PetscFunctionBegin;
  PetscCall(VecCopy(ic, rdy->u_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyCreatePrognosticVec(RDy rdy, Vec *prog_vec) {
  PetscFunctionBegin;
  PetscCall(VecDuplicate(rdy->u_global, prog_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyCreateOneDOFGlobalVec(RDy rdy, Vec *global) {
  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector(rdy->aux_dm, global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyWriteOneDOFGlobalVecToBinaryFile(RDy rdy, const char filename[], Vec *global) {
  PetscFunctionBegin;

  // create a naturally-ordered vector with a stride equal to the number of
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(rdy->aux_dm, &natural));

  // scatter global-to-natural
  PetscCall(DMPlexGlobalToNaturalBegin(rdy->aux_dm, *global, natural));
  PetscCall(DMPlexGlobalToNaturalEnd(rdy->aux_dm, *global, natural));

  // write the data to file
  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(natural, viewer));

  // free up memory
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&natural));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// reads data for a single DOF from a binary file into a global Vec
PetscErrorCode RDyReadOneDOFGlobalVecFromBinaryFile(RDy rdy, const char filename[], Vec *global) {
  PetscFunctionBegin;

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(rdy->comm, filename, FILE_MODE_READ, &viewer));

  // create a naturally-ordered vector with a stride equal to the number of
  Vec natural;

  PetscCall(DMPlexCreateNaturalVector(rdy->aux_dm, &natural));
  PetscCall(DMCreateGlobalVector(rdy->aux_dm, global));

  // load the properties into the vector and copy them into place
  PetscCall(VecLoad(natural, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // scatter natural-to-global
  PetscCall(DMPlexNaturalToGlobalBegin(rdy->aux_dm, natural, *global));
  PetscCall(DMPlexNaturalToGlobalEnd(rdy->aux_dm, natural, *global));

  PetscCall(VecDestroy(&natural));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// reads data for a single DOF from a binary file into a local Vec
PetscErrorCode RDyReadOneDOFLocalVecFromBinaryFile(RDy rdy, const char filename[], Vec *local) {
  PetscFunctionBegin;

  Vec global;
  PetscCall(RDyReadOneDOFGlobalVecFromBinaryFile(rdy, filename, &global));

  PetscCall(DMCreateLocalVector(rdy->aux_dm, local));

  // scatter global-to-local
  PetscCall(DMGlobalToLocalBegin(rdy->aux_dm, global, INSERT_VALUES, *local));
  PetscCall(DMGlobalToLocalEnd(rdy->aux_dm, global, INSERT_VALUES, *local));

  PetscCall(VecDestroy(&global));

  PetscFunctionReturn(PETSC_SUCCESS);
}
