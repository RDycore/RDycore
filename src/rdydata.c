#include <private/rdycoreimpl.h>
#include <private/rdysweimpl.h>
#include <rdycore.h>

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

static PetscErrorCode CheckBoundaryConditionIndex(RDy rdy, PetscInt boundary_index) {
  PetscFunctionBegin;
  PetscCheck(boundary_index < rdy->num_boundaries, rdy->comm, PETSC_ERR_USER,
             "Boundary condition index (%d) exceeds the max number of boundary conditions (%d)", boundary_index, rdy->num_boundaries);
  PetscCheck(boundary_index >= 0, rdy->comm, PETSC_ERR_USER, "Boundary condition index (%d) cannot be less than zero.", boundary_index);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetNumBoundaryEdges(RDy rdy, PetscInt boundary_index, PetscInt *num_edges) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));
  RDyBoundary *boundary = &rdy->boundaries[boundary_index];
  *num_edges            = boundary->num_edges;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetBoundaryConditionFlowType(RDy rdy, PetscInt boundary_index, PetscInt *bc_type) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));
  RDyCondition *boundary_cond = &rdy->boundary_conditions[boundary_index];
  *bc_type                    = boundary_cond->flow->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetDirichletBoundaryValues(RDy rdy, PetscInt boundary_index, PetscInt num_edges, PetscInt ndof, PetscReal *boundary_values) {
  PetscFunctionBegin;

  PetscCall(CheckBoundaryConditionIndex(rdy, boundary_index));

  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "The number of DOFs (%d) for the boundary condition need to be three.", ndof);

  RDyBoundary boundary = rdy->boundaries[boundary_index];
  PetscCheck(boundary.num_edges == num_edges, rdy->comm, PETSC_ERR_USER,
             "The given number of edges (%d) for boundary with index %d is incorrect (should be %d)", num_edges, boundary_index, boundary.num_edges);

  RDyCondition boundary_cond = rdy->boundary_conditions[boundary_index];
  PetscCheck(boundary_cond.flow->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
             "Trying to set dirichlet values for boundary with index %d, but it has a different type (%d)", boundary_index, boundary_cond.flow->type);

  // dispatch this call to CEED or PETSc
  PetscReal tiny_h = rdy->config.physics.flow.tiny_h;
  if (rdy->ceed_resource[0]) {  // ceed
    PetscCall(SWEFluxOperatorSetDirichletBoundaryValues(rdy->ceed_rhs.op_edges, rdy->boundaries[boundary_index], boundary_values));
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
        bdata.h[e] = boundary_values[3 * e];

        if (bdata.h[e] > tiny_h) {
          bdata.u[e] = boundary_values[3 * e + 1] / bdata.h[e];
          bdata.v[e] = boundary_values[3 * e + 2] / bdata.h[e];
        } else {
          bdata.u[e] = 0.0;
          bdata.v[e] = 0.0;
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetHeight(RDy rdy, PetscReal *h) {
  PetscFunctionBegin;

  PetscReal *x;
  PetscCall(VecGetArray(rdy->X, &x));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    h[i] = x[3 * i];
  }
  PetscCall(VecRestoreArray(rdy->X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetXVelocity(RDy rdy, PetscReal *vx) {
  PetscFunctionBegin;

  PetscReal *x;
  PetscCall(VecGetArray(rdy->X, &x));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    vx[i] = x[3 * i + 1];
  }
  PetscCall(VecRestoreArray(rdy->X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDyGetYVelocity(RDy rdy, PetscReal *vy) {
  PetscFunctionBegin;

  PetscReal *x;
  PetscCall(VecGetArray(rdy->X, &x));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    vy[i] = x[3 * i + 2];
  }
  PetscCall(VecRestoreArray(rdy->X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RDySetWaterSource(RDy rdy, PetscReal *watsrc) {
  PetscFunctionBegin;

  if (rdy->ceed_resource[0]) {  // ceed
    PetscCall(SWESourceOperatorSetWaterSource(rdy->ceed_rhs.op_src, watsrc));
  } else {  // petsc
    PetscReal *s;
    PetscCall(VecGetArray(rdy->water_src, &s));
    for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
      s[i] = watsrc[i];
    }
    PetscCall(VecRestoreArray(rdy->water_src, &s));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
