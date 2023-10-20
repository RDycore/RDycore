#include <private/rdycoreimpl.h>
#include <private/rdysweimpl.h>
#include <rdycore.h>

PetscErrorCode RDyGetNumLocalCells(RDy rdy, PetscInt *num_cells) {
  PetscFunctionBegin;
  *num_cells = rdy->mesh.num_cells_local;
  PetscFunctionReturn(0);
}

PetscErrorCode RDyGetNumBoundaryConditions(RDy rdy, PetscInt *num_bnd_conds) {
  PetscFunctionBegin;
  *num_bnd_conds = rdy->num_boundaries;
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckBoundaryConditionID(RDy rdy, PetscInt bnd_cond_id) {
  PetscFunctionBegin;
  PetscCheck(bnd_cond_id < rdy->num_boundaries, rdy->comm, PETSC_ERR_USER,
             "Boundary condition ID (%d) exceeds the max number of boundary conditions (%d)", bnd_cond_id, rdy->num_boundaries);

  PetscCheck(bnd_cond_id >= 0, rdy->comm, PETSC_ERR_USER, "Boundary condition ID (%d) cannot be less than zero.", bnd_cond_id);
  PetscFunctionReturn(0);
}

PetscErrorCode RDyGetNumBoundaryEdges(RDy rdy, PetscInt boundary_id, PetscInt *num_edges) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionID(rdy, boundary_id));
  RDyBoundary *boundary = &rdy->boundaries[boundary_id];
  *num_edges            = boundary->num_edges;
  PetscFunctionReturn(0);
}

PetscErrorCode RDyGetBoundaryConditionFlowType(RDy rdy, PetscInt boundary_id, PetscInt *bc_type) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionID(rdy, boundary_id));
  RDyCondition *boundary_cond = &rdy->boundary_conditions[boundary_id];
  *bc_type                    = boundary_cond->flow->type;
  PetscFunctionReturn(0);
}

PetscErrorCode RDySetDirichletBoundaryValues(RDy rdy, PetscInt boundary_id, PetscInt num_edges, PetscInt ndof, PetscReal *boundary_values) {
  PetscFunctionBegin;

  PetscCall(CheckBoundaryConditionID(rdy, boundary_id));

  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "The number of DOFs (%d) for the boundary condition need to be three.", ndof);

  // find the index of the boundary with the given ID
  PetscInt b_index = -1;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    if (rdy->boundaries[b].id == boundary_id) {
      b_index = b;
      break;
    }
  }
  PetscCheck(b_index != -1, rdy->comm, PETSC_ERR_USER, "Invalid boundary ID: %d", boundary_id);

  RDyBoundary boundary = rdy->boundaries[b_index];
  PetscCheck(boundary.num_edges == num_edges, rdy->comm, PETSC_ERR_USER, "The given number of edges (%d) for boundary %d is incorrect (should be %d)",
             num_edges, boundary_id, boundary.num_edges);

  RDyCondition boundary_cond = rdy->boundary_conditions[b_index];
  PetscCheck(boundary_cond.flow->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
             "Trying to set dirichlet values for boundary %d, but it has a different type (%d)", boundary_id, boundary_cond.flow->type);

  // dispatch this call to CEED or PETSc
  PetscReal tiny_h = rdy->config.physics.flow.tiny_h;
  if (rdy->ceed_resource[0]) {  // ceed
    // fetch the array storing the boundary values
    CeedOperatorField dirichlet_field;
    PetscCall(GetSWEFluxOperatorDirichletBoundaryValues(rdy->ceed_rhs.op_edges, b_index, &dirichlet_field));
    CeedVector dirichlet_vector;
    CeedOperatorFieldGetVector(dirichlet_field, &dirichlet_vector);
    PetscInt num_comp = 3;
    CeedScalar(*dirichlet_ceed)[num_comp];
    CeedVectorGetArray(dirichlet_vector, CEED_MEM_HOST, (CeedScalar **)&dirichlet_ceed);

    // set the boundary values
    for (PetscInt i = 0; i < boundary.num_edges; ++i) {
      dirichlet_ceed[i][0] = boundary_values[num_comp * i];
      dirichlet_ceed[i][1] = boundary_values[num_comp * i + 1];
      dirichlet_ceed[i][2] = boundary_values[num_comp * i + 2];
    }

    // copy the values into the CEED operator
    CeedVectorRestoreArray(dirichlet_vector, (CeedScalar **)&dirichlet_ceed);
  } else {  // petsc
    // fetch the boundary data
    RiemannDataSWE bdata;
    PetscCall(GetPetscSWEDirichletBoundaryValues(rdy->petsc_rhs, b_index, &bdata));

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
  PetscFunctionReturn(0);
}

PetscErrorCode RDyGetHeight(RDy rdy, PetscReal *h) {
  PetscFunctionBegin;

  PetscReal *x;
  PetscCall(VecGetArray(rdy->X, &x));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    h[i] = x[3 * i];
  }
  PetscCall(VecRestoreArray(rdy->X, &x));
  PetscFunctionReturn(0);
}

PetscErrorCode RDyGetXVelocity(RDy rdy, PetscReal *vx) {
  PetscFunctionBegin;

  PetscReal *x;
  PetscCall(VecGetArray(rdy->X, &x));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    vx[i] = x[3 * i + 1];
  }
  PetscCall(VecRestoreArray(rdy->X, &x));
  PetscFunctionReturn(0);
}

PetscErrorCode RDyGetYVelocity(RDy rdy, PetscReal *vy) {
  PetscFunctionBegin;

  PetscReal *x;
  PetscCall(VecGetArray(rdy->X, &x));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    vy[i] = x[3 * i + 2];
  }
  PetscCall(VecRestoreArray(rdy->X, &x));
  PetscFunctionReturn(0);
}

PetscErrorCode RDySetWaterSource(RDy rdy, PetscReal *watsrc) {
  PetscFunctionBegin;

  PetscReal *s;
  PetscCall(VecGetArray(rdy->water_src, &s));
  for (PetscInt i = 0; i < rdy->mesh.num_cells_local; ++i) {
    s[i] = watsrc[i];
  }
  PetscCall(VecRestoreArray(rdy->water_src, &s));
  rdy->ceed_rhs.water_src_updated = PETSC_FALSE;
  PetscFunctionReturn(0);
}
