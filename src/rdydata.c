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

PetscErrorCode RDyGetNumBoundaryConditionEdges(RDy rdy, PetscInt bnd_cond_id, PetscInt *num_edges) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionID(rdy, bnd_cond_id));
  RDyBoundary *boundary = &rdy->boundaries[bnd_cond_id];
  *num_edges            = boundary->num_edges;
  PetscFunctionReturn(0);
}

PetscErrorCode RDyGetBoundaryConditionFlowType(RDy rdy, PetscInt bnd_cond_id, PetscInt *bc_type) {
  PetscFunctionBegin;
  PetscCall(CheckBoundaryConditionID(rdy, bnd_cond_id));
  RDyCondition *boundary_cond = &rdy->boundary_conditions[bnd_cond_id];
  *bc_type                    = boundary_cond->flow->type;
  PetscFunctionReturn(0);
}

PetscErrorCode RDySetDirichletBoundaryConditionValues(RDy rdy, PetscInt bnd_cond_id, PetscInt num_edges, PetscInt ndof, PetscReal *bc_values) {
  PetscFunctionBegin;

  PetscCall(CheckBoundaryConditionID(rdy, bnd_cond_id));

  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "The number of DOFs (%d) for the boundary condition ID (%d) need to be three.", ndof, bnd_cond_id);

  RDyBoundary *boundary = &rdy->boundaries[bnd_cond_id];
  PetscCheck(boundary->num_edges == num_edges, rdy->comm, PETSC_ERR_USER,
             "The number of edges (%d) in the data to set boundary condition for ID = %d, do not match the actual number of edge (%d)", num_edges,
             bnd_cond_id, boundary->num_edges);

  RDyCondition *boundary_cond = &rdy->boundary_conditions[bnd_cond_id];
  PetscCheck(boundary_cond->flow->type == CONDITION_DIRICHLET, rdy->comm, PETSC_ERR_USER,
             "Trying to set dirichlet values for boundary condition (%d), but it is of a different type (%d)", bnd_cond_id,
             boundary_cond->flow->type);

  RDyCells            *cells    = &rdy->mesh.cells;
  RDyEdges            *edges    = &rdy->mesh.edges;
  PetscRiemannDataSWE *data_swe = rdy->petsc_rhs;
  RiemannDataSWE      *datar    = &data_swe->datar_bnd_edges[bnd_cond_id];
  PetscReal            tiny_h   = rdy->config.physics.flow.tiny_h;

  for (PetscInt e = 0; e < boundary->num_edges; ++e) {
    PetscInt iedge = boundary->edge_ids[e];
    PetscInt icell = edges->cell_ids[2 * iedge];

    if (cells->is_local[icell]) {
      datar->h[e] = bc_values[e * ndof];

      if (datar->h[e] > tiny_h) {
        datar->u[e] = bc_values[e * ndof + 1] / datar->h[e];
        datar->v[e] = bc_values[e * ndof + 2] / datar->h[e];
      } else {
        datar->u[e] = 0.0;
        datar->v[e] = 0.0;
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
