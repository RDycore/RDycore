#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

#include "swe_roe_petsc_impl.h"

static PetscErrorCode CreateRiemannStateData(PetscInt num_states, RiemannStateData *data) {
  PetscFunctionBegin;

  data->num_states = num_states;
  PetscCall(PetscCalloc1(num_states, &data->h));
  PetscCall(PetscCalloc1(num_states, &data->hu));
  PetscCall(PetscCalloc1(num_states, &data->hv));
  PetscCall(PetscCalloc1(num_states, &data->u));
  PetscCall(PetscCalloc1(num_states, &data->v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyRiemannStateData(RiemannStateData data) {
  PetscFunctionBegin;

  data.num_states = 0;
  PetscCall(PetscFree(data.h));
  PetscCall(PetscFree(data.hu));
  PetscCall(PetscFree(data.hv));
  PetscCall(PetscFree(data.u));
  PetscCall(PetscFree(data.v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateRiemannEdgeData(PetscInt num_edges, PetscInt num_comp, RiemannEdgeData *data) {
  PetscFunctionBegin;

  data->num_edges = num_edges;
  PetscCall(PetscCalloc1(num_edges, &data->cn));
  PetscCall(PetscCalloc1(num_edges, &data->sn));
  PetscCall(PetscCalloc1(num_edges * num_comp, &data->fluxes));
  PetscCall(PetscCalloc1(num_edges, &data->amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyRiemannEdgeData(RiemannEdgeData data) {
  PetscFunctionBegin;

  data.num_edges = 0;
  PetscCall(PetscFree(data.cn));
  PetscCall(PetscFree(data.sn));
  PetscCall(PetscFree(data.fluxes));
  PetscCall(PetscFree(data.amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRiemannVelocities(PetscReal tiny_h, PetscReal h_anuga, RiemannStateData *data) {
  PetscFunctionBeginUser;
  PetscReal denom;

  for (PetscInt n = 0; n < data->num_states; n++) {
    if (data->h[n] < tiny_h) {
      data->u[n] = 0.0;
      data->v[n] = 0.0;
    } else {
      denom      = Square(data->h[n]) + Square(h_anuga);
      data->u[n] = data->hu[n] * data->h[n] / denom;
      data->v[n] = data->hv[n] * data->h[n] / denom;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Interior Flux Operator
//------------------------

typedef struct {
  RDyNumericsRiemann   riemann;          // Riemann solver method
  RDyMesh             *mesh;             // domain mesh
  PetscReal            tiny_h;           // minimum water height for wet conditions
  PetscReal            h_anuga_regular;  // ANUGA height parameter used in velocity regularization
  RiemannStateData     left_states;      // "left" riemann states on interior edges
  RiemannStateData     right_states;     // "right" riemann states on interior edges
  RiemannEdgeData      edges;            // riemann fluxes on interior edges
  OperatorDiagnostics *diagnostics;      // courant number, etc
} InteriorFluxOperator;

static PetscErrorCode ApplyInteriorFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  InteriorFluxOperator *interior_flux_op = context;

  RDyMesh  *mesh  = interior_flux_op->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // get pointers to vector data
  PetscScalar *u_ptr, *f_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == 3, comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  RiemannStateData *datal        = &interior_flux_op->left_states;
  RiemannStateData *datar        = &interior_flux_op->right_states;
  RiemannEdgeData  *data_edge    = &interior_flux_op->edges;
  PetscReal        *sn_vec_int   = data_edge->sn;
  PetscReal        *cn_vec_int   = data_edge->cn;
  PetscReal        *amax_vec_int = data_edge->amax;
  PetscReal        *flux_vec_int = data_edge->fluxes;

  // Collect the h/hu/hv for left and right cells to compute u/v
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id             = edges->internal_edge_ids[e];
    PetscInt left_local_cell_id  = edges->cell_ids[2 * edge_id];
    PetscInt right_local_cell_id = edges->cell_ids[2 * edge_id + 1];

    if (right_local_cell_id != -1) {
      datal->h[e]  = u_ptr[n_dof * left_local_cell_id + 0];
      datal->hu[e] = u_ptr[n_dof * left_local_cell_id + 1];
      datal->hv[e] = u_ptr[n_dof * left_local_cell_id + 2];

      datar->h[e]  = u_ptr[n_dof * right_local_cell_id + 0];
      datar->hu[e] = u_ptr[n_dof * right_local_cell_id + 1];
      datar->hv[e] = u_ptr[n_dof * right_local_cell_id + 2];
    }
  }

  const PetscReal tiny_h  = interior_flux_op->tiny_h;
  const PetscReal h_anuga = interior_flux_op->h_anuga_regular;
  PetscCall(ComputeRiemannVelocities(tiny_h, h_anuga, datal));
  PetscCall(ComputeRiemannVelocities(tiny_h, h_anuga, datar));

  // call Riemann solver
  switch (interior_flux_op->riemann) {
    case RIEMANN_ROE:
      PetscCall(ComputeSWERoeFlux(datal, datar, sn_vec_int, cn_vec_int, flux_vec_int, amax_vec_int));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Unsupported Riemann solver");
  }

  // accummulate the flux values in the global flux vector
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id             = edges->internal_edge_ids[e];
    PetscInt left_local_cell_id  = edges->cell_ids[2 * edge_id];
    PetscInt right_local_cell_id = edges->cell_ids[2 * edge_id + 1];

    if (right_local_cell_id != -1) {  // internal edge
      PetscReal edge_len = edges->lengths[edge_id];

      PetscReal hl = u_ptr[n_dof * left_local_cell_id + 0];
      PetscReal hr = u_ptr[n_dof * right_local_cell_id + 0];

      if (!(hr < tiny_h && hl < tiny_h)) {  // either cell is "wet"
        PetscReal areal = cells->areas[left_local_cell_id];
        PetscReal arear = cells->areas[right_local_cell_id];

        PetscReal                 cnum              = amax_vec_int[e] * edge_len / fmin(areal, arear) * dt;
        CourantNumberDiagnostics *courant_num_diags = &interior_flux_op->diagnostics->courant_number;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[e];
          if (areal < arear) courant_num_diags->global_cell_id = cells->global_ids[left_local_cell_id];
          else courant_num_diags->global_cell_id = cells->global_ids[right_local_cell_id];
        }

        for (PetscInt i_dof = 0; i_dof < n_dof; i_dof++) {
          if (cells->is_owned[left_local_cell_id]) {
            PetscInt left_owned_cell_id = cells->local_to_owned[left_local_cell_id];
            f_ptr[n_dof * left_owned_cell_id + i_dof] += flux_vec_int[n_dof * e + i_dof] * (-edge_len / areal);
          }

          if (cells->is_owned[right_local_cell_id]) {
            PetscInt right_owned_cell_id = cells->local_to_owned[right_local_cell_id];
            f_ptr[n_dof * right_owned_cell_id + i_dof] += flux_vec_int[n_dof * e + i_dof] * (edge_len / arear);
          }
        }
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyInteriorFlux(void *context) {
  PetscFunctionBegin;
  InteriorFluxOperator *interior_flux_op = context;
  DestroyRiemannStateData(interior_flux_op->left_states);
  DestroyRiemannStateData(interior_flux_op->right_states);
  DestroyRiemannEdgeData(interior_flux_op->edges);
  PetscCall(PetscFree(interior_flux_op));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that computes fluxes between pairs of cells on the
/// domain's interior, suitable for the shallow water equations.
/// @param [in]    mesh        mesh defining the computational domain of the operator
/// @param [in]    config      RDycore's configuration
/// @param [inout] diagnostics a set of diagnostics that can be updated by the PetscOperator
/// @param [out]   petsc_op    the newly created PetscOperator
PetscErrorCode CreateSWEPetscInteriorFluxOperator(RDyMesh *mesh, const RDyConfig config, OperatorDiagnostics *diagnostics, PetscOperator *petsc_op) {
  PetscFunctionBegin;

  const PetscInt num_comp = 3;

  InteriorFluxOperator *interior_flux_op;
  PetscCall(PetscCalloc1(1, &interior_flux_op));
  *interior_flux_op = (InteriorFluxOperator){
      .riemann         = config.numerics.riemann,
      .mesh            = mesh,
      .diagnostics     = diagnostics,
      .tiny_h          = config.physics.flow.tiny_h,
      .h_anuga_regular = config.physics.flow.h_anuga_regular,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateRiemannStateData(mesh->num_internal_edges, &interior_flux_op->left_states));
  PetscCall(CreateRiemannStateData(mesh->num_internal_edges, &interior_flux_op->right_states));
  PetscCall(CreateRiemannEdgeData(mesh->num_internal_edges, num_comp, &interior_flux_op->edges));

  // copy mesh geometry data into place
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id       = edges->internal_edge_ids[e];
    PetscInt right_cell_id = edges->cell_ids[2 * edge_id + 1];

    if (right_cell_id != -1) {
      interior_flux_op->edges.cn[e] = edges->cn[edge_id];
      interior_flux_op->edges.sn[e] = edges->sn[edge_id];
    }
  }

  PetscCall(PetscOperatorCreate(interior_flux_op, ApplyInteriorFlux, DestroyInteriorFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Boundary Flux Operator
//------------------------

typedef struct {
  RDyNumericsRiemann   riemann;                // Riemann solver method
  RDyMesh             *mesh;                   // domain mesh
  RDyBoundary          boundary;               // boundary associated with this sub-operator
  RDyCondition         boundary_condition;     // boundary condition associated with this sub-operator
  Vec                  boundary_values;        // Dirichlet boundary values vector
  Vec                  boundary_fluxes;        // boundary flux values vector
  Vec                  boundary_fluxes_accum;  // boundary flux values vector
  OperatorDiagnostics *diagnostics;            // courant number, boundary fluxes
  PetscReal            tiny_h;                 // minimum water height for wet conditions
  PetscReal            h_anuga_regular;        // ANUGA height parameter used in velocity regularization
  RiemannStateData     left_states;
  RiemannStateData     right_states;
  RiemannEdgeData      edges;
  PetscReal           *cosines, *sines;  // cosine and sine of the angle between the edge and y-axis
  PetscReal           *a_max;            // maximum courant number
} BoundaryFluxOperator;

// applies a reflecting boundary condition on the given boundary, computing
// fluxes F for the solution vector components X
static PetscErrorCode ApplyReflectingBC(RDyMesh *mesh, RDyBoundary boundary, RiemannStateData *datal, RiemannStateData *datar,
                                        RiemannEdgeData *data_edge) {
  PetscFunctionBeginUser;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscReal *sn_vec_bnd = data_edge->sn;
  PetscReal *cn_vec_bnd = data_edge->cn;

  // compute h/u/v for right cells
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id            = boundary.edge_ids[e];
    PetscInt left_local_cell_id = edges->cell_ids[2 * edge_id];

    if (cells->is_owned[left_local_cell_id]) {
      datar->h[e] = datal->h[e];

      PetscReal dum1 = Square(sn_vec_bnd[e]) - Square(cn_vec_bnd[e]);
      PetscReal dum2 = 2.0 * sn_vec_bnd[e] * cn_vec_bnd[e];

      datar->u[e] = datal->u[e] * dum1 - datal->v[e] * dum2;
      datar->v[e] = -datal->u[e] * dum2 - datal->v[e] * dum1;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// applies a critical outflow boundary condition, computing
// fluxes F for the solution vector components X
static PetscErrorCode ApplyCriticalOutflowBC(RDyMesh *mesh, RDyBoundary boundary, RiemannStateData *datal, RiemannStateData *datar,
                                             RiemannEdgeData *data_edge) {
  PetscFunctionBeginUser;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscReal *sn_vec_bnd = data_edge->sn;
  PetscReal *cn_vec_bnd = data_edge->cn;

  // Compute h/u/v for right cells
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id            = boundary.edge_ids[e];
    PetscInt left_local_cell_id = edges->cell_ids[2 * edge_id];

    if (cells->is_owned[left_local_cell_id]) {
      PetscReal uperp = datal->u[e] * cn_vec_bnd[e] + datal->v[e] * sn_vec_bnd[e];
      PetscReal q     = datal->h[e] * fabs(uperp);

      datar->h[e] = PetscPowReal(Square(q) / GRAVITY, 1.0 / 3.0);

      PetscReal velocity = PetscPowReal(GRAVITY * datar->h[e], 0.5);
      datar->u[e]        = velocity * cn_vec_bnd[e];
      datar->v[e]        = velocity * sn_vec_bnd[e];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// application of boundary flux operator for its specific boundary
static PetscErrorCode ApplyBoundaryFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  BoundaryFluxOperator *boundary_flux_op = context;

  RDyBoundary  boundary              = boundary_flux_op->boundary;
  RDyCondition boundary_condition    = boundary_flux_op->boundary_condition;
  Vec          boundary_values       = boundary_flux_op->boundary_values;
  Vec          boundary_fluxes       = boundary_flux_op->boundary_fluxes;
  Vec          boundary_fluxes_accum = boundary_flux_op->boundary_fluxes_accum;

  // get pointers to vector data
  PetscScalar *u_ptr, *f_ptr, *boundary_values_ptr, *boundary_fluxes_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));
  PetscCall(VecGetArray(boundary_values, &boundary_values_ptr));
  PetscCall(VecGetArray(boundary_fluxes, &boundary_fluxes_ptr));

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == 3, comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  // apply boundary conditions
  RiemannStateData *datal     = &boundary_flux_op->left_states;
  RiemannStateData *datar     = &boundary_flux_op->right_states;
  RiemannEdgeData  *data_edge = &boundary_flux_op->edges;

  // copy the "left cell" values into the "left states"
  const PetscReal tiny_h  = boundary_flux_op->tiny_h;
  const PetscReal h_anuga = boundary_flux_op->h_anuga_regular;

  RDyEdges *edges = &boundary_flux_op->mesh->edges;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id            = boundary.edge_ids[e];
    PetscInt left_local_cell_id = edges->cell_ids[2 * edge_id];
    datal->h[e]                 = u_ptr[n_dof * left_local_cell_id + 0];
    datal->hu[e]                = u_ptr[n_dof * left_local_cell_id + 1];
    datal->hv[e]                = u_ptr[n_dof * left_local_cell_id + 2];
  }
  PetscCall(ComputeRiemannVelocities(tiny_h, h_anuga, datal));

  // compute the "right" Riemann cell values using the boundary condition
  switch (boundary_condition.flow->type) {
    case CONDITION_DIRICHLET:
      // copy Dirichlet boundary values into the "right states"
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        datar->h[e]  = boundary_values_ptr[n_dof * e + 0];
        datar->hu[e] = boundary_values_ptr[n_dof * e + 1];
        datar->hv[e] = boundary_values_ptr[n_dof * e + 2];
      }
      PetscCall(ComputeRiemannVelocities(tiny_h, h_anuga, datar));
      break;
    case CONDITION_REFLECTING:
      PetscCall(ApplyReflectingBC(boundary_flux_op->mesh, boundary, datal, datar, data_edge));
      break;
    case CONDITION_CRITICAL_OUTFLOW:
      PetscCall(ApplyCriticalOutflowBC(boundary_flux_op->mesh, boundary, datal, datar, data_edge));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n", boundary.id);
  }

  // solve the Riemann problem
  switch (boundary_flux_op->riemann) {
    case RIEMANN_ROE:
      PetscCall(ComputeSWERoeFlux(datal, datar, data_edge->sn, data_edge->cn, boundary_fluxes_ptr, data_edge->amax));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Unsupported Riemann solver");
  }

  // accumulate the flux values in f_global
  RDyCells                 *cells             = &boundary_flux_op->mesh->cells;
  CourantNumberDiagnostics *courant_num_diags = &boundary_flux_op->diagnostics->courant_number;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt  edge_id       = boundary.edge_ids[e];
    PetscReal edge_len      = edges->lengths[edge_id];
    PetscInt  local_cell_id = edges->cell_ids[2 * edge_id];

    if (cells->is_owned[local_cell_id]) {
      PetscReal cell_area = cells->areas[local_cell_id];
      PetscReal hl        = datal->h[e];
      PetscReal hr        = datar->h[e];

      if (!(hl < tiny_h && hr < tiny_h)) {
        PetscReal cnum = data_edge->amax[e] * edge_len / cell_area * dt;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[e];
          courant_num_diags->global_cell_id  = cells->global_ids[local_cell_id];
        }

        PetscInt owned_cell_id = cells->local_to_owned[local_cell_id];
        for (PetscInt i_dof = 0; i_dof < n_dof; i_dof++) {
          f_ptr[n_dof * owned_cell_id + i_dof] += boundary_fluxes_ptr[n_dof * e + i_dof] * (-edge_len / cell_area);
        }
      }
    }
  }

  // restore vectors
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));
  PetscCall(VecRestoreArray(boundary_values, &boundary_values_ptr));
  PetscCall(VecRestoreArray(boundary_fluxes, &boundary_fluxes_ptr));

  // accumulate boundary fluxes
  PetscCall(VecAYPX(boundary_fluxes_accum, 1.0, boundary_fluxes));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyBoundaryFlux(void *context) {
  PetscFunctionBegin;
  BoundaryFluxOperator *boundary_flux_op = context;
  DestroyRiemannStateData(boundary_flux_op->left_states);
  DestroyRiemannStateData(boundary_flux_op->right_states);
  DestroyRiemannEdgeData(boundary_flux_op->edges);
  PetscCall(PetscFree(boundary_flux_op));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that computes fluxes through edges on the boundary
/// of a domain, suitable for the shallow water equations.
/// @param [in]    mesh                  mesh defining the computational domain of the operator
/// @param [in]    config                RDycore's configuration
/// @param [in]    boundary              the boundary through which fluxes are to be computed
/// @param [in]    boundary_condition    the boundary condition associated with the boundary of interest
/// @param [in]    boundary_values       a Vec storing Dirichlet values (if any) for this boundary
/// @param [inout] boundary_fluxes       a Vec storing fluxes for this boundary
/// @param [inout] boundary_fluxes_accum a Vec storing accumulated fluxes for this boundary
/// @param [inout] diagnostics           a set of diagnostics that can be updated by the PetscOperator
/// @param [out]   petsc_op              the newly created PetscOperator
PetscErrorCode CreateSWEPetscBoundaryFluxOperator(RDyMesh *mesh, const RDyConfig config, RDyBoundary boundary, RDyCondition boundary_condition,
                                                  Vec boundary_values, Vec boundary_fluxes, Vec boundary_fluxes_accum,
                                                  OperatorDiagnostics *diagnostics, PetscOperator *petsc_op) {
  PetscFunctionBegin;
  BoundaryFluxOperator *boundary_flux_op;
  PetscCall(PetscCalloc1(1, &boundary_flux_op));
  *boundary_flux_op = (BoundaryFluxOperator){
      .riemann               = config.numerics.riemann,
      .mesh                  = mesh,
      .boundary              = boundary,
      .boundary_condition    = boundary_condition,
      .boundary_values       = boundary_values,
      .boundary_fluxes       = boundary_fluxes,
      .boundary_fluxes_accum = boundary_fluxes_accum,
      .diagnostics           = diagnostics,
      .tiny_h                = config.physics.flow.tiny_h,
      .h_anuga_regular       = config.physics.flow.h_anuga_regular,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateRiemannStateData(boundary.num_edges, &boundary_flux_op->left_states));
  PetscCall(CreateRiemannStateData(boundary.num_edges, &boundary_flux_op->right_states));
  PetscCall(CreateRiemannEdgeData(boundary.num_edges, 3, &boundary_flux_op->edges));

  // copy mesh geometry data into place
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id              = boundary.edge_ids[e];
    boundary_flux_op->edges.cn[e] = edges->cn[edge_id];
    boundary_flux_op->edges.sn[e] = edges->sn[edge_id];
  }
  PetscCall(PetscOperatorCreate(boundary_flux_op, ApplyBoundaryFlux, DestroyBoundaryFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//-----------------
// Source Operator
//-----------------

typedef struct {
  RDyMesh  *mesh;  // domain mesh
  Vec       external_sources;
  Vec       material_properties;
  PetscReal tiny_h;            // minimum water height for wet conditions
  PetscReal xq2018_threshold;  // threshold for the XQ2018's implicit time integration of source term
} SourceOperator;

// adds source terms to the right hand side vector F
static PetscErrorCode ApplySourceSemiImplicit(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  SourceOperator *source_op     = context;
  Vec             source_vec    = source_op->external_sources;
  Vec             mat_props_vec = source_op->material_properties;
  RDyMesh        *mesh          = source_op->mesh;
  RDyCells       *cells         = &mesh->cells;
  PetscReal       tiny_h        = source_op->tiny_h;

  // access Vec data
  PetscScalar *source_ptr, *mat_props_ptr, *u_ptr, *f_ptr;
  PetscCall(VecGetArray(source_vec, &source_ptr));        // sequential vector
  PetscCall(VecGetArray(mat_props_vec, &mat_props_ptr));  // sequential vector
  PetscCall(VecGetArray(u_local, &u_ptr));                // domain local vector (indexed by local cells)
  PetscCall(VecGetArray(f_global, &f_ptr));               // domain global vector (indexed by owned cells)

  // access previously-computed flux divergence data
  Vec flux_div;
  PetscCall(PetscOperatorFieldsGet(fields, "riemannf", &flux_div));
  PetscCheck(flux_div, comm, PETSC_ERR_USER, "No 'riemannf' field found in source operator!");
  PetscScalar *flux_div_ptr;
  PetscCall(VecGetArray(flux_div, &flux_div_ptr));  // domain global vector

  PetscInt size;
  PetscCall(VecGetSize(source_vec, &size));
  PetscInt n_dof = size / mesh->num_owned_cells;
  PetscCheck(n_dof == 3, comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  PetscInt num_mat_props = NUM_MATERIAL_PROPERTIES;
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (cells->is_owned[c]) {
      PetscInt owned_cell_id = cells->local_to_owned[c];

      PetscReal h  = u_ptr[n_dof * c + 0];
      PetscReal hu = u_ptr[n_dof * c + 1];
      PetscReal hv = u_ptr[n_dof * c + 2];

      PetscReal dz_dx = cells->dz_dx[c];
      PetscReal dz_dy = cells->dz_dy[c];

      PetscReal bedx = dz_dx * GRAVITY * h;
      PetscReal bedy = dz_dy * GRAVITY * h;

      PetscReal Fsum_x = flux_div_ptr[n_dof * owned_cell_id + 1];
      PetscReal Fsum_y = flux_div_ptr[n_dof * owned_cell_id + 2];

      PetscReal tbx = 0.0, tby = 0.0;

      if (h >= tiny_h) {  // wet conditions
        PetscReal u = hu / h;
        PetscReal v = hv / h;

        // Manning's coefficient
        PetscReal N_mannings = mat_props_ptr[num_mat_props * owned_cell_id + MATERIAL_PROPERTY_MANNINGS];

        // Cd = g n^2 h^{-1/3}, where n is Manning's coefficient
        PetscReal Cd = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);

        PetscReal velocity = PetscSqrtReal(Square(u) + Square(v));
        PetscReal tb       = Cd * velocity / h;
        PetscReal factor   = tb / (1.0 + dt * tb);

        tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
        tby = (hv + dt * Fsum_y - dt * bedy) * factor;
      }

      // NOTE: we accumulate everything into the RHS vector by convention.
      f_ptr[n_dof * owned_cell_id + 0] += source_ptr[n_dof * owned_cell_id + 0];
      f_ptr[n_dof * owned_cell_id + 1] += -bedx - tbx + source_ptr[n_dof * owned_cell_id + 1];
      f_ptr[n_dof * owned_cell_id + 2] += -bedy - tby + source_ptr[n_dof * owned_cell_id + 2];
    }
  }

  // restore vectors
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));
  PetscCall(VecRestoreArray(source_vec, &source_ptr));
  PetscCall(VecRestoreArray(mat_props_vec, &mat_props_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Adds contribution of the source-term using implicit time integration approach of:
///        Xia, Xilin, and Qiuhua Liang. "A new efficient implicit scheme for discretising the stiff
///        friction terms in the shallow water equations." Advances in water resources 117 (2018): 87-97.
///        https://www.sciencedirect.com/science/article/pii/S0309170818302124?ref=cra_js_challenge&fr=RR-1
/// @param [in] context   a context for the SourceOperator
/// @param [in] fields    a PetscOperatorFields from which includes the "rimeannf" field
/// @param [in] dt        time step
/// @param [in] u_local   a Vec that contain unknowns for locally-present and ghost cells
/// @param [out] f_global a Vec that stores the fluxes for only locally-present cells
/// @return
static PetscErrorCode ApplySourceImplicitXQ2018(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  SourceOperator *source_op        = context;
  Vec             source_vec       = source_op->external_sources;
  Vec             mat_props_vec    = source_op->material_properties;
  RDyMesh        *mesh             = source_op->mesh;
  RDyCells       *cells            = &mesh->cells;
  PetscReal       tiny_h           = source_op->tiny_h;
  PetscReal       xq2018_threshold = source_op->xq2018_threshold;

  // access Vec data
  PetscScalar *source_ptr, *mat_props_ptr, *u_ptr, *f_ptr;
  PetscCall(VecGetArray(source_vec, &source_ptr));        // sequential vector
  PetscCall(VecGetArray(mat_props_vec, &mat_props_ptr));  // sequential vector
  PetscCall(VecGetArray(u_local, &u_ptr));                // domain local vector (indexed by local cells)
  PetscCall(VecGetArray(f_global, &f_ptr));               // domain global vector (indexed by owned cells)

  // access previously-computed flux divergence data
  Vec flux_div;
  PetscCall(PetscOperatorFieldsGet(fields, "riemannf", &flux_div));
  PetscCheck(flux_div, comm, PETSC_ERR_USER, "No 'riemannf' field found in source operator!");
  PetscScalar *flux_div_ptr;
  PetscCall(VecGetArray(flux_div, &flux_div_ptr));  // domain global vector

  PetscInt size;
  PetscCall(VecGetSize(source_vec, &size));
  PetscInt n_dof = size / mesh->num_owned_cells;
  PetscCheck(n_dof == 3, comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  PetscInt num_mat_props = NUM_MATERIAL_PROPERTIES;
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (cells->is_owned[c]) {
      PetscInt owned_cell_id = cells->local_to_owned[c];

      PetscReal h  = u_ptr[n_dof * c + 0];
      PetscReal hu = u_ptr[n_dof * c + 1];
      PetscReal hv = u_ptr[n_dof * c + 2];

      PetscReal dz_dx = cells->dz_dx[c];
      PetscReal dz_dy = cells->dz_dy[c];

      PetscReal bedx = dz_dx * GRAVITY * h;
      PetscReal bedy = dz_dy * GRAVITY * h;

      PetscReal tbx = 0.0, tby = 0.0;

      if (h >= tiny_h) {  // wet conditions

        // Manning's coefficient
        PetscReal N_mannings = mat_props_ptr[num_mat_props * owned_cell_id + MATERIAL_PROPERTY_MANNINGS];

        PetscReal Fsum_x = flux_div_ptr[n_dof * owned_cell_id + 1];
        PetscReal Fsum_y = flux_div_ptr[n_dof * owned_cell_id + 2];

        // defined in the text below equation 22 of XQ2018
        PetscReal Ax = Fsum_x - bedx;
        PetscReal Ay = Fsum_y - bedy;

        // equation 27 of XQ2018
        PetscReal mx = hu + Ax * dt;
        PetscReal my = hv + Ay * dt;

        PetscReal lambda = GRAVITY * Square(N_mannings) * PetscPowReal(h, -4.0 / 3.0) * PetscPowReal(Square(mx / h) + Square(my / h), 0.5);

        PetscReal qx_nplus1, qy_nplus1;

        // equation 36 and 37 of XQ2018
        if (dt * lambda < xq2018_threshold) {
          qx_nplus1 = mx;
          qy_nplus1 = my;
        } else {
          qx_nplus1 = (mx - mx * PetscPowReal(1.0 + 4.0 * dt * lambda, 0.5)) / (-2.0 * dt * lambda);
          qy_nplus1 = (my - my * PetscPowReal(1.0 + 4.0 * dt * lambda, 0.5)) / (-2.0 * dt * lambda);
        }

        PetscReal q_magnitude = PetscPowReal(Square(qx_nplus1) + Square(qy_nplus1), 0.5);

        // equation 21 and 22 of XQ2018
        tbx = GRAVITY * Square(N_mannings) * PetscPowReal(h, -7.0 / 3.0) * qx_nplus1 * q_magnitude;
        tby = GRAVITY * Square(N_mannings) * PetscPowReal(h, -7.0 / 3.0) * qy_nplus1 * q_magnitude;
      }

      // NOTE: we accumulate everything into the RHS vector by convention.
      f_ptr[n_dof * owned_cell_id + 0] += source_ptr[n_dof * owned_cell_id + 0];
      f_ptr[n_dof * owned_cell_id + 1] += -bedx - tbx + source_ptr[n_dof * owned_cell_id + 1];
      f_ptr[n_dof * owned_cell_id + 2] += -bedy - tby + source_ptr[n_dof * owned_cell_id + 2];
    }
  }

  // restore vectors
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));
  PetscCall(VecRestoreArray(source_vec, &source_ptr));
  PetscCall(VecRestoreArray(mat_props_vec, &mat_props_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySource(void *context) {
  PetscFunctionBegin;
  SourceOperator *source_op = context;
  PetscFree(source_op);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that computes sources within a region in a domain,
/// suitable for the shallow water equations.
/// @param [in]  mesh                mesh defining the computational domain of the operator
/// @param [in]  config              RDycore's configuration
/// @param [in]  external_sources    a Vec storing external source values (if any) for the domain
/// @param [in]  material_properties a Vec storing material properties for the domain
/// @param [out] petsc_op            the newly created PetscOperator
PetscErrorCode CreateSWEPetscSourceOperator(RDyMesh *mesh, const RDyConfig config, Vec external_sources, Vec material_properties,
                                            PetscOperator *petsc_op) {
  PetscFunctionBegin;
  SourceOperator *source_op;
  PetscCall(PetscCalloc1(1, &source_op));
  *source_op = (SourceOperator){
      .mesh                = mesh,
      .external_sources    = external_sources,
      .material_properties = material_properties,
      .tiny_h              = config.physics.flow.tiny_h,
      .xq2018_threshold    = config.physics.flow.source.xq2018_threshold,
  };

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)external_sources, &comm));

  switch (config.physics.flow.source.method) {
    case SOURCE_SEMI_IMPLICIT:
      PetscCall(PetscOperatorCreate(source_op, ApplySourceSemiImplicit, DestroySource, petsc_op));
      break;
    case SOURCE_IMPLICIT_XQ2018:
      PetscCall(PetscOperatorCreate(source_op, ApplySourceImplicitXQ2018, DestroySource, petsc_op));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only semi_implicit and implicit_xq2018 are supported in the PETSc version");
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
