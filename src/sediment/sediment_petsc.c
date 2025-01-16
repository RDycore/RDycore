#ifndef SEDIMENT_PETSC_H
#define SEDIMENT_PETSC_H

#include <petscsys.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

// gravitational acceleration [m/s/s]
static const PetscReal GRAVITY = 9.806;

// riemann left and right states
typedef struct {
  PetscInt   num_states;         // number of states
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_sediment_comp;  // number of sediment components
  PetscReal *h, *hu, *hv, *hci;  // prognostic variables
  PetscReal *u, *v, *ci;         // diagnostic variables
} SedimentRiemannStateData;

typedef struct {
  PetscInt   num_edges;          // number of edges
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_sediment_comp;  // number of sediment components
  PetscReal *cn, *sn;            // cosine and sine of the angle between edges and y-axis
  PetscReal *fluxes;             // fluxes through the edge
  PetscReal *amax;               // courant number on edges
} SedimentRiemannEdgeData;

static PetscErrorCode CreateSedimentRiemannStateData(PetscInt num_states, PetscInt num_flow_comp, PetscInt num_sediment_comp,
                                                     SedimentRiemannStateData *data) {
  PetscFunctionBegin;

  data->num_states        = num_states;
  data->num_flow_comp     = num_flow_comp;
  data->num_sediment_comp = num_sediment_comp;

  PetscCall(PetscCalloc1(num_states, &data->h));
  PetscCall(PetscCalloc1(num_states, &data->hu));
  PetscCall(PetscCalloc1(num_states, &data->hv));
  PetscCall(PetscCalloc1(num_states, &data->u));
  PetscCall(PetscCalloc1(num_states, &data->v));

  PetscCall(PetscCalloc1(num_states * num_sediment_comp, &data->hci));
  PetscCall(PetscCalloc1(num_states * num_sediment_comp, &data->ci));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentRiemannStateData(SedimentRiemannStateData data) {
  PetscFunctionBegin;

  data.num_states        = 0;
  data.num_flow_comp     = 0;
  data.num_sediment_comp = 0;
  PetscCall(PetscFree(data.h));
  PetscCall(PetscFree(data.hu));
  PetscCall(PetscFree(data.hv));
  PetscCall(PetscFree(data.hci));
  PetscCall(PetscFree(data.u));
  PetscCall(PetscFree(data.v));
  PetscCall(PetscFree(data.ci));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentRiemannEdgeData(SedimentRiemannEdgeData data) {
  PetscFunctionBegin;

  data.num_edges         = 0;
  data.num_flow_comp     = 0;
  data.num_sediment_comp = 0;

  PetscCall(PetscFree(data.cn));
  PetscCall(PetscFree(data.sn));
  PetscCall(PetscFree(data.fluxes));
  PetscCall(PetscFree(data.amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSedimentRiemannEdgeData(PetscInt num_edges, PetscInt num_flow_comp, PetscInt num_sediment_comp,
                                                    SedimentRiemannEdgeData *data) {
  PetscFunctionBegin;

  data->num_edges         = num_edges;
  data->num_flow_comp     = num_flow_comp;
  data->num_sediment_comp = num_sediment_comp;

  PetscCall(PetscCalloc1(num_edges, &data->cn));
  PetscCall(PetscCalloc1(num_edges, &data->sn));

  PetscCall(PetscCalloc1(num_edges * (num_flow_comp + num_sediment_comp), &data->fluxes));
  PetscCall(PetscCalloc1(num_edges, &data->amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRiemannVelocitiesAndConcentration(PetscReal tiny_h, SedimentRiemannStateData *data) {
  PetscFunctionBeginUser;

  PetscInt index;
  for (PetscInt n = 0; n < data->num_states; n++) {
    if (data->h[n] < tiny_h) {
      data->u[n] = 0.0;
      data->v[n] = 0.0;
      for (PetscInt s = 0; s < data->num_sediment_comp; s++) {
        index           = n * data->num_sediment_comp + s;
        data->ci[index] = 0.0;
      }
    } else {
      data->u[n] = data->hu[n] / data->h[n];
      data->v[n] = data->hv[n] / data->h[n];
      for (PetscInt s = 0; s < data->num_sediment_comp; s++) {
        index           = n * data->num_sediment_comp + s;
        data->ci[index] = data->hci[index] / data->h[n];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#define MAX_NUM_SECTION_FIELD_COMPONENTS 10

static PetscErrorCode ComputeSedimentRoeFlux(SedimentRiemannStateData *datal, SedimentRiemannStateData *datar, const PetscReal *sn,
                                             const PetscReal *cn, PetscReal *fij, PetscReal *amax) {
  PetscFunctionBeginUser;

  PetscReal *hl  = datal->h;
  PetscReal *ul  = datal->u;
  PetscReal *vl  = datal->v;
  PetscReal *cil = datal->ci;

  PetscReal *hr  = datar->h;
  PetscReal *ur  = datar->u;
  PetscReal *vr  = datar->v;
  PetscReal *cir = datar->ci;

  PetscAssert(datal->num_states == datar->num_states, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Size of data left and right of edges is not the same!");

  PetscInt num_states = datal->num_states;
  PetscInt flow_ncomp = datal->num_flow_comp;
  PetscInt sed_ncomp  = datal->num_sediment_comp;
  PetscInt soln_ncomp = flow_ncomp + sed_ncomp;

  PetscInt ci_index_offset;

  PetscReal cihat[MAX_NUM_SECTION_FIELD_COMPONENTS];
  PetscReal dch[MAX_NUM_SECTION_FIELD_COMPONENTS];
  PetscReal R[MAX_NUM_SECTION_FIELD_COMPONENTS][MAX_NUM_SECTION_FIELD_COMPONENTS];
  PetscReal A[MAX_NUM_SECTION_FIELD_COMPONENTS][MAX_NUM_SECTION_FIELD_COMPONENTS];
  PetscReal dW[MAX_NUM_SECTION_FIELD_COMPONENTS];
  PetscReal FL[MAX_NUM_SECTION_FIELD_COMPONENTS], FR[MAX_NUM_SECTION_FIELD_COMPONENTS];

  // initialize
  for (PetscInt i = 0; i < MAX_NUM_SECTION_FIELD_COMPONENTS; i++) {
    for (PetscInt j = 0; j < MAX_NUM_SECTION_FIELD_COMPONENTS; j++) {
      R[i][j] = 0.0;
    }
  }

  for (PetscInt i = 0; i < num_states; ++i) {
    // compute Roe averages
    PetscReal duml  = pow(hl[i], 0.5);
    PetscReal dumr  = pow(hr[i], 0.5);
    PetscReal cl    = pow(GRAVITY * hl[i], 0.5);
    PetscReal cr    = pow(GRAVITY * hr[i], 0.5);
    PetscReal hhat  = duml * dumr;
    PetscReal uhat  = (duml * ul[i] + dumr * ur[i]) / (duml + dumr);
    PetscReal vhat  = (duml * vl[i] + dumr * vr[i]) / (duml + dumr);
    PetscReal chat  = pow(0.5 * GRAVITY * (hl[i] + hr[i]), 0.5);
    PetscReal uperp = uhat * cn[i] + vhat * sn[i];

    PetscReal dh     = hr[i] - hl[i];
    PetscReal du     = ur[i] - ul[i];
    PetscReal dv     = vr[i] - vl[i];
    PetscReal dupar  = -du * sn[i] + dv * cn[i];
    PetscReal duperp = du * cn[i] + dv * sn[i];

    ci_index_offset = i * sed_ncomp;
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      cihat[j] = (duml * cil[ci_index_offset + j] + dumr * cir[ci_index_offset + j]) / (duml + dumr);
      dch[j]   = cir[ci_index_offset + j] * hr[i] - cil[ci_index_offset + j] * hl[i];
    }

    dW[0] = 0.5 * (dh - hhat * duperp / chat);
    dW[1] = hhat * dupar;
    dW[2] = 0.5 * (dh + hhat * duperp / chat);
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      dW[j + 3] = dch[j] - cihat[j] * dh;
    }

    PetscReal uperpl = ul[i] * cn[i] + vl[i] * sn[i];
    PetscReal uperpr = ur[i] * cn[i] + vr[i] * sn[i];
    PetscReal al1    = uperpl - cl;
    PetscReal al3    = uperpl + cl;
    PetscReal ar1    = uperpr - cr;
    PetscReal ar3    = uperpr + cr;

    R[0][0] = 1.0;
    R[0][1] = 0.0;
    R[0][2] = 1.0;
    R[1][0] = uhat - chat * cn[i];
    R[1][1] = -sn[i];
    R[1][2] = uhat + chat * cn[i];
    R[2][0] = vhat - chat * sn[i];
    R[2][1] = cn[i];
    R[2][2] = vhat + chat * sn[i];
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      R[j + 3][0]     = cihat[j];
      R[j + 3][2]     = cihat[j];
      R[j + 3][j + 3] = 1.0;
    }

    PetscReal da1 = fmax(0.0, 2.0 * (ar1 - al1));
    PetscReal da3 = fmax(0.0, 2.0 * (ar3 - al3));
    PetscReal a1  = fabs(uperp - chat);
    PetscReal a2  = fabs(uperp);
    PetscReal a3  = fabs(uperp + chat);

    // Critical flow fix
    if (a1 < da1) {
      a1 = 0.5 * (a1 * a1 / da1 + da1);
    }
    if (a3 < da3) {
      a3 = 0.5 * (a3 * a3 / da3 + da3);
    }

    // Compute interface flux
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) {
        A[i][j] = 0.0;
      }
    }
    A[0][0] = a1;
    A[1][1] = a2;
    A[2][2] = a3;
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      A[j + 3][j + 3] = a2;
    }

    FL[0] = uperpl * hl[i];
    FL[1] = ul[i] * uperpl * hl[i] + 0.5 * GRAVITY * hl[i] * hl[i] * cn[i];
    FL[2] = vl[i] * uperpl * hl[i] + 0.5 * GRAVITY * hl[i] * hl[i] * sn[i];

    FR[0] = uperpr * hr[i];
    FR[1] = ur[i] * uperpr * hr[i] + 0.5 * GRAVITY * hr[i] * hr[i] * cn[i];
    FR[2] = vr[i] * uperpr * hr[i] + 0.5 * GRAVITY * hr[i] * hr[i] * sn[i];

    ci_index_offset = i * sed_ncomp;
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      FL[j + 3] = hl[i] * uperpl * cil[ci_index_offset + j];
      FR[j + 3] = hr[i] * uperpr * cir[ci_index_offset + j];
    }

    // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
    for (PetscInt dof1 = 0; dof1 < soln_ncomp; dof1++) {
      for (PetscInt dof2 = 0; dof2 < soln_ncomp; dof2++) {
        if (dof2 == 0) {
          fij[soln_ncomp * i + dof1] = 0.5 * (FL[dof1] + FR[dof1]);
        }
        fij[soln_ncomp * i + dof1] = fij[soln_ncomp * i + dof1] - 0.5 * R[dof1][dof2] * A[dof2][dof2] * dW[dof2];
      }
    }

    amax[i] = chat + fabs(uperp);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Interior Flux Operator
//------------------------

typedef struct {
  RDyMesh                 *mesh;          // domain mesh
  PetscReal                tiny_h;        // minimum water height for wet conditions
  SedimentRiemannStateData left_states;   // "left" riemann states on interior edges
  SedimentRiemannStateData right_states;  // "right" riemann states on interior edges
  SedimentRiemannEdgeData  edges;         // riemann fluxes on interior edges
  OperatorDiagnostics     *diagnostics;   // courant number, etc
} SedimentInteriorFluxOperator;

static PetscErrorCode ApplySedimentInteriorFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  SedimentInteriorFluxOperator *interior_flux_op = context;

  RDyMesh  *mesh  = interior_flux_op->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // get pointers to vector data
  PetscScalar *u_ptr, *f_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));

  SedimentRiemannStateData *datal        = &interior_flux_op->left_states;
  SedimentRiemannStateData *datar        = &interior_flux_op->right_states;
  SedimentRiemannEdgeData  *data_edge    = &interior_flux_op->edges;
  PetscReal                *sn_vec_int   = data_edge->sn;
  PetscReal                *cn_vec_int   = data_edge->cn;
  PetscReal                *amax_vec_int = data_edge->amax;
  PetscReal                *flux_vec_int = data_edge->fluxes;

  PetscInt num_flow_comp     = datal->num_flow_comp;
  PetscInt num_sediment_comp = datal->num_sediment_comp;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == num_flow_comp + num_sediment_comp, comm, PETSC_ERR_USER, "Number of dof in local vector do not match flow and sediment dof!");

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

      for (PetscInt s = 0; s < num_sediment_comp; s++) {
        datal->hci[e * num_sediment_comp + s] = u_ptr[n_dof * left_local_cell_id + 3 + s];
        datar->hci[e * num_sediment_comp + s] = u_ptr[n_dof * right_local_cell_id + 3 + s];
      }
    }
  }

  const PetscReal tiny_h = interior_flux_op->tiny_h;
  PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, datal));
  PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, datar));

  // call Riemann solver (only Roe currently supported)
  PetscCall(ComputeSedimentRoeFlux(datal, datar, sn_vec_int, cn_vec_int, flux_vec_int, amax_vec_int));

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

static PetscErrorCode DestroySedimentInteriorFlux(void *context) {
  PetscFunctionBegin;

  SedimentInteriorFluxOperator *interior_flux_op = context;

  DestroySedimentRiemannStateData(interior_flux_op->left_states);
  DestroySedimentRiemannStateData(interior_flux_op->right_states);
  DestroySedimentRiemannEdgeData(interior_flux_op->edges);

  PetscCall(PetscFree(interior_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSedimentPetscInteriorFluxOperator(RDyMesh *mesh, PetscInt num_flow_comp, PetscInt num_sediment_comp,
                                                       OperatorDiagnostics *diagnostics, PetscReal tiny_h, PetscOperator *petsc_op) {
  PetscFunctionBegin;

  SedimentInteriorFluxOperator *interior_flux_op;
  PetscCall(PetscCalloc1(1, &interior_flux_op));
  *interior_flux_op = (SedimentInteriorFluxOperator){
      .mesh        = mesh,
      .diagnostics = diagnostics,
      .tiny_h      = tiny_h,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateSedimentRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_sediment_comp, &interior_flux_op->left_states));
  PetscCall(CreateSedimentRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_sediment_comp, &interior_flux_op->right_states));
  PetscCall(CreateSedimentRiemannEdgeData(mesh->num_internal_edges, num_flow_comp, num_sediment_comp, &interior_flux_op->edges));

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

  PetscCall(PetscOperatorCreate(interior_flux_op, ApplySedimentInteriorFlux, DestroySedimentInteriorFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Boundary Flux Operator
//------------------------

typedef struct {
  RDyMesh                 *mesh;                // domain mesh
  RDyBoundary              boundary;            // boundary associated with this sub-operator
  RDyCondition             boundary_condition;  // boundary condition associated with this sub-operator
  Vec                      boundary_values;     // Dirichlet boundary values vector
  Vec                      boundary_fluxes;     // boundary flux values vector
  OperatorDiagnostics     *diagnostics;         // courant number, boundary fluxes
  PetscReal                tiny_h;              // minimum water height for wet conditions
  SedimentRiemannStateData left_states;
  SedimentRiemannStateData right_states;
  SedimentRiemannEdgeData  edges;
  PetscReal               *cosines, *sines;  // cosine and sine of the angle between the edge and y-axis
  PetscReal               *a_max;            // maximum courant number
} SedimentBoundaryFluxOperator;

static PetscErrorCode ApplySedimentReflectingBC(RDyMesh *mesh, RDyBoundary boundary, SedimentRiemannStateData *datal, SedimentRiemannStateData *datar,
                                                SedimentRiemannEdgeData *data_edge) {
  PetscFunctionBeginUser;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscReal *sn_vec_bnd = data_edge->sn;
  PetscReal *cn_vec_bnd = data_edge->cn;

  PetscInt num_sediment_comp = datal->num_sediment_comp;

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

      for (PetscInt s = 0; s < num_sediment_comp; s++) {
        datar->ci[e * num_sediment_comp + s] = datal->ci[e * num_sediment_comp + s];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ApplySedimentBoundaryFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  SedimentBoundaryFluxOperator *boundary_flux_op = context;

  RDyBoundary  boundary           = boundary_flux_op->boundary;
  RDyCondition boundary_condition = boundary_flux_op->boundary_condition;
  Vec          boundary_values    = boundary_flux_op->boundary_values;
  Vec          boundary_fluxes    = boundary_flux_op->boundary_fluxes;

  // get pointers to vector data
  PetscScalar *u_ptr, *f_ptr, *boundary_values_ptr, *boundary_fluxes_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));
  PetscCall(VecGetArray(boundary_values, &boundary_values_ptr));
  PetscCall(VecGetArray(boundary_fluxes, &boundary_fluxes_ptr));

  // apply boundary conditions
  SedimentRiemannStateData *datal     = &boundary_flux_op->left_states;
  SedimentRiemannStateData *datar     = &boundary_flux_op->right_states;
  SedimentRiemannEdgeData  *data_edge = &boundary_flux_op->edges;

  PetscInt num_flow_comp     = datal->num_flow_comp;
  PetscInt num_sediment_comp = datal->num_sediment_comp;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == num_flow_comp + num_sediment_comp, comm, PETSC_ERR_USER, "Number of dof in local vector do not match flow and sediment dof!");

  // copy the "left cell" values into the "left states"
  RDyEdges *edges = &boundary_flux_op->mesh->edges;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id            = boundary.edge_ids[e];
    PetscInt left_local_cell_id = edges->cell_ids[2 * edge_id];
    datal->h[e]                 = u_ptr[n_dof * left_local_cell_id + 0];
    datal->hu[e]                = u_ptr[n_dof * left_local_cell_id + 1];
    datal->hv[e]                = u_ptr[n_dof * left_local_cell_id + 2];

    for (PetscInt s = 0; s < num_sediment_comp; s++) {
      datal->hci[e * num_sediment_comp + s] = u_ptr[n_dof * left_local_cell_id + 3 + s];
    }
  }

  const PetscReal tiny_h = boundary_flux_op->tiny_h;
  PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, datal));

  // compute the "right" Riemann cell values using the boundary condition
  switch (boundary_condition.flow->type) {
    case CONDITION_DIRICHLET:
      // copy Dirichlet boundary values into the "right states"
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        datar->h[e]  = boundary_values_ptr[n_dof * e + 0];
        datar->hu[e] = boundary_values_ptr[n_dof * e + 1];
        datar->hv[e] = boundary_values_ptr[n_dof * e + 2];
        for (PetscInt s = 0; s < num_sediment_comp; s++) {
          datal->hci[e * num_sediment_comp + s] = u_ptr[n_dof * e + 3 + s];
        }
      }
      PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, datar));
      break;
    case CONDITION_REFLECTING:
      PetscCall(ApplySedimentReflectingBC(boundary_flux_op->mesh, boundary, datal, datar, data_edge));
      break;
    case CONDITION_CRITICAL_OUTFLOW:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "CONDITION_CRITICAL_OUTFLOW not supported for sediment");
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n", boundary.id);
  }

  // call Riemann solver (only Roe currently supported)
  PetscCall(ComputeSedimentRoeFlux(datal, datar, data_edge->sn, data_edge->cn, boundary_fluxes_ptr, data_edge->amax));

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

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentBoundaryFlux(void *context) {
  PetscFunctionBegin;

  SedimentBoundaryFluxOperator *boundary_flux_op = context;

  DestroySedimentRiemannStateData(boundary_flux_op->left_states);
  DestroySedimentRiemannStateData(boundary_flux_op->right_states);
  DestroySedimentRiemannEdgeData(boundary_flux_op->edges);

  PetscCall(PetscFree(boundary_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSedimentPetscBoundaryFluxOperator(RDyMesh *mesh, PetscInt num_flow_comp, PetscInt num_sediment_comp, RDyBoundary boundary,
                                                       RDyCondition boundary_condition, Vec boundary_values, Vec boundary_fluxes,
                                                       OperatorDiagnostics *diagnostics, PetscReal tiny_h, PetscOperator *petsc_op) {
  PetscFunctionBegin;
  SedimentBoundaryFluxOperator *boundary_flux_op;
  PetscCall(PetscCalloc1(1, &boundary_flux_op));
  *boundary_flux_op = (SedimentBoundaryFluxOperator){
      .mesh               = mesh,
      .boundary           = boundary,
      .boundary_condition = boundary_condition,
      .boundary_values    = boundary_values,
      .boundary_fluxes    = boundary_fluxes,
      .diagnostics        = diagnostics,
      .tiny_h             = tiny_h,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateSedimentRiemannStateData(boundary.num_edges, num_flow_comp, num_sediment_comp, &boundary_flux_op->left_states));
  PetscCall(CreateSedimentRiemannStateData(boundary.num_edges, num_flow_comp, num_sediment_comp, &boundary_flux_op->right_states));
  PetscCall(CreateSedimentRiemannEdgeData(boundary.num_edges, num_flow_comp, num_sediment_comp, &boundary_flux_op->edges));

  // copy mesh geometry data into place
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id              = boundary.edge_ids[e];
    boundary_flux_op->edges.cn[e] = edges->cn[edge_id];
    boundary_flux_op->edges.sn[e] = edges->sn[edge_id];
  }
  PetscCall(PetscOperatorCreate(boundary_flux_op, ApplySedimentBoundaryFlux, DestroySedimentBoundaryFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//-----------------
// Source Operator
//-----------------

typedef struct {
  RDyMesh  *mesh;               // domain mesh
  PetscInt  num_flow_comp;      // number of flow components
  PetscInt  num_sediment_comp;  // number of sediment components
  Vec       external_sources;   // external source vector
  Vec       mannings;           // mannings coefficient vector
  PetscReal tiny_h;             // minimum water height for wet conditions
  PetscReal xq2018_threshold;   // threshold for the XQ2018's implicit time integration of source term
} SedimentSourceOperator;

static PetscErrorCode ApplySedimentSourceSemiImplicit(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  SedimentSourceOperator *source_op         = context;
  Vec                     source_vec        = source_op->external_sources;
  Vec                     mannings_vec      = source_op->mannings;
  RDyMesh                *mesh              = source_op->mesh;
  RDyCells               *cells             = &mesh->cells;
  PetscReal               tiny_h            = source_op->tiny_h;
  PetscInt                num_sediment_comp = source_op->num_sediment_comp;

  const PetscReal kp_constant             = 0.001;
  const PetscReal settling_velocity       = 0.01;
  const PetscReal tau_critical_erosion    = 0.1;
  const PetscReal tau_critical_deposition = 1000.0;
  const PetscReal rhow                    = 1000.0;

  // access Vec data
  PetscScalar *source_ptr, *mannings_ptr, *u_ptr, *f_ptr;
  PetscCall(VecGetArray(source_vec, &source_ptr));      // sequential vector
  PetscCall(VecGetArray(mannings_vec, &mannings_ptr));  // sequential vector
  PetscCall(VecGetArray(u_local, &u_ptr));              // domain local vector (indexed by local cells)
  PetscCall(VecGetArray(f_global, &f_ptr));             // domain global vector (indexed by owned cells)

  // access previously-computed flux divergence data
  Vec flux_div;
  PetscCall(PetscOperatorFieldsGet(fields, "riemannf", &flux_div));
  PetscCheck(flux_div, comm, PETSC_ERR_USER, "No 'riemannf' field found in source operator!");
  PetscScalar *flux_div_ptr;
  PetscCall(VecGetArray(flux_div, &flux_div_ptr));  // domain global vector

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));

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
        PetscReal N_mannings = mannings_ptr[c];

        // Cd = g n^2 h^{-1/3}, where n is Manning's coefficient
        PetscReal Cd = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);

        PetscReal velocity = PetscSqrtReal(Square(u) + Square(v));
        PetscReal tb       = Cd * velocity / h;
        PetscReal factor   = tb / (1.0 + dt * tb);

        tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
        tby = (hv + dt * Fsum_y - dt * bedy) * factor;

        for (PetscInt s = 0; s < num_sediment_comp; s++) {
          PetscReal ci    = u_ptr[n_dof * c + 3 + s] / h;
          PetscReal tau_b = 0.5 * rhow * Cd * (Square(u) + Square(v));
          PetscReal ei    = kp_constant * (tau_b - tau_critical_erosion) / tau_critical_erosion;
          PetscReal di    = settling_velocity * ci * (1.0 - tau_b / tau_critical_deposition);

          f_ptr[n_dof * owned_cell_id + 3 + s] += (ei - di) + source_ptr[n_dof * owned_cell_id + 3 + s];
        }
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
  PetscCall(VecRestoreArray(mannings_vec, &mannings_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentSource(void *context) {
  PetscFunctionBegin;
  SedimentSourceOperator *source_op = context;
  PetscFree(source_op);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSedimentPetscSourceOperator(RDyMesh *mesh, PetscInt num_flow_comp, PetscInt num_sediment_comp, Vec external_sources,
                                                 Vec mannings, RDyFlowSourceMethod method, PetscReal tiny_h, PetscReal xq2018_threshold,
                                                 PetscOperator *petsc_op) {
  PetscFunctionBegin;
  SedimentSourceOperator *source_op;
  PetscCall(PetscCalloc1(1, &source_op));
  *source_op = (SedimentSourceOperator){
      .mesh              = mesh,
      .num_flow_comp     = num_flow_comp,
      .num_sediment_comp = num_sediment_comp,
      .external_sources  = external_sources,
      .mannings          = mannings,
      .tiny_h            = tiny_h,
      .xq2018_threshold  = xq2018_threshold,
  };

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)external_sources, &comm));

  switch (method) {
    case SOURCE_SEMI_IMPLICIT:
      PetscCall(PetscOperatorCreate(source_op, ApplySedimentSourceSemiImplicit, DestroySedimentSource, petsc_op));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only semi_implicit and implicit_xq2018 are supported in the PETSc version");
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif