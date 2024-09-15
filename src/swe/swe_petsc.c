#ifndef swe_flux_petsc_h
#define swe_flux_petsc_h

#include <petscsys.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

// gravitational acceleration [m/s/s]
static const PetscReal GRAVITY = 9.806;

/// For computing fluxes, allocates structs to hold values left and right
/// of internal and boundary edges. This must be called before CreatePetscSWESource.
PetscErrorCode CreatePetscSWEFlux(RDyCells *cells, RDyEdges *edges, PetscInt ncomp, PetscInt num_internal_edges, PetscInt num_boundaries,
                                  RDyBoundary boundaries[num_boundaries], PetscBool ceed_enabled, void **petsc_rhs) {
  PetscFunctionBegin;

  RiemannDataSWE datal, datar;
  if (!ceed_enabled) {
    PetscCall(RiemannDataSWECreate(num_internal_edges, &datal));
    PetscCall(RiemannDataSWECreate(num_internal_edges, &datar));
  }

  RiemannEdgeDataSWE data_edge_internal;
  if (!ceed_enabled) {
    PetscCall(RiemannEdgeDataSWECreate(num_internal_edges, ncomp, &data_edge_internal));
    for (PetscInt ii = 0; ii < num_internal_edges; ii++) {
      PetscInt iedge = edges->internal_edge_ids[ii];
      PetscInt r     = edges->cell_ids[2 * iedge + 1];

      if (r != -1) {
        data_edge_internal.cn[ii] = edges->cn[iedge];
        data_edge_internal.sn[ii] = edges->sn[iedge];
      }
    }
  }

  RiemannDataSWE *datal_bnd, *datar_bnd;
  if (!ceed_enabled) {
    PetscCall(PetscCalloc1(num_boundaries, &datal_bnd));
    PetscCall(PetscCalloc1(num_boundaries, &datar_bnd));
  }

  RiemannEdgeDataSWE *data_edge_bnd;
  PetscCall(PetscCalloc1(num_boundaries, &data_edge_bnd));

  for (PetscInt b = 0; b < num_boundaries; b++) {
    PetscInt num_edges = boundaries[b].num_edges;
    if (!ceed_enabled) {
      PetscCall(RiemannDataSWECreate(num_edges, &datal_bnd[b]));
      PetscCall(RiemannDataSWECreate(num_edges, &datar_bnd[b]));
    }

    PetscCall(RiemannEdgeDataSWECreate(num_edges, ncomp, &data_edge_bnd[b]));

    if (!ceed_enabled) {
      for (PetscInt e = 0; e < num_edges; e++) {
        PetscInt iedge = boundaries[b].edge_ids[e];

        data_edge_bnd[b].cn[e] = edges->cn[iedge];
        data_edge_bnd[b].sn[e] = edges->sn[iedge];
      }
    }
  }

  PetscRiemannDataSWE *data_swe;
  PetscCall(PetscCalloc1(1, &data_swe));

  if (!ceed_enabled) {
    // set pointers for internal edges
    data_swe->datal_internal_edges = datal;
    data_swe->datar_internal_edges = datar;
    data_swe->data_internal_edges  = data_edge_internal;

    // set pointers for boundary edges
    data_swe->datal_bnd_edges = datal_bnd;
    data_swe->datar_bnd_edges = datar_bnd;
  }
  data_swe->data_bnd_edges = data_edge_bnd;

  // set the pointer
  *petsc_rhs = data_swe;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Allocates a struct for computing the source/sink term
PetscErrorCode CreatePetscSWESource(RDyMesh *mesh, void *petsc_rhs) {
  PetscFunctionBegin;

  PetscInt num = mesh->num_cells;

  RiemannDataSWE *data;
  PetscCall(PetscCalloc1(1, &data));

  PetscCall(RiemannDataSWECreate(num, data));

  PetscRiemannDataSWE *data_swe = petsc_rhs;
  data_swe->data_cells          = *data;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DestroyPetscSWEFlux(void *petsc_rhs, PetscBool ceed_enabled, PetscInt num_boundaries) {
  PetscFunctionBeginUser;

  PetscRiemannDataSWE *data_swe = petsc_rhs;

  if (!ceed_enabled) {
    PetscCall(RiemannDataSWEDestroy(data_swe->datal_internal_edges));
    PetscCall(RiemannDataSWEDestroy(data_swe->datar_internal_edges));
    PetscCall(RiemannEdgeDataSWEDestroy(data_swe->data_internal_edges));
  }

  for (PetscInt b = 0; b < num_boundaries; b++) {
    if (!ceed_enabled) {
      PetscCall(RiemannDataSWEDestroy(data_swe->datal_bnd_edges[b]));
      PetscCall(RiemannDataSWEDestroy(data_swe->datar_bnd_edges[b]));
    }
    PetscCall(RiemannEdgeDataSWEDestroy(data_swe->data_bnd_edges[b]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// computes velocities in x and y-dir based on momentum in x and y-dir
// N - Size of the array
// tiny_h - Threshold value for height
// h - Height
// hu - Momentum in x-dir
// hv - Momentum in y-dir
// u - Velocity in x-dir
// v - Velocity in y-dir
static PetscErrorCode GetVelocityFromMomentum(PetscReal tiny_h, RiemannDataSWE *data) {
  PetscFunctionBeginUser;

  for (PetscInt n = 0; n < data->N; n++) {
    if (data->h[n] < tiny_h) {
      data->u[n] = 0.0;
      data->v[n] = 0.0;
    } else {
      data->u[n] = data->hu[n] / data->h[n];
      data->v[n] = data->hv[n] / data->h[n];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Computes flux based on Roe solver
/// @param [in] N Size of the array
/// @param [in] *datal A RiemannDataSWE for values left of the edges
/// @param [in] *datar A RiemannDataSWE for values right of the edges
/// @param [in] sn Sine of the angle between edge and y-axis
/// @param [in] cn Cosine of the angle between edge and y-axis
/// @param [out] fij Flux through the edges
/// @param [out] amax Maximum courant number
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ComputeRoeFlux(PetscInt N, RiemannDataSWE *datal, RiemannDataSWE *datar, const PetscReal sn[N], const PetscReal cn[N],
                                     PetscReal fij[N * 3], PetscReal amax[N]) {
  PetscFunctionBeginUser;

  PetscReal *hl = datal->h;
  PetscReal *ul = datal->u;
  PetscReal *vl = datal->v;

  PetscReal *hr = datar->h;
  PetscReal *ur = datar->u;
  PetscReal *vr = datar->v;

  PetscAssert(datal->N == datar->N, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Size of data left and right of edges is not the same!");
  PetscAssert(N == datal->N, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Size of data left/right of edges is not the same as sn/cn");

  for (PetscInt n = 0; n < N; n++) {
    // Compute Roe averages
    PetscReal duml  = pow(hl[n], 0.5);
    PetscReal dumr  = pow(hr[n], 0.5);
    PetscReal cl    = pow(GRAVITY * hl[n], 0.5);
    PetscReal cr    = pow(GRAVITY * hr[n], 0.5);
    PetscReal hhat  = duml * dumr;
    PetscReal uhat  = (duml * ul[n] + dumr * ur[n]) / (duml + dumr);
    PetscReal vhat  = (duml * vl[n] + dumr * vr[n]) / (duml + dumr);
    PetscReal chat  = pow(0.5 * GRAVITY * (hl[n] + hr[n]), 0.5);
    PetscReal uperp = uhat * cn[n] + vhat * sn[n];

    PetscReal dh     = hr[n] - hl[n];
    PetscReal du     = ur[n] - ul[n];
    PetscReal dv     = vr[n] - vl[n];
    PetscReal dupar  = -du * sn[n] + dv * cn[n];
    PetscReal duperp = du * cn[n] + dv * sn[n];

    PetscReal dW[3];
    dW[0] = 0.5 * (dh - hhat * duperp / chat);
    dW[1] = hhat * dupar;
    dW[2] = 0.5 * (dh + hhat * duperp / chat);

    PetscReal uperpl = ul[n] * cn[n] + vl[n] * sn[n];
    PetscReal uperpr = ur[n] * cn[n] + vr[n] * sn[n];
    PetscReal al1    = uperpl - cl;
    PetscReal al3    = uperpl + cl;
    PetscReal ar1    = uperpr - cr;
    PetscReal ar3    = uperpr + cr;

    PetscReal R[3][3];
    R[0][0] = 1.0;
    R[0][1] = 0.0;
    R[0][2] = 1.0;
    R[1][0] = uhat - chat * cn[n];
    R[1][1] = -sn[n];
    R[1][2] = uhat + chat * cn[n];
    R[2][0] = vhat - chat * sn[n];
    R[2][1] = cn[n];
    R[2][2] = vhat + chat * sn[n];

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
    PetscReal A[3][3];
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) {
        A[i][j] = 0.0;
      }
    }
    A[0][0] = a1;
    A[1][1] = a2;
    A[2][2] = a3;

    PetscReal FL[3], FR[3];
    FL[0] = uperpl * hl[n];
    FL[1] = ul[n] * uperpl * hl[n] + 0.5 * GRAVITY * hl[n] * hl[n] * cn[n];
    FL[2] = vl[n] * uperpl * hl[n] + 0.5 * GRAVITY * hl[n] * hl[n] * sn[n];

    FR[0] = uperpr * hr[n];
    FR[1] = ur[n] * uperpr * hr[n] + 0.5 * GRAVITY * hr[n] * hr[n] * cn[n];
    FR[2] = vr[n] * uperpr * hr[n] + 0.5 * GRAVITY * hr[n] * hr[n] * sn[n];

    // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
    fij[n * 3 + 0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0][0] * dW[0] - R[0][1] * A[1][1] * dW[1] - R[0][2] * A[2][2] * dW[2]);
    fij[n * 3 + 1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0][0] * dW[0] - R[1][1] * A[1][1] * dW[1] - R[1][2] * A[2][2] * dW[2]);
    fij[n * 3 + 2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0][0] * dW[0] - R[2][1] * A[1][1] * dW[1] - R[2][2] * A[2][2] * dW[2]);

    amax[n] = chat + fabs(uperp);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// computes RHS on internal edges
// rdy               - an RDy object
// F                 - a global vector that stores the fluxes between internal edges
// courant_num_diags - diagnostics struct for tracking maximum courant number
PetscErrorCode SWERHSFunctionForInternalEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // Get pointers to vector data
  PetscScalar *x_ptr, *f_ptr;
  PetscCall(VecGetArray(rdy->X_local, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));

  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->X_local, &ndof));
  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  PetscInt num = mesh->num_internal_edges;

  PetscRiemannDataSWE *data_swe = rdy->petsc_rhs;
  RiemannDataSWE      *datal    = &data_swe->datal_internal_edges;
  RiemannDataSWE      *datar    = &data_swe->datar_internal_edges;
  RiemannDataSWE      *datac    = &data_swe->data_cells;

  RiemannEdgeDataSWE *data_edge    = &data_swe->data_internal_edges;
  PetscReal          *sn_vec_int   = data_edge->sn;
  PetscReal          *cn_vec_int   = data_edge->cn;
  PetscReal          *amax_vec_int = data_edge->amax;
  PetscReal          *flux_vec_int = data_edge->flux;

  // Collect the h/hu/hv for left and right cells to compute u/v
  for (PetscInt ii = 0; ii < mesh->num_internal_edges; ii++) {
    PetscInt iedge = edges->internal_edge_ids[ii];
    PetscInt l     = edges->cell_ids[2 * iedge];
    PetscInt r     = edges->cell_ids[2 * iedge + 1];

    if (r != -1) {
      datal->h[ii]  = datac->h[l];
      datal->u[ii]  = datac->u[l];
      datal->v[ii]  = datac->v[l];
      datal->hu[ii] = datac->hu[l];
      datal->hv[ii] = datac->hv[l];

      datar->h[ii]  = datac->h[r];
      datar->u[ii]  = datac->u[r];
      datar->v[ii]  = datac->v[r];
      datar->hu[ii] = datac->hu[r];
      datar->hv[ii] = datac->hv[r];
    }
  }

  // Compute u/v for left and right cells
  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;

  // Call Riemann solver (only Roe currently supported)
  PetscCheck(rdy->config.numerics.riemann == RIEMANN_ROE, rdy->comm, PETSC_ERR_USER, "Invalid Riemann solver selected! (Only roe is supported)");
  PetscCall(ComputeRoeFlux(num, datal, datar, sn_vec_int, cn_vec_int, flux_vec_int, amax_vec_int));

  // Save the flux values in the Vec based by TS
  for (PetscInt ii = 0; ii < mesh->num_internal_edges; ii++) {
    PetscInt iedge = edges->internal_edge_ids[ii];
    PetscInt l     = edges->cell_ids[2 * iedge];
    PetscInt r     = edges->cell_ids[2 * iedge + 1];

    if (r != -1) {  // internal edge
      PetscReal edge_len = edges->lengths[iedge];

      PetscReal hl = x_ptr[l * ndof + 0];
      PetscReal hr = x_ptr[r * ndof + 0];

      if (!(hr < tiny_h && hl < tiny_h)) {
        PetscReal areal = cells->areas[l];
        PetscReal arear = cells->areas[r];

        PetscReal cnum = amax_vec_int[ii] * edge_len / fmin(areal, arear) * rdy->dt;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[ii];
          if (areal < arear) courant_num_diags->global_cell_id = cells->global_ids[l];
          else courant_num_diags->global_cell_id = cells->global_ids[r];
        }

        for (PetscInt idof = 0; idof < ndof; idof++) {
          if (cells->is_local[l]) {
            PetscInt idx = cells->local_to_owned[l];
            f_ptr[idx * ndof + idof] += flux_vec_int[ii * ndof + idof] * (-edge_len / areal);
          }

          if (cells->is_local[r]) {
            PetscInt idx = cells->local_to_owned[r];
            f_ptr[idx * ndof + idof] += flux_vec_int[ii * ndof + idof] * (edge_len / arear);
          }
        }
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(rdy->X_local, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Before computing BC fluxes, perform common precomputation irrespective of BC type that include:
// (i) extracting h/hu/hv from the solution vector X, and
// (ii) compute velocities (u/v) from momentum (hu/hv).
static PetscErrorCode PerformPrecomputationForBC(RDy rdy, RDyBoundary boundary, PetscReal tiny_h, PetscInt N, RiemannDataSWE *datal,
                                                 RiemannDataSWE *datac) {
  PetscFunctionBeginUser;

  RDyEdges *edges = &rdy->mesh.edges;

  // Collect the h/hu/hv for left cells to compute u/v
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt iedge = boundary.edge_ids[e];
    PetscInt icell = edges->cell_ids[2 * iedge];

    datal->h[e]  = datac->h[icell];
    datal->u[e]  = datac->u[icell];
    datal->v[e]  = datac->v[icell];
    datal->hu[e] = datac->hu[icell];
    datal->hv[e] = datac->hv[icell];
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// After the right values (hr/ur/vr) have been computed based on the different type of BCs,
// compute the fluxes and add contribution in the F vector.
static PetscErrorCode ComputeBC(RDy rdy, RDyBoundary boundary, PetscReal tiny_h, CourantNumberDiagnostics *courant_num_diags, PetscInt N,
                                RiemannDataSWE *datal, RiemannDataSWE *datar, const PetscReal sn[N], const PetscReal cn[N],
                                PetscReal flux_vec_bnd[N * 3], PetscReal amax_vec_bnd[N], PetscReal *F) {
  PetscFunctionBeginUser;

  RDyCells *cells = &rdy->mesh.cells;
  RDyEdges *edges = &rdy->mesh.edges;

  // Call Riemann solver (only Roe is currently supported)
  PetscCheck(rdy->config.numerics.riemann == RIEMANN_ROE, rdy->comm, PETSC_ERR_USER, "Invalid Riemann solver selected! (Only roe is supported)");
  PetscCall(ComputeRoeFlux(boundary.num_edges, datal, datar, sn, cn, flux_vec_bnd, amax_vec_bnd));

  // Save the flux values in the Vec based by TS
  PetscInt ndof      = 3;
  PetscInt num_edges = boundary.num_edges;
  for (PetscInt e = 0; e < num_edges; ++e) {
    PetscInt  iedge     = boundary.edge_ids[e];
    PetscInt  icell     = edges->cell_ids[2 * iedge];
    PetscReal edge_len  = edges->lengths[iedge];
    PetscReal cell_area = cells->areas[icell];

    if (cells->is_local[icell]) {
      PetscReal hl = datal->h[e];
      PetscReal hr = datar->h[e];

      if (!(hl < tiny_h && hr < tiny_h)) {
        PetscReal cnum = amax_vec_bnd[e] * edge_len / cell_area * rdy->dt;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[e];
          courant_num_diags->global_cell_id  = cells->global_ids[icell];
        }

        PetscInt idx = cells->local_to_owned[icell];
        for (PetscInt idof = 0; idof < ndof; idof++) {
          F[ndof * idx + idof] += flux_vec_bnd[e * ndof + idof] * (-edge_len / cell_area);
        }
      }
    }
  }

  PetscCall(AccumulateBoundaryFluxes(rdy, boundary, num_edges, ndof, flux_vec_bnd));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// applies a reflecting boundary condition on the given boundary, computing
// fluxes F for the solution vector components X
static PetscErrorCode ApplyReflectingBC(RDy rdy, RDyBoundary boundary, RiemannDataSWE *datal, RiemannDataSWE *datar, RiemannDataSWE *datac,
                                        RiemannEdgeDataSWE *data_edge, PetscReal tiny_h, CourantNumberDiagnostics *courant_num_diags, PetscReal *F) {
  PetscFunctionBeginUser;

  RDyCells *cells = &rdy->mesh.cells;
  RDyEdges *edges = &rdy->mesh.edges;

  PetscInt num = boundary.num_edges;

  PetscCall(PerformPrecomputationForBC(rdy, boundary, tiny_h, num, datal, datac));

  PetscReal *sn_vec_bnd   = data_edge->sn;
  PetscReal *cn_vec_bnd   = data_edge->cn;
  PetscReal *flux_vec_bnd = data_edge->flux;
  PetscReal *amax_vec_bnd = data_edge->amax;

  // Compute h/u/v for right cells
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt iedge = boundary.edge_ids[e];
    PetscInt icell = edges->cell_ids[2 * iedge];

    if (cells->is_local[icell]) {
      datar->h[e] = datal->h[e];

      PetscReal dum1 = Square(sn_vec_bnd[e]) - Square(cn_vec_bnd[e]);
      PetscReal dum2 = 2.0 * sn_vec_bnd[e] * cn_vec_bnd[e];

      datar->u[e] = datal->u[e] * dum1 - datal->v[e] * dum2;
      datar->v[e] = -datal->u[e] * dum2 - datal->v[e] * dum1;
    }
  }

  PetscCall(ComputeBC(rdy, boundary, tiny_h, courant_num_diags, num, datal, datar, sn_vec_bnd, cn_vec_bnd, flux_vec_bnd, amax_vec_bnd, F));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// applies a dirichlet boundary condition on the given boundary, computing
// fluxes F for the solution vector components X
static PetscErrorCode ApplyDirichletBC(RDy rdy, RDyBoundary boundary, RiemannDataSWE *datal, RiemannDataSWE *datar, RiemannDataSWE *datac,
                                       RiemannEdgeDataSWE *data_edge, PetscReal tiny_h, CourantNumberDiagnostics *courant_num_diags, PetscReal *F) {
  PetscFunctionBeginUser;

  PetscInt num = boundary.num_edges;

  PetscCall(PerformPrecomputationForBC(rdy, boundary, tiny_h, num, datal, datac));

  PetscReal *sn_vec_bnd   = data_edge->sn;
  PetscReal *cn_vec_bnd   = data_edge->cn;
  PetscReal *flux_vec_bnd = data_edge->flux;
  PetscReal *amax_vec_bnd = data_edge->amax;

  // Currently, only time-invariant BC is supported and the values in 'datar' (aka BC) was set during model setup

  PetscCall(ComputeBC(rdy, boundary, tiny_h, courant_num_diags, num, datal, datar, sn_vec_bnd, cn_vec_bnd, flux_vec_bnd, amax_vec_bnd, F));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// applies a critical outflow boundary condition, computing
// fluxes F for the solution vector components X
static PetscErrorCode ApplyCriticalOutflowBC(RDy rdy, RDyBoundary boundary, RiemannDataSWE *datal, RiemannDataSWE *datar, RiemannDataSWE *datac,
                                             RiemannEdgeDataSWE *data_edge, PetscReal tiny_h, CourantNumberDiagnostics *courant_num_diags,
                                             PetscReal *F) {
  PetscFunctionBeginUser;

  RDyCells *cells = &rdy->mesh.cells;
  RDyEdges *edges = &rdy->mesh.edges;

  PetscInt num = boundary.num_edges;

  PetscCall(PerformPrecomputationForBC(rdy, boundary, tiny_h, num, datal, datac));

  PetscReal *sn_vec_bnd   = data_edge->sn;
  PetscReal *cn_vec_bnd   = data_edge->cn;
  PetscReal *flux_vec_bnd = data_edge->flux;
  PetscReal *amax_vec_bnd = data_edge->amax;

  // Compute h/u/v for right cells
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt iedge = boundary.edge_ids[e];
    PetscInt icell = edges->cell_ids[2 * iedge];

    if (cells->is_local[icell]) {
      PetscReal uperp = datal->u[e] * cn_vec_bnd[e] + datal->v[e] * sn_vec_bnd[e];
      PetscReal q     = datal->h[e] * fabs(uperp);

      datar->h[e] = PetscPowReal(Square(q) / GRAVITY, 1.0 / 3.0);

      PetscReal velocity = PetscPowReal(GRAVITY * datar->h[e], 0.5);
      datar->u[e]        = velocity * cn_vec_bnd[e];
      datar->v[e]        = velocity * sn_vec_bnd[e];
    }
  }

  PetscCall(ComputeBC(rdy, boundary, tiny_h, courant_num_diags, num, datal, datar, sn_vec_bnd, cn_vec_bnd, flux_vec_bnd, amax_vec_bnd, F));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// computes RHS on boundary edges
// rdy               - an RDy object
// F                 - a global vector that stores the fluxes between boundary edges
// courant_num_diags - diagnostics struct for tracking maximum courant number
PetscErrorCode SWERHSFunctionForBoundaryEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags) {
  PetscFunctionBeginUser;

  // Get pointers to vector data
  PetscScalar *x_ptr, *f_ptr;
  PetscCall(VecGetArray(rdy->X_local, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));

  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->X_local, &ndof));
  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  const PetscReal tiny_h = rdy->config.physics.flow.tiny_h;

  // loop over all boundaries and apply boundary conditions
  PetscRiemannDataSWE *data_swe = rdy->petsc_rhs;
  for (PetscInt b = 0; b < rdy->num_boundaries; ++b) {
    RDyBoundary         boundary      = rdy->boundaries[b];
    RDyCondition        boundary_cond = rdy->boundary_conditions[b];
    RiemannDataSWE     *datal         = &data_swe->datal_bnd_edges[b];
    RiemannDataSWE     *datar         = &data_swe->datar_bnd_edges[b];
    RiemannDataSWE     *datac         = &data_swe->data_cells;
    RiemannEdgeDataSWE *data_edge     = &data_swe->data_bnd_edges[b];

    switch (boundary_cond.flow->type) {
      case CONDITION_DIRICHLET:
        PetscCall(ApplyDirichletBC(rdy, boundary, datal, datar, datac, data_edge, tiny_h, courant_num_diags, f_ptr));
        break;
      case CONDITION_REFLECTING:
        PetscCall(ApplyReflectingBC(rdy, boundary, datal, datar, datac, data_edge, tiny_h, courant_num_diags, f_ptr));
        break;
      case CONDITION_CRITICAL_OUTFLOW:
        PetscCall(ApplyCriticalOutflowBC(rdy, boundary, datal, datar, datac, data_edge, tiny_h, courant_num_diags, f_ptr));
        break;
      default:
        PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n", boundary.id);
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(rdy->X_local, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Compute u/v based on hu/hv
PetscErrorCode ComputeSWEDiagnosticVariables(RDy rdy) {
  PetscFunctionBeginUser;

  RDyMesh *mesh = &rdy->mesh;

  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->X_local, &ndof));
  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  // Get access to Vec
  PetscScalar *x_ptr;
  PetscCall(VecGetArray(rdy->X_local, &x_ptr));

  PetscRiemannDataSWE *data_swe = rdy->petsc_rhs;
  RiemannDataSWE      *data     = &data_swe->data_cells;

  // Collect the h/hu/hv for cells to compute u/v
  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    data->h[icell]  = x_ptr[icell * ndof + 0];
    data->hu[icell] = x_ptr[icell * ndof + 1];
    data->hv[icell] = x_ptr[icell * ndof + 2];
  }

  // Compute u/v for cells
  PetscCall(GetVelocityFromMomentum(rdy->config.physics.flow.tiny_h, data));

  PetscCall(VecRestoreArray(rdy->X_local, &x_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// adds source terms to the right hand side vector F
PetscErrorCode AddSWESourceTerm(RDy rdy, Vec F) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;

  // Get access to Vec
  PetscScalar *x_ptr, *f_ptr, *swe_src_ptr;
  PetscCall(VecGetArray(rdy->X_local, &x_ptr));
  PetscCall(VecGetArray(rdy->swe_src, &swe_src_ptr));
  PetscCall(VecGetArray(F, &f_ptr));

  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->X_local, &ndof));
  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  PetscRiemannDataSWE *data_swe = rdy->petsc_rhs;
  RiemannDataSWE      *data     = &data_swe->data_cells;

  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    if (cells->is_local[icell]) {
      PetscReal h  = data->h[icell];
      PetscReal hu = data->hu[icell];
      PetscReal hv = data->hv[icell];

      PetscReal dz_dx = cells->dz_dx[icell];
      PetscReal dz_dy = cells->dz_dy[icell];

      PetscReal bedx = dz_dx * GRAVITY * h;
      PetscReal bedy = dz_dy * GRAVITY * h;

      PetscReal u = data->u[icell];
      PetscReal v = data->v[icell];

      PetscInt  idx    = cells->local_to_owned[icell];
      PetscReal Fsum_x = f_ptr[idx * ndof + 1];
      PetscReal Fsum_y = f_ptr[idx * ndof + 2];

      PetscReal tbx = 0.0, tby = 0.0;

      if (h >= rdy->config.physics.flow.tiny_h) {
        // Manning's coefficient
        PetscReal N_mannings = rdy->materials_by_cell[icell].manning;

        // Cd = g n^2 h^{-1/3}, where n is Manning's coefficient
        PetscReal Cd = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);

        PetscReal velocity = PetscSqrtReal(Square(u) + Square(v));

        PetscReal tb = Cd * velocity / h;

        PetscReal dt     = rdy->dt;
        PetscReal factor = tb / (1.0 + dt * tb);

        tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
        tby = (hv + dt * Fsum_y - dt * bedy) * factor;
      }

      f_ptr[idx * ndof + 0] += swe_src_ptr[idx * ndof + 0];
      f_ptr[idx * ndof + 1] += -bedx - tbx + swe_src_ptr[idx * ndof + 1];
      f_ptr[idx * ndof + 2] += -bedy - tby + swe_src_ptr[idx * ndof + 2];
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(rdy->X_local, &x_ptr));
  PetscCall(VecRestoreArray(rdy->swe_src, &swe_src_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Sets Dirichlet boundary values on the boundary with the given index. NOTE
// that the boundary index b identifies the bth boundary in RDycore's array of
// boundaries and boundary conditions, and NOT the boundary with ID b.
PetscErrorCode GetPetscSWEDirichletBoundaryValues(void *petsc_rhs, PetscInt boundary_index, RiemannDataSWE *boundary_data) {
  PetscFunctionBegin;

  PetscRiemannDataSWE *data_swe = petsc_rhs;
  *boundary_data                = data_swe->datar_bnd_edges[boundary_index];

  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif  // swe_flux_petsc_h
