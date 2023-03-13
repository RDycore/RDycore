#include <private/rdycoreimpl.h>
#include <private/rdymathimpl.h>
#include <stddef.h>  // for offsetof

// gravitational acceleration [m/s/s]
static const PetscReal GRAVITY = 9.806;

//-----------------------
// Debugging diagnostics
//-----------------------

// Diagnostic structure that captures information about the conditions under
// which the maximum courant number is encountered. If you change this struct,
// update the call to MPI_Type_create_struct in InitMPITypesAndOps below.
typedef struct {
  PetscReal max_courant_num;  // maximum courant number
  PetscInt  global_edge_id;   // edge at which the max courant number was encountered
  PetscInt  global_cell_id;   // cell in which the max courant number was encountered
} CourantNumberDiagnostics;

// MPI datatype corresponding to CourantNumberDiagnostics. Created during InitSWE.
static MPI_Datatype courant_num_diags_type;

// MPI operator used to determine the prevailing diagnostics for the maximum
// courant number on all processes. Created during InitSWE.
static MPI_Op courant_num_diags_op;

// function backing the above MPI operator
static void FindCourantNumberDiagnostics(void *in_vec, void *result_vec, int *len, MPI_Datatype *type) {
  CourantNumberDiagnostics *in_diags     = in_vec;
  CourantNumberDiagnostics *result_diags = result_vec;

  // select the item with the maximum courant number
  for (int i = 0; i < *len; ++i) {
    if (in_diags[i].max_courant_num > result_diags[i].max_courant_num) {
      result_diags[i] = in_diags[i];
    }
  }
}

// this function destroys the MPI type and operator associated with CourantNumberDiagnostics
static void DestroyCourantNumberDiagnostics(void) {
  MPI_Op_free(&courant_num_diags_op);
  MPI_Type_free(&courant_num_diags_type);
}

// this function is called by InitSWE to initialize the above type(s) and op(s).
static PetscErrorCode InitMPITypesAndOps(void) {
  PetscFunctionBegin;

  // create an MPI data type for the CourantNumberDiagnostics struct
  const int      num_blocks             = 3;
  const int      block_lengths[3]       = {1, 1, 1};
  const MPI_Aint block_displacements[3] = {
      offsetof(CourantNumberDiagnostics, max_courant_num),
      offsetof(CourantNumberDiagnostics, global_edge_id),
      offsetof(CourantNumberDiagnostics, global_cell_id),
  };
  MPI_Datatype block_types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
  MPI_Type_create_struct(num_blocks, block_lengths, block_displacements, block_types, &courant_num_diags_type);
  MPI_Type_commit(&courant_num_diags_type);

  // create a corresponding reduction operator for the new type
  MPI_Op_create(FindCourantNumberDiagnostics, 1, &courant_num_diags_op);

  // make sure the operator and the type are destroyed upon exit
  PetscCall(RDyOnFinalize(DestroyCourantNumberDiagnostics));

  PetscFunctionReturn(0);
}

//---------------------------
// End debugging diagnostics
//---------------------------

// computes velocities in x and y-dir based on momentum in x and y-dir
// N - Size of the array
// tiny_h - Threshold value for height
// h - Height
// hu - Momentum in x-dir
// hv - Momentum in y-dir
// u - Velocity in x-dir
// v - Velocity in y-dir
static PetscErrorCode GetVelocityFromMomentum(PetscInt N, PetscReal tiny_h, const PetscReal h[N], const PetscReal hu[N], const PetscReal hv[N],
                                              PetscReal u[N], PetscReal v[N]) {
  PetscFunctionBeginUser;

  for (PetscInt n = 0; n < N; n++) {
    if (h[n] < tiny_h) {
      u[n] = 0.0;
      v[n] = 0.0;
    } else {
      u[n] = hu[n] / h[n];
      v[n] = hv[n] / h[n];
    }
  }

  PetscFunctionReturn(0);
}

// Computes flux based on Roe solver
// N    - Size of the array
// hl   - Height left of the edge
// hr   - Height right of the edge
// ul   - Velocity in x-dir left of the edge
// ur   - Velocity in x-dir right of the edge
// vl   - Velocity in y-dir left of the edge
// vr   - Velocity in y-dir right of the edge
// sn   - sine of the angle between edge and y-axis
// cn   - cosine of the angle between edge and y-axis
// fij  - flux
// amax - maximum courant number
static PetscErrorCode ComputeRoeFlux(PetscInt N, const PetscReal hl[N], const PetscReal hr[N], const PetscReal ul[N], const PetscReal ur[N],
                                     const PetscReal vl[N], const PetscReal vr[N], const PetscReal sn[N], const PetscReal cn[N], PetscReal fij[N][3],
                                     PetscReal amax[N]) {
  PetscFunctionBeginUser;

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
    fij[n][0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0][0] * dW[0] - R[0][1] * A[1][1] * dW[1] - R[0][2] * A[2][2] * dW[2]);
    fij[n][1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0][0] * dW[0] - R[1][1] * A[1][1] * dW[1] - R[1][2] * A[2][2] * dW[2]);
    fij[n][2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0][0] * dW[0] - R[2][1] * A[1][1] * dW[1] - R[2][2] * A[2][2] * dW[2]);

    amax[n] = chat + fabs(uperp);
  }

  PetscFunctionReturn(0);
}

// computes RHS on internal edges
// rdy               - an RDy object
// F                 - a global vector that stores the fluxes between internal edges
// courant_num_diags - diagnostics struct for tracking maximum courant number
static PetscErrorCode RHSFunctionForInternalEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags) {
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

  PetscInt  num = mesh->num_internal_edges;
  PetscReal hl_vec_int[num], hul_vec_int[num], hvl_vec_int[num], ul_vec_int[num], vl_vec_int[num];
  PetscReal hr_vec_int[num], hur_vec_int[num], hvr_vec_int[num], ur_vec_int[num], vr_vec_int[num];
  PetscReal sn_vec_int[num], cn_vec_int[num];
  PetscReal flux_vec_int[num][3], amax_vec_int[num];

  // Collect the h/hu/hv for left and right cells to compute u/v
  for (PetscInt ii = 0; ii < mesh->num_internal_edges; ii++) {
    PetscInt iedge = edges->internal_edge_ids[ii];
    PetscInt l     = edges->cell_ids[2 * iedge];
    PetscInt r     = edges->cell_ids[2 * iedge + 1];

    if (r != -1) {
      hl_vec_int[ii]  = x_ptr[l * ndof + 0];
      hul_vec_int[ii] = x_ptr[l * ndof + 1];
      hvl_vec_int[ii] = x_ptr[l * ndof + 2];

      hr_vec_int[ii]  = x_ptr[r * ndof + 0];
      hur_vec_int[ii] = x_ptr[r * ndof + 1];
      hvr_vec_int[ii] = x_ptr[r * ndof + 2];

      cn_vec_int[ii] = edges->cn[iedge];
      sn_vec_int[ii] = edges->sn[iedge];
    }
  }

  // Compute u/v for left and right cells
  const PetscReal tiny_h = rdy->config.tiny_h;
  PetscCall(GetVelocityFromMomentum(num, tiny_h, hl_vec_int, hul_vec_int, hvl_vec_int, ul_vec_int, vl_vec_int));
  PetscCall(GetVelocityFromMomentum(num, tiny_h, hr_vec_int, hur_vec_int, hvr_vec_int, ur_vec_int, vr_vec_int));

  // Call Riemann solver (only Roe currently supported)
  PetscCheck(rdy->config.riemann == RIEMANN_ROE, rdy->comm, PETSC_ERR_USER, "Invalid Riemann solver selected! (Only roe is supported)");
  PetscCall(ComputeRoeFlux(num, hl_vec_int, hr_vec_int, ul_vec_int, ur_vec_int, vl_vec_int, vr_vec_int, sn_vec_int, cn_vec_int, flux_vec_int,
                           amax_vec_int));

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
          if (cells->is_local[l]) f_ptr[l * ndof + idof] -= flux_vec_int[ii][idof] * edge_len / areal;
          if (cells->is_local[r]) f_ptr[r * ndof + idof] += flux_vec_int[ii][idof] * edge_len / arear;
        }
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(rdy->X_local, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));

  PetscFunctionReturn(0);
}

// computes RHS on boundary edges
// rdy               - an RDy object
// F                 - a global vector that stores the fluxes between boundary edges
// courant_num_diags - diagnostics struct for tracking maximum courant number
static PetscErrorCode RHSFunctionForBoundaryEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags) {
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

  const PetscReal tiny_h = rdy->config.tiny_h;

  // loop over all surfaces that correspond to reflecting boundary conditions
  for (PetscInt s = 0; s < rdy->num_surfaces; ++s) {
    // skip non-reflecting surfaces
    if (rdy->boundary_conditions[s].flow->type != CONDITION_REFLECTING) continue;

    RDySurface *surface = &rdy->surfaces[s];
    PetscInt    num     = surface->num_edges;

    PetscReal hl_vec_bnd[num], hul_vec_bnd[num], hvl_vec_bnd[num], ul_vec_bnd[num], vl_vec_bnd[num];
    PetscReal hr_vec_bnd[num], ur_vec_bnd[num], vr_vec_bnd[num];
    PetscReal sn_vec_bnd[num], cn_vec_bnd[num];
    PetscReal flux_vec_bnd[num][3], amax_vec_bnd[num];

    // Collect the h/hu/hv for left cells to compute u/v
    for (PetscInt e = 0; e < surface->num_edges; ++e) {
      PetscInt iedge = surface->edge_ids[e];
      PetscInt icell = edges->cell_ids[2 * iedge];

      hl_vec_bnd[e]  = x_ptr[icell * ndof + 0];
      hul_vec_bnd[e] = x_ptr[icell * ndof + 1];
      hvl_vec_bnd[e] = x_ptr[icell * ndof + 2];
    }

    // Compute u/v for left cells
    PetscCall(GetVelocityFromMomentum(num, tiny_h, hl_vec_bnd, hul_vec_bnd, hvl_vec_bnd, ul_vec_bnd, vl_vec_bnd));

    // Compute h/u/v for right cells
    for (PetscInt e = 0; e < surface->num_edges; ++e) {
      PetscInt iedge = surface->edge_ids[e];
      PetscInt icell = edges->cell_ids[2 * iedge];

      cn_vec_bnd[e] = edges->cn[iedge];
      sn_vec_bnd[e] = edges->sn[iedge];

      if (cells->is_local[icell]) {
        hr_vec_bnd[e] = hl_vec_bnd[e];

        PetscReal dum1 = Square(sn_vec_bnd[e]) - Square(cn_vec_bnd[e]);
        PetscReal dum2 = 2.0 * sn_vec_bnd[e] * cn_vec_bnd[e];

        ur_vec_bnd[e] = ul_vec_bnd[e] * dum1 - vl_vec_bnd[e] * dum2;
        vr_vec_bnd[e] = -ul_vec_bnd[e] * dum2 - vl_vec_bnd[e] * dum1;
      }
    }

    // Call Riemann solver (only Roe is currently supported)
    PetscCheck(rdy->config.riemann == RIEMANN_ROE, rdy->comm, PETSC_ERR_USER, "Invalid Riemann solver selected! (Only roe is supported)");
    PetscCall(ComputeRoeFlux(num, hl_vec_bnd, hr_vec_bnd, ul_vec_bnd, ur_vec_bnd, vl_vec_bnd, vr_vec_bnd, sn_vec_bnd, cn_vec_bnd, flux_vec_bnd,
                             amax_vec_bnd));

    // Save the flux values in the Vec based by TS
    for (PetscInt e = 0; e < surface->num_edges; ++e) {
      PetscInt  iedge     = surface->edge_ids[e];
      PetscInt  icell     = edges->cell_ids[2 * iedge];
      PetscReal edge_len  = edges->lengths[iedge];
      PetscReal cell_area = cells->areas[icell];

      if (cells->is_local[icell]) {
        PetscReal hl = x_ptr[icell * ndof + 0];

        if (!(hl < tiny_h)) {
          PetscReal cnum = amax_vec_bnd[icell] * edge_len / cell_area * rdy->dt;
          if (cnum > courant_num_diags->max_courant_num) {
            courant_num_diags->max_courant_num = cnum;
            courant_num_diags->global_edge_id  = edges->global_ids[e];
            courant_num_diags->global_cell_id  = cells->global_ids[icell];
          }

          for (PetscInt idof = 0; idof < ndof; idof++) {
            f_ptr[icell * ndof + idof] -= flux_vec_bnd[e][idof] * edge_len / cell_area;
          }
        }
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(rdy->X_local, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));

  PetscFunctionReturn(0);
}

// adds source terms to the right hand side vector F
static PetscErrorCode AddSourceTerm(RDy rdy, Vec F) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;

  // Get access to Vec
  PetscScalar *x_ptr, *f_ptr;
  PetscCall(VecGetArray(rdy->X_local, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));

  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->X_local, &ndof));
  PetscCheck(ndof == 3, rdy->comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  PetscInt  N = mesh->num_cells;
  PetscReal h_vec[N], hu_vec[N], hv_vec[N], u_vec[N], v_vec[N];

  // Collect the h/hu/hv for cells to compute u/v
  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    h_vec[icell]  = x_ptr[icell * ndof + 0];
    hu_vec[icell] = x_ptr[icell * ndof + 1];
    hv_vec[icell] = x_ptr[icell * ndof + 2];
  }

  // Compute u/v for cells
  PetscCall(GetVelocityFromMomentum(N, rdy->config.tiny_h, h_vec, hu_vec, hv_vec, u_vec, v_vec));

  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    if (cells->is_local[icell]) {
      PetscReal h  = h_vec[icell];
      PetscReal hu = hu_vec[icell];
      PetscReal hv = hv_vec[icell];

      PetscReal dz_dx = cells->dz_dx[icell];
      PetscReal dz_dy = cells->dz_dy[icell];

      PetscReal bedx = dz_dx * GRAVITY * h;
      PetscReal bedy = dz_dy * GRAVITY * h;

      PetscReal u = u_vec[icell];
      PetscReal v = v_vec[icell];

      PetscReal Fsum_x = f_ptr[icell * ndof + 1];
      PetscReal Fsum_y = f_ptr[icell * ndof + 2];

      PetscReal tbx = 0.0, tby = 0.0;

      if (h >= rdy->config.tiny_h) {
        // Manning's coefficient
        PetscReal N_mannings = 0.015;

        // Cd = g n^2 h^{-1/3}, where n is Manning's coefficient
        PetscReal Cd = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);

        PetscReal velocity = PetscSqrtReal(Square(u) + Square(v));

        PetscReal tb = Cd * velocity / h;

        PetscReal dt     = rdy->dt;
        PetscReal factor = tb / (1.0 + dt * tb);

        tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
        tby = (hv + dt * Fsum_y - dt * bedy) * factor;
      }

      f_ptr[icell * ndof + 0] += 0.0;
      f_ptr[icell * ndof + 1] += -bedx - tbx;
      f_ptr[icell * ndof + 2] += -bedy - tby;
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(rdy->X_local, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));

  PetscFunctionReturn(0);
}

// This function initializes SWE physics for the given dycore.
PetscErrorCode InitSWE(RDy rdy) {
  PetscFunctionBeginUser;

  // set up MPI types and operators used by SWE physics
  PetscCall(InitMPITypesAndOps());

  PetscFunctionReturn(0);
}

// This is the right-hand-side function used by our timestepping solver for
// the shallow water equations.
// Parameters:
//  ts  - the solver
//  t   - the simulation time [minutes]
//  X   - the solution vector at time t
//  F   - the right hand side vector to be evaluated at time t
//  ctx - a generic pointer to our RDy object
PetscErrorCode RHSFunctionSWE(TS ts, PetscReal t, Vec X, Vec F, void *ctx) {
  PetscFunctionBegin;

  RDy rdy = ctx;
  DM  dm  = rdy->dm;

  PetscCall(VecZeroEntries(F));

  // populate the local X vector
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, rdy->X_local));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, rdy->X_local));

  // compute the right hand side
  CourantNumberDiagnostics courant_num_diags = {
      .max_courant_num = 0.0,
      .global_edge_id  = -1,
      .global_cell_id  = -1,
  };
  PetscCall(RHSFunctionForInternalEdges(rdy, F, &courant_num_diags));
  PetscCall(RHSFunctionForBoundaryEdges(rdy, F, &courant_num_diags));
  PetscCall(AddSourceTerm(rdy, F));

  // write out debugging info for maximum courant number
  if (rdy->config.log_level >= LOG_DEBUG) {
    MPI_Allreduce(MPI_IN_PLACE, &courant_num_diags, 1, courant_num_diags_type, courant_num_diags_op, rdy->comm);
    PetscReal dt;
    PetscCall(TSGetTimeStep(ts, &dt));
    RDyLogDebug(rdy, "Max courant number %g encountered at edge %d of cell %d is %f", courant_num_diags.max_courant_num,
                courant_num_diags.global_edge_id, courant_num_diags.global_cell_id, courant_num_diags.max_courant_num);
  }
  rdy->step++;

  PetscFunctionReturn(0);
}
