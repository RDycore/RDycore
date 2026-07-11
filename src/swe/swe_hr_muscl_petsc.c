// Hydrostatic reconstruction (HR) with second-order MUSCL reconstruction for the
// PETSc/CPU backend. This is the opt-in (-hr_muscl) well-balanced, positivity-
// preserving scheme of Audusse & Bristeau (2004).
//
// The reconstruction operates on the free surface eta = h + z (rather than the
// depth h) so that lake-at-rest (eta = const) is preserved to second order: a
// constant eta has zero gradient, so the reconstruction returns cell values and
// HR reproduces the first-order well-balanced state exactly.
//
// Structure mirrors ApplyInteriorFlux2R in swe_petsc.c: only the owning rank
// solves each interior edge, contributions are accumulated into a local vector
// (including ghost cells), and DMLocalToGlobal(ADD_VALUES) adds the ghost-side
// contributions back onto their owning ranks.

#include <private/rdymathimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysweimpl.h>

#include "swe_riemann_petsc.h"
#include "swe_roe_flux_petsc.h"

//----------------------
// MUSCL slope limiters
//----------------------

// Returns minmod(a, b): zero if opposite signs, else value with smaller magnitude.
static inline PetscReal Minmod(PetscReal a, PetscReal b) {
  if (a * b <= 0.0) return 0.0;
  return PetscAbsReal(a) < PetscAbsReal(b) ? a : b;
}

// Returns the van Leer-limited slope: 2ab/(a+b) for ab>0, else 0.
static inline PetscReal VanLeer(PetscReal a, PetscReal b) {
  if (a * b <= 0.0) return 0.0;
  return 2.0 * a * b / (a + b);
}

// Applies the selected slope limiter to a local extrapolation `extrap` against the
// neighbor half-jump `half_dq` (signed toward the same face).
static inline PetscReal LimitSlope(RDyLimiterType limiter, PetscReal extrap, PetscReal half_dq) {
  switch (limiter) {
    case LIMITER_NONE:
      return extrap;
    case LIMITER_VANLEER:
      return VanLeer(extrap, half_dq);
    case LIMITER_MINMOD:
    default:
      return Minmod(extrap, half_dq);
  }
}

// Reconstructs face-centered (eta, hu, hv) states for all owned interior edges by
// limited linear extrapolation from cell centroids. Identical in spirit to
// ReconstructFaceValues (operator_fluxes_ceed.c) but reconstructs the free
// surface eta in component 0 and therefore does NOT clamp it to >= 0 (beds, and
// hence eta, may be negative).
//
// q_eta layout per cell:  [eta, hu, hv]  (size num_cells * 3)
// q_face layout per owned edge: [eta_L, hu_L, hv_L, eta_R, hu_R, hv_R] (size num_owned_internal_edges * 6)
static PetscErrorCode ReconstructFaceValuesEta(RDyMesh *mesh, const PetscScalar *q_eta, const PetscScalar *grad_eta, const PetscScalar *grad_hu,
                                               const PetscScalar *grad_hv, RDyLimiterType limiter, PetscScalar *q_face) {
  PetscFunctionBeginUser;

  RDyCells    *cells    = &mesh->cells;
  RDyEdges    *edges    = &mesh->edges;
  RDyVertices *vertices = &mesh->vertices;

  PetscInt owned_edge = 0;
  for (PetscInt ie = 0; ie < mesh->num_internal_edges; ie++) {
    PetscInt e = edges->internal_edge_ids[ie];
    if (!edges->is_owned[e]) continue;

    PetscInt cl = edges->cell_ids[2 * e];
    PetscInt cr = edges->cell_ids[2 * e + 1];

    // Edge midpoint from its two vertices
    PetscInt  v0    = edges->vertex_ids[2 * e];
    PetscInt  v1    = edges->vertex_ids[2 * e + 1];
    PetscReal x_mid = 0.5 * (vertices->points[v0].X[0] + vertices->points[v1].X[0]);
    PetscReal y_mid = 0.5 * (vertices->points[v0].X[1] + vertices->points[v1].X[1]);

    // Displacement vectors from cell centroids to edge midpoint
    PetscReal dx_L = x_mid - cells->centroids[cl].X[0];
    PetscReal dy_L = y_mid - cells->centroids[cl].X[1];
    PetscReal dx_R = x_mid - cells->centroids[cr].X[0];
    PetscReal dy_R = y_mid - cells->centroids[cr].X[1];

    // Ghost-cell gradients are filled by CommunicateCellGradients before this
    // routine runs, so both sides carry complete gradients.
    const PetscScalar *grads[3] = {grad_eta, grad_hu, grad_hv};
    for (PetscInt k = 0; k < 3; k++) {
      PetscReal extrap_L = grads[k][cl * 2 + 0] * dx_L + grads[k][cl * 2 + 1] * dy_L;
      PetscReal extrap_R = grads[k][cr * 2 + 0] * dx_R + grads[k][cr * 2 + 1] * dy_R;

      // Limit each extrapolation against half the cell-to-cell jump.
      PetscReal dq                   = q_eta[cr * 3 + k] - q_eta[cl * 3 + k];
      q_face[owned_edge * 6 + k]     = q_eta[cl * 3 + k] + LimitSlope(limiter, extrap_L, 0.5 * dq);
      q_face[owned_edge * 6 + 3 + k] = q_eta[cr * 3 + k] + LimitSlope(limiter, extrap_R, -0.5 * dq);
    }
    // NOTE: no clamp on component 0 here -- it is the free surface eta, not depth.

    owned_edge++;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//--------------------------------------------
// HR + MUSCL Interior Flux Operator (2nd order)
//--------------------------------------------

typedef struct {
  RDyNumericsRiemann   riemann;
  RDyMesh             *mesh;
  PetscReal            tiny_h;
  PetscReal            h_anuga_regular;
  PetscReal           *zc;             // per-cell bed elevation (local indexing)
  RDyLimiterType       limiter;        // MUSCL slope limiter
  PetscReal           *ls_grad_coeffs; // [num_internal_edges * 4] precomputed LS gradient coeffs
  PetscScalar         *q_eta;          // [num_cells * 3]: (eta, hu, hv) per cell
  PetscScalar         *grad_eta;       // [num_cells * 2]: cell-centered gradient of eta
  PetscScalar         *grad_hu;        // [num_cells * 2]
  PetscScalar         *grad_hv;        // [num_cells * 2]
  PetscScalar         *q_reconstructed;// [num_owned_internal_edges * 6]: reconstructed face states
  RiemannStateData     left_states;    // reconstructed "left" states on owned interior edges
  RiemannStateData     right_states;   // reconstructed "right" states on owned interior edges
  RiemannEdgeData      edges;
  OperatorDiagnostics *diagnostics;
} InteriorFluxHR2ROperator;

static PetscErrorCode ApplyInteriorFluxHR2R(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  InteriorFluxHR2ROperator *op = context;

  RDyMesh  *mesh  = op->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscScalar *u_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == 3, comm, PETSC_ERR_USER, "Number of dof in local vector must be 3!");

  RiemannStateData *datal        = &op->left_states;
  RiemannStateData *datar        = &op->right_states;
  RiemannEdgeData  *data_edge    = &op->edges;
  PetscReal        *sn_vec_int   = data_edge->sn;
  PetscReal        *cn_vec_int   = data_edge->cn;
  PetscReal        *amax_vec_int = data_edge->amax;
  PetscReal        *flux_vec_int = data_edge->fluxes;

  const PetscReal  tiny_h  = op->tiny_h;
  const PetscReal  h_anuga = op->h_anuga_regular;
  const PetscReal *zc      = op->zc;

  // Build the reconstruction variables (eta = h + z, hu, hv) over ALL local
  // cells (including ghosts), then compute limited cell gradients, communicate
  // them into ghost cells, and reconstruct face states for owned edges.
  for (PetscInt c = 0; c < mesh->num_cells; c++) {
    op->q_eta[3 * c + 0] = u_ptr[n_dof * c + 0] + zc[c];
    op->q_eta[3 * c + 1] = u_ptr[n_dof * c + 1];
    op->q_eta[3 * c + 2] = u_ptr[n_dof * c + 2];
  }

  DM dm;
  PetscCall(VecGetDM(u_local, &dm));
  PetscCall(ComputeLeastSquaresGradients(mesh, op->ls_grad_coeffs, op->q_eta, op->grad_eta, op->grad_hu, op->grad_hv));
  PetscCall(CommunicateCellGradients(dm, mesh, op->grad_eta, op->grad_hu, op->grad_hv));
  PetscCall(ReconstructFaceValuesEta(mesh, op->q_eta, op->grad_eta, op->grad_hu, op->grad_hv, op->limiter, op->q_reconstructed));

  // Hydrostatic reconstruction of the face states, packed by owned-edge index.
  PetscInt owned_e = 0;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    if (!edges->is_owned[edge_id]) continue;

    PetscInt l = edges->cell_ids[2 * edge_id];
    PetscInt r = edges->cell_ids[2 * edge_id + 1];

    PetscReal eta_L = op->q_reconstructed[owned_e * 6 + 0];
    PetscReal hu_L  = op->q_reconstructed[owned_e * 6 + 1];
    PetscReal hv_L  = op->q_reconstructed[owned_e * 6 + 2];
    PetscReal eta_R = op->q_reconstructed[owned_e * 6 + 3];
    PetscReal hu_R  = op->q_reconstructed[owned_e * 6 + 4];
    PetscReal hv_R  = op->q_reconstructed[owned_e * 6 + 5];

    // physical reconstructed depths (relative to each cell's own bed)
    PetscReal hphys_L = fmax(0.0, eta_L - zc[l]);
    PetscReal hphys_R = fmax(0.0, eta_R - zc[r]);

    // hydrostatic reconstruction against the higher bed at the interface
    PetscReal z_max  = fmax(zc[l], zc[r]);
    PetscReal hL_rec = fmax(0.0, eta_L - z_max);
    PetscReal hR_rec = fmax(0.0, eta_R - z_max);

    // velocities via ANUGA regularization from the physical reconstructed depth
    PetscReal denom_L = Square(hphys_L) + Square(h_anuga);
    PetscReal denom_R = Square(hphys_R) + Square(h_anuga);
    PetscReal uL      = (hphys_L > tiny_h) ? hu_L * hphys_L / denom_L : 0.0;
    PetscReal vL      = (hphys_L > tiny_h) ? hv_L * hphys_L / denom_L : 0.0;
    PetscReal uR      = (hphys_R > tiny_h) ? hu_R * hphys_R / denom_R : 0.0;
    PetscReal vR      = (hphys_R > tiny_h) ? hv_R * hphys_R / denom_R : 0.0;

    datal->h[owned_e] = hL_rec;
    datal->u[owned_e] = uL;
    datal->v[owned_e] = vL;
    datar->h[owned_e] = hR_rec;
    datar->u[owned_e] = uR;
    datar->v[owned_e] = vR;

    owned_e++;
  }

  // Riemann solver on the reconstructed states
  switch (op->riemann) {
    case RIEMANN_ROE:
      PetscCall(ComputeSWERoeFlux(datal, datar, sn_vec_int, cn_vec_int, flux_vec_int, amax_vec_int));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Unsupported Riemann solver");
  }

  // Accumulate fluxes + hydrostatic pressure correction into a local vector,
  // then add ghost-cell contributions back to their owning ranks.
  Vec          rhs_local;
  PetscScalar *rhs_local_ptr;
  PetscCall(DMGetLocalVector(dm, &rhs_local));
  PetscCall(VecZeroEntries(rhs_local));
  PetscCall(VecGetArray(rhs_local, &rhs_local_ptr));

  owned_e = 0;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    if (!edges->is_owned[edge_id]) continue;

    PetscInt l = edges->cell_ids[2 * edge_id];
    PetscInt r = edges->cell_ids[2 * edge_id + 1];

    PetscReal eta_L   = op->q_reconstructed[owned_e * 6 + 0];
    PetscReal eta_R   = op->q_reconstructed[owned_e * 6 + 3];
    PetscReal hphys_L = fmax(0.0, eta_L - zc[l]);
    PetscReal hphys_R = fmax(0.0, eta_R - zc[r]);
    PetscReal z_max   = fmax(zc[l], zc[r]);
    PetscReal hL_rec  = fmax(0.0, eta_L - z_max);
    PetscReal hR_rec  = fmax(0.0, eta_R - z_max);

    if (!(hphys_R < tiny_h && hphys_L < tiny_h)) {
      PetscReal edge_len     = edges->lengths[edge_id];
      PetscReal areal        = cells->areas[l];
      PetscReal arear        = cells->areas[r];
      PetscReal flux_scale_l = -edge_len / areal;
      PetscReal flux_scale_r = edge_len / arear;

      if (hL_rec > tiny_h || hR_rec > tiny_h) {
        PetscReal                 cnum              = amax_vec_int[owned_e] * edge_len / fmin(areal, arear) * dt;
        CourantNumberDiagnostics *courant_num_diags = &op->diagnostics->courant_number;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[edge_id];
          if (areal < arear) courant_num_diags->global_cell_id = cells->global_ids[l];
          else courant_num_diags->global_cell_id = cells->global_ids[r];
        }

        for (PetscInt i_dof = 0; i_dof < n_dof; i_dof++) {
          rhs_local_ptr[n_dof * l + i_dof] += flux_vec_int[n_dof * owned_e + i_dof] * flux_scale_l;
          rhs_local_ptr[n_dof * r + i_dof] += flux_vec_int[n_dof * owned_e + i_dof] * flux_scale_r;
        }
      }

      // hydrostatic pressure correction applied per cell (even when both
      // reconstructed heights are dry, matching the first-order HR operator).
      PetscReal corr_L = 0.5 * GRAVITY * (Square(hphys_L) - Square(hL_rec));
      PetscReal corr_R = 0.5 * GRAVITY * (Square(hphys_R) - Square(hR_rec));
      PetscReal cn     = cn_vec_int[owned_e];
      PetscReal sn     = sn_vec_int[owned_e];

      rhs_local_ptr[n_dof * l + 1] += corr_L * cn * flux_scale_l;
      rhs_local_ptr[n_dof * l + 2] += corr_L * sn * flux_scale_l;
      rhs_local_ptr[n_dof * r + 1] += corr_R * cn * flux_scale_r;
      rhs_local_ptr[n_dof * r + 2] += corr_R * sn * flux_scale_r;
    }

    owned_e++;
  }

  PetscCall(VecRestoreArray(rhs_local, &rhs_local_ptr));
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(DMLocalToGlobal(dm, rhs_local, ADD_VALUES, f_global));
  PetscCall(DMRestoreLocalVector(dm, &rhs_local));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyInteriorFluxHR2R(void *context) {
  PetscFunctionBegin;
  InteriorFluxHR2ROperator *op = context;
  PetscCall(DestroyRiemannStateData(op->left_states));
  PetscCall(DestroyRiemannStateData(op->right_states));
  PetscCall(DestroyRiemannEdgeData(op->edges));
  PetscCall(PetscFree(op->zc));
  PetscCall(PetscFree(op->ls_grad_coeffs));
  PetscCall(PetscFree(op->q_eta));
  PetscCall(PetscFree(op->grad_eta));
  PetscCall(PetscFree(op->grad_hu));
  PetscCall(PetscFree(op->grad_hv));
  PetscCall(PetscFree(op->q_reconstructed));
  PetscCall(PetscFree(op));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that computes interior fluxes with hydrostatic
/// reconstruction AND second-order MUSCL reconstruction of the free surface,
/// for well-balanced, positivity-preserving shallow water equations (opt-in via
/// the -hr_muscl flag). PETSc/CPU backend only.
PetscErrorCode CreatePetscSWEInteriorFluxHR2ROperator(RDyMesh *mesh, MPI_Comm comm, const RDyConfig config, OperatorDiagnostics *diagnostics,
                                                      PetscOperator *petsc_op) {
  PetscFunctionBegin;

  const PetscInt num_comp = 3;

  InteriorFluxHR2ROperator *op;
  PetscCall(PetscCalloc1(1, &op));
  *op = (InteriorFluxHR2ROperator){
      .riemann         = config.numerics.riemann,
      .mesh            = mesh,
      .diagnostics     = diagnostics,
      .tiny_h          = config.physics.flow.tiny_h,
      .h_anuga_regular = config.physics.flow.h_anuga_regular,
  };

  // Resolve the slope limiter (same precedence as CreatePetscSWEInteriorFluxOperator).
  RDyLimiterType limiter = config.numerics.limiter;
  if (config.numerics.no_limiter) limiter = LIMITER_NONE;
  PetscBool no_limiter = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_limiter", &no_limiter, NULL));
  if (no_limiter) limiter = LIMITER_NONE;
  PetscBool van_leer = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-van_leer", &van_leer, NULL));
  if (van_leer) limiter = LIMITER_VANLEER;
  op->limiter = limiter;

  // Reconstructed states are owned-only, so size the Riemann batch accordingly.
  PetscCall(CreateRiemannStateData(mesh->num_owned_internal_edges, &op->left_states));
  PetscCall(CreateRiemannStateData(mesh->num_owned_internal_edges, &op->right_states));
  PetscCall(CreateRiemannEdgeData(mesh->num_owned_internal_edges, num_comp, &op->edges));

  // Reconstruction buffers
  PetscCall(PetscCalloc1(mesh->num_internal_edges * 4, &op->ls_grad_coeffs));
  PetscCall(PetscCalloc1(mesh->num_cells * 3, &op->q_eta));
  PetscCall(PetscCalloc1(mesh->num_cells * 2, &op->grad_eta));
  PetscCall(PetscCalloc1(mesh->num_cells * 2, &op->grad_hu));
  PetscCall(PetscCalloc1(mesh->num_cells * 2, &op->grad_hv));
  PetscCall(PetscCalloc1(mesh->num_owned_internal_edges * 6, &op->q_reconstructed));
  PetscCall(PrecomputeLSGradCoeffs(comm, mesh, op->ls_grad_coeffs));

  // Copy edge normals in owned-edge order (matching the Apply loop indexing).
  RDyEdges *edges   = &mesh->edges;
  PetscInt  owned_e = 0;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    if (!edges->is_owned[edge_id]) continue;
    op->edges.cn[owned_e] = edges->cn[edge_id];
    op->edges.sn[owned_e] = edges->sn[edge_id];
    owned_e++;
  }

  // per-cell bed elevation (shared helper with the first-order HR operator)
  PetscCall(ComputeCellBedElevation(mesh, config, &op->zc));

  PetscCall(PetscOperatorCreate(op, ApplyInteriorFluxHR2R, DestroyInteriorFluxHR2R, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}
