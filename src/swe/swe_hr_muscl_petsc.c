// Hydrostatic reconstruction (HR) with second-order MUSCL reconstruction for the
// PETSc/CPU backend. This is the opt-in (-hr_muscl) well-balanced, positivity-
// preserving scheme of Audusse & Bristeau (2004).
//
// The reconstruction operates on the free surface eta = h + z (rather than the
// depth h) so that lake-at-rest (eta = const) is preserved to second order: a
// constant eta has zero gradient, so the reconstruction returns cell values and
// HR reproduces the first-order well-balanced state exactly.
//
// Momentum is reconstructed as the PRIMITIVE velocity (u, v) = (hu/h, hv/h),
// NOT the conserved momentum (hu, hv). This is the Audusse & Bristeau (2005)
// positivity-preserving choice: the face velocity is then a bounded, limited
// reconstruction of the cell velocity, rather than the ratio hu_face/h_face,
// which blows up when a reconstructed face depth is thin -- the singularity that
// (a) manufactures huge velocities at wet/dry fronts and (b) drives the outgoing
// flux (and hence the positivity timestep constraint) to collapse. The cell
// velocity is formed from the true conserved cell depth (optionally ANUGA-
// regularized), so no reconstructed thin depth ever appears in a denominator.
//
// The water DEPTH h = eta - z is ALSO reconstructed with an independent limited
// slope (Buttinger-Kreuzhuber et al. 2019, eq 14/20-22). The interface bottom is
// b* = max(eta_L - h_face_L, eta_R - h_face_R) and the hydrostatic interface depth
// is capped by the reconstructed depth: h* = max(0, min(eta - b*, h_face)), giving
// 0 <= h* <= h_face by construction. This bounds h* by the cell's own reconstructed
// depth instead of the unbounded eta - max(z), which a thin cell on a slope can
// inflate into an over-large interface depth that collapses the timestep. The
// pressure correction stays the (cell-referenced) Audusse well-balanced form, so
// lake-at-rest is preserved bit-for-bit -- the depth reconstruction only tightens
// positivity, it does not perturb well-balancing.
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

// Reconstructs face-centered states for all owned interior edges by limited
// linear extrapolation from cell centroids. Identical in spirit to
// ReconstructFaceValues (operator_fluxes_ceed.c) but reconstructs the free
// surface eta in component 0 and therefore does NOT clamp it to >= 0 (beds, and
// hence eta, may be negative). Components 1,2 are the PRIMITIVE velocities u, v
// (not conserved momentum); they need no clamp either.
//
// It ALSO reconstructs the water DEPTH h = eta - z with an INDEPENDENT limited
// slope (Buttinger-Kreuzhuber et al. 2019, eq 14/20-22). The reconstructed depth
// bounds the hydrostatic interface depth (0 <= h* <= h_face) in the Apply loop,
// which is the positivity property that a single-field eta reconstruction lacks:
// eta - z_max can produce an over-large interface depth at a thin cell and drive
// the positivity/Courant timestep to collapse.
//
// q_eta layout per cell:  [eta, u, v]  (size num_cells * 3); depth is eta - zc.
// q_face layout per owned edge (size num_owned_internal_edges * 8):
//   [eta_L, u_L, v_L, eta_R, u_R, v_R, h_face_L, h_face_R]
static PetscErrorCode ReconstructFaceValuesEta(RDyMesh *mesh, const PetscScalar *q_eta, const PetscReal *zc, const PetscScalar *grad_eta,
                                               const PetscScalar *grad_hu, const PetscScalar *grad_hv, const PetscScalar *grad_depth,
                                               RDyLimiterType limiter, PetscScalar *q_face) {
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
      q_face[owned_edge * 8 + k]     = q_eta[cl * 3 + k] + LimitSlope(limiter, extrap_L, 0.5 * dq);
      q_face[owned_edge * 8 + 3 + k] = q_eta[cr * 3 + k] + LimitSlope(limiter, extrap_R, -0.5 * dq);
    }
    // NOTE: no clamp on component 0 here -- it is the free surface eta, not depth.

    // Independent limited reconstruction of the DEPTH h = eta - z (cell-referenced),
    // Buttinger 2019 eq 14/20-22. No clamp here -- the Apply loop takes
    // max(0, min(eta - b*, h_face)), which is where positivity is enforced.
    PetscReal hcell_L   = q_eta[cl * 3 + 0] - zc[cl];
    PetscReal hcell_R   = q_eta[cr * 3 + 0] - zc[cr];
    PetscReal extrapd_L = grad_depth[cl * 2 + 0] * dx_L + grad_depth[cl * 2 + 1] * dy_L;
    PetscReal extrapd_R = grad_depth[cr * 2 + 0] * dx_R + grad_depth[cr * 2 + 1] * dy_R;
    PetscReal dqd       = hcell_R - hcell_L;
    q_face[owned_edge * 8 + 6] = hcell_L + LimitSlope(limiter, extrapd_L, 0.5 * dqd);
    q_face[owned_edge * 8 + 7] = hcell_R + LimitSlope(limiter, extrapd_R, -0.5 * dqd);

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
  PetscReal            wetdry_h;        // depth below which an edge falls back to first order (wet/dry robustness)
  PetscReal           *zc;             // per-cell bed elevation (local indexing)
  RDyLimiterType       limiter;        // MUSCL slope limiter
  PetscReal           *ls_grad_coeffs; // [num_internal_edges * 4] precomputed LS gradient coeffs
  PetscScalar         *q_eta;          // [num_cells * 3]: (eta, u, v) per cell -- primitive velocity in comps 1,2
  PetscScalar         *grad_eta;       // [num_cells * 2]: cell-centered gradient of eta
  PetscScalar         *grad_hu;        // [num_cells * 2]: gradient of primitive u
  PetscScalar         *grad_hv;        // [num_cells * 2]: gradient of primitive v
  PetscScalar         *grad_depth;     // [num_cells * 2]: gradient of depth h = eta - z (Buttinger 2019 eq 14)
  PetscScalar         *q_reconstructed;// [num_owned_internal_edges * 8]: reconstructed face states {eta,u,v}_L/R + h_face_L/R
  RiemannStateData     left_states;    // reconstructed "left" states on owned interior edges
  RiemannStateData     right_states;   // reconstructed "right" states on owned interior edges
  RiemannEdgeData      edges;
  OperatorDiagnostics *diagnostics;
} InteriorFluxHR2ROperator;

// Wet/dry-aware least-squares gradients for HR+SO. A dry cell's reconstruction
// variable eta = h + z = z is the BED, not a water surface, so it must not
// contribute to a neighbor's surface gradient -- that pollution manufactures a
// spurious eta slope at wet/dry fronts and breaks well-balancing (lake-at-rest
// generates flow from rest). We skip the dq accumulation across any edge that
// touches a dry cell. For lake-at-rest every remaining (wet-wet) edge has dq=0,
// so the gradient is exactly zero -> well-balancing is preserved at fronts; only
// moving-water accuracy at the front drops to first order (standard, positivity-safe).
// The same skip also protects components 1,2 (primitive velocity u = hu/h), which
// is undefined in a dry cell, from polluting a wet neighbor's velocity gradient.
// (grad_eta passed as grad_h below; the extra grad_depth output is the gradient of
// the water depth h = eta - z, reconstructed independently per Buttinger 2019 eq 14.)
static PetscErrorCode ComputeLeastSquaresGradientsWetDry(RDyMesh *mesh, const PetscReal *ls_grad_coeffs, const PetscScalar *q,
                                                         const PetscReal *zc, PetscReal tiny_h, PetscScalar *grad_h,
                                                         PetscScalar *grad_hu, PetscScalar *grad_hv, PetscScalar *grad_depth) {
  PetscFunctionBeginUser;
  RDyEdges *edges = &mesh->edges;
  PetscCall(PetscArrayzero(grad_h, mesh->num_cells * 2));
  PetscCall(PetscArrayzero(grad_hu, mesh->num_cells * 2));
  PetscCall(PetscArrayzero(grad_hv, mesh->num_cells * 2));
  PetscCall(PetscArrayzero(grad_depth, mesh->num_cells * 2));
  for (PetscInt ie = 0; ie < mesh->num_internal_edges; ie++) {
    PetscInt e  = edges->internal_edge_ids[ie];
    PetscInt cl = edges->cell_ids[2 * e];
    PetscInt cr = edges->cell_ids[2 * e + 1];
    PetscReal hl = q[cl * 3 + 0] - zc[cl], hr = q[cr * 3 + 0] - zc[cr];
    if (hl < tiny_h || hr < tiny_h) continue;  // skip wet/dry edges
    PetscReal cx_LR = ls_grad_coeffs[ie * 4 + 0], cy_LR = ls_grad_coeffs[ie * 4 + 1];
    PetscReal cx_RL = ls_grad_coeffs[ie * 4 + 2], cy_RL = ls_grad_coeffs[ie * 4 + 3];
    PetscReal dq_h = q[cr * 3 + 0] - q[cl * 3 + 0], dq_hu = q[cr * 3 + 1] - q[cl * 3 + 1], dq_hv = q[cr * 3 + 2] - q[cl * 3 + 2];
    PetscReal dq_d = hr - hl;  // depth jump (eta jump minus bed jump)
    grad_h[cl * 2 + 0] += cx_LR * dq_h;  grad_h[cl * 2 + 1] += cy_LR * dq_h;
    grad_hu[cl * 2 + 0] += cx_LR * dq_hu; grad_hu[cl * 2 + 1] += cy_LR * dq_hu;
    grad_hv[cl * 2 + 0] += cx_LR * dq_hv; grad_hv[cl * 2 + 1] += cy_LR * dq_hv;
    grad_depth[cl * 2 + 0] += cx_LR * dq_d; grad_depth[cl * 2 + 1] += cy_LR * dq_d;
    grad_h[cr * 2 + 0] += cx_RL * dq_h;  grad_h[cr * 2 + 1] += cy_RL * dq_h;
    grad_hu[cr * 2 + 0] += cx_RL * dq_hu; grad_hu[cr * 2 + 1] += cy_RL * dq_hu;
    grad_hv[cr * 2 + 0] += cx_RL * dq_hv; grad_hv[cr * 2 + 1] += cy_RL * dq_hv;
    grad_depth[cr * 2 + 0] += cx_RL * dq_d; grad_depth[cr * 2 + 1] += cy_RL * dq_d;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

  // Build the reconstruction variables (eta = h + z, and PRIMITIVE velocities
  // u = hu/h, v = hv/h) over ALL local cells (including ghosts), then compute
  // limited cell gradients, communicate them into ghost cells, and reconstruct
  // face states for owned edges. The velocity is regularized with the CELL depth
  // (the true conserved value) -- never a reconstructed thin face depth -- so the
  // Audusse-Bristeau positivity property holds. Dry cells carry u = v = 0.
  for (PetscInt c = 0; c < mesh->num_cells; c++) {
    PetscReal h   = u_ptr[n_dof * c + 0];
    PetscReal hu  = u_ptr[n_dof * c + 1];
    PetscReal hv  = u_ptr[n_dof * c + 2];
    PetscReal reg = Square(h) + Square(h_anuga);
    op->q_eta[3 * c + 0] = h + zc[c];
    op->q_eta[3 * c + 1] = (h > tiny_h) ? hu * h / reg : 0.0;
    op->q_eta[3 * c + 2] = (h > tiny_h) ? hv * h / reg : 0.0;
  }

  DM dm;
  PetscCall(VecGetDM(u_local, &dm));
  PetscCall(ComputeLeastSquaresGradientsWetDry(mesh, op->ls_grad_coeffs, op->q_eta, op->zc, op->tiny_h, op->grad_eta, op->grad_hu, op->grad_hv,
                                               op->grad_depth));
  PetscCall(CommunicateCellGradients(dm, mesh, op->grad_eta, op->grad_hu, op->grad_hv));
  // Communicate the 4th (depth) gradient through the same bs=3 halo exchange by
  // packing it into all three slots (an idempotent round-trip for a single field).
  PetscCall(CommunicateCellGradients(dm, mesh, op->grad_depth, op->grad_depth, op->grad_depth));
  PetscCall(ReconstructFaceValuesEta(mesh, op->q_eta, op->zc, op->grad_eta, op->grad_hu, op->grad_hv, op->grad_depth, op->limiter,
                                     op->q_reconstructed));

  // Wet/dry first-order fallback (Godunov barrier): where either cell is shallower
  // than wetdry_h, replace the MUSCL face states with the first-order cell-centered
  // states IN PLACE in q_reconstructed, so the Roe flux AND the pressure correction
  // (which both read q_reconstructed) stay consistent. Off by default (wetdry_h=0).
  if (op->wetdry_h > 0.0) {
    PetscInt oe = 0;
    for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
      PetscInt edge_id = edges->internal_edge_ids[e];
      if (!edges->is_owned[edge_id]) continue;
      PetscInt l = edges->cell_ids[2 * edge_id];
      PetscInt r = edges->cell_ids[2 * edge_id + 1];
      if ((op->q_eta[l * 3 + 0] - op->zc[l]) < op->wetdry_h || (op->q_eta[r * 3 + 0] - op->zc[r]) < op->wetdry_h) {
        for (PetscInt k = 0; k < 3; k++) {
          op->q_reconstructed[oe * 8 + k]     = op->q_eta[l * 3 + k];
          op->q_reconstructed[oe * 8 + 3 + k] = op->q_eta[r * 3 + k];
        }
        // first-order depth too: fall back to the cell depths h = eta - z
        op->q_reconstructed[oe * 8 + 6] = op->q_eta[l * 3 + 0] - op->zc[l];
        op->q_reconstructed[oe * 8 + 7] = op->q_eta[r * 3 + 0] - op->zc[r];
      }
      oe++;
    }
  }

  // Hydrostatic reconstruction of the face states, packed by owned-edge index.
  PetscInt owned_e = 0;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    if (!edges->is_owned[edge_id]) continue;

    PetscInt l = edges->cell_ids[2 * edge_id];
    PetscInt r = edges->cell_ids[2 * edge_id + 1];

    PetscReal eta_L    = op->q_reconstructed[owned_e * 8 + 0];
    PetscReal u_rec_L  = op->q_reconstructed[owned_e * 8 + 1];  // reconstructed primitive velocity
    PetscReal v_rec_L  = op->q_reconstructed[owned_e * 8 + 2];
    PetscReal eta_R    = op->q_reconstructed[owned_e * 8 + 3];
    PetscReal u_rec_R  = op->q_reconstructed[owned_e * 8 + 4];
    PetscReal v_rec_R  = op->q_reconstructed[owned_e * 8 + 5];
    PetscReal hface_L  = op->q_reconstructed[owned_e * 8 + 6];  // independently reconstructed depth
    PetscReal hface_R  = op->q_reconstructed[owned_e * 8 + 7];

    // physical reconstructed depths (relative to each cell's own bed). These stay
    // CELL-referenced (eta_face - zc[cell]) so the pressure correction below is the
    // Audusse well-balanced form; at lake-at-rest they equal the constant cell depth
    // and the correction cancels the flux to give zero net momentum.
    PetscReal hphys_L = fmax(0.0, eta_L - zc[l]);
    PetscReal hphys_R = fmax(0.0, eta_R - zc[r]);

    // Hydrostatic reconstruction with the sloped interface bottom of Buttinger 2019
    // (eq 20-22/66-67): the interface bottom b* = max of the RECONSTRUCTED bottoms
    // b = eta - h_face, and the HR depth is CAPPED by the reconstructed depth,
    // h* = max(0, min(eta - b*, h_face)), so 0 <= h* <= h_face (positivity). This
    // replaces the unbounded h* = eta - max(zc) that lets a thin cell manufacture an
    // over-large interface depth and collapse the timestep. At rest both sides reduce
    // to min(h_face_L, h_face_R) -> equal -> no Roe dissipation -> well-balancing holds.
    PetscReal b_recon_L = eta_L - hface_L;
    PetscReal b_recon_R = eta_R - hface_R;
    PetscReal b_star    = fmax(b_recon_L, b_recon_R);
    PetscReal hL_rec    = fmax(0.0, fmin(eta_L - b_star, hface_L));
    PetscReal hR_rec    = fmax(0.0, fmin(eta_R - b_star, hface_R));

    // Velocities are the reconstructed PRIMITIVE values directly (Audusse-Bristeau):
    // no reconstructed thin depth in a denominator, so no velocity singularity.
    // Zero the velocity where the physical reconstructed depth is dry.
    PetscReal uL = (hphys_L > tiny_h) ? u_rec_L : 0.0;
    PetscReal vL = (hphys_L > tiny_h) ? v_rec_L : 0.0;
    PetscReal uR = (hphys_R > tiny_h) ? u_rec_R : 0.0;
    PetscReal vR = (hphys_R > tiny_h) ? v_rec_R : 0.0;

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

    PetscReal eta_L   = op->q_reconstructed[owned_e * 8 + 0];
    PetscReal eta_R   = op->q_reconstructed[owned_e * 8 + 3];
    PetscReal hphys_L = fmax(0.0, eta_L - zc[l]);
    PetscReal hphys_R = fmax(0.0, eta_R - zc[r]);
    // capped HR interface depths computed in the reconstruction loop above
    PetscReal hL_rec  = datal->h[owned_e];
    PetscReal hR_rec  = datar->h[owned_e];

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
  PetscCall(PetscFree(op->grad_depth));
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

  // Optional wet/dry first-order fallback threshold [m]. DEFAULT 0 = disabled
  // (HR+SO unchanged). Opt in with -hr_wetdry_h <value> if a run needs extra
  // wet/dry robustness; below this cell depth an edge uses first order.
  op->wetdry_h = 0.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hr_wetdry_h", &op->wetdry_h, NULL));

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
  PetscCall(PetscCalloc1(mesh->num_cells * 2, &op->grad_depth));
  PetscCall(PetscCalloc1(mesh->num_owned_internal_edges * 8, &op->q_reconstructed));
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
