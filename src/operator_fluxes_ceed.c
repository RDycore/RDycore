#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysedimentimpl.h>
#include <private/rdysweimpl.h>

#include "sediment/sediment_fluxes_ceed.h"
#include "swe/swe_fluxes_ceed.h"
#include "swe/swe_well_balance.h"

// The CEED flux operator consists of the following sub-operators:
//
// * for the entire domain: an interior flux sub-operator that accepts the local
//   solution vector as input and computes the fluxes on pairs of cells on the
//   interior of the computational domain. This sub-operator is created by the
//   function CreateSWECeedInteriorFluxOperator.
// * for each domain boundary: a boundary flux sub-operator that accepts the
//   local solution vector as input and computes the fluxes into/out of cells
//   adjacent to boundary edges. Each sub-operator is created by the function
//   CreateSWECeedBoundaryFluxOperator.
//
// The relevant function documentation includes a comprehensive list of input
// and output fields for each sub-operator.

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

static PetscErrorCode CreateInteriorFluxQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  if (num_sediment_comp == 0) {  // flow only, and SWE is it!
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEFlux_Roe, SWEFlux_Roe_loc, qf));
    PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
  } else {
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentFlux_Roe, SedimentFlux_Roe_loc, qf));
    PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for solving governing equations by computing
/// fluxes on interior edges
/// Creates a CeedOperator that computes fluxes between pairs of cells on the
/// domain's interior.
///
/// Active input fields:
///    * `q_left[num_interior_edges][3]` - an array associating a 3-DOF left cell
///      input state with each edge separating two interior cells
///    * `q_right[num_interior_edges][3]` - an array associating a 3-DOF right cell
///      input state with each edge separating two interior cells
///
/// Passive input fields:
///    * `geom[num_interior_edges][4]` - an array associating 4 geometric factors
///      with each edge separating two interior cells:
///        1. sin(theta), where theta is the angle between the edge and the y axis
///        2. cos(theta), where theta is the angle between the edge and the y axis
///        3. -L / A_l, where L is the edge's length and A_l is the area of the "left" cell
///        4. L / A_r, where L is the edge's length and A_r is the area of the "right" cell
///
/// Active output fields:
///    * `cell_left[num_interior_edges][3]` - an array associating a 3-DOF left cell
///      output state with each edge separating two interior cells
///    * `cell_right[num_interior_edges][3]` - an array associating a 3-DOF right cell
///      output state with each edge separating two interior cells
///
/// Passive output fields:
///    * `flux[num_owned_cells][3]` - an array associating riemann fluxes
///      with each owned cell
///    * `courant_number[num_interior_edges][1]` - an array associating the
///      Courant number (max wave speed) with each edge separating two interior
///      cells
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config  RDycore's configuration
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [out] subop   the CeedOperator representing the newly created suboperator
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateCeedInteriorFluxSuboperator(const RDyConfig config, RDyMesh *mesh, CeedVector *eta_vertices, CeedOperator *subop) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells    *cells    = &mesh->cells;
  RDyEdges    *edges    = &mesh->edges;
  RDyVertices *vertices = &mesh->vertices;

  CeedQFunction qf;
  PetscCall(CreateInteriorFluxQFunction(ceed, config, &qf));

  // add inputs and outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation (swe_ceed_impl.h)
  CeedInt num_comp_geom = 6;  // sn, cn, -L/A_l, L/A_r, z_beg_vertex, z_end_vertex
  CeedInt num_comp_cnum = 2;  // courant number for the left and right cells
  CeedInt num_comp_eta  = 1;  // h_vertex
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "eta_vert_beg", num_comp_eta, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "eta_vert_end", num_comp_eta, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction q_restrict_l, q_restrict_r, c_restrict_l, c_restrict_r, restrict_geom, restrict_flux, restrict_cnum, eta_beg_restrict,
      eta_end_restrict;
  CeedVector geom, flux, cnum;
  {
    CeedInt num_edges = mesh->num_owned_internal_edges;

    // create a vector of geometric factors that transform fluxes to cell states
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    CeedScalar(*g)[6];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l        = edges->cell_ids[2 * iedge];
      CeedInt r        = edges->cell_ids[2 * iedge + 1];
      g[owned_edge][0] = edges->sn[iedge];
      g[owned_edge][1] = edges->cn[iedge];
      g[owned_edge][2] = -edges->lengths[iedge] / cells->areas[l];
      g[owned_edge][3] = edges->lengths[iedge] / cells->areas[r];

      CeedInt vid_beg  = edges->vertex_ids[2 * iedge];
      CeedInt vid_end  = edges->vertex_ids[2 * iedge + 1];
      g[owned_edge][4] = vertices->points[vid_beg].X[2];  // z value for beginning vertex
      g[owned_edge][5] = vertices->points[vid_end].X[2];  // z value for ending vertex

      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector to store inter-cell fluxes
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));

    // create a vector to store the courant number for each edge
    CeedInt cnum_strides[] = {num_comp_cnum, 1, num_comp_cnum};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_cnum, num_edges * num_comp_cnum, cnum_strides, &restrict_cnum));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_cnum, &cnum, NULL));
    PetscCallCEED(CeedVectorSetValue(cnum, 0.0));

    // create element restrictions for (active) left and right input/output states
    CeedInt *q_offset_l, *q_offset_r, *c_offset_l, *c_offset_r, *eta_beg_offset, *eta_end_offset;
    PetscCall(PetscMalloc2(num_edges, &q_offset_l, num_edges, &q_offset_r));
    PetscCall(PetscMalloc2(num_edges, &c_offset_l, num_edges, &c_offset_r));
    PetscCall(PetscMalloc2(num_edges, &eta_beg_offset, num_edges, &eta_end_offset));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l              = edges->cell_ids[2 * iedge];
      CeedInt r              = edges->cell_ids[2 * iedge + 1];
      q_offset_l[owned_edge] = l * num_comp;
      q_offset_r[owned_edge] = r * num_comp;
      c_offset_l[owned_edge] = cells->local_to_owned[l] * num_comp;
      c_offset_r[owned_edge] = cells->local_to_owned[r] * num_comp;

      CeedInt vid_beg            = edges->vertex_ids[2 * iedge];
      CeedInt vid_end            = edges->vertex_ids[2 * iedge + 1];
      eta_beg_offset[owned_edge] = vid_beg * num_comp_eta;
      eta_end_offset[owned_edge] = vid_end * num_comp_eta;
      owned_edge++;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, q_offset_l,
                                            &q_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, q_offset_r,
                                            &q_restrict_r));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, c_offset_l,
                                            &c_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, c_offset_r,
                                            &c_restrict_r));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp_eta, 1, mesh->num_vertices * num_comp_eta, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            eta_beg_offset, &eta_beg_restrict));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp_eta, 1, mesh->num_vertices * num_comp_eta, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            eta_end_offset, &eta_end_restrict));
    PetscCall(PetscFree2(q_offset_l, q_offset_r));
    PetscCall(PetscFree2(c_offset_l, c_offset_r));
    PetscCall(PetscFree2(eta_beg_offset, eta_end_offset));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(q_restrict_r, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_r, stdout));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, subop));
  PetscCallCEED(CeedOperatorSetField(*subop, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*subop, "q_left", q_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "q_right", q_restrict_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "eta_vert_beg", eta_beg_restrict, CEED_BASIS_NONE, *eta_vertices));
  PetscCallCEED(CeedOperatorSetField(*subop, "eta_vert_end", eta_end_restrict, CEED_BASIS_NONE, *eta_vertices));
  PetscCallCEED(CeedOperatorSetField(*subop, "cell_left", c_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "cell_right", c_restrict_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "flux", restrict_flux, CEED_BASIS_NONE, flux));
  PetscCallCEED(CeedOperatorSetField(*subop, "courant_number", restrict_cnum, CEED_BASIS_NONE, cnum));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&eta_beg_restrict));
  PetscCallCEED(CeedElemRestrictionDestroy(&eta_end_restrict));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

static PetscErrorCode CreateBoundaryFluxQFunction(Ceed ceed, const RDyConfig config, RDyBoundary boundary, RDyCondition boundary_condition,
                                                  CeedQFunction *qf) {
  PetscFunctionBeginUser;

  int num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  switch (boundary_condition.flow->type) {
    case CONDITION_DIRICHLET:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEBoundaryFlux_Dirichlet_Roe, SWEBoundaryFlux_Dirichlet_Roe_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {  // sediment dynamics
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentBoundaryFlux_Dirichlet_Roe, SedimentBoundaryFlux_Dirichlet_Roe_loc, qf));
        PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
      }
      break;
    case CONDITION_REFLECTING:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEBoundaryFlux_Reflecting_Roe, SWEBoundaryFlux_Reflecting_Roe_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {  // sediment dynamics
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentBoundaryFlux_Reflecting_Roe, SedimentBoundaryFlux_Reflecting_Roe_loc, qf));
        PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
      }
      break;
    case CONDITION_CRITICAL_OUTFLOW:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEBoundaryFlux_Outflow_Roe, SWEBoundaryFlux_Outflow_Roe_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {  // sediment dynamics
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "CONDITION_CRITICAL_OUTFLOW not implemented");
      }
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n",
                 boundary.id);
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator that computes fluxes through edges on the boundary of a domain.
/// Creates a CeedOperator that computes fluxes through edges on the boundary
/// of a domain.
///
/// Active input fields:
///    * `q_left[num_boundary_edges][3]` - an array associating a 3-DOF left
///      (interior) cell input state with each boundary edge
///
/// Passive input fields:
///    * `geom[num_boundary_edges][3]` - an array associating 3 geometric factors
///      with each boundary edge:
///        1. sin(theta), where theta is the angle between the edge and the y axis
///        2. cos(theta), where theta is the angle between the edge and the y axis
///        3. -L / A_l, where L is the edge's length and A_l is the area of the
///           "left" (interior) cell
///    * `q_dirichlet[num_boundary_edges][3]` - an array associating 3 boundary
///      values with each boundary edge (**iff the boundary associated with the
///      sub-operator is assigned a dirichlet boundary condition**)
///
/// Active output fields:
///    * `cell_left[num_boundary_edges][3]` - an array associating a 3-DOF left
///      (interior) cell output state with each boundary edge
///
/// Passive output fields:
///    * `flux[num_boundary_edges][3]` - an array associating riemann fluxes
///      with each boundary edge (to be summed to compute their divergence)
///    * `courant_number[num_interior_edges][1]` - an array associating the
///      Courant number (max wave speed) with each boundary edge
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config             RDycore's configuration
/// @param [in]  mesh               mesh defining the computational domain of the operator
/// @param [in]  boundary           a RDyBoundary struct describing the boundary on which the boundary condition is applied
/// @param [in]  eta_vertices       a CeedVector containing the water heights at mesh vertices
/// @param [in]  boundary_condition a RDyCondition describing the type of boundary condition
/// @param [out] subop   the CeedOperator representing the newly created suboperator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedBoundaryFluxSuboperator(const RDyConfig config, RDyMesh *mesh, CeedVector *eta_vertices, RDyBoundary boundary,
                                                 RDyCondition boundary_condition, CeedOperator *subop) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells    *cells    = &mesh->cells;
  RDyEdges    *edges    = &mesh->edges;
  RDyVertices *vertices = &mesh->vertices;

  CeedQFunction qf;
  PetscCall(CreateBoundaryFluxQFunction(ceed, config, boundary, boundary_condition, &qf));

  // add inputs/outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation
  CeedInt num_comp_geom = 5;  // sn, cn, -L/A_l, z_beg_vertex, z_end_vertex
  CeedInt num_comp_cnum = 1;
  CeedInt num_comp_eta  = 1;  // h_vertex
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "eta_vert_beg", num_comp_eta, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "eta_vert_end", num_comp_eta, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    PetscCallCEED(CeedQFunctionAddInput(qf, "q_dirichlet", num_comp, CEED_EVAL_NONE));
  }
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux_accumulated", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction q_restrict_l, c_restrict_l, restrict_dirichlet, restrict_geom, restrict_flux, restrict_cnum, eta_beg_restrict, eta_end_restrict;
  CeedVector          geom, flux, flux_accumulated, dirichlet, cnum;
  {
    CeedInt num_edges = boundary.num_edges;

    // create element restrictions for left and right input/output states
    CeedInt *q_offset_l, *c_offset_l, *offset_dirichlet = NULL, *eta_beg_offset, *eta_end_offset;
    PetscCall(PetscMalloc1(num_edges, &q_offset_l));
    PetscCall(PetscMalloc1(num_edges, &c_offset_l));
    PetscCall(PetscMalloc2(num_edges, &eta_beg_offset, num_edges, &eta_end_offset));
    if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
      PetscCall(PetscMalloc1(num_edges, &offset_dirichlet));
    }

    // create a vector of geometric factors that transform fluxes to cell states
    CeedInt num_owned_edges = 0;
    for (CeedInt e = 0; e < boundary.num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (edges->is_owned[iedge]) num_owned_edges++;
    }
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    CeedScalar(*g)[5];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, owned_edge = 0; e < num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l        = edges->cell_ids[2 * iedge];
      g[owned_edge][0] = edges->sn[iedge];
      g[owned_edge][1] = edges->cn[iedge];
      g[owned_edge][2] = -edges->lengths[iedge] / cells->areas[l];

      CeedInt vid_beg  = edges->vertex_ids[2 * iedge];
      CeedInt vid_end  = edges->vertex_ids[2 * iedge + 1];
      g[owned_edge][3] = vertices->points[vid_beg].X[2];  // z value for beginning vertex
      g[owned_edge][4] = vertices->points[vid_end].X[2];  // z value for ending vertex
      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector to store accumulated fluxes (flux divergences)
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux_accumulated, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));
    PetscCallCEED(CeedVectorSetValue(flux_accumulated, 0.0));

    // create a vector to store the courant number for each edge
    CeedInt cnum_strides[] = {num_comp_cnum, 1, num_comp_cnum};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp_cnum, num_edges * num_comp_cnum, cnum_strides, &restrict_cnum));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_cnum, &cnum, NULL));
    PetscCallCEED(CeedVectorSetValue(cnum, 0.0));

    // create an element restriction for the (active) "left" (interior) input/output states
    for (CeedInt e = 0, owned_edge = 0; e < num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l              = edges->cell_ids[2 * iedge];
      q_offset_l[owned_edge] = l * num_comp;
      c_offset_l[owned_edge] = cells->local_to_owned[l] * num_comp;
      if (offset_dirichlet) {  // Dirichlet boundary values
        offset_dirichlet[owned_edge] = e * num_comp;
      }

      CeedInt vid_beg            = edges->vertex_ids[2 * iedge];
      CeedInt vid_end            = edges->vertex_ids[2 * iedge + 1];
      eta_beg_offset[owned_edge] = vid_beg * num_comp_eta;
      eta_end_offset[owned_edge] = vid_end * num_comp_eta;

      owned_edge++;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            q_offset_l, &q_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            c_offset_l, &c_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp_eta, 1, mesh->num_vertices * num_comp_eta, CEED_MEM_HOST,
                                            CEED_COPY_VALUES, eta_beg_offset, &eta_beg_restrict));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp_eta, 1, mesh->num_vertices * num_comp_eta, CEED_MEM_HOST,
                                            CEED_COPY_VALUES, eta_end_offset, &eta_end_restrict));
    PetscCall(PetscFree(q_offset_l));
    PetscCall(PetscFree(c_offset_l));
    PetscCall(PetscFree(eta_beg_offset));
    PetscCall(PetscFree(eta_end_offset));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
    }

    // create a vector to store (Dirichlet) boundary values if needed
    if (offset_dirichlet) {
      PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, num_edges * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                              offset_dirichlet, &restrict_dirichlet));
      PetscCall(PetscFree(offset_dirichlet));
      if (0) PetscCallCEED(CeedElemRestrictionView(restrict_dirichlet, stdout));
      PetscCallCEED(CeedElemRestrictionCreateVector(restrict_dirichlet, &dirichlet, NULL));
      PetscCallCEED(CeedVectorSetValue(dirichlet, 0.0));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, subop));
  PetscCallCEED(CeedOperatorSetField(*subop, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*subop, "q_left", q_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "eta_vert_beg", eta_beg_restrict, CEED_BASIS_NONE, *eta_vertices));
  PetscCallCEED(CeedOperatorSetField(*subop, "eta_vert_end", eta_end_restrict, CEED_BASIS_NONE, *eta_vertices));
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    PetscCallCEED(CeedOperatorSetField(*subop, "q_dirichlet", restrict_dirichlet, CEED_BASIS_NONE, dirichlet));
  }
  PetscCallCEED(CeedOperatorSetField(*subop, "cell_left", c_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "flux", restrict_flux, CEED_BASIS_NONE, flux));
  PetscCallCEED(CeedOperatorSetField(*subop, "flux_accumulated", restrict_flux, CEED_BASIS_NONE, flux_accumulated));
  PetscCallCEED(CeedOperatorSetField(*subop, "courant_number", restrict_cnum, CEED_BASIS_NONE, cnum));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&eta_beg_restrict));
  PetscCallCEED(CeedElemRestrictionDestroy(&eta_end_restrict));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));
  PetscCallCEED(CeedVectorDestroy(&flux_accumulated));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// Creates a CEED flux operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [in]    num_boundaries      the number of distinct boundaries bounding the computational domain
/// @param [in]    boundaries          an array of distinct boundaries bounding the computational domain
/// @param [in]    boundary_conditions an array of boundary conditions corresponding to the domain boundaries
/// @param [out]   flux_op             the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedFluxOperator(RDyConfig *config, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries,
                                      RDyCondition *boundary_conditions, CeedVector *eta_vertices, CeedOperator *flux_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedOperatorCreateComposite(ceed, flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }

  // flux suboperator 0: fluxes between interior cells

  CeedOperator interior_flux_op;
  PetscCall(CreateCeedInteriorFluxSuboperator(*config, mesh, eta_vertices, &interior_flux_op));
  PetscCall(CeedOperatorCompositeAddSub(*flux_op, interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    CeedOperator boundary_flux_op;
    RDyBoundary  boundary  = boundaries[b];
    RDyCondition condition = boundary_conditions[b];
    PetscCall(CreateCeedBoundaryFluxSuboperator(*config, mesh, eta_vertices, boundary, condition, &boundary_flux_op));
    PetscCall(CeedOperatorCompositeAddSub(*flux_op, boundary_flux_op));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sorts three CeedScalar values in ascending order. after function call,
///        *data1 <= *data2 <= *data3
/// @param data1 [inout] pointer to the first value
/// @param data2 [inout] pointer to the second value
/// @param data3 [inout] pointer to the third value
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode sort3(CeedScalar *data1, CeedScalar *data2, CeedScalar *data3) {
  PetscFunctionBegin;

  CeedScalar tmp;
  if (*data1 > *data2) {
    tmp    = *data1;
    *data1 = *data2;
    *data2 = tmp;
  }
  if (*data2 > *data3) {
    tmp    = *data2;
    *data2 = *data3;
    *data3 = tmp;
  }
  if (*data1 > *data2) {
    tmp    = *data1;
    *data1 = *data2;
    *data2 = tmp;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CeedVector for storing eta (water surface elevation) at mesh vertices
///
/// Creates a CeedVector with the appropriate element restriction for storing vertex-centered
/// eta values. The vector size is determined by the number of vertices and the connectivity
/// between vertices and cells.
///
/// @param [in]  mesh         mesh defining vertices and their connectivity
/// @param [out] eta_vertices CeedVector for vertex-centered eta values
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedEtaVerticesVector(RDyMesh *mesh, CeedVector *eta_vertices) {
  PetscFunctionBegin;

  Ceed         ceed         = CeedContext();
  CeedInt      num_comp_eta = 1;  // only for water surface elevation
  CeedInt      num_vertices = mesh->num_vertices;
  RDyVertices *vertices     = &mesh->vertices;

  // count the total number of vertex-cell connections
  CeedInt total_vertex_cell_conns = 0;
  for (CeedInt v = 0; v < num_vertices; v++) {
    total_vertex_cell_conns += vertices->num_cells[v];
  }

  // allocate and compute eta_offset
  CeedInt *eta_offset;
  PetscCall(PetscMalloc1(total_vertex_cell_conns, &eta_offset));

  CeedInt conn_id = 0;
  for (CeedInt v = 0; v < num_vertices; v++) {
    for (CeedInt i = 0; i < vertices->num_cells[v]; i++) {
      eta_offset[conn_id] = v * num_comp_eta;
      conn_id++;
    }
  }

  // create restriction for eta at vertices
  CeedElemRestriction eta_restrict;
  PetscCallCEED(CeedElemRestrictionCreate(ceed, total_vertex_cell_conns, 1, num_comp_eta, 1, num_vertices * num_comp_eta, CEED_MEM_HOST,
                                          CEED_COPY_VALUES, eta_offset, &eta_restrict));
  if (0) CeedElemRestrictionView(eta_restrict, stdout);

  // create the vector
  PetscCallCEED(CeedElemRestrictionCreateVector(eta_restrict, eta_vertices, NULL));
  PetscCallCEED(CeedVectorSetValue(*eta_vertices, 0.0));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&eta_restrict));
  PetscCall(PetscFree(eta_offset));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for computing vertex-averaged water surface elevation (eta)
///
/// This operator computes eta at mesh vertices by averaging the eta values from all
/// cells surrounding each vertex. It implements a **scatter-gather-reduce pattern** where
/// cell-centered data is scattered to vertex-cell connection points, transformed, and then
/// gathered (with accumulation) to produce vertex-centered results.
///
/// Key Concept:
///   The operator uses three restriction spaces of different sizes:
///   - Input from cell space (lsize = num_cells)
///   - Processing in vertex-cell connection space (nelem = total_vertex_cell_conns)
///   - Output to vertex space (lsize = num_vertices)
///
/// Algorithm:
///   For each vertex v:
///     For each cell c connected to v:
///       1. Scatter: read q[c] (cell-centered state)
///       2. Compute: calculate eta from q[c] and geometric factors
///       3. Gather: accumulate (1/num_cells_at_v) * eta into eta_vertices[v]
///     Result: eta_vertices[v] = average of eta over all neighboring cells
///
/// Restriction Space Details:
///   * Input q: lsize = num_cells * num_comp
///     - Indexed by cell IDs
///     - Contains cell-centered flow state
///
///   * Input geom: lsize = total_vertex_cell_conns * num_comp_geom (strided)
///     - One entry per (vertex, cell) pair
///     - Contains [z1, z2, z3, weight] where weight = 1/num_cells_at_vertex
///     - Enables weighted averaging during the gather phase
///
///   * Output eta: lsize = num_vertices * num_comp_eta
///     - Indexed by vertex IDs
///     - Accumulates weighted eta contributions from all connected cells
///
/// @param [in]  config       RDycore's configuration
/// @param [in]  mesh         mesh defining vertices, cells, and their connectivity
/// @param [out] eta_vertices CeedVector for vertex-centered eta values
/// @param [out] op           the newly created CeedOperator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedEtaVerticesOperator(RDyConfig *config, RDyMesh *mesh, CeedVector *eta_vertices, CeedOperator *op) {
  PetscFunctionBegin;
  Ceed ceed = CeedContext();

  PetscCall(CeedOperatorCreateComposite(ceed, op));

  RDyCells    *cells    = &mesh->cells;
  RDyVertices *vertices = &mesh->vertices;

  CeedInt num_sediment_comp = config->physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;
  CeedInt num_comp_geom     = 4;
  CeedInt num_comp_eta      = 1;  // only for water surface elevation

  CeedInt num_vertices = mesh->num_vertices;

  // create QFunction
  CeedQFunction        qf;
  CeedQFunctionContext qf_context;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEEtaVertex, SWEEtaVertex_loc, &qf));
  PetscCall(CreateSWEQFunctionContext(ceed, *config, &qf_context));
  PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "eta", num_comp_eta, CEED_EVAL_NONE));

  // count the total number of vertex-cell connections
  CeedInt total_vertex_cell_conns = 0;
  for (CeedInt v = 0; v < num_vertices; v++) {
    total_vertex_cell_conns += vertices->num_cells[v];
  }

  // allocate offsets array for the unknowns
  CeedInt *q_offset, *eta_offset;
  PetscCall(PetscMalloc1(total_vertex_cell_conns, &q_offset));
  PetscCall(PetscMalloc1(total_vertex_cell_conns, &eta_offset));

  // create a vector of geometric factors that transform fluxes to cell states
  CeedElemRestriction geom_restrict;
  CeedVector          geom;
  CeedInt             g_strides[] = {num_comp_geom, 1, num_comp_geom};
  PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, total_vertex_cell_conns, 1, num_comp_geom, total_vertex_cell_conns * num_comp_geom, g_strides,
                                                 &geom_restrict));
  if (0) CeedElemRestrictionView(geom_restrict, stdout);
  PetscCallCEED(CeedElemRestrictionCreateVector(geom_restrict, &geom, NULL));
  PetscCallCEED(CeedVectorSetValue(geom, 0.0));
  CeedScalar(*g)[4];
  PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));

  CeedInt conn_id = 0;
  for (CeedInt v = 0; v < num_vertices; v++) {
    CeedInt offset_for_cell_id = vertices->cell_offsets[v];

    for (CeedInt i = 0; i < vertices->num_cells[v]; i++) {
      CeedInt icell = vertices->cell_ids[offset_for_cell_id + i];

      CeedInt offset_for_vertex_ids = cells->vertex_offsets[icell];

      CeedInt iv1 = cells->vertex_ids[offset_for_vertex_ids];
      CeedInt iv2 = cells->vertex_ids[offset_for_vertex_ids + 1];
      CeedInt iv3 = cells->vertex_ids[offset_for_vertex_ids + 2];

      CeedScalar z1 = vertices->points[iv1].X[2];
      CeedScalar z2 = vertices->points[iv2].X[2];
      CeedScalar z3 = vertices->points[iv3].X[2];

      PetscCall(sort3(&z1, &z2, &z3));

      g[conn_id][0] = z1;
      g[conn_id][1] = z2;
      g[conn_id][2] = z3;
      g[conn_id][3] = 1.0 / vertices->num_cells[v];

      q_offset[conn_id] = icell * num_comp;

      eta_offset[conn_id] = v * num_comp_eta;
      conn_id++;
    }
  }

  PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

  CeedElemRestriction q_restrict;
  PetscCallCEED(CeedElemRestrictionCreate(ceed, total_vertex_cell_conns, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                          q_offset, &q_restrict));
  if (0) CeedElemRestrictionView(q_restrict, stdout);
  PetscCall(PetscFree(q_offset));

  // output: create vector and restriction for eta at vertices
  CeedElemRestriction eta_restrict;
  PetscCallCEED(CeedElemRestrictionCreate(ceed, total_vertex_cell_conns, 1, num_comp_eta, 1, num_vertices * num_comp_eta, CEED_MEM_HOST,
                                          CEED_COPY_VALUES, eta_offset, &eta_restrict));
  PetscCall(PetscFree(eta_offset));
  if (0) CeedElemRestrictionView(eta_restrict, stdout);
  PetscCallCEED(CeedElemRestrictionCreateVector(eta_restrict, eta_vertices, NULL));

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, op));
  PetscCallCEED(CeedOperatorSetField(*op, "geom", geom_restrict, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*op, "q", q_restrict, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*op, "eta", eta_restrict, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&geom_restrict));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict));
  PetscCallCEED(CeedElemRestrictionDestroy(&eta_restrict));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
