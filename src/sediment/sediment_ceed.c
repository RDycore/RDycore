#include <petscdmceed.h>
#include <private/rdysedimentimpl.h>

#include "rdycore.h"
#include "sediment_ceed_impl.h"

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

static const PetscReal GRAVITY          = 9.806;   // gravitational acceleration [m/s^2]
static const PetscReal DENSITY_OF_WATER = 1000.0;  // [kg/m^3]

// frees a data context allocated using PETSc, returning a libCEED error code
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// creates a QFunction context for a flux or source operator with the given
// minimum parameters
static PetscErrorCode CreateSedimentQFunctionContext(Ceed ceed, const RDyConfig config, CeedQFunctionContext *qf_context) {
  PetscFunctionBeginUser;

  SedimentContext sediment_ctx;
  PetscCall(PetscCalloc1(1, &sediment_ctx));

  sediment_ctx->dtime                   = 0.0;
  sediment_ctx->tiny_h                  = config.physics.flow.tiny_h;
  sediment_ctx->gravity                 = GRAVITY;
  sediment_ctx->xq2018_threshold        = config.physics.flow.source.xq2018_threshold;
  sediment_ctx->kp_constant             = 0.001;
  sediment_ctx->settling_velocity       = 0.01;
  sediment_ctx->tau_critical_erosion    = 0.1;
  sediment_ctx->tau_critical_deposition = 1000.0;
  sediment_ctx->rhow                    = DENSITY_OF_WATER;
  sediment_ctx->sed_ndof                = config.physics.sediment.num_classes;
  sediment_ctx->flow_ndof               = 3;  // NOTE: SWE assumed!

  PetscCallCEED(CeedQFunctionContextCreate(ceed, qf_context));

  PetscCallCEED(CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*sediment_ctx), sediment_ctx));

  PetscCallCEED(CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "time step", offsetof(struct SedimentContext_, dtime), 1, "Time step of TS"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "small h value", offsetof(struct SedimentContext_, tiny_h), 1,
                                                   "Height threshold below which dry condition is assumed"));
  PetscCallCEED(
      CeedQFunctionContextRegisterDouble(*qf_context, "gravity", offsetof(struct SedimentContext_, gravity), 1, "Accelaration due to gravity"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "xq2018_threshold", offsetof(struct SedimentContext_, xq2018_threshold), 1,
                                                   "Threshold for the treatment of Implicit XQ2018 method"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "kp_constant", offsetof(struct SedimentContext_, kp_constant), 1,
                                                   "Krone-Partheniades erosion law constant [kg/m2/s]"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "settling_velocity", offsetof(struct SedimentContext_, settling_velocity), 1,
                                                   "settling velocity of sediment class"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "tau_critical_erosion", offsetof(struct SedimentContext_, tau_critical_erosion), 1,
                                                   "critical shear stress for erosion (N/m2)"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "tau_critical_deposition", offsetof(struct SedimentContext_, tau_critical_deposition),
                                                   1, "critical shear stress for deposition (N/m2)"));

  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "rhow", offsetof(struct SedimentContext_, rhow), 1, "density of water"));

  PetscCallCEED(
      CeedQFunctionContextRegisterInt32(*qf_context, "sed_ndof", offsetof(struct SedimentContext_, sed_ndof), 1, "number of sediment classes"));
  PetscCallCEED(CeedQFunctionContextRegisterInt32(*qf_context, "flow_ndof", offsetof(struct SedimentContext_, flow_ndof), 1, "number of flow DoF"));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// @brief Creates a CEED operator for solving flow and sediment dynamics equation for interior edges
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [in]  config  RDycore's configuration
/// @param [out] ceed_op a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateSedimentCeedInteriorFluxOperator(RDyMesh *mesh, const RDyConfig config, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // create the Q-function that underlies the operator, and set its inputs and outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation (swe_ceed_impl.h)
  CeedQFunction qf;
  CeedInt       num_comp_geom = 4, num_comp_cnum = 2;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentFlux_Roe, SedimentFlux_Roe_loc, &qf));
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // create a context for the Q-function
  CeedQFunctionContext qf_context;
  PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction q_restrict_l, q_restrict_r, c_restrict_l, c_restrict_r, restrict_geom, restrict_flux, restrict_cnum;
  CeedVector          geom, flux, cnum;
  {
    CeedInt num_edges = mesh->num_owned_internal_edges;

    // create a vector of geometric factors that transform fluxes to cell states
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    CeedScalar(*g)[4];
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
    CeedInt *q_offset_l, *q_offset_r, *c_offset_l, *c_offset_r;
    PetscCall(PetscMalloc2(num_edges, &q_offset_l, num_edges, &q_offset_r));
    PetscCall(PetscMalloc2(num_edges, &c_offset_l, num_edges, &c_offset_r));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l              = edges->cell_ids[2 * iedge];
      CeedInt r              = edges->cell_ids[2 * iedge + 1];
      q_offset_l[owned_edge] = l * num_comp;
      q_offset_r[owned_edge] = r * num_comp;
      c_offset_l[owned_edge] = cells->local_to_owned[l] * num_comp;
      c_offset_r[owned_edge] = cells->local_to_owned[r] * num_comp;
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
    PetscCall(PetscFree2(q_offset_l, q_offset_r));
    PetscCall(PetscFree2(c_offset_l, c_offset_r));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(q_restrict_r, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_r, stdout));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_left", q_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_right", q_restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_left", c_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_right", c_restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "flux", restrict_flux, CEED_BASIS_COLLOCATED, flux));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "courant_number", restrict_cnum, CEED_BASIS_COLLOCATED, cnum));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_r));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// @brief Creates a CEED operator for solving flow and sediment dynamics equation for a set of boundary edges
/// @param [in]  mesh               mesh defining the computational domain of the operator
/// @param [in]  config             RDycore's configuration
/// @param [in]  boundary           a RDyBoundary struct describing the boundary on which the boundary condition is applied
/// @param [in]  boundary_condition a RDyCondition describing the type of boundary condition
/// @param [out] ceed_op            a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateSedimentCeedBoundaryFluxOperator(RDyMesh *mesh, const RDyConfig config, RDyBoundary boundary, RDyCondition boundary_condition,
                                                      CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // create the Q-function that underlies the operator, and set its inputs and outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation (swe_ceed_impl.h)
  CeedQFunctionUser func;
  const char       *func_loc;
  switch (boundary_condition.flow->type) {
    case CONDITION_DIRICHLET:
      func     = SedimentBoundaryFlux_Dirichlet_Roe;
      func_loc = SedimentBoundaryFlux_Dirichlet_Roe_loc;
      break;
    case CONDITION_REFLECTING:
      func     = SedimentBoundaryFlux_Reflecting_Roe;
      func_loc = SedimentBoundaryFlux_Reflecting_Roe_loc;
      break;
    case CONDITION_CRITICAL_OUTFLOW:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "CONDITION_CRITICAL_OUTFLOW not implemented");
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n",
                 boundary.id);
  }
  CeedQFunction qf;
  CeedInt       num_comp_geom = 3, num_comp_cnum = 1;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, func, func_loc, &qf));
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    PetscCallCEED(CeedQFunctionAddInput(qf, "q_dirichlet", num_comp, CEED_EVAL_NONE));
  }
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // create a context for the Q-function
  CeedQFunctionContext qf_context;
  PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction q_restrict_l, c_restrict_l, restrict_geom, restrict_cnum;
  CeedVector          geom, cnum;
  {
    CeedInt num_edges = boundary.num_edges;

    // create element restrictions for left and right input/output states
    CeedInt *q_offset_l, *c_offset_l, *offset_dirichlet = NULL;
    PetscCall(PetscMalloc1(num_edges, &q_offset_l));
    PetscCall(PetscMalloc1(num_edges, &c_offset_l));
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
    CeedScalar(*g)[3];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, owned_edge = 0; e < num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l        = edges->cell_ids[2 * iedge];
      g[owned_edge][0] = edges->sn[iedge];
      g[owned_edge][1] = edges->cn[iedge];
      g[owned_edge][2] = -edges->lengths[iedge] / cells->areas[l];
      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

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
      owned_edge++;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            q_offset_l, &q_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            c_offset_l, &c_restrict_l));
    PetscCall(PetscFree(q_offset_l));
    PetscCall(PetscFree(c_offset_l));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_left", q_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_left", c_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "courant_number", restrict_cnum, CEED_BASIS_COLLOCATED, cnum));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// @brief Creates a CEED operator for solving flow and sediment dynamics equation for the source-sink term
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [in]  config  RDycore's configuration
/// @param [out] ceed_op a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateSedimentCeedSourceOperator(RDyMesh *mesh, RDyConfig config, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;

  // create the Q-function that underlies the operator, and set its inputs and outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation (swe_ceed_impl.h)
  CeedQFunction qf;
  switch (config.physics.flow.source.method) {
    case SOURCE_SEMI_IMPLICIT:
      PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentSourceTermSemiImplicit, SedimentSourceTermSemiImplicit_loc, &qf));
      break;
    case SOURCE_IMPLICIT_XQ2018:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SOURCE_IMPLICIT_XQ2018 is not supported in sediment CEED version");
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Only semi_implicit source-term is supported in the CEED version");
      break;
  }

  CeedInt num_mat_props = OPERATOR_NUM_MATERIAL_PROPERTIES;
  PetscCallCEED(CeedQFunctionAddInput(qf, "ext_src", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "mat_props", num_mat_props, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE));

  // create a context for the Q-function
  CeedQFunctionContext qf_context;
  PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction restrict_c, restrict_q;
  {
    PetscInt num_local_cells = mesh->num_cells;
    PetscInt num_owned_cells = mesh->num_owned_cells;

    // create element restrictions for (active) input/output cell states
    CeedInt *offset_c, *offset_q;
    PetscCall(PetscMalloc1(num_owned_cells, &offset_q));
    PetscCall(PetscMalloc1(num_owned_cells, &offset_c));
    for (CeedInt c = 0, owned_cell = 0; c < num_local_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      offset_q[owned_cell] = c * num_comp;
      offset_c[owned_cell] = cells->local_to_owned[c] * num_comp;
      ++owned_cell;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_cells, 1, num_comp, 1, num_local_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            offset_q, &restrict_q));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_cells, 1, num_comp, 1, num_owned_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            offset_c, &restrict_c));
    PetscCall(PetscFree(offset_c));
    PetscCall(PetscFree(offset_q));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(restrict_q, stdout));
      PetscCallCEED(CeedElemRestrictionView(restrict_c, stdout));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  // NOTE: "ext_src" and "mat_props" fields are added via CreateOperator.
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q", restrict_q, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell", restrict_c, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_c));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
