#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysedimentimpl.h>
#include <private/rdysweimpl.h>

#include "sediment/sediment_sources_ceed.h"
#include "swe/swe_sources_ceed.h"

// The source operator consists of the following sub-operators:
//
// * a flux divergence source sub-operator that incorporates the summed riemann fluxes (always on)
// * an external source sub-operator that handles e.g. net runoff production
// * an bed elevation slope source sub-operator
// * a bed friction roughness source sub-operator
//
// The relevant function documentation includes a comprehensive list of input
// and output fields for each sub-operator.

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

static PetscErrorCode CreateFluxDivergenceSourceQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  CeedInt num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  if (num_sediment_comp == 0) {  // flow only
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEFluxDivergenceSourceTerm, SWEFluxDivergenceSourceTerm_loc, qf));
    PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
  } else {
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentFluxDivergenceSourceTerm, SedimentFluxDivergenceSourceTerm_loc, qf));
    PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator that handles the flux divergence contribution within a domain.
///
/// Active input fields:
///    * `q[num_owned_cells][3]` - an array associating a 3-DOF solution input
///      state with each (owned) cell in the domain
///
/// Passive input fields:
///    * `riemannf[num_owned_cells][3]` - an array associating a 3-component
///      flux divergence with each (owned) cell in the domain
///
/// Active output fields:
///    * `cell[num_owned_cells][3]` - an array associating a 3-component source
///      value with each (owned) cell in the domain
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
static PetscErrorCode CreateFluxDivergenceSourceSuboperator(const RDyConfig config, RDyMesh *mesh, CeedOperator *subop) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;

  CeedQFunction qf;
  PetscCall(CreateSourceQFunction(ceed, config, &qf));

  // add inputs/outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation
  CeedInt num_comp_geom = 2, num_comp_ext_src = num_comp;
  CeedInt num_mat_props = NUM_MATERIAL_PROPERTIES;
  PetscCallCEED(CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction restrict_c, restrict_q, restrict_geom, restrict_ext_src, restrict_mat_props;
  CeedVector          geom, ext_src, mat_props;
  {
    PetscInt num_local_cells = mesh->num_cells;
    PetscInt num_owned_cells = mesh->num_owned_cells;

    // create a vector of geometric factors (elevation function derivatives)
    CeedScalar(*g)[num_comp_geom];
    CeedInt strides_geom[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(
        CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt c = 0, owned_cell = 0; c < num_local_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      g[owned_cell][0] = cells->dz_dx[c];
      g[owned_cell][1] = cells->dz_dy[c];
      ++owned_cell;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector of external source terms
    CeedInt strides_ext_src[] = {num_comp_ext_src, 1, num_comp_ext_src};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_ext_src, num_owned_cells * num_comp_ext_src, strides_ext_src,
                                                   &restrict_ext_src));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_ext_src, &ext_src, NULL));
    PetscCallCEED(CeedVectorSetValue(ext_src, 0.0));

    // create a vector that stores Manning's coefficient for the region of interest
    // NOTE: we zero-initialize this coefficient here; it must be set before use
    // NOTE: using (Get/Restore)OperatorMaterialProperty
    CeedInt strides_mat_props[] = {num_mat_props, 1, num_mat_props};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_mat_props, num_owned_cells * num_mat_props, strides_mat_props,
                                                   &restrict_mat_props));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_mat_props, &mat_props, NULL));
    PetscCallCEED(CeedVectorSetValue(mat_props, 0.0));

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
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "ext_src", restrict_ext_src, CEED_BASIS_NONE, ext_src));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "mat_props", restrict_mat_props, CEED_BASIS_NONE, mat_props));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q", restrict_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell", restrict_c, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_ext_src));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_mat_props));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_c));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&ext_src));
  PetscCallCEED(CeedVectorDestroy(&mat_props));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

static PetscErrorCode CreateExternalSourceQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  CeedInt num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  if (num_sediment_comp == 0) {  // flow only
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEExternalSourceTerm, SWEExternalSourceTerm_loc, qf));
    PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
  } else {
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentExternalSourceTerm, SedimentExternalSourceTerm_loc, qf));
    PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for computing external sources within a domain.
///
/// Active input fields:
///    * `q[num_owned_cells][3]` - an array associating a 3-DOF solution input
///      state with each (owned) cell in the domain
///
/// Passive input fields:
///    * `riemannf[num_owned_cells][3]` - an array associating a 3-component
///      flux divergence with each (owned) cell in the domain
///    * `ext_src[num_owned_cells][3]` - an array associating 3 external source
///      components with each (owned) cell in the domain
///
/// Active output fields:
///    * `cell[num_owned_cells][3]` - an array associating a 3-component source
///      value with each (owned) cell in the domain
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config  RDycore's configuration
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [out] ceed_op a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateCeedSource0Operator(const RDyConfig config, RDyMesh *mesh, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;

  CeedQFunction qf;
  PetscCall(CreateSourceQFunction(ceed, config, &qf));

  // add inputs/outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation
  CeedInt num_comp_geom = 2, num_comp_ext_src = num_comp;
  CeedInt num_mat_props = NUM_MATERIAL_PROPERTIES;
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "ext_src", num_comp_ext_src, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "mat_props", num_mat_props, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction restrict_c, restrict_q, restrict_geom, restrict_ext_src, restrict_mat_props;
  CeedVector          geom, ext_src, mat_props;
  {
    PetscInt num_local_cells = mesh->num_cells;
    PetscInt num_owned_cells = mesh->num_owned_cells;

    // create a vector of geometric factors (elevation function derivatives)
    CeedScalar(*g)[num_comp_geom];
    CeedInt strides_geom[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(
        CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt c = 0, owned_cell = 0; c < num_local_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      g[owned_cell][0] = cells->dz_dx[c];
      g[owned_cell][1] = cells->dz_dy[c];
      ++owned_cell;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector of external source terms
    CeedInt strides_ext_src[] = {num_comp_ext_src, 1, num_comp_ext_src};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_ext_src, num_owned_cells * num_comp_ext_src, strides_ext_src,
                                                   &restrict_ext_src));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_ext_src, &ext_src, NULL));
    PetscCallCEED(CeedVectorSetValue(ext_src, 0.0));

    // create a vector that stores Manning's coefficient for the region of interest
    // NOTE: we zero-initialize this coefficient here; it must be set before use
    // NOTE: using (Get/Restore)OperatorMaterialProperty
    CeedInt strides_mat_props[] = {num_mat_props, 1, num_mat_props};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_mat_props, num_owned_cells * num_mat_props, strides_mat_props,
                                                   &restrict_mat_props));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_mat_props, &mat_props, NULL));
    PetscCallCEED(CeedVectorSetValue(mat_props, 0.0));

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
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "ext_src", restrict_ext_src, CEED_BASIS_NONE, ext_src));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "mat_props", restrict_mat_props, CEED_BASIS_NONE, mat_props));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q", restrict_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell", restrict_c, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_ext_src));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_mat_props));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_c));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&ext_src));
  PetscCallCEED(CeedVectorDestroy(&mat_props));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

static PetscErrorCode CreateSourceQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  CeedInt num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  switch (config.physics.flow.source.method) {
    case SOURCE_SEMI_IMPLICIT:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourceTermSemiImplicit, SWESourceTermSemiImplicit_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentSourceTermSemiImplicit, SedimentSourceTermSemiImplicit_loc, qf));
        PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
      }
      break;
    case SOURCE_IMPLICIT_XQ2018:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourceTermImplicitXQ2018, SWESourceTermImplicitXQ2018_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SOURCE_IMPLICIT_XQ2018 is not supported in sediment CEED version");
      }
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Only semi_implicit source-term is supported in the CEED version");
      break;
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for computing source terms within a domain.
/// Creates a CeedOperator that computes sources for a domain.
///
/// Active input fields:
///    * `q[num_owned_cells][3]` - an array associating a 3-DOF solution input
///      state with each (owned) cell in the domain
///
/// Passive input fields:
///    * `geom[num_owned_cells][2]` - an array associating 2 geometric factors
///      with each (owned) cell in the domain:
///        1. dz/dx, the derivative of the elevation function z(x, y) w.r.t. x,
///           evaluated at the cell center
///        2. dz/dy, the derivative of the elevation function z(x, y) w.r.t. y,
///           evaluated at the cell center
///    * `mat_props[num_owned_cells][N]` - an array associating material
///      properties with each (owned) cell in the domain
///    * `riemannf[num_owned_cells][3]` - an array associating a 3-component
///      flux divergence with each (owned) cell in the domain
///    * `ext_src[num_owned_cells][3]` - an array associating 3 external source
///      components with each (owned) cell in the domain
///
/// Active output fields:
///    * `cell[num_owned_cells][3]` - an array associating a 3-component source
///      value with each (owned) cell in the domain
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config  RDycore's configuration
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [out] ceed_op a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateCeedSource0Operator(const RDyConfig config, RDyMesh *mesh, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;

  CeedQFunction qf;
  PetscCall(CreateSourceQFunction(ceed, config, &qf));

  // add inputs/outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation
  CeedInt num_comp_geom = 2, num_comp_ext_src = num_comp;
  CeedInt num_mat_props = NUM_MATERIAL_PROPERTIES;
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "ext_src", num_comp_ext_src, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "mat_props", num_mat_props, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction restrict_c, restrict_q, restrict_geom, restrict_ext_src, restrict_mat_props;
  CeedVector          geom, ext_src, mat_props;
  {
    PetscInt num_local_cells = mesh->num_cells;
    PetscInt num_owned_cells = mesh->num_owned_cells;

    // create a vector of geometric factors (elevation function derivatives)
    CeedScalar(*g)[num_comp_geom];
    CeedInt strides_geom[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(
        CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt c = 0, owned_cell = 0; c < num_local_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      g[owned_cell][0] = cells->dz_dx[c];
      g[owned_cell][1] = cells->dz_dy[c];
      ++owned_cell;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector of external source terms
    CeedInt strides_ext_src[] = {num_comp_ext_src, 1, num_comp_ext_src};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_ext_src, num_owned_cells * num_comp_ext_src, strides_ext_src,
                                                   &restrict_ext_src));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_ext_src, &ext_src, NULL));
    PetscCallCEED(CeedVectorSetValue(ext_src, 0.0));

    // create a vector that stores Manning's coefficient for the region of interest
    // NOTE: we zero-initialize this coefficient here; it must be set before use
    // NOTE: using (Get/Restore)OperatorMaterialProperty
    CeedInt strides_mat_props[] = {num_mat_props, 1, num_mat_props};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_mat_props, num_owned_cells * num_mat_props, strides_mat_props,
                                                   &restrict_mat_props));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_mat_props, &mat_props, NULL));
    PetscCallCEED(CeedVectorSetValue(mat_props, 0.0));

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
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "ext_src", restrict_ext_src, CEED_BASIS_NONE, ext_src));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "mat_props", restrict_mat_props, CEED_BASIS_NONE, mat_props));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q", restrict_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell", restrict_c, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_ext_src));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_mat_props));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_c));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&ext_src));
  PetscCallCEED(CeedVectorDestroy(&mat_props));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

static PetscErrorCode CreateSourceQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  CeedInt num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  switch (config.physics.flow.source.method) {
    case SOURCE_SEMI_IMPLICIT:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourceTermSemiImplicit, SWESourceTermSemiImplicit_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentSourceTermSemiImplicit, SedimentSourceTermSemiImplicit_loc, qf));
        PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
      }
      break;
    case SOURCE_IMPLICIT_XQ2018:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourceTermImplicitXQ2018, SWESourceTermImplicitXQ2018_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SOURCE_IMPLICIT_XQ2018 is not supported in sediment CEED version");
      }
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Only semi_implicit source-term is supported in the CEED version");
      break;
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for computing source terms within a domain.
/// Creates a CeedOperator that computes sources for a domain.
///
/// Active input fields:
///    * `q[num_owned_cells][3]` - an array associating a 3-DOF solution input
///      state with each (owned) cell in the domain
///
/// Passive input fields:
///    * `geom[num_owned_cells][2]` - an array associating 2 geometric factors
///      with each (owned) cell in the domain:
///        1. dz/dx, the derivative of the elevation function z(x, y) w.r.t. x,
///           evaluated at the cell center
///        2. dz/dy, the derivative of the elevation function z(x, y) w.r.t. y,
///           evaluated at the cell center
///    * `mat_props[num_owned_cells][N]` - an array associating material
///      properties with each (owned) cell in the domain
///    * `riemannf[num_owned_cells][3]` - an array associating a 3-component
///      flux divergence with each (owned) cell in the domain
///    * `ext_src[num_owned_cells][3]` - an array associating 3 external source
///      components with each (owned) cell in the domain
///
/// Active output fields:
///    * `cell[num_owned_cells][3]` - an array associating a 3-component source
///      value with each (owned) cell in the domain
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config  RDycore's configuration
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [out] ceed_op a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateCeedSource0Operator(const RDyConfig config, RDyMesh *mesh, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;

  CeedQFunction qf;
  PetscCall(CreateSourceQFunction(ceed, config, &qf));

  // add inputs/outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation
  CeedInt num_comp_geom = 2, num_comp_ext_src = num_comp;
  CeedInt num_mat_props = NUM_MATERIAL_PROPERTIES;
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "ext_src", num_comp_ext_src, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "mat_props", num_mat_props, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction restrict_c, restrict_q, restrict_geom, restrict_ext_src, restrict_mat_props;
  CeedVector          geom, ext_src, mat_props;
  {
    PetscInt num_local_cells = mesh->num_cells;
    PetscInt num_owned_cells = mesh->num_owned_cells;

    // create a vector of geometric factors (elevation function derivatives)
    CeedScalar(*g)[num_comp_geom];
    CeedInt strides_geom[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(
        CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt c = 0, owned_cell = 0; c < num_local_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      g[owned_cell][0] = cells->dz_dx[c];
      g[owned_cell][1] = cells->dz_dy[c];
      ++owned_cell;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector of external source terms
    CeedInt strides_ext_src[] = {num_comp_ext_src, 1, num_comp_ext_src};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_ext_src, num_owned_cells * num_comp_ext_src, strides_ext_src,
                                                   &restrict_ext_src));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_ext_src, &ext_src, NULL));
    PetscCallCEED(CeedVectorSetValue(ext_src, 0.0));

    // create a vector that stores Manning's coefficient for the region of interest
    // NOTE: we zero-initialize this coefficient here; it must be set before use
    // NOTE: using (Get/Restore)OperatorMaterialProperty
    CeedInt strides_mat_props[] = {num_mat_props, 1, num_mat_props};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_mat_props, num_owned_cells * num_mat_props, strides_mat_props,
                                                   &restrict_mat_props));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_mat_props, &mat_props, NULL));
    PetscCallCEED(CeedVectorSetValue(mat_props, 0.0));

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
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "ext_src", restrict_ext_src, CEED_BASIS_NONE, ext_src));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "mat_props", restrict_mat_props, CEED_BASIS_NONE, mat_props));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q", restrict_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell", restrict_c, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_ext_src));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_mat_props));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_c));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&ext_src));
  PetscCallCEED(CeedVectorDestroy(&mat_props));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// Creates a CEED source operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [out]   source_op           the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedSourceOperator(RDyConfig *config, RDyMesh *mesh, CeedOperator *source_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedOperatorCreateComposite(ceed, source_op));

  CeedOperator source_0;
  PetscCall(CreateCeedSource0Operator(*config, mesh, &source_0));
  PetscCall(CeedOperatorCompositeAddSub(*source_op, source_0));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
