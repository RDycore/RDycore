#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysweimpl.h>
#include <private/rdytracerimpl.h>

#include "swe/swe_sources_hydro_recon_ceed.h"
#include "tracer/tracer_sources_hydro_recon_ceed.h"

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

static inline CeedInt NumTracers(const RDyConfig config) {
  return config.physics.sediment.num_classes + ((config.physics.salinity) ? 1 : 0) + ((config.physics.heat) ? 1 : 0);
}

static PetscErrorCode CreateHRSourceQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;

  CeedInt num_tracers = NumTracers(config);

  CeedQFunctionContext qf_context;
  if (num_tracers == 0) {  // flow only
    switch (config.physics.flow.source.method) {
      case SOURCE_SEMI_IMPLICIT:
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourcesHydroReconWithSemiImplicitBedFriction,
                                                  SWESourcesHydroReconWithSemiImplicitBedFriction_loc, qf));
        break;
      case SOURCE_IMPLICIT_XQ2018:
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourcesHydroReconWithImplicitBedFrictionXQ2018,
                                                  SWESourcesHydroReconWithImplicitBedFrictionXQ2018_loc, qf));
        break;
      case SOURCE_ARK_IMEX:
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourcesHydroReconWithoutBedFriction, SWESourcesHydroReconWithoutBedFriction_loc, qf));
        break;
      default:
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported source-term method for HR");
        break;
    }
    PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
  } else {  // flow + tracers
    switch (config.physics.flow.source.method) {
      case SOURCE_SEMI_IMPLICIT:
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, TracerSourcesHydroReconWithSemiImplicitBedFriction,
                                                  TracerSourcesHydroReconWithSemiImplicitBedFriction_loc, qf));
        break;
      case SOURCE_ARK_IMEX:
        PetscCallCEED(
            CeedQFunctionCreateInterior(ceed, 1, TracerSourcesHydroReconWithoutBedFriction, TracerSourcesHydroReconWithoutBedFriction_loc, qf));
        break;
      default:
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported source-term method for tracer HR");
        break;
    }
    PetscCall(CreateTracerQFunctionContext(ceed, config, &qf_context));
  }
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED source sub-operator for hydrostatic reconstruction.
///
/// The Q-function interface is identical to the standard source operator
/// (same fields: geom, ext_src, mat_props, riemannf, q -> cell) so that
/// the riemannf wiring in AddOperatorFluxDivergence works unchanged.
/// The geom field (dz/dx, dz/dy) is present but unused because the bed-slope
/// source terms are already accounted for by the HR pressure correction in
/// the flux operator.
///
/// @param [in]  config  RDycore's configuration
/// @param [in]  mesh    mesh defining the computational domain
/// @param [out] ceed_op the newly created operator
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateCeedSourceHydroReconSuboperator(const RDyConfig config, RDyMesh *mesh, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_flow_comp = 3;  // NOTE: SWE assumed!
  CeedInt num_tracers   = NumTracers(config);
  CeedInt num_comp      = num_flow_comp + num_tracers;

  RDyCells *cells = &mesh->cells;

  CeedQFunction qf;
  PetscCall(CreateHRSourceQFunction(ceed, config, &qf));

  // add inputs/outputs (same layout as standard source operator)
  CeedInt num_comp_geom    = 2;
  CeedInt num_comp_ext_src = num_comp;
  CeedInt num_mat_props    = NUM_MATERIAL_PROPERTIES;
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

    // create a vector of geometric factors (unused by HR but kept for interface compatibility)
    CeedScalar(*g)[2];
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

    // create a vector for material properties (Manning's n, etc.)
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

/// Creates a CEED source operator for hydrostatic reconstruction.
/// @param [in]    config    the configuration defining physics and numerics
/// @param [in]    mesh      a mesh containing geometric and topological information
/// @param [out]   source_op the newly created composite source operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedSourceHROperator(RDyConfig *config, RDyMesh *mesh, CeedOperator *source_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedOperatorCreateComposite(ceed, source_op));

  CeedOperator source_0;
  PetscCall(CreateCeedSourceHydroReconSuboperator(*config, mesh, &source_0));
  PetscCall(CeedOperatorCompositeAddSub(*source_op, source_0));
  PetscCall(CeedOperatorDestroy(&source_0));

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
