#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdysolversimpl.h>
#include <private/rdysweimpl.h>

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

// these are used to register CEED operator events for profiling
static PetscClassId RDY_CLASSID;
PetscLogEvent       RDY_CeedOperatorApply;

// initializes solvers for physics specified in the input configuration
PetscErrorCode InitSolvers(RDy rdy) {
  PetscFunctionBegin;

  // register a logging event for applying our CEED operator
  PetscCall(PetscClassIdRegister("RDycore", &RDY_CLASSID));
  PetscCall(PetscLogEventRegister("CeedOperatorApp", RDY_CLASSID, &RDY_CeedOperatorApply));

  // just pass the call along for now
  InitSWE(rdy);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// frees all resources devoted to solvers
PetscErrorCode DestroySolvers(RDy rdy) {
  PetscFunctionBegin;

  PetscBool ceed_enabled = CeedEnabled(rdy);

  if (rdy->petsc.context) {
    PetscCall(DestroyPetscSWEFlux(rdy->petsc.context, ceed_enabled, rdy->num_boundaries));
  }

  if (ceed_enabled) {
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.flux_operator));
    PetscCallCEED(CeedOperatorDestroy(&rdy->ceed.source_operator));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.u_local));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.rhs));
    PetscCallCEED(CeedVectorDestroy(&rdy->ceed.sources));
    if (rdy->ceed.host_fluxes) VecDestroy(&rdy->ceed.host_fluxes);
    // the CEED context belongs to RDycore itself, so we don't delete it
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// produces a BoundaryData object corresponding to the boundary with the given
// index
PetscErrorCode GetBoundaryData(RDy rdy, PetscInt boundary_index, BoundaryData *boundary_data) {
  PetscFunctionBegin;

  PetscCheck(boundary_index >= 0 && boundary_index < rdy->num_boundaries, rdy->comm, PETSC_ERR_USER,
             "Invalid boundary index for boundary data: %" PetscInt_FMT, boundary_index);
  boundary_data->rdy      = rdy;
  boundary_data->boundary = rdy->boundaries[boundary_index];
  PetscCall(VecGetBlockSize(rdy->u_global, &boundary_data->num_components));

  if (CeedEnabled(rdy)) {
    // get the relevant boundary sub-operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(rdy->ceed.flux_operator, &sub_ops));
    CeedOperator flux_op = sub_ops[1 + boundary_data->boundary.index];

    // fetch the array storing the boundary values
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(flux_op, "q_dirichlet", &field));
    CeedVector vec;
    PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));
    PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &boundary_data->ceed.data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets boundary values for the given component on the boundary associated with
// the given boundary data
PetscErrorCode SetBoundaryValues(BoundaryData boundary_data, PetscInt component, PetscReal *boundary_values) {
  PetscFunctionBegin;

  if (CeedEnabled(boundary_data.rdy)) {
    CeedScalar(*dirichlet_values)[boundary_data.num_components];
    *((CeedScalar **)&dirichlet_values) = boundary_data.ceed.data;

    // copy the boundary values into place
    for (CeedInt i = 0; i < boundary_data.boundary.num_edges; ++i) {
      dirichlet_values[i][component] = boundary_values[i];
    }
  } else {
    // FIXME: PETSc stuff goes here
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CommitBoundaryValues(BoundaryData boundary_data) {
  PetscFunctionBegin;
  if (CeedEnabled(boundary_data.rdy)) {
    // send the boundary values to the device
    PetscCallCEED(CeedVectorRestoreArray(boundary_data.ceed.vec, &boundary_data.ceed.data));
  } else {
    // FIXME: PETSc stuff goes here
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode GetSourceData(RDy rdy, PetscInt region_index, SourceData *source_data) {
  PetscFunctionBegin;

  PetscCheck(region_index >= 0 && region_index < rdy->num_regions, rdy->comm, PETSC_ERR_USER, "Invalid region index for source data: %" PetscInt_FMT,
             region_index);
  source_data->rdy    = rdy;
  source_data->region = rdy->regions[region_index];
  PetscCall(VecGetBlockSize(rdy->u_global, &source_data->num_components));

  if (CeedEnabled(rdy)) {
    // get the relevant source sub-operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(rdy->ceed.flux_operator, &sub_ops));
    CeedOperator source_op = sub_ops[0];

    // fetch the array storing the source values
    // FIXME: this is only valid for SWE
    CeedOperatorField field;
    PetscCallCEED(CeedOperatorGetFieldByName(source_op, "swe_src", &field));
    CeedVector vec;
    PetscCallCEED(CeedOperatorFieldGetVector(field, &vec));
    PetscCallCEED(CeedVectorGetArray(vec, CEED_MEM_HOST, &source_data->ceed.data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets source values for the given component in the region associated with the
// given source data
PetscErrorCode SetSourceValues(SourceData source_data, PetscInt component, PetscReal *source_values) {
  PetscFunctionBegin;

  if (CeedEnabled(source_data.rdy)) {
    CeedScalar(*values)[source_data.num_components];
    *((CeedScalar **)&values) = source_data.ceed.data;

    // copy the boundary values into place
    for (CeedInt i = 0; i < source_data.region.num_cells; ++i) {
      values[i][component] = source_values[i];
    }
  } else {
    // FIXME: PETSc stuff goes here
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CommitSourceValues(SourceData source_data) {
  PetscFunctionBegin;
  if (CeedEnabled(source_data.rdy)) {
    // send the boundary values to the device
    PetscCallCEED(CeedVectorRestoreArray(source_data.ceed.vec, &source_data.ceed.data));
  } else {
    // FIXME: PETSc stuff goes here
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
