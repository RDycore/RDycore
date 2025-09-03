#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysedimentimpl.h>
#include <private/rdysweimpl.h>

/// Creates a new PetscOperator with behavior defined by the given arguments.
/// @param [in]  context a pointer to a data structure used by the operator implementation
/// @param [in]  apply   the function called by PetscOperatorApply
/// @param [in]  destroy the function called by PetscOperatorDestroy
/// @param [out] op      the PetscOperator created by this call
PetscErrorCode PetscOperatorCreate(void *context, PetscErrorCode (*apply)(void *, PetscOperatorFields, PetscReal, Vec, Vec),
                                   PetscErrorCode (*destroy)(void *), PetscOperator *op) {
  PetscFunctionBegin;
  PetscCall(PetscCalloc1(1, op));
  (*op)->fields       = (PetscOperatorFields){0};
  (*op)->is_composite = PETSC_FALSE;
  (*op)->context      = context;
  (*op)->apply        = apply;
  (*op)->destroy      = destroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Releases resources allocated to the given PetscOperator.
/// @param [inout] op the PetscOperator to be destroyed
PetscErrorCode PetscOperatorDestroy(PetscOperator *op) {
  PetscFunctionBegin;
  if ((*op)->fields.fields) {
    PetscFree((*op)->fields.fields);
  }
  PetscCall((*op)->destroy((*op)->context));
  PetscFree(*op);
  *op = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Applies the given PetscOperator to a local solution vector, storing the
/// results in a global "right-hand-side" vector.
/// @param [inout] op       the operator being applyed to the solution vector
/// @param [in]    dt       the timestep over which the operator is applied
/// @param [in]    u_local  the local solution vector to which the operator is applied
/// @param [inout] f_global the global vector storing the results of the application
PetscErrorCode PetscOperatorApply(PetscOperator op, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;
  PetscCall(op->apply(op->context, op->fields, dt, u_local, f_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt       num_suboperators, capacity;
  PetscOperator *suboperators;
} PetscCompositeOperator;

static PetscErrorCode PetscCompositeOperatorApply(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = context;
  for (PetscInt i = 0; i < composite->num_suboperators; ++i) {
    PetscCall(PetscOperatorApply(composite->suboperators[i], dt, u_local, f_global));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscCompositeOperatorDestroy(void *context) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = context;
  for (PetscInt i = 0; i < composite->num_suboperators; ++i) {
    PetscCall(PetscOperatorDestroy(&composite->suboperators[i]));
  }
  PetscFree(composite->suboperators);
  PetscFree(composite);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that applies sub-operators in sequence.
/// @param [out] a pointer to the created (empty) composite operator
PetscErrorCode PetscCompositeOperatorCreate(PetscOperator *op) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite;
  PetscCall(PetscCalloc1(1, &composite));
  *composite = (PetscCompositeOperator){0};
  PetscCall(PetscOperatorCreate(composite, PetscCompositeOperatorApply, PetscCompositeOperatorDestroy, op));
  (*op)->is_composite = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Appends a sub-operator to the given composite PetscOperator (which takes ownership).
/// @param [inout] op     the composite operator to which a sub-operator is appended
/// @param [in]    sub_op the suboperator to be appended to the composite operator
PetscErrorCode PetscCompositeOperatorAddSub(PetscOperator op, PetscOperator sub_op) {
  PetscFunctionBegin;
  PetscCompositeOperator *composite = op->context;
  if (composite->num_suboperators + 1 > composite->capacity) {
    composite->capacity = (composite->capacity > 0) ? 2 * composite->capacity : 8;
    PetscCall(PetscRealloc(sizeof(PetscOperator) * composite->capacity, &composite->suboperators));
  }
  composite->suboperators[composite->num_suboperators] = sub_op;
  ++composite->num_suboperators;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves a PETSc Vec corresponding to the PetscOperator field with the given name.
/// This is useful for extracting PETSc Vecs for use within operator apply() functions.
/// @param [in]  fields the set of available fields
/// @param [in]  name   the name of the desired field
/// @param [out] vec    points to the field with the desired name, or to NULL if no such field exists
PetscErrorCode PetscOperatorFieldsGet(PetscOperatorFields fields, const char *name, Vec *vec) {
  PetscFunctionBegin;
  for (PetscInt f = 0; f < fields.num_fields; ++f) {
    if (!strncmp(name, fields.fields[f].name, MAX_NAME_LEN)) {  // found it!
      *vec = fields.fields[f].vec;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  *vec = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppendOperatorField(PetscOperatorFields *fields, PetscOperatorField field) {
  PetscFunctionBegin;

  if (fields->num_fields + 1 > fields->capacity) {
    fields->capacity = (fields->capacity > 0) ? 2 * fields->capacity : 8;
    PetscCall(PetscRealloc(sizeof(PetscOperatorField) * fields->capacity, &fields->fields));
  }
  fields->fields[fields->num_fields] = field;
  ++fields->num_fields;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Adds a field with the given name and Vec to the given PETSc operator. If
/// this is a composite operator, the field is set on all sub-operators. The
/// operator does NOT manage memory for the PETSc Vec.
/// @param [inout] op   the operator for which a field is set
/// @param [in]    name the name of the field
/// @param [in]    vec  the PETSc Vec storing the field's data (NOT managed by the operator)
PetscErrorCode PetscOperatorSetField(PetscOperator op, const char *name, Vec vec) {
  PetscFunctionBegin;
  // look for an existing field of this name
  PetscInt index = -1;
  if (op->fields.num_fields > 0) {
    for (PetscInt f = 0; f < op->fields.num_fields; ++f) {
      if (!strncmp(name, op->fields.fields[f].name, MAX_NAME_LEN)) {
        index = f;
      }
    }
  }

  // replace the existing field or append a new one
  PetscOperatorField field = {
      .vec = vec,
  };
  strncpy(field.name, name, MAX_NAME_LEN);
  if (index != -1) {
    op->fields.fields[index] = field;
  } else {
    PetscCall(AppendOperatorField(&op->fields, field));
  }

  if (op->is_composite) {
    PetscCompositeOperator *composite = op->context;
    for (PetscInt i = 0; i < composite->num_suboperators; ++i) {
      PetscCall(PetscOperatorSetField(composite->suboperators[i], name, vec));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PETSc flux operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [in]    num_boundaries      the number of distinct boundaries bounding the computational domain
/// @param [in]    boundaries          an array of distinct boundaries bounding the computational domain
/// @param [in]    boundary_conditions an array of boundary conditions corresponding to the domain boundaries
/// @param [inout] boundary_values     an array of sequential Vecs that can store boundary values for each boundary
/// @param [inout] boundary_fluxes     an array of sequential Vecs that can store boundary fluxes for each boundary
/// @param [out]   flux_op             the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreatePetscFluxOperator(RDyConfig *config, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries,
                                       RDyCondition *boundary_conditions, Vec *boundary_values, Vec *boundary_fluxes,
                                       OperatorDiagnostics *diagnostics, PetscOperator *flux_op) {
  PetscFunctionBegin;

  PetscCall(PetscCompositeOperatorCreate(flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }

  // flux suboperator 0: fluxes between interior cells

  PetscOperator interior_flux_op;
  if (config->physics.sediment.num_classes > 0) {
    PetscCall(CreateSedimentPetscInteriorFluxOperator(mesh, *config, diagnostics, &interior_flux_op));
  } else {
    PetscCall(CreateSWEPetscInteriorFluxOperator(mesh, *config, diagnostics, &interior_flux_op));
  }
  PetscCall(PetscCompositeOperatorAddSub(*flux_op, interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    PetscOperator boundary_flux_op;
    RDyBoundary   boundary  = boundaries[b];
    RDyCondition  condition = boundary_conditions[b];
    if (config->physics.sediment.num_classes > 0) {
      PetscCall(CreateSedimentPetscBoundaryFluxOperator(mesh, *config, boundary, condition, boundary_values[b], boundary_fluxes[b], diagnostics,
                                                        &boundary_flux_op));
    } else {
      PetscCall(CreateSWEPetscBoundaryFluxOperator(mesh, *config, boundary, condition, boundary_values[b], boundary_fluxes[b], diagnostics,
                                                   &boundary_flux_op));
    }
    PetscCall(PetscCompositeOperatorAddSub(*flux_op, boundary_flux_op));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreatePetscFluxOperatorReconstructed(RDyConfig *config, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries,
                                       RDyCondition *boundary_conditions, Vec *boundary_values, Vec *boundary_fluxes,
                                       OperatorDiagnostics *diagnostics, PetscOperator *flux_op) {
  PetscFunctionBegin;

  PetscCall(PetscCompositeOperatorCreate(flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }

  // flux suboperator 0: fluxes between interior cells (THIS IS THE KEY CHANGE)
  PetscOperator interior_flux_op;
  if (config->physics.sediment.num_classes > 0) {
    // You'll need to create a reconstructed sediment version too
    PetscCall(CreateSedimentPetscInteriorFluxOperator(mesh, *config, diagnostics, &interior_flux_op));
  } else {
    // ** CHANGED: Call the reconstructed version (removed dm parameter) **
    PetscCall(CreateSWEPetscInteriorFluxOperatorReconstructed(mesh, *config, diagnostics, &interior_flux_op));
  }
  PetscCall(PetscCompositeOperatorAddSub(*flux_op, interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges (same as original)
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    PetscOperator boundary_flux_op;
    RDyBoundary   boundary  = boundaries[b];
    RDyCondition  condition = boundary_conditions[b];
    if (config->physics.sediment.num_classes > 0) {
      PetscCall(CreateSedimentPetscBoundaryFluxOperator(mesh, *config, boundary, condition, boundary_values[b], boundary_fluxes[b], diagnostics,
                                                        &boundary_flux_op));
    } else {
      PetscCall(CreateSWEPetscBoundaryFluxOperator(mesh, *config, boundary, condition, boundary_values[b], boundary_fluxes[b], diagnostics,
                                                   &boundary_flux_op));
    }
    PetscCall(PetscCompositeOperatorAddSub(*flux_op, boundary_flux_op));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
/// Creates a PETSc source operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [in]    external_sources    a sequential Vec that can store external sources for the local process
/// @param [in]    material_properties a sequential Vec that can store material properties for the local process
/// @param [out]   source_op           the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreatePetscSourceOperator(RDyConfig *config, RDyMesh *mesh, Vec external_sources, Vec material_properties, PetscOperator *source_op) {
  PetscFunctionBegin;

  PetscCall(PetscCompositeOperatorCreate(source_op));
  PetscOperator source_0;
  if (config->physics.sediment.num_classes > 0) {
    PetscCall(CreateSedimentPetscSourceOperator(mesh, *config, external_sources, material_properties, &source_0));
  } else {
    PetscCall(CreateSWEPetscSourceOperator(mesh, *config, external_sources, material_properties, &source_0));
  }
  PetscCall(PetscCompositeOperatorAddSub(*source_op, source_0));

  PetscFunctionReturn(PETSC_SUCCESS);
}
