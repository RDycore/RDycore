#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdytracerimpl.h>
#include <private/rdysweimpl.h>

#include "tracer/tracer_sources_ceed.h"

/// Creates a PETSc source operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [in]    external_sources    a sequential Vec that can store external sources for the local process
/// @param [in]    material_properties a sequential Vec that can store material properties for the local process
/// @param [out]   source_op           the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreatePetscSourceOperator(RDyConfig *config, RDyMesh *mesh, Vec external_sources, Vec material_properties, PetscOperator *source_op) {
  PetscFunctionBegin;

  PetscCall(PetscOperatorCreateComposite(source_op));

  PetscOperator source_0;
  if (config->physics.sediment.num_classes > 0) {
    PetscCall(CreatePetscTracerSourceOperator(mesh, *config, external_sources, material_properties, &source_0));
  } else {
    PetscCall(CreatePetscSWESourceOperator(mesh, *config, external_sources, material_properties, &source_0));
  }
  PetscCall(PetscCompositeOperatorAddSub(*source_op, source_0));

  PetscFunctionReturn(PETSC_SUCCESS);
}
