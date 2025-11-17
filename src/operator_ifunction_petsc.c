#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysedimentimpl.h>
#include <private/rdysweimpl.h>

PetscErrorCode CreatePetscIFunctionOperator(RDyConfig *config, RDyMesh *mesh, Vec material_properties, PetscOperator *ifunction_op) {
  PetscFunctionBegin;

  PetscCall(PetscCompositeOperatorCreate(ifunction_op));
  PetscOperator ifunction_0;
  PetscCall(CreateSWEPetscIFunctionOperator(mesh, *config, material_properties, &ifunction_0));
  PetscCall(PetscCompositeOperatorAddSub(*ifunction_op, ifunction_0));

  PetscFunctionReturn(PETSC_SUCCESS);
}
