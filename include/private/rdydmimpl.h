#ifndef RDYDMIMPL_H
#define RDYDMIMPL_H

#include <petsc.h>
#include <rdycore.h>

// this struct specifies the number, degrees of freedom, and names of fields in
// a section within a DM
typedef struct {
  PetscInt num_fields;
  char     field_names[MAX_NUM_FIELDS][MAX_NAME_LEN + 1];
  PetscInt num_field_components[MAX_NUM_FIELDS];
  char     field_component_names[MAX_NUM_FIELDS][MAX_NUM_FIELD_COMPONENTS][MAX_NAME_LEN + 1];
} SectionFieldSpec;

PETSC_INTERN PetscErrorCode CreateCellCenteredDMFromDM(DM, const SectionFieldSpec, DM *);
PETSC_INTERN PetscErrorCode CreateDM(RDy);
PETSC_INTERN PetscErrorCode CreateAuxiliaryDM(RDy);
PETSC_INTERN PetscErrorCode CreateFlowDM(RDy);
PETSC_INTERN PetscErrorCode CreateSedimentDM(RDy);
PETSC_INTERN PetscErrorCode CreateVectors(RDy);
PETSC_INTERN PetscErrorCode CreateSectionForSWE(RDy, PetscSection *);

#endif
