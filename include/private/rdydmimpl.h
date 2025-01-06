#ifndef RDYDMIMPL_H
#define RDYDMIMPL_H

#include <petsc.h>
#include <rdycore.h>

// maximum number of supported fields in a DM section
#define MAX_NUM_SECTION_FIELDS 5

// this struct specifies the number, degrees of freedom, and names of fields in
// a section within a DM
typedef struct {
  PetscInt num_fields;                                             // number of fields
  PetscInt num_field_dof[MAX_NUM_SECTION_FIELDS];                  // number of degrees of freedom for fields
  char     field_names[MAX_NUM_SECTION_FIELDS][MAX_NAME_LEN + 1];  // names of fields
} SectionFieldSpec;

PETSC_INTERN PetscErrorCode CloneAndCreateCellCenteredDM(DM dm, const SectionFieldSpec cc_spec, DM *cc_dm);
PETSC_INTERN PetscErrorCode CreateDM(RDy rdy);
PETSC_INTERN PetscErrorCode CreateAuxiliaryDM(RDy rdy);
PETSC_INTERN PetscErrorCode CreateVectors(RDy rdy);
PETSC_INTERN PetscErrorCode CreateSectionForSWE(RDy, PetscSection *);

#endif
