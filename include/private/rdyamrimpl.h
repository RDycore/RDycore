#ifndef RDYAMRIMPL_H
#define RDYAMRIMPL_H

#include <private/rdymeshimpl.h>

// conservative transfer of a cell-centred field between two adapted meshes
// (src/rdyamr_transfer.c)
PETSC_INTERN PetscErrorCode RDyAMRCreateTransferMatrix(RDyMesh *, RDyMesh *, Mat, Mat, PetscInt, PetscInt, Mat *);
PETSC_INTERN PetscErrorCode RDyAMRCreateSiblingAssign(PetscInt, PetscInt, Mat, Mat, PetscInt, PetscInt, Mat *);

#endif
