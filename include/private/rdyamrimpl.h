#ifndef RDYAMRIMPL_H
#define RDYAMRIMPL_H

#include <petscdmadaptor.h>
#include <private/rdymeshimpl.h>

// RDycore's DMAdaptor type: refines on the jump in water-surface elevation
#define RDYADAPTORSURFACEJUMP "surfacejump"

// RDycore's mesh adaptor: rebuilds from the base mesh, so it can coarsen
#define RDYMESHADAPTORREBUILD "rdycore_rebuild"

// conservative transfer of a cell-centred field between two adapted meshes
// (src/rdyamr_transfer.c)
PETSC_INTERN PetscErrorCode RDyAMRCreateTransferMatrix(RDyMesh *, RDyMesh *, Mat, Mat, PetscInt, PetscInt, Mat *);
PETSC_INTERN PetscErrorCode RDyAMRCreateSiblingAssign(PetscInt, PetscInt, Mat, Mat, PetscInt, PetscInt, Mat *);

// interpolation matrices between a mesh and the mesh refined from it
PETSC_INTERN PetscErrorCode RDyAMRBuildInterpolationMatrices(DM, DM, const PetscInt, Mat *, Mat *, Mat *, Mat *);

// rebuilds the mesh from the base mesh at the levels rdy->amr.base_level asks for
PETSC_INTERN PetscErrorCode RDyAMRRebuildMeshFromBase(RDy, const PetscInt, PetscInt, DM *, Mat *, Mat *, Mat *, Mat *);

// the DMAdaptor that drives AMR (src/rdyamr_adaptor.c)
PETSC_INTERN PetscErrorCode RDyAMRCreateAdaptor(RDy, DMAdaptor *);

#endif
