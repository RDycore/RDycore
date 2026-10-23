#ifndef RDYAMRIMPL_H
#define RDYAMRIMPL_H

#include <petscdmplex.h>
#include <private/rdymeshimpl.h>

// conservative transfer of a cell-centred field between two adapted meshes
// (src/rdyamr_transfer.c)
PETSC_INTERN PetscErrorCode RDyAMRCreateTransferMatrix(RDyMesh *, RDyMesh *, Mat, Mat, PetscInt, PetscInt, Mat *);

// interpolation matrices between a mesh and the mesh refined from it
PETSC_INTERN PetscErrorCode RDyAMRBuildInterpolationMatrices(DM, DM, const PetscInt, Mat *, Mat *, Mat *, Mat *);

// refines a mesh by simplex bisection where a label marks it, and builds the
// interpolation matrices; the label is consumed
PETSC_INTERN PetscErrorCode RDyAMRRefineByLabel(DM, DMLabel, const PetscInt, DM *, Mat *, Mat *, Mat *, Mat *);

// rebuilds the mesh from the base mesh at the levels rdy->amr.base_level asks for
PETSC_INTERN PetscErrorCode RDyAMRRebuildMeshFromBase(RDy, const PetscInt, PetscInt, DM *, Mat *, Mat *, Mat *, Mat *);

// the base cell a local cell of the current mesh descends from, or -1
PETSC_INTERN PetscErrorCode RDyAMRBaseCellOfLocalCell(RDy, PetscInt, PetscInt *);

// the refinement criterion and the two ways to act on it (src/rdyamr_adaptor.c)
PETSC_INTERN PetscErrorCode RDyAMRMarkBySurfaceJump(RDy, DMLabel *);
PETSC_INTERN PetscErrorCode RDyAMRRefineFromLabel(RDy, DMLabel, DM *, Mat *, Mat *, Mat *, Mat *);
PETSC_INTERN PetscErrorCode RDyAMRRebuildFromLabel(RDy, DMLabel, DM *, Mat *, Mat *, Mat *, Mat *);

#endif
