#ifndef RDYAMRIMPL_H
#define RDYAMRIMPL_H

#include <petscdmplex.h>
#include <private/rdymeshimpl.h>

// Conservative transfer of a cell-centred field between two adapted meshes
// (src/rdyamr_transfer.c). The old mesh enters as a packed array -- per base
// cell, its descendants' polygons and values -- so that it can have come from
// another rank through DMPlexDistributeField().
#define RDY_AMR_PACK_MAX_V 4  // vertices a packed cell holds; simplex bisection gives 3
static inline PetscInt RDyAMRPackStride(PetscInt ndof) { return 1 + 2 * RDY_AMR_PACK_MAX_V + ndof; }

PETSC_INTERN PetscErrorCode RDyAMRPackCellsByBaseCell(RDyMesh *, DM, Mat, PetscInt, Vec, PetscSection *, PetscScalar **);
PETSC_INTERN PetscErrorCode RDyAMRTransferFromPacked(RDyMesh *, DM, Mat, PetscInt, PetscSection, const PetscScalar *, Vec);

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
PETSC_INTERN PetscErrorCode RDyAMRUpdateBaseLevelsFromLabel(RDy, DMLabel);

#endif
