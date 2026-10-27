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

// injection matrices (ndof and 1 dof) from a mesh to the mesh refined from it
PETSC_INTERN PetscErrorCode RDyAMRBuildInterpolationMatrices(DM, DM, const PetscInt, Mat *, Mat *);

// refines a mesh by simplex bisection where a label marks it, and builds the
// injection matrices; the label is consumed
PETSC_INTERN PetscErrorCode RDyAMRRefineByLabel(DM, DMLabel, const PetscInt, DM *, Mat *, Mat *);

// rebuilds the mesh from the base mesh at the levels rdy->amr.base_level asks for
PETSC_INTERN PetscErrorCode RDyAMRRebuildMeshFromBase(RDy, const PetscInt, PetscInt, DM *, Mat *, Mat *);

// the base cell a local cell of the current mesh descends from, or -1
PETSC_INTERN PetscErrorCode RDyAMRBaseCellOfLocalCell(RDy, PetscInt, PetscInt *);

// reads a per-base-cell elevation file onto the base mesh and injects it onto
// the mesh in hand; the setup path calls this once there is a base mesh
PETSC_INTERN PetscErrorCode RDyAMROverrideCellElevationFromBase(RDy);

// repartitions the base mesh by the cells each base cell is about to carry,
// bringing base_level, the base areas and the packed solution with it
// (src/rdyamr_balance.c)
PETSC_INTERN PetscErrorCode RDyAMRBalanceBaseMesh(RDy, DM, const PetscInt *, PetscInt, PetscSection *, PetscScalar **);

// the refinement criterion and the two ways to act on it (src/rdyamr_adaptor.c)
PETSC_INTERN PetscErrorCode RDyAMRMarkBySurfaceJump(RDy, DMLabel *);
PETSC_INTERN PetscErrorCode RDyAMRRefineFromLabel(RDy, DMLabel, DM *, Mat *, Mat *);
PETSC_INTERN PetscErrorCode RDyAMRUpdateBaseLevelsFromLabel(RDy, DMLabel);

#endif
