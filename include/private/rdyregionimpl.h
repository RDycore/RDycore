#ifndef RDYCOREREGIONIMPL_H
#define RDYCOREREGIONIMPL_H

#include <petsc/private/petscimpl.h>
#include <rdycore.h>

// This type defines a region consisting of cells referenced by specific IDs.
typedef struct {
  char     name[MAX_NAME_LEN + 1];  // region name
  PetscInt id;                      // region ID (as specified in mesh file)
  PetscInt index;                   // index of region witin RDycore region list

  PetscInt  num_owned_cells;        // number of owned cells in the region
  PetscInt *owned_cell_global_ids;  // global IDs for owned cells in the region,
                                    // for use with a global PETSc Vec
                                    // (this is redundant, since cells belonging
                                    //  to the global topology are all owned, but
                                    //  it serves as a reminder of our complicated
                                    //  indexing nomenclature)

  PetscInt num_local_cells;  // number of cells in the region that belong to
                             // the local topology
  PetscInt *cell_local_ids;  // local IDs for cells in the region, for use with
                             // a local PETSc Vec
} RDyRegion;

PETSC_INTERN PetscErrorCode DestroyRegion(RDyRegion *region);

#endif
