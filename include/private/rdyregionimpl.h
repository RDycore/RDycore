#ifndef RDYCOREREGIONIMPL_H
#define RDYCOREREGIONIMPL_H

#include <petsc/private/petscimpl.h>
#include <rdycore.h>

// This type defines a region consisting of cells identified by their local
// indices.
typedef struct {
  char      name[MAX_NAME_LEN + 1];  // region name
  PetscInt  id;                      // region ID (as specified in mesh file)
  PetscInt  index;                   // index of region witin RDycore region list
  PetscInt *cell_ids;
  PetscInt  num_cells;
} RDyRegion;

#endif
