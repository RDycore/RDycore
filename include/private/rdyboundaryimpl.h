#ifndef RDYBOUNDARYIMPL_H
#define RDYBOUNDARYIMPL_H

#include <petsc/private/petscimpl.h>
#include <rdycore.h>  // for MAX_NAME_LEN

// This type defines a boundary consisting of edges identified by their local
// indices.
typedef struct {
  char      name[MAX_NAME_LEN + 1];  // boundary name
  PetscInt  id;                      // boundary ID (as specified in mesh file)
  PetscInt  index;                   // index of boundary within RDycore boundary list
  PetscInt *edge_ids;
  PetscInt  num_edges;
  CeedVector flux_accumulated;
} RDyBoundary;

#endif
