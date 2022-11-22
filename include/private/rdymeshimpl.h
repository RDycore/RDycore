#ifndef RDYMESHIMPL_H
#define RDYMESHIMPL_H

#include <rdycore.h>

#include <petsc.h>
#include <petsc/private/petscimpl.h>

/// a point in R^3
typedef struct {
  PetscReal X[3];
} RDyPoint;

/// a vector in R^3
typedef struct {
  PetscReal V[3];
} RDyVector;

/// a type indicating one of a set of supported cell types
typedef enum {
  CELL_TRI_TYPE = 0,  // tetrahedron cell for a 3D cell
  CELL_QUAD_TYPE      // hexahedron cell for a 3D cell
} RDyCellType;

/// A struct of arrays storing information about mesh cells. The ith element in
/// each array stores a property for mesh cell i.
typedef struct {
  /// local IDs of cells in local numbering
  PetscInt *ids;
  /// global IDs of cells in local numbering
  PetscInt *global_ids;
  /// natural IDs of cells in local numbering
  PetscInt *natural_ids;

  /// PETSC_TRUE iff corresponding cell is locally stored
  PetscBool *is_local;

  /// numbers of cell vertices
  PetscInt *num_vertices;
  /// numbers of cell edges
  PetscInt *num_edges;
  /// numbers of cell neigbors (themselves cells)
  PetscInt *num_neighbors;

  /// offsets of first cell vertices in vertex_ids
  PetscInt *vertex_offsets;
  /// IDs of vertices for cells
  PetscInt *vertex_ids;

  /// offsets of first cell edges in edge_ids
  PetscInt *edge_offsets;
  /// IDs of edges for cells
  PetscInt *edge_ids;

  /// offsets of first neighbors in neighbor_ids for cells
  PetscInt *neighbor_offsets;
  /// IDs of neighbors for cells
  PetscInt *neighbor_ids;

  /// cell centroids
  RDyPoint *centroids;
  /// cell areas
  PetscReal *areas;
} RDyCells;

/// A struct of arrays storing information about mesh vertices. The ith element
/// in each array stores a property for vertex i.
typedef struct {
  /// local IDs of vertices in local numbering
  PetscInt *ids;
  /// global IDs of vertices in local numbering
  PetscInt *global_ids;

  /// PETSC_TRUE iff vertex is attached to a local cell
  PetscBool *is_local;

  /// numbers of cells attached to vertices
  PetscInt *num_cells;
  /// numbers of edges attached to vertices
  PetscInt *num_edges;

  /// offsets of first vertex edges in edge_ids
  PetscInt *edge_offsets;
  /// IDs of edges attached to vertices
  PetscInt *edge_ids;

  /// offsets of first vertex cells in cell_ids
  PetscInt *cell_offsets;
  /// IDs of local cells attached to vertices
  PetscInt *cell_ids;

  /// vertex positions
  RDyPoint *points;
} RDyVertices;

/// A struct of arrays storing information about edges separating mesh cells.
/// The ith element in each array stores a property for edge i.
typedef struct {
  /// local IDs of edges in local numbering
  PetscInt *ids;
  /// global IDs of edges in local numbering
  PetscInt *global_ids;

  /// PETSC_TRUE iff edge is shared by locally owned cells, OR
  /// if it is shared by a local cell c1 and non-local cell c2 such that
  /// global ID(c1) < global ID(c2).
  PetscBool *is_local;

  /// numbers of cells attached to edges
  PetscInt *num_cells;
  /// numbers of vertices attached to edges
  PetscInt *vertex_ids;

  /// offsets of first edge cells in cell_ids
  PetscInt *cell_offsets;
  /// IDs of cells attached to edges
  PetscInt *cell_ids;

  /// false if the edge is on the domain boundary
  PetscBool *is_internal;

  /// unit vector pointing out of one cell into another for each edge
  RDyVector *normals;
  /// edge centroids
  RDyPoint *centroids;
  /// edge lengths
  PetscReal *lengths;
} RDyEdges;

/// A mesh representing a computational domain consisting of a set of cells
/// connected by edges and vertices
typedef struct RDyMesh {
  /// spatial dimension of the mesh (1, 2, or 3)
  PetscInt dim;

  /// number of cells in the mesh (across all processes)
  PetscInt num_cells;
  /// number of cells in the mesh stored on the local process
  PetscInt num_cells_local;
  /// number of edges in the mesh attached to locally stored cells
  PetscInt num_edges;
  /// number of vertices in the mesh attached to locally stored cells
  PetscInt num_vertices;
  /// number of faces on the domain boundary attached to locally stored cells
  PetscInt num_boundary_faces;

  /// the maximum number of vertices attached to any cell
  PetscInt max_vertex_cells;
  /// the maximum number of vertices attached to any face
  PetscInt max_vertex_faces;

  /// cell information
  RDyCells cells;
  /// vertex information
  RDyVertices vertices;
  /// edge information
  RDyEdges edges;

  /// closure sizes and data for locally stored cells
  PetscInt *closureSize, **closure;
  /// the maximum closure size for any cell (locally stored?)
  PetscInt maxClosureSize;
} RDyMesh;

PETSC_INTERN PetscErrorCode RDyCellsCreate(PetscInt, RDyCells*);
PETSC_INTERN PetscErrorCode RDyCellsCreateFromDM(DM, RDyCells*);
PETSC_INTERN PetscErrorCode RDyCellsDestroy(RDyCells);

PETSC_INTERN PetscErrorCode RDyVerticesCreate(PetscInt, RDyVertices*);
PETSC_INTERN PetscErrorCode RDyVerticesCreateFromDM(DM, RDyVertices*);
PETSC_INTERN PetscErrorCode RDyVerticesDestroy(RDyVertices);

PETSC_INTERN PetscErrorCode RDyEdgesCreate(PetscInt, RDyEdges*);
PETSC_INTERN PetscErrorCode RDyEdgesCreateFromDM(DM, RDyEdges*);
PETSC_INTERN PetscErrorCode RDyEdgesDestroy(RDyEdges);

PETSC_INTERN PetscErrorCode RDyMeshCreateFromDM(DM, RDyMesh*);
PETSC_INTERN PetscErrorCode RDyMeshDestroy(RDyMesh);

#endif
