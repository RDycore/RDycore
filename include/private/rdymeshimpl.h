#ifndef RDYMESHIMPL_H
#define RDYMESHIMPL_H

#include <petsc.h>
#include <petsc/private/petscimpl.h>
#include <rdycore.h>

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

  /// PETSC_TRUE iff corresponding cell is owned (locally stored)
  PetscBool *is_owned;

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
  // IDs of neighbors for cells
  PetscInt *neighbor_ids;

  // cell centroids
  RDyPoint *centroids;
  // cell areas
  PetscReal *areas;

  // surface x and y slopes
  PetscReal *dz_dx, *dz_dy;

  // mapping cells of a local PETSc Vec that includes owned and ghost cells
  // to a global PETSc Vec that only includes owned cells
  PetscInt *local_to_owned;

  // mapping cells of a global PETSc Vec that only includes owned cells
  // to a local PETSc Vec that includes owned and ghost cells
  PetscInt *owned_to_local;

} RDyCells;

// A struct of arrays storing information about mesh vertices. The ith element
// in each array stores a property for vertex i.
typedef struct {
  // local IDs of vertices in local numbering
  PetscInt *ids;
  // global IDs of vertices in local numbering
  PetscInt *global_ids;

  // numbers of cells attached to vertices
  PetscInt *num_cells;
  // numbers of edges attached to vertices
  PetscInt *num_edges;

  // offsets of first vertex edges in edge_ids
  PetscInt *edge_offsets;
  // IDs of edges attached to vertices
  PetscInt *edge_ids;

  // offsets of first vertex cells in cell_ids
  PetscInt *cell_offsets;
  // IDs of local cells attached to vertices
  PetscInt *cell_ids;

  // vertex positions
  RDyPoint *points;
} RDyVertices;

// A struct of arrays storing information about edges separating mesh cells.
// The ith element in each array stores a property for edge i.
typedef struct {
  // local IDs of edges in local numbering
  PetscInt *ids;
  // global IDs of edges in local numbering
  PetscInt *global_ids;
  // local IDs of internal edges
  PetscInt *internal_edge_ids;
  // local IDs of boundary edges
  PetscInt *boundary_edge_ids;

  // IDs of vertices attached to edges
  PetscInt *vertex_ids;

  // IDs of cells attached to edges:
  // * cell_ids[2*i]   is the first cell attached to edge i
  // * cell_ids[2*i+1] is the second cell attached to edge i (-1 if none)
  PetscInt *cell_ids;

  // false if the edge is on the domain boundary
  PetscBool *is_internal;

  // true if edge is owned
  PetscBool *is_owned;

  // unit vector pointing out of one cell into another for each edge
  RDyVector *normals;
  // edge centroids
  RDyPoint *centroids;
  // edge lengths
  PetscReal *lengths;
  // cosine of the angle between edge and y-axis
  PetscReal *cn;
  // sine of the angle between edge and y-axis
  PetscReal *sn;
} RDyEdges;

// A mesh representing a computational domain consisting of a set of cells
// connected by edges and vertices
typedef struct RDyMesh {
  // number of mesh refinements based on DMRefine
  PetscInt refine_level;

  // spatial dimension of the mesh (1, 2, or 3)
  PetscInt dim;

  // number of cells in the mesh (across ghost cells owned by other processes)
  PetscInt num_cells;
  // number of cells in the mesh owned by the local process
  PetscInt num_owned_cells;
  // number of total cells in the global mesh
  PetscInt num_cells_global;
  // number of edges in the mesh attached to locally stored cells
  PetscInt num_edges;
  /// number of edges that are internal (i.e. shared by two cells)
  PetscInt num_internal_edges;
  /// number of owned internal edges
  PetscInt num_owned_internal_edges;
  /// number of edges that are on the boundary
  PetscInt num_boundary_edges;
  // total number of vertices in the global mesh
  PetscInt num_vertices_global;
  // number of vertices in the mesh attached to locally stored cells
  PetscInt num_vertices;
  // number of faces on the domain boundary attached to locally stored cells
  PetscInt num_boundary_faces;

  // the maximum number of vertices that form a cell
  PetscInt max_nvertices_per_cell;
  // the maximum number of vertices that form a cell
  PetscInt max_nedges_per_cell;
  // the maximum number of cells that a vertex is a part
  PetscInt max_ncells_per_vertex;
  // the maximum number of edges that a vertex is a part
  PetscInt max_nedges_per_vertex;

  // cell information
  RDyCells cells;
  // vertex information
  RDyVertices vertices;
  // edge information
  RDyEdges edges;

  // closure sizes and data for locally stored cells
  PetscInt *closureSize, **closure;
  // the maximum closure size for any cell (locally stored?)
  PetscInt maxClosureSize;

  struct {
    // for output: coordinates of vertices (in vertex natural order)
    Vec vertices_xyz_norder;
    // for output: connections of vertices forming the cells (in cell natural order)
    Vec cell_conns_norder;
    // for output: cell centroids and area (in cell natural order)
    Vec xc, yc, zc, area;
  } output;

} RDyMesh;

PETSC_INTERN PetscErrorCode RDyMeshCreateFromDM(DM, PetscInt, RDyMesh *);
PETSC_INTERN PetscErrorCode RDyMeshDestroy(RDyMesh);
PETSC_INTERN PetscErrorCode RDyMeshGetLocalCellXCentroids(RDyMesh *, const PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RDyMeshGetLocalCellYCentroids(RDyMesh *, const PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RDyMeshGetLocalCellZCentroids(RDyMesh *, const PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RDyMeshGetLocalCellAreas(RDyMesh *, const PetscInt, PetscReal *);

#endif
