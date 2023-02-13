#include <petscdmplex.h>
#include <private/rdymemoryimpl.h>
#include <private/rdymeshimpl.h>

// Returns true iff start <= closure < end.
static PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start, PetscInt end) { return (closure >= start) && (closure < end); }

/// Allocates and initializes an RDyCells struct.
/// @param [in] num_cells Number of cells
/// @param [out] cells A pointer to an RDyCells that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyCellsCreate(PetscInt num_cells, RDyCells *cells) {
  PetscFunctionBegin;

  PetscInt vertices_per_cell  = 4;
  PetscInt edges_per_cell     = 4;
  PetscInt neighbors_per_cell = 4;

  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->ids));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->global_ids));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->natural_ids));
  PetscCall(RDyFill(PetscInt, cells->global_ids, num_cells, -1));
  PetscCall(RDyFill(PetscInt, cells->natural_ids, num_cells, -1));

  PetscCall(RDyAlloc(PetscBool, num_cells, &cells->is_local));
  PetscCall(RDyFill(PetscInt, cells->is_local, num_cells, PETSC_FALSE));

  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->num_vertices));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->num_edges));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->num_neighbors));
  PetscCall(RDyFill(PetscInt, cells->num_vertices, num_cells, -1));
  PetscCall(RDyFill(PetscInt, cells->num_edges, num_cells, -1));
  PetscCall(RDyFill(PetscInt, cells->num_neighbors, num_cells, -1));

  PetscCall(RDyAlloc(PetscInt, num_cells + 1, &cells->vertex_offsets));
  PetscCall(RDyAlloc(PetscInt, num_cells + 1, &cells->edge_offsets));
  PetscCall(RDyAlloc(PetscInt, num_cells + 1, &cells->neighbor_offsets));
  PetscCall(RDyFill(PetscInt, cells->vertex_offsets, num_cells + 1, -1));
  PetscCall(RDyFill(PetscInt, cells->edge_offsets, num_cells + 1, -1));
  PetscCall(RDyFill(PetscInt, cells->neighbor_offsets, num_cells + 1, -1));

  PetscCall(RDyAlloc(PetscInt, num_cells * vertices_per_cell, &cells->vertex_ids));
  PetscCall(RDyAlloc(PetscInt, num_cells * edges_per_cell, &cells->edge_ids));
  PetscCall(RDyAlloc(PetscInt, num_cells * neighbors_per_cell, &cells->neighbor_ids));
  PetscCall(RDyFill(PetscInt, cells->vertex_ids, num_cells * vertices_per_cell, -1));
  PetscCall(RDyFill(PetscInt, cells->edge_ids, num_cells * edges_per_cell, -1));
  PetscCall(RDyFill(PetscInt, cells->neighbor_ids, num_cells * neighbors_per_cell, -1));

  PetscCall(RDyAlloc(RDyPoint, num_cells, &cells->centroids));
  PetscCall(RDyAlloc(PetscReal, num_cells, &cells->areas));

  for (PetscInt icell = 0; icell < num_cells; icell++) {
    cells->ids[icell]           = icell;
    cells->num_vertices[icell]  = vertices_per_cell;
    cells->num_edges[icell]     = edges_per_cell;
    cells->num_neighbors[icell] = neighbors_per_cell;
  }

  for (PetscInt icell = 0; icell <= num_cells; icell++) {
    cells->vertex_offsets[icell]   = icell * vertices_per_cell;
    cells->edge_offsets[icell]     = icell * edges_per_cell;
    cells->neighbor_offsets[icell] = icell * neighbors_per_cell;
  }

  PetscFunctionReturn(0);
}

/// Creates a fully initialized RDyCells struct from a given DM.
/// @param [in] dm A DM that provides cell data
/// @param [out] cells A pointer to an RDyCells that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyCellsCreateFromDM(DM dm, RDyCells *cells) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  // allocate cell storage
  PetscCall(RDyCellsCreate(cEnd - cStart, cells));

  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt  icell = c - cStart;
    PetscInt  dim   = 2;
    PetscReal centroid[dim], normal[dim];
    DMPlexComputeCellGeometryFVM(dm, c, &cells->areas[icell], &centroid[0], &normal[0]);

    for (PetscInt idim = 0; idim < dim; idim++) {
      cells->centroids[icell].X[idim] = centroid[idim];
    }

    PetscInt  pSize;
    PetscInt *p        = NULL;
    PetscInt  use_cone = PETSC_TRUE;

    cells->num_vertices[icell] = 0;
    cells->num_edges[icell]    = 0;

    // Get information about which cells are local.
    PetscInt gref, junkInt;
    PetscCall(DMPlexGetPointGlobal(dm, c, &gref, &junkInt));
    if (gref >= 0) {
      cells->is_local[icell] = PETSC_TRUE;
    } else {
      cells->is_local[icell] = PETSC_FALSE;
    }

    PetscCall(DMPlexGetTransitiveClosure(dm, c, use_cone, &pSize, &p));
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset        = cells->edge_offsets[icell];
        PetscInt index         = offset + cells->num_edges[icell];
        cells->edge_ids[index] = p[i] - eStart;
        cells->num_edges[icell]++;
      } else {
        PetscInt offset          = cells->vertex_offsets[icell];
        PetscInt index           = offset + cells->num_vertices[icell];
        cells->vertex_ids[index] = p[i] - vStart;
        cells->num_vertices[icell]++;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, use_cone, &pSize, &p));
  }

  PetscFunctionReturn(0);
}

/// Destroys an RDyCells struct, freeing its resources.
/// @param [inout] cells An RDyCells struct whose resources will be freed.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyCellsDestroy(RDyCells cells) {
  PetscFunctionBegin;

  PetscCall(RDyFree(cells.ids));
  PetscCall(RDyFree(cells.global_ids));
  PetscCall(RDyFree(cells.natural_ids));
  PetscCall(RDyFree(cells.is_local));
  PetscCall(RDyFree(cells.num_vertices));
  PetscCall(RDyFree(cells.num_edges));
  PetscCall(RDyFree(cells.num_neighbors));
  PetscCall(RDyFree(cells.vertex_offsets));
  PetscCall(RDyFree(cells.edge_offsets));
  PetscCall(RDyFree(cells.neighbor_offsets));
  PetscCall(RDyFree(cells.vertex_ids));
  PetscCall(RDyFree(cells.edge_ids));
  PetscCall(RDyFree(cells.neighbor_ids));
  PetscCall(RDyFree(cells.centroids));
  PetscCall(RDyFree(cells.areas));

  PetscFunctionReturn(0);
}

/// Allocates and initializes an RDyVertices struct.
/// @param [in] num_vertices Number of vertices
/// @param [out] vertices A pointer to an RDyVertices that stores data
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyVerticesCreate(PetscInt num_vertices, RDyVertices *vertices) {
  PetscFunctionBegin;

  PetscInt cells_per_vertex = 4;
  PetscInt edges_per_vertex = 4;

  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->ids));
  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->global_ids));
  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->num_cells));
  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->num_edges));
  PetscCall(RDyFill(PetscInt, vertices->global_ids, num_vertices, -1));

  PetscCall(RDyAlloc(PetscBool, num_vertices, &vertices->is_local));

  PetscCall(RDyAlloc(RDyPoint, num_vertices, &vertices->points));

  PetscCall(RDyAlloc(PetscInt, num_vertices + 1, &vertices->edge_offsets));
  PetscCall(RDyAlloc(PetscInt, num_vertices + 1, &vertices->cell_offsets));

  PetscCall(RDyAlloc(PetscInt, num_vertices * edges_per_vertex, &vertices->edge_ids));
  PetscCall(RDyAlloc(PetscInt, num_vertices * cells_per_vertex, &vertices->cell_ids));
  PetscCall(RDyFill(PetscInt, vertices->edge_ids, num_vertices * edges_per_vertex, -1));
  PetscCall(RDyFill(PetscInt, vertices->cell_ids, num_vertices * cells_per_vertex, -1));

  for (PetscInt ivertex = 0; ivertex < num_vertices; ivertex++) {
    vertices->ids[ivertex] = ivertex;
  }

  for (PetscInt ivertex = 0; ivertex <= num_vertices; ivertex++) {
    vertices->edge_offsets[ivertex] = ivertex * edges_per_vertex;
    vertices->cell_offsets[ivertex] = ivertex * cells_per_vertex;
  }

  PetscFunctionReturn(0);
}

/// Creates a fully initialized RDyVertices struct from a given DM.
/// @param [in] dm A DM that provides vertex data
/// @param [out] vertices A pointer to an RDyVertices that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyVerticesCreateFromDM(DM dm, RDyVertices *vertices) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  // allocate vertex storage
  PetscCall(RDyVerticesCreate(vEnd - vStart, vertices));

  PetscSection coordSection;
  Vec          coordinates;
  DMGetCoordinateSection(dm, &coordSection);
  DMGetCoordinatesLocal(dm, &coordinates);
  PetscReal *coords;
  VecGetArray(coordinates, &coords);

  for (PetscInt v = vStart; v < vEnd; v++) {
    PetscInt  ivertex = v - vStart;
    PetscInt  pSize;
    PetscInt *p = NULL;

    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));

    PetscInt coordOffset, dim = 2;
    PetscSectionGetOffset(coordSection, v, &coordOffset);
    for (PetscInt idim = 0; idim < dim; idim++) {
      vertices->points[ivertex].X[idim] = coords[coordOffset + idim];
    }

    vertices->num_edges[ivertex] = 0;
    vertices->num_cells[ivertex] = 0;

    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset           = vertices->edge_offsets[ivertex];
        PetscInt index            = offset + vertices->num_edges[ivertex];
        vertices->edge_ids[index] = p[i] - eStart;
        vertices->num_edges[ivertex]++;
      } else {
        PetscInt offset           = vertices->cell_offsets[ivertex];
        PetscInt index            = offset + vertices->num_cells[ivertex];
        vertices->cell_ids[index] = p[i] - cStart;
        vertices->num_cells[ivertex]++;
      }
    }

    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));
  }

  VecRestoreArray(coordinates, &coords);

  PetscFunctionReturn(0);
}

/// Destroys an RDyVertices struct, freeing its resources.
/// @param [inout] vertices An RDyVertices struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyVerticesDestroy(RDyVertices vertices) {
  PetscFunctionBegin;

  PetscCall(RDyFree(vertices.ids));
  PetscCall(RDyFree(vertices.global_ids));
  PetscCall(RDyFree(vertices.is_local));
  PetscCall(RDyFree(vertices.num_cells));
  PetscCall(RDyFree(vertices.num_edges));
  PetscCall(RDyFree(vertices.cell_offsets));
  PetscCall(RDyFree(vertices.edge_offsets));
  PetscCall(RDyFree(vertices.cell_ids));
  PetscCall(RDyFree(vertices.edge_ids));
  PetscCall(RDyFree(vertices.points));

  PetscFunctionReturn(0);
}

/// Allocates and initializes an RDyEdges struct.
/// @param [in] num_edges Number of edges
/// @param [out] edges A pointer to an RDyEdges that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyEdgesCreate(PetscInt num_edges, RDyEdges *edges) {
  PetscFunctionBegin;

  PetscInt cells_per_edge = 2;

  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->ids));
  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->global_ids));
  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->num_cells));
  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->vertex_ids));
  PetscCall(RDyFill(PetscInt, edges->global_ids, num_edges, -1));
  PetscCall(RDyFill(PetscInt, edges->num_cells, num_edges, -1));
  PetscCall(RDyFill(PetscInt, edges->vertex_ids, num_edges, -1));

  PetscCall(RDyAlloc(PetscBool, num_edges, &edges->is_local));
  PetscCall(RDyAlloc(PetscBool, num_edges, &edges->is_internal));

  PetscCall(RDyAlloc(PetscInt, num_edges + 1, &edges->cell_offsets));
  PetscCall(RDyAlloc(PetscInt, num_edges * cells_per_edge, &edges->cell_ids));
  PetscCall(RDyFill(PetscInt, edges->cell_ids, num_edges * cells_per_edge, -1));

  PetscCall(RDyAlloc(RDyPoint, num_edges, &edges->centroids));
  PetscCall(RDyAlloc(RDyVector, num_edges, &edges->normals));
  PetscCall(RDyAlloc(PetscReal, num_edges, &edges->lengths));

  for (PetscInt iedge = 0; iedge < num_edges; iedge++) {
    edges->ids[iedge] = iedge;
  }

  for (PetscInt iedge = 0; iedge <= num_edges; iedge++) {
    edges->cell_offsets[iedge] = iedge * cells_per_edge;
  }

  PetscFunctionReturn(0);
}

/// Creates a fully initialized RDyEdges struct from a given DM.
/// @param [in] dm A DM that provides edge data
/// @param [out] edges A pointer to an RDyEdges that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyEdgesCreateFromDM(DM dm, RDyEdges *edges) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  // allocate edge storage
  PetscCall(RDyEdgesCreate(eEnd - eStart, edges));

  for (PetscInt e = eStart; e < eEnd; e++) {
    PetscInt  iedge = e - eStart;
    PetscInt  dim   = 2;
    PetscReal centroid[dim], normal[dim];
    DMPlexComputeCellGeometryFVM(dm, e, &edges->lengths[iedge], &centroid[0], &normal[0]);

    for (PetscInt idim = 0; idim < dim; idim++) {
      edges->centroids[iedge].X[idim] = centroid[idim];
      edges->normals[iedge].V[idim]   = normal[idim];
    }

    // edge-to-vertex
    PetscInt  pSize;
    PetscInt *p        = NULL;
    PetscInt  use_cone = PETSC_TRUE;
    PetscCall(DMPlexGetTransitiveClosure(dm, e, use_cone, &pSize, &p));
    PetscAssert(pSize == 3, comm, PETSC_ERR_ARG_SIZ, "Incorrect transitive closure size!");
    PetscInt index               = iedge * 2;
    edges->vertex_ids[index + 0] = p[2] - vStart;
    edges->vertex_ids[index + 1] = p[4] - vStart;
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, use_cone, &pSize, &p));

    // edge-to-cell
    edges->num_cells[iedge] = 0;
    PetscCall(DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
    PetscAssert(pSize == 2 || pSize == 3, comm, PETSC_ERR_ARG_SIZ, "Incorrect transitive closure size!");
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      PetscInt offset        = edges->cell_offsets[iedge];
      PetscInt index         = offset + edges->num_cells[iedge];
      edges->cell_ids[index] = p[i] - cStart;
      edges->num_cells[iedge]++;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
  }

  PetscFunctionReturn(0);
}

/// Destroys an RDyEdges struct, freeing its resources.
/// @param [inout] edges An RDyEdges struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyEdgesDestroy(RDyEdges edges) {
  PetscFunctionBegin;

  PetscCall(RDyFree(edges.ids));
  PetscCall(RDyFree(edges.global_ids));
  PetscCall(RDyFree(edges.is_local));
  PetscCall(RDyFree(edges.num_cells));
  PetscCall(RDyFree(edges.vertex_ids));
  PetscCall(RDyFree(edges.cell_offsets));
  PetscCall(RDyFree(edges.cell_ids));
  PetscCall(RDyFree(edges.is_internal));
  PetscCall(RDyFree(edges.normals));
  PetscCall(RDyFree(edges.centroids));
  PetscCall(RDyFree(edges.lengths));

  PetscFunctionReturn(0);
}

static PetscErrorCode SaveNaturalCellIDs(DM dm, RDyCells *cells, PetscInt rank) {
  PetscFunctionBegin;

  PetscBool useNatural;
  PetscCall(DMGetUseNatural(dm, &useNatural));

  if (useNatural) {
    PetscInt num_fields;

    PetscCall(DMGetNumFields(dm, &num_fields));

    // Create the natural vector
    Vec      natural;
    PetscInt natural_size = 0, natural_start;
    PetscCall(DMPlexCreateNaturalVector(dm, &natural));
    PetscCall(PetscObjectSetName((PetscObject)natural, "Natural Vec"));
    PetscCall(VecGetLocalSize(natural, &natural_size));
    PetscCall(VecGetOwnershipRange(natural, &natural_start, NULL));

    // Add entries in the natural vector
    PetscScalar *entries;
    PetscCall(VecGetArray(natural, &entries));
    for (PetscInt i = 0; i < natural_size; ++i) {
      if (i % num_fields == 0) {
        entries[i] = (natural_start + i) / num_fields;
      } else {
        entries[i] = -(rank + 1);
      }
    }
    PetscCall(VecRestoreArray(natural, &entries));

    // Map natural IDs in global order
    Vec global;
    PetscCall(DMCreateGlobalVector(dm, &global));
    PetscCall(PetscObjectSetName((PetscObject)global, "Global Vec"));
    PetscCall(DMPlexNaturalToGlobalBegin(dm, natural, global));
    PetscCall(DMPlexNaturalToGlobalEnd(dm, natural, global));

    // Map natural IDs in local order
    Vec         local;
    PetscViewer selfviewer;
    PetscCall(DMCreateLocalVector(dm, &local));
    PetscCall(PetscObjectSetName((PetscObject)local, "Local Vec"));
    PetscCall(DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local));
    PetscCall(DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &selfviewer));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &selfviewer));

    // Save natural IDs
    PetscInt local_size;
    PetscCall(VecGetLocalSize(local, &local_size));
    PetscCall(VecGetArray(local, &entries));
    for (PetscInt i = 0; i < local_size / num_fields; ++i) {
      cells->natural_ids[i] = entries[i * num_fields];
    }
    PetscCall(VecRestoreArray(local, &entries));

    // Cleanup
    PetscCall(VecDestroy(&natural));
    PetscCall(VecDestroy(&global));
    PetscCall(VecDestroy(&local));
  }

  PetscFunctionReturn(0);
}

/// Creates an RDyMesh from a PETSc DM.
/// @param [in] dm A PETSc DM
/// @param [out] mesh A pointer to an RDyMesh that stores allocated data.
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyMeshCreateFromDM(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  // Determine the number of cells in the mesh
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  mesh->num_cells = cEnd - cStart;

  // Determine the number of edges in the mesh
  PetscInt eStart, eEnd;
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  mesh->num_edges = eEnd - eStart;

  // Determine the number of vertices in the mesh
  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
  mesh->num_vertices = vEnd - vStart;

  // Create mesh elements from the DM
  PetscCall(RDyCellsCreateFromDM(dm, &mesh->cells));
  PetscCall(RDyEdgesCreateFromDM(dm, &mesh->edges));
  PetscCall(RDyVerticesCreateFromDM(dm, &mesh->vertices));

  // Count up local cells.
  mesh->num_cells_local = 0;
  for (PetscInt icell = 0; icell < mesh->num_cells; ++icell) {
    if (mesh->cells.is_local[icell]) {
      ++mesh->num_cells_local;
    }
  }

  // Extract natural cell IDs from the DM.
  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscInt rank;
  MPI_Comm_rank(comm, &rank);
  PetscCall(SaveNaturalCellIDs(dm, &mesh->cells, rank));

  PetscFunctionReturn(0);
}

/// Destroys an RDyMesh struct, freeing its resources.
/// @param [inout] edges An RDyMesh struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyMeshDestroy(RDyMesh mesh) {
  PetscFunctionBegin;
  PetscCall(RDyCellsDestroy(mesh.cells));
  PetscCall(RDyEdgesDestroy(mesh.edges));
  PetscCall(RDyVerticesDestroy(mesh.vertices));
  PetscFunctionReturn(0);
}
