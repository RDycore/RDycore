#include <petscdmplex.h>
#include <private/rdydmimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdymeshimpl.h>

// Returns true iff start <= closure < end.
static PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start, PetscInt end) { return (closure >= start) && (closure < end); }

static PetscInt TRI_ID_EXODUS  = 4;
static PetscInt QUAD_ID_EXODUS = 5;

// fills the given array of length n with the given value
#define FILL(n, array, value)      \
  for (size_t i = 0; i < n; ++i) { \
    array[i] = value;              \
  }

/// Allocates and initializes an RDyCells struct.
/// @param [in] num_cells Number of cells
/// @param [in] nvertices_per_cell Maximum number of vertices per cell
/// @param [in] nedges_per_cell Maximum number of edges per cell
/// @param [out] cells A pointer to an RDyCells that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyCellsCreate(PetscInt num_cells, PetscInt nvertices_per_cell, PetscInt nedges_per_cell, RDyCells *cells) {
  PetscFunctionBegin;

  PetscInt neighbors_per_cell = nedges_per_cell;

  PetscCall(PetscCalloc1(num_cells, &cells->ids));
  PetscCall(PetscCalloc1(num_cells, &cells->global_ids));
  PetscCall(PetscCalloc1(num_cells, &cells->natural_ids));
  FILL(num_cells, cells->global_ids, -1);
  FILL(num_cells, cells->natural_ids, -1);

  PetscCall(PetscCalloc1(num_cells, &cells->is_local));
  PetscCall(PetscCalloc1(num_cells, &cells->is_local));

  PetscCall(PetscCalloc1(num_cells, &cells->local_to_owned));
  FILL(num_cells, cells->local_to_owned, -1);

  PetscCall(PetscCalloc1(num_cells, &cells->num_vertices));
  PetscCall(PetscCalloc1(num_cells, &cells->num_edges));
  PetscCall(PetscCalloc1(num_cells, &cells->num_neighbors));
  FILL(num_cells, cells->num_vertices, -1);
  FILL(num_cells, cells->num_edges, -1);
  FILL(num_cells, cells->num_neighbors, -1);

  PetscCall(PetscCalloc1(num_cells + 1, &cells->vertex_offsets));
  PetscCall(PetscCalloc1(num_cells + 1, &cells->edge_offsets));
  PetscCall(PetscCalloc1(num_cells + 1, &cells->neighbor_offsets));
  FILL(num_cells + 1, cells->vertex_offsets, -1);
  FILL(num_cells + 1, cells->edge_offsets, -1);
  FILL(num_cells + 1, cells->neighbor_offsets, -1);

  PetscCall(PetscCalloc1(num_cells * nvertices_per_cell, &cells->vertex_ids));
  PetscCall(PetscCalloc1(num_cells * nedges_per_cell, &cells->edge_ids));
  PetscCall(PetscCalloc1(num_cells * neighbors_per_cell, &cells->neighbor_ids));
  FILL(num_cells * nvertices_per_cell, cells->vertex_ids, -1);
  FILL(num_cells * nedges_per_cell, cells->edge_ids, -1);
  FILL(num_cells * neighbors_per_cell, cells->neighbor_ids, -1);

  PetscCall(PetscCalloc1(num_cells, &cells->centroids));
  PetscCall(PetscCalloc1(num_cells, &cells->areas));
  PetscCall(PetscCalloc1(num_cells, &cells->dz_dx));
  PetscCall(PetscCalloc1(num_cells, &cells->dz_dy));

  for (PetscInt icell = 0; icell < num_cells; icell++) {
    cells->ids[icell]           = icell;
    cells->num_vertices[icell]  = nvertices_per_cell;
    cells->num_edges[icell]     = nedges_per_cell;
    cells->num_neighbors[icell] = neighbors_per_cell;
  }

  for (PetscInt icell = 0; icell <= num_cells; icell++) {
    cells->vertex_offsets[icell]   = icell * nvertices_per_cell;
    cells->edge_offsets[icell]     = icell * nedges_per_cell;
    cells->neighbor_offsets[icell] = icell * neighbors_per_cell;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a fully initialized RDyCells struct from a given DM.
/// @param [in] dm A DM that provides cell data
/// @param [out] cells A pointer to an RDyCells that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyCellsCreateFromDM(DM dm, PetscInt nvertices_per_cell, PetscInt nedges_per_cell, RDyCells *cells) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscInt dim;
  PetscCall(DMGetCoordinateDim(dm, &dim));

  PetscInt c_start, c_end;
  PetscInt e_start, e_end;
  PetscInt v_start, v_end;
  DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
  DMPlexGetDepthStratum(dm, 1, &e_start, &e_end);
  DMPlexGetDepthStratum(dm, 0, &v_start, &v_end);

  // allocate cell storage
  PetscInt num_cells = c_end - c_start;
  PetscCall(RDyCellsCreate(num_cells, nvertices_per_cell, nedges_per_cell, cells));

  PetscInt num_cells_local = 0;
  for (PetscInt c = c_start; c < c_end; c++) {
    PetscInt  icell = c - c_start;
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
      num_cells_local++;
    } else {
      cells->is_local[icell] = PETSC_FALSE;
    }

    PetscCall(DMPlexGetTransitiveClosure(dm, c, use_cone, &pSize, &p));
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], e_start, e_end)) {
        PetscInt offset        = cells->edge_offsets[icell];
        PetscInt index         = offset + cells->num_edges[icell];
        cells->edge_ids[index] = p[i] - e_start;
        cells->num_edges[icell]++;
      } else {
        PetscInt offset          = cells->vertex_offsets[icell];
        PetscInt index           = offset + cells->num_vertices[icell];
        cells->vertex_ids[index] = p[i] - v_start;
        cells->num_vertices[icell]++;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, use_cone, &pSize, &p));
  }

  // make a first pass to put all local cells at the beginning
  PetscCall(PetscCalloc1(num_cells_local, &cells->owned_to_local));
  FILL(num_cells_local, cells->owned_to_local, -1);

  PetscInt count = 0;
  for (PetscInt icell = 0; icell < num_cells; icell++) {
    if (cells->is_local[icell]) {
      cells->local_to_owned[icell] = count;
      cells->owned_to_local[count] = icell;
      count++;
    }
  }

  for (PetscInt icell = 0; icell < num_cells; icell++) {
    if (!cells->is_local[icell]) {
      cells->local_to_owned[icell] = count;
      count++;
    }
  }

  // fetch global cell IDs.
  ISLocalToGlobalMapping map;
  PetscCall(DMGetLocalToGlobalMapping(dm, &map));
  PetscCall(ISLocalToGlobalMappingApply(map, num_cells, cells->ids, cells->global_ids));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys an RDyCells struct, freeing its resources.
/// @param [inout] cells An RDyCells struct whose resources will be freed.
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyCellsDestroy(RDyCells cells) {
  PetscFunctionBegin;

  PetscCall(PetscFree(cells.ids));
  PetscCall(PetscFree(cells.global_ids));
  PetscCall(PetscFree(cells.natural_ids));
  PetscCall(PetscFree(cells.is_local));
  PetscCall(PetscFree(cells.num_vertices));
  PetscCall(PetscFree(cells.num_edges));
  PetscCall(PetscFree(cells.num_neighbors));
  PetscCall(PetscFree(cells.vertex_offsets));
  PetscCall(PetscFree(cells.edge_offsets));
  PetscCall(PetscFree(cells.neighbor_offsets));
  PetscCall(PetscFree(cells.vertex_ids));
  PetscCall(PetscFree(cells.edge_ids));
  PetscCall(PetscFree(cells.neighbor_ids));
  PetscCall(PetscFree(cells.centroids));
  PetscCall(PetscFree(cells.areas));
  PetscCall(PetscFree(cells.dz_dx));
  PetscCall(PetscFree(cells.dz_dy));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Allocates and initializes an RDyVertices struct.
/// @param [in] num_vertices Number of vertices
/// @param [in] ncells_per_vertex Maximum number of cells per vertex
/// @param [in] nedges_per_vertex Maximum number of edges per vertex
/// @param [out] vertices A pointer to an RDyVertices that stores data
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyVerticesCreate(PetscInt num_vertices, PetscInt ncells_per_vertex, PetscInt nedges_per_vertex, RDyVertices *vertices) {
  PetscFunctionBegin;

  PetscCall(PetscCalloc1(num_vertices, &vertices->ids));
  PetscCall(PetscCalloc1(num_vertices, &vertices->global_ids));
  PetscCall(PetscCalloc1(num_vertices, &vertices->num_cells));
  PetscCall(PetscCalloc1(num_vertices, &vertices->num_edges));
  FILL(num_vertices, vertices->global_ids, -1);

  PetscCall(PetscCalloc1(num_vertices, &vertices->is_local));

  PetscCall(PetscCalloc1(num_vertices, &vertices->points));

  PetscCall(PetscCalloc1(num_vertices + 1, &vertices->edge_offsets));
  PetscCall(PetscCalloc1(num_vertices + 1, &vertices->cell_offsets));

  PetscCall(PetscCalloc1(num_vertices * nedges_per_vertex, &vertices->edge_ids));
  PetscCall(PetscCalloc1(num_vertices * ncells_per_vertex, &vertices->cell_ids));
  FILL(num_vertices * nedges_per_vertex, vertices->edge_ids, -1);
  FILL(num_vertices * ncells_per_vertex, vertices->cell_ids, -1);

  for (PetscInt ivertex = 0; ivertex < num_vertices; ivertex++) {
    vertices->ids[ivertex] = ivertex;
  }

  for (PetscInt ivertex = 0; ivertex <= num_vertices; ivertex++) {
    vertices->edge_offsets[ivertex] = ivertex * nedges_per_vertex;
    vertices->cell_offsets[ivertex] = ivertex * ncells_per_vertex;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a fully initialized RDyVertices struct from a given DM.
/// @param [in] dm A DM that provides vertex data
/// @param [out] vertices A pointer to an RDyVertices that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyVerticesCreateFromDM(DM dm, PetscInt ncells_per_vertex, PetscInt nedges_per_vertex, RDyVertices *vertices,
                                              PetscInt *num_vertices_global) {
  PetscFunctionBegin;

  PetscInt dim;
  PetscCall(DMGetCoordinateDim(dm, &dim));

  PetscInt c_start, c_end;
  PetscInt e_start, e_end;
  PetscInt v_start, v_end;
  DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
  DMPlexGetDepthStratum(dm, 1, &e_start, &e_end);
  DMPlexGetDepthStratum(dm, 0, &v_start, &v_end);

  // allocate vertex storage
  PetscInt num_vertices = v_end - v_start;
  PetscCall(RDyVerticesCreate(num_vertices, ncells_per_vertex, nedges_per_vertex, vertices));

  PetscSection coordSection;
  Vec          coordinates;
  DMGetCoordinateSection(dm, &coordSection);
  DMGetCoordinatesLocal(dm, &coordinates);
  PetscReal *coords;
  VecGetArray(coordinates, &coords);

  for (PetscInt v = v_start; v < v_end; v++) {
    PetscInt  ivertex = v - v_start;
    PetscInt  pSize;
    PetscInt *p = NULL;

    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));
    PetscInt coordOffset;
    PetscSectionGetOffset(coordSection, v, &coordOffset);

    for (PetscInt idim = 0; idim < dim; idim++) {
      vertices->points[ivertex].X[idim] = coords[coordOffset + idim];
    }
    if (dim < 3) {
      vertices->points[ivertex].X[2] = 0.0;
    }

    vertices->num_edges[ivertex] = 0;
    vertices->num_cells[ivertex] = 0;

    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], e_start, e_end)) {
        PetscInt offset           = vertices->edge_offsets[ivertex];
        PetscInt index            = offset + vertices->num_edges[ivertex];
        vertices->edge_ids[index] = p[i] - e_start;
        vertices->num_edges[ivertex]++;
      } else {
        PetscInt offset           = vertices->cell_offsets[ivertex];
        PetscInt index            = offset + vertices->num_cells[ivertex];
        vertices->cell_ids[index] = p[i] - c_start;
        vertices->num_cells[ivertex]++;
      }
    }

    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));
  }

  VecRestoreArray(coordinates, &coords);

  // fetch global vertex IDs if mesh is not refined
  PetscInt refine_level;
  PetscCall(DMGetRefineLevel(dm, &refine_level));
  if (!refine_level) {
    PetscMPIInt commsize;
    MPI_Comm    comm;
    PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
    PetscCallMPI(MPI_Comm_size(comm, &commsize));

    if (commsize == 1) {
      for (PetscInt v = v_start; v < v_end; v++) {
        PetscInt ivertex              = v - v_start;
        vertices->global_ids[ivertex] = ivertex;
      }
    } else {
      PetscSF            sf;
      const PetscInt    *local;
      const PetscSFNode *natural;
      PetscInt           p_start, p_end, Nl;

      PetscCall(DMPlexGetMigrationSF(dm, &sf));
      PetscCheck(sf, comm, PETSC_ERR_ARG_WRONGSTATE, "DM must have a migration SF");

      PetscCall(DMPlexGetChart(dm, &p_start, &p_end));
      PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &local, &natural));
      PetscCheck(p_end - p_start == Nl, comm, PETSC_ERR_PLIB,
                 "The number of mesh points %" PetscInt_FMT " != %" PetscInt_FMT " the number of migration leaves", p_end - p_start, Nl);

      PetscMPIInt min_vertex_idx;  // to save the min v_start

      for (PetscInt v = v_start; v < v_end; v++) {
        PetscInt ivertex = v - v_start;
        if (local) PetscCall(PetscFindInt(v, Nl, local, &v));
        PetscCheck(v >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Vertex %" PetscInt_FMT " not found in migration SF", v);
        PetscCheck(!natural[v].rank, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                   "Natural ID for vertex %" PetscInt_FMT " should come from rank 0 not %" PetscInt_FMT, v, natural[v].rank);
        vertices->global_ids[ivertex] = natural[v].index;

        if (v == v_start) {
          min_vertex_idx = natural[v].index;
        } else {
          min_vertex_idx = PetscMin(min_vertex_idx, natural[v].index);
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, &min_vertex_idx, 1, MPI_INT, MPI_MIN, comm);

      // substract the min v_start
      for (PetscInt v = v_start; v < v_end; v++) {
        PetscInt ivertex = v - v_start;
        vertices->global_ids[ivertex] -= min_vertex_idx;
      }
    }
  }

  // compute total number of vertices
  DMGetCoordinates(dm, &coordinates);
  VecGetSize(coordinates, num_vertices_global);
  *num_vertices_global = *num_vertices_global / dim;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys an RDyVertices struct, freeing its resources.
/// @param [inout] vertices An RDyVertices struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyVerticesDestroy(RDyVertices vertices) {
  PetscFunctionBegin;

  PetscCall(PetscFree(vertices.ids));
  PetscCall(PetscFree(vertices.global_ids));
  PetscCall(PetscFree(vertices.is_local));
  PetscCall(PetscFree(vertices.num_cells));
  PetscCall(PetscFree(vertices.num_edges));
  PetscCall(PetscFree(vertices.cell_offsets));
  PetscCall(PetscFree(vertices.edge_offsets));
  PetscCall(PetscFree(vertices.cell_ids));
  PetscCall(PetscFree(vertices.edge_ids));
  PetscCall(PetscFree(vertices.points));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Allocates and initializes an RDyEdges struct.
/// @param [in] num_edges Number of edges
/// @param [out] edges A pointer to an RDyEdges that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyEdgesCreate(PetscInt num_edges, RDyEdges *edges) {
  PetscFunctionBegin;

  PetscCall(PetscCalloc1(num_edges, &edges->ids));
  PetscCall(PetscCalloc1(num_edges, &edges->global_ids));
  PetscCall(PetscCalloc1(2 * num_edges, &edges->vertex_ids));
  FILL(num_edges, edges->global_ids, -1);
  FILL(2 * num_edges, edges->vertex_ids, -1);

  PetscCall(PetscCalloc1(num_edges, &edges->is_local));
  PetscCall(PetscCalloc1(num_edges, &edges->is_internal));
  PetscCall(PetscCalloc1(num_edges, &edges->is_owned));

  PetscCall(PetscCalloc1(2 * num_edges, &edges->cell_ids));
  FILL(2 * num_edges, edges->cell_ids, -1);

  PetscCall(PetscCalloc1(num_edges, &edges->centroids));
  PetscCall(PetscCalloc1(num_edges, &edges->normals));
  PetscCall(PetscCalloc1(num_edges, &edges->lengths));
  PetscCall(PetscCalloc1(num_edges, &edges->cn));
  PetscCall(PetscCalloc1(num_edges, &edges->sn));

  for (PetscInt iedge = 0; iedge < num_edges; iedge++) {
    edges->ids[iedge] = iedge;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a fully initialized RDyEdges struct from a given DM.
/// @param [in] dm A DM that provides edge data
/// @param [out] edges A pointer to an RDyEdges that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyEdgesCreateFromDM(DM dm, RDyEdges *edges) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscInt dim;
  PetscCall(DMGetCoordinateDim(dm, &dim));

  PetscInt c_start, c_end;
  PetscInt e_start, e_end;
  PetscInt v_start, v_end;
  DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
  DMPlexGetDepthStratum(dm, 1, &e_start, &e_end);
  DMPlexGetDepthStratum(dm, 0, &v_start, &v_end);

  // allocate edge storage
  PetscInt num_edges = e_end - e_start;
  PetscCall(RDyEdgesCreate(num_edges, edges));

  for (PetscInt e = e_start; e < e_end; e++) {
    PetscInt  iedge = e - e_start;
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
    edges->vertex_ids[index + 0] = p[2] - v_start;
    edges->vertex_ids[index + 1] = p[4] - v_start;
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, use_cone, &pSize, &p));

    // edge-to-cell
    PetscCall(DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
    PetscAssert(pSize == 2 || pSize == 3, comm, PETSC_ERR_ARG_SIZ, "Incorrect transitive closure size!");
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      PetscInt cell_id = p[i] - c_start;
      if (edges->cell_ids[2 * iedge] != -1) {  // we already have one cell
        edges->cell_ids[2 * iedge + 1] = cell_id;
      } else {  // no cells attached yet
        edges->cell_ids[2 * iedge] = cell_id;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
  }

  // fetch global edge IDs.
  ISLocalToGlobalMapping map;
  PetscCall(DMGetLocalToGlobalMapping(dm, &map));
  PetscCall(ISLocalToGlobalMappingApply(map, num_edges, edges->ids, edges->global_ids));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Destroys an RDyEdges struct, freeing its resources.
/// @param [inout] edges An RDyEdges struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode RDyEdgesDestroy(RDyEdges edges) {
  PetscFunctionBegin;

  PetscCall(PetscFree(edges.ids));
  PetscCall(PetscFree(edges.global_ids));
  PetscCall(PetscFree(edges.internal_edge_ids));
  PetscCall(PetscFree(edges.boundary_edge_ids));
  PetscCall(PetscFree(edges.is_local));
  PetscCall(PetscFree(edges.vertex_ids));
  PetscCall(PetscFree(edges.cell_ids));
  PetscCall(PetscFree(edges.is_internal));
  PetscCall(PetscFree(edges.is_owned));
  PetscCall(PetscFree(edges.normals));
  PetscCall(PetscFree(edges.centroids));
  PetscCall(PetscFree(edges.lengths));
  PetscCall(PetscFree(edges.cn));
  PetscCall(PetscFree(edges.sn));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// computes attributes about edges needed by RDycore.
static PetscErrorCode ComputeAdditionalEdgeAttributes(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  RDyCells    *cells    = &mesh->cells;
  RDyEdges    *edges    = &mesh->edges;
  RDyVertices *vertices = &mesh->vertices;

  PetscInt c_start, c_end;
  PetscInt e_start, e_end;
  PetscInt v_start, v_end;
  DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
  DMPlexGetDepthStratum(dm, 1, &e_start, &e_end);
  DMPlexGetDepthStratum(dm, 0, &v_start, &v_end);

  PetscSF     point_sf;
  PetscInt   *leaf_owner = NULL;
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(DMGetPointSF(dm, &point_sf));
  {
    PetscInt nroots, nleaves;
    PetscCall(PetscSFGetGraph(point_sf, &nroots, &nleaves, NULL, NULL));
    if (nroots >= 0) {  // graph has been set
      PetscInt *root_owner;
      PetscCall(PetscMalloc1(nroots, &root_owner));
      PetscCall(PetscMalloc1(nroots, &leaf_owner));
      for (PetscInt i = 0; i < nroots; i++) leaf_owner[i] = root_owner[i] = rank;
      PetscCall(PetscSFBcastBegin(point_sf, MPIU_INT, root_owner, leaf_owner, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(point_sf, MPIU_INT, root_owner, leaf_owner, MPI_REPLACE));
      PetscCall(PetscFree(root_owner));
    }
  }

  for (PetscInt e = e_start; e < e_end; e++) {
    PetscInt iedge = e - e_start;

    PetscInt l = edges->cell_ids[2 * iedge];
    PetscInt r = edges->cell_ids[2 * iedge + 1];

    PetscCheck(l >= 0, comm, PETSC_ERR_USER, "non-internal 'left' edge %" PetscInt_FMT " encountered (expected internal edge)", l);
    PetscBool is_internal_edge = (r >= 0);

    edges->is_owned[iedge] = !leaf_owner || leaf_owner[e] == rank;
    if (is_internal_edge) {
      mesh->num_internal_edges++;
      if (edges->is_owned[iedge]) mesh->num_owned_internal_edges++;
    } else {
      mesh->num_boundary_edges++;
    }

    /*
                 Case-1                      Case-2                       Update Case-2

                    v2                         v2                             v1
                   /|\                        /|\                             |
                    |                          |                              |
                    |---> normal               | ----> normal     normal <----|
                    |                          |                              |
             L -----|-----> R          R <-----|----- L               R <-----|----- L
                    |                          |                              |
                    |                          |                              |
                    |                          |                             \|/
                    v1                         v1                             v2

    In DMPlex, the cross product of the normal vector to the edge and vector joining the
    vertices of the edge (i.e. v1Tov2)  always points in the positive z-direction.
    However, the vector joining the left and the right cell may not be in the same direction
    as the normal vector to the edge (Case-2). Thus, the edge information in the Case-2 is
    updated by spawing the vertex ids and flipping the edge normal.
    */

    PetscInt v_offset = iedge * 2;
    PetscInt vid_1    = edges->vertex_ids[v_offset + 0];
    PetscInt vid_2    = edges->vertex_ids[v_offset + 1];

    RDyVector edge_parallel;  // a vector parallel along the edge in 2D
    for (PetscInt idim = 0; idim < 2; idim++) {
      edge_parallel.V[idim] = vertices->points[vid_2].X[idim] - vertices->points[vid_1].X[idim];
    }
    edge_parallel.V[2] = 0.0;

    // In case of an internal edge, a vector from the left cell to the right cell.
    // In case of a boundary edge, a vector from the left cell to edge centroid.
    // Note: This is a vector in 2D.
    RDyVector vec_L2RorEC;

    if (is_internal_edge) {
      for (PetscInt idim = 0; idim < 2; idim++) {
        vec_L2RorEC.V[idim] = cells->centroids[r].X[idim] - cells->centroids[l].X[idim];
      }

    } else {
      for (PetscInt idim = 0; idim < 2; idim++) {
        vec_L2RorEC.V[idim] = (vertices->points[vid_2].X[idim] + vertices->points[vid_1].X[idim]) / 2.0 - cells->centroids[l].X[idim];
      }
    }
    vec_L2RorEC.V[2] = 0.0;

    // Compute a vector perpendicular to the edge_parallel vector via a clockwise
    // 90 degree rotation
    RDyVector edge_perp;
    edge_perp.V[0] = edge_parallel.V[1];
    edge_perp.V[1] = -edge_parallel.V[0];

    // Compute the dot product to check if vector joining L-to-R is pointing
    // in the direction of the vector perpendicular to the edge.
    PetscReal dot_prod = vec_L2RorEC.V[0] * edge_perp.V[0] + vec_L2RorEC.V[1] * edge_perp.V[1];

    if (dot_prod < 0.0) {
      // The angle between edge_perp and vec_L2RorEC is greater than 90 deg.
      // Thus, flip vertex ids and the normal vector
      edges->vertex_ids[v_offset + 0] = vid_2;
      edges->vertex_ids[v_offset + 1] = vid_1;
      for (PetscInt idim = 0; idim < 3; idim++) {
        edges->normals[iedge].V[idim] *= -1.0;
      }
    }

    vid_1 = edges->vertex_ids[v_offset + 0];
    vid_2 = edges->vertex_ids[v_offset + 1];

    PetscReal x1 = vertices->points[vid_1].X[0];
    PetscReal y1 = vertices->points[vid_1].X[1];
    PetscReal x2 = vertices->points[vid_2].X[0];
    PetscReal y2 = vertices->points[vid_2].X[1];

    PetscReal dx = x2 - x1;
    PetscReal dy = y2 - y1;
    PetscReal ds = PetscSqrtReal(Square(dx) + Square(dy));

    edges->sn[iedge] = -dx / ds;
    edges->cn[iedge] = dy / ds;
  }
  PetscCall(PetscFree(leaf_owner));

  // allocate memory to save IDs of internal and boundary edges
  PetscCall(PetscCalloc1(mesh->num_internal_edges, &edges->internal_edge_ids));
  PetscCall(PetscCalloc1(mesh->num_boundary_edges, &edges->boundary_edge_ids));

  // now save the IDs
  mesh->num_internal_edges = 0;
  mesh->num_boundary_edges = 0;

  for (PetscInt e = e_start; e < e_end; e++) {
    PetscInt iedge = e - e_start;
    PetscInt l     = edges->cell_ids[2 * iedge];
    PetscInt r     = edges->cell_ids[2 * iedge + 1];

    if (r >= 0 && l >= 0) {
      edges->internal_edge_ids[mesh->num_internal_edges++] = iedge;
    } else {
      edges->boundary_edge_ids[mesh->num_boundary_edges++] = iedge;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// returns true if the vertices forming the triangle are in counter clockwise
// direction, false otherwise
// xyz0 - coordinates of the first vertex of the triangle
// xyz1 - coordinates of the second vertex of the triangle
// xyz2 - coordinates of the third vertex of the triangle
static PetscBool AreVerticesOrientedCounterClockwise(PetscReal xyz0[3], PetscReal xyz1[3], PetscReal xyz2[3]) {
  PetscFunctionBegin;

  PetscBool result = PETSC_TRUE;

  PetscReal x0, y0;
  PetscReal x1, y1;
  PetscReal x2, y2;

  x0 = xyz0[0];
  y0 = xyz0[1];
  x1 = xyz1[0];
  y1 = xyz1[1];
  x2 = xyz2[0];
  y2 = xyz2[1];

  PetscFunctionReturn((y1 - y0) * (x2 - x1) - (y2 - y1) * (x1 - x0) < 0);

  PetscFunctionReturn(result);
}

// computes slopes in the x and y directions for a triangle
// xyz0 - Coordinates of the first vertex of the triangle
// xyz1 - Coordinates of the second vertex of the triangle
// xyz2 - Coordinates of the third vertex of the triangle
// dz_dx - Slope in x-direction
// dz_dy - Slope in y-direction
static PetscErrorCode ComputeXYSlopesForTriangle(PetscReal xyz0[3], PetscReal xyz1[3], PetscReal xyz2[3], PetscReal *dz_dx, PetscReal *dz_dy) {
  PetscFunctionBegin;

  PetscReal x0, y0, z0;
  PetscReal x1, y1, z1;
  PetscReal x2, y2, z2;

  x0 = xyz0[0];
  y0 = xyz0[1];
  z0 = xyz0[2];

  if (AreVerticesOrientedCounterClockwise(xyz0, xyz1, xyz2)) {
    x1 = xyz1[0];
    y1 = xyz1[1];
    z1 = xyz1[2];
    x2 = xyz2[0];
    y2 = xyz2[1];
    z2 = xyz2[2];
  } else {
    x1 = xyz2[0];
    y1 = xyz2[1];
    z1 = xyz2[2];
    x2 = xyz1[0];
    y2 = xyz1[1];
    z2 = xyz1[2];
  }

  PetscReal num, den;
  num    = (y2 - y0) * (z1 - z0) - (y1 - y0) * (z2 - z0);
  den    = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
  *dz_dx = num / den;

  num    = (x2 - x0) * (z1 - z0) - (x1 - x0) * (z2 - z0);
  den    = (x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0);
  *dz_dy = num / den;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// computes geometric attributes about cells needed by RDycore
static PetscErrorCode ComputeAdditionalCellAttributes(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  RDyCells    *cells    = &mesh->cells;
  RDyVertices *vertices = &mesh->vertices;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    PetscInt nverts = cells->num_vertices[icell];

    PetscCheck((nverts == 3) || (nverts == 4), comm, PETSC_ERR_USER, "Cell has %" PetscInt_FMT " vertices (must be 3 or 4)", nverts);

    if (nverts == 3) {
      PetscInt offset = cells->vertex_offsets[icell];
      PetscInt v0     = cells->vertex_ids[offset + 0];
      PetscInt v1     = cells->vertex_ids[offset + 1];
      PetscInt v2     = cells->vertex_ids[offset + 2];

      PetscCall(ComputeXYSlopesForTriangle(vertices->points[v0].X, vertices->points[v1].X, vertices->points[v2].X, &cells->dz_dx[icell],
                                           &cells->dz_dy[icell]));

    } else {  // nverts == 4
      PetscInt offset = cells->vertex_offsets[icell];
      PetscInt v0     = cells->vertex_ids[offset + 0];
      PetscInt v1     = cells->vertex_ids[offset + 1];
      PetscInt v2     = cells->vertex_ids[offset + 2];
      PetscInt v3     = cells->vertex_ids[offset + 3];

      PetscInt vertexIDs[4][2];
      vertexIDs[0][0] = v0;
      vertexIDs[0][1] = v1;
      vertexIDs[1][0] = v1;
      vertexIDs[1][1] = v2;
      vertexIDs[2][0] = v2;
      vertexIDs[2][1] = v3;
      vertexIDs[3][0] = v3;
      vertexIDs[3][1] = v0;

      PetscReal dz_dx, dz_dy;
      cells->dz_dx[icell] = 0.0;
      cells->dz_dy[icell] = 0.0;

      // TODO: Revisit the approach to compute dz/dx and dz/y for quad cells.
      for (PetscInt ii = 0; ii < 4; ii++) {
        PetscInt a = vertexIDs[ii][0];
        PetscInt b = vertexIDs[ii][1];

        PetscCall(ComputeXYSlopesForTriangle(vertices->points[a].X, vertices->points[b].X, cells->centroids[icell].X, &dz_dx, &dz_dy));
        cells->dz_dx[icell] += 0.5 * dz_dx;
        cells->dz_dy[icell] += 0.5 * dz_dy;
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SaveNaturalCellIDs(DM dm, RDyCells *cells, PetscMPIInt rank) {
  PetscFunctionBegin;

  PetscBool useNatural;
  PetscCall(DMGetUseNatural(dm, &useNatural));

  if (useNatural) {
    PetscInt num_fields;

    // Create the natural vector
    Vec      natural;
    PetscInt natural_size = 0, natural_start;
    PetscCall(DMPlexCreateNaturalVector(dm, &natural));
    PetscCall(PetscObjectSetName((PetscObject)natural, "Natural Vec"));
    PetscCall(VecGetLocalSize(natural, &natural_size));
    PetscCall(VecGetBlockSize(natural, &num_fields));
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

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PETSc Vec (mesh->output.vertices_xyz_norder) with block size of 3 that saves 3D coordinate
/// values of vertices.
/// @param [in] dm A PETSc DM object
/// @param [inout] mesh A pointer to an RDyMesh that is updated
/// @return PETSC_SUCCESS on success
static PetscErrorCode CreateCoordinatesVectorInNaturalOrder(MPI_Comm comm, RDyMesh *mesh) {
  PetscFunctionBegin;

  Vec xcoord_nat, ycoord_nat, zcoord_nat;
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, mesh->num_vertices_global, &xcoord_nat));
  PetscCall(VecDuplicate(xcoord_nat, &ycoord_nat));
  PetscCall(VecDuplicate(xcoord_nat, &zcoord_nat));

  PetscInt  num_vertices = mesh->num_vertices;
  PetscInt  indices[num_vertices];
  PetscReal x[num_vertices], y[num_vertices], z[num_vertices];

  RDyVertices *vertices = &mesh->vertices;
  for (PetscInt v = 0; v < num_vertices; v++) {
    indices[v] = vertices->global_ids[v];
    PetscCheck(
        (indices[v] < mesh->num_vertices_global), comm, PETSC_ERR_USER,
        "The global vertex id (= %d) is greater than the total number of vertices (= %d). Remove vertices from the mesh that are not part of any"
        " grid cells.\n",
        indices[v], mesh->num_vertices_global);
    x[v] = vertices->points[v].X[0];
    y[v] = vertices->points[v].X[1];
    z[v] = vertices->points[v].X[2];
  }

  PetscCall(VecSetValues(xcoord_nat, num_vertices, indices, x, INSERT_VALUES));
  PetscCall(VecSetValues(ycoord_nat, num_vertices, indices, y, INSERT_VALUES));
  PetscCall(VecSetValues(zcoord_nat, num_vertices, indices, z, INSERT_VALUES));

  PetscCall(VecAssemblyBegin(xcoord_nat));
  PetscCall(VecAssemblyEnd(xcoord_nat));
  PetscCall(VecAssemblyBegin(ycoord_nat));
  PetscCall(VecAssemblyEnd(ycoord_nat));
  PetscCall(VecAssemblyBegin(zcoord_nat));
  PetscCall(VecAssemblyEnd(zcoord_nat));

  if (0) {
    VecView(xcoord_nat, PETSC_VIEWER_STDOUT_WORLD);
    VecView(ycoord_nat, PETSC_VIEWER_STDOUT_WORLD);
    VecView(zcoord_nat, PETSC_VIEWER_STDOUT_WORLD);
  }

  PetscInt local_size;
  PetscCall(VecGetLocalSize(xcoord_nat, &local_size));
  PetscInt ndim = 3;

  Vec *vertices_xyz_norder = &mesh->output.vertices_xyz_norder;
  PetscCall(VecCreate(comm, vertices_xyz_norder));
  PetscCall(VecSetSizes(*vertices_xyz_norder, local_size * ndim, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(*vertices_xyz_norder, ndim));
  PetscCall(VecSetFromOptions(*vertices_xyz_norder));

  PetscScalar *x_ptr, *y_ptr, *z_ptr, *xyz_ptr;

  PetscCall(VecGetArray(xcoord_nat, &x_ptr));
  PetscCall(VecGetArray(ycoord_nat, &y_ptr));
  PetscCall(VecGetArray(zcoord_nat, &z_ptr));
  PetscCall(VecGetArray(*vertices_xyz_norder, &xyz_ptr));

  for (PetscInt v = 0; v < local_size; v++) {
    xyz_ptr[v * ndim]     = x_ptr[v];
    xyz_ptr[v * ndim + 1] = y_ptr[v];
    xyz_ptr[v * ndim + 2] = z_ptr[v];
  }

  PetscCall(VecRestoreArray(xcoord_nat, &x_ptr));
  PetscCall(VecRestoreArray(ycoord_nat, &y_ptr));
  PetscCall(VecRestoreArray(zcoord_nat, &z_ptr));
  PetscCall(VecRestoreArray(*vertices_xyz_norder, &xyz_ptr));

  if (0) VecView(*vertices_xyz_norder, PETSC_VIEWER_STDOUT_WORLD);

  PetscCall((PetscObjectSetName((PetscObject)mesh->output.vertices_xyz_norder, "Vertices")));

  PetscCall(VecDestroy(&xcoord_nat));
  PetscCall(VecDestroy(&ycoord_nat));
  PetscCall(VecDestroy(&zcoord_nat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a 1D PETSc Vec (mesh->output.cell_conns_norder) that saves information about
/// cell connection for XDMF output.
/// @param [in] dm A PETSc DM object
/// @param [inout] mesh A pointer to an RDyMesh that is updated
/// @return PETSC_SUCCESS on success
static PetscErrorCode CreateCellConnectionVector(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  // create a local DM
  DM       local_dm;
  PetscInt max_num_vertices       = 4;
  PetscInt n_aux_field            = 1;
  PetscInt n_aux_field_dof[1]     = {max_num_vertices};
  char     aux_field_names[1][20] = {"Cell Connections"};

  PetscCall(CloneAndCreateCellCenteredDM(dm, n_aux_field, n_aux_field_dof, 20, &aux_field_names[0], &local_dm));

  Vec          global_vec, natural_vec;
  PetscScalar *vec_ptr;
  PetscCall(DMCreateGlobalVector(local_dm, &global_vec));
  PetscCall(DMPlexCreateNaturalVector(local_dm, &natural_vec));

  PetscCall(VecSet(global_vec, -1));
  PetscCall(VecGetArray(global_vec, &vec_ptr));

  RDyCells    *cells    = &mesh->cells;
  RDyVertices *vertices = &mesh->vertices;
  for (PetscInt c = 0; c < mesh->num_cells_local; c++) {
    PetscInt icell = cells->owned_to_local[c];
    for (PetscInt v = 0; v < cells->num_vertices[icell]; v++) {
      PetscInt offset    = cells->vertex_offsets[icell];
      PetscInt vertex_id = cells->vertex_ids[offset + v];
      PetscInt index     = c * max_num_vertices + v;
      vec_ptr[index]     = vertices->global_ids[vertex_id];
    }
  }

  PetscCall(VecRestoreArray(global_vec, &vec_ptr));

  PetscCall(DMPlexGlobalToNaturalBegin(local_dm, global_vec, natural_vec));
  PetscCall(DMPlexGlobalToNaturalEnd(local_dm, global_vec, natural_vec));

  if (0) {
    PetscCall(VecView(global_vec, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecView(natural_vec, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(VecDestroy(&global_vec));

  PetscInt size, count = 0;
  PetscCall(VecGetLocalSize(natural_vec, &size));

  PetscCall(VecGetArray(natural_vec, &vec_ptr));

  // Determine the number of vertices that are valid (i.e. vertex id > -1).
  PetscInt ncells = size / max_num_vertices;
  for (PetscInt i = 0; i < ncells; i++) {
    for (PetscInt j = 0; j < max_num_vertices; j++) {
      if (vec_ptr[i * max_num_vertices + j] > -1) count++;
    }
  }
  // Add the number of cells
  count += ncells;

  // The *cell_conns_norder vector is a long 1D distributed vector that will hold information about
  // cell vertices. For an i-th cell, the first entry will denoted a valid XMDF element ID
  // followed by the ID of vertices in the natural order that form the i-th cell.
  // The supported element types include:
  // - Triangles (TRI_ID_EXODUS)
  // - Quadrilaterals (QUAD_ID_EXODUS)

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  Vec *cell_conns_norder = &mesh->output.cell_conns_norder;
  PetscCall(VecCreate(comm, cell_conns_norder));
  PetscCall(VecSetSizes(*cell_conns_norder, count, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(*cell_conns_norder));

  PetscScalar *cell_conns_norder_ptr;
  PetscInt     idx = 0;

  VecGetArray(*cell_conns_norder, &cell_conns_norder_ptr);
  for (PetscInt i = 0; i < ncells; i++) {
    PetscInt nvertices = 0;
    for (PetscInt j = 0; j < max_num_vertices; j++) {
      if (vec_ptr[i * max_num_vertices + j] > -1) nvertices++;
    }
    switch (nvertices) {
      case 3:
        cell_conns_norder_ptr[idx++] = TRI_ID_EXODUS;
        break;
      case 4:
        cell_conns_norder_ptr[idx++] = QUAD_ID_EXODUS;
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unsupported cell type");
        break;
    }
    for (PetscInt j = 0; j < max_num_vertices; j++) {
      if (vec_ptr[i * max_num_vertices + j] > -1) {
        cell_conns_norder_ptr[idx++] = vec_ptr[i * max_num_vertices + j];
      }
    }
  }
  VecRestoreArray(*cell_conns_norder, &cell_conns_norder_ptr);
  if (0) {
    PetscCall(VecView(*cell_conns_norder, PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall((PetscObjectSetName((PetscObject)mesh->output.cell_conns_norder, "Cells")));

  PetscCall(VecRestoreArray(natural_vec, &vec_ptr));
  PetscCall(VecDestroy(&natural_vec));

  PetscCall(DMDestroy(&local_dm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates three PETSc Vec that saves x,y,z at cell centroid. These vectors are
/// use for output.
/// @param [in] dm A PETSc DM object
/// @param [inout] mesh A pointer to an RDyMesh that is updated
/// @return PETSC_SUCCESS on success
static PetscErrorCode CreateCellCentroidVectors(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  // create a local DM
  DM       local_dm;
  PetscInt n_aux_field            = 1;
  PetscInt n_aux_field_dof[1]     = {1};
  char     aux_field_names[1][20] = {"Cell Coordinates"};

  PetscCall(CloneAndCreateCellCenteredDM(dm, n_aux_field, n_aux_field_dof, 20, &aux_field_names[0], &local_dm));

  Vec global_vec, natural_vec;
  PetscCall(DMCreateGlobalVector(local_dm, &global_vec));
  PetscCall(DMPlexCreateNaturalVector(local_dm, &natural_vec));

  // create the Vec for storing coordinates in nautral order
  PetscCall(DMPlexCreateNaturalVector(local_dm, &mesh->output.xc));
  PetscCall(DMPlexCreateNaturalVector(local_dm, &mesh->output.yc));
  PetscCall(DMPlexCreateNaturalVector(local_dm, &mesh->output.zc));

  // set names to the Vecs
  PetscCall((PetscObjectSetName((PetscObject)mesh->output.xc, "XC")));
  PetscCall((PetscObjectSetName((PetscObject)mesh->output.yc, "YC")));
  PetscCall((PetscObjectSetName((PetscObject)mesh->output.zc, "ZC")));

  RDyCells *cells = &mesh->cells;

  for (PetscInt idim = 0; idim < 3; idim++) {
    PetscScalar *vec_ptr;
    PetscCall(VecGetArray(global_vec, &vec_ptr));

    // pack up the idim-th coordinates in global order
    for (PetscInt c = 0; c < mesh->num_cells_local; c++) {
      PetscInt icell = cells->owned_to_local[c];
      vec_ptr[c]     = cells->centroids[icell].X[idim];
    }
    PetscCall(VecGetArray(global_vec, &vec_ptr));

    // scatter the data from global to natural order
    PetscCall(DMPlexGlobalToNaturalBegin(local_dm, global_vec, natural_vec));
    PetscCall(DMPlexGlobalToNaturalEnd(local_dm, global_vec, natural_vec));

    // save the coordinate in appropriate Vec
    switch (idim) {
      case 0:
        PetscCall(VecCopy(natural_vec, mesh->output.xc));
        break;
      case 1:
        PetscCall(VecCopy(natural_vec, mesh->output.yc));
        break;
      case 2:
        PetscCall(VecCopy(natural_vec, mesh->output.zc));
        break;
    }
  }

  PetscCall(VecDestroy(&global_vec));
  PetscCall(VecDestroy(&natural_vec));
  PetscCall(DMDestroy(&local_dm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Determines the max number of edges/vertices per cell and max number of
/// cells/edges per vertices
/// @param [in] dm A PETSc DM object
/// @param [inout] mesh A pointer to an RDyMesh that is updated
/// @return PETSC_SUCCESS on success
static PetscErrorCode DetermineMaxAttributesForCellsAndVertices(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  mesh->max_nvertices_per_cell = 0;
  mesh->max_nedges_per_cell    = 0;
  mesh->max_ncells_per_vertex  = 0;
  mesh->max_nedges_per_vertex  = 0;

  PetscInt c_start, c_end;
  PetscInt e_start, e_end;
  PetscInt v_start, v_end;
  DMPlexGetHeightStratum(dm, 0, &c_start, &c_end);
  DMPlexGetDepthStratum(dm, 1, &e_start, &e_end);
  DMPlexGetDepthStratum(dm, 0, &v_start, &v_end);

  PetscInt  pSize;
  PetscInt *p        = NULL;
  PetscInt  use_cone = PETSC_TRUE;

  // loop over the cells to find max number of vertices/edges per cells
  for (PetscInt c = c_start; c < c_end; c++) {
    PetscInt nvertices_per_cell = 0, nedges_per_cell = 0;

    PetscCall(DMPlexGetTransitiveClosure(dm, c, use_cone, &pSize, &p));
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], e_start, e_end)) {
        nedges_per_cell++;
      } else {
        nvertices_per_cell++;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, use_cone, &pSize, &p));

    if (nvertices_per_cell > mesh->max_nvertices_per_cell) mesh->max_nvertices_per_cell = nvertices_per_cell;
    if (nedges_per_cell > mesh->max_nedges_per_cell) mesh->max_nedges_per_cell = nedges_per_cell;
  }

  // loop over the edges to find max number of cells/edges per edges
  for (PetscInt v = v_start; v < v_end; v++) {
    PetscInt nedges_per_vertex = 0, ncells_per_vertex = 0;

    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], e_start, e_end)) {
        nedges_per_vertex++;
      } else {
        ncells_per_vertex++;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, use_cone, &pSize, &p));

    if (nedges_per_vertex > mesh->max_nedges_per_vertex) mesh->max_nedges_per_vertex = nedges_per_vertex;
    if (ncells_per_vertex > mesh->max_ncells_per_vertex) mesh->max_ncells_per_vertex = ncells_per_vertex;
  }

  // find the max values across all MPI ranks
  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &mesh->max_nvertices_per_cell, 1, MPI_INT, MPI_MAX, comm));
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &mesh->max_nedges_per_cell, 1, MPI_INT, MPI_MAX, comm));
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &mesh->max_ncells_per_vertex, 1, MPI_INT, MPI_MAX, comm));
  PetscCall(MPI_Allreduce(MPI_IN_PLACE, &mesh->max_nedges_per_vertex, 1, MPI_INT, MPI_MAX, comm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates an RDyMesh from a PETSc DM.
/// @param [in] dm A PETSc DM
/// @param [out] mesh A pointer to an RDyMesh that stores allocated data.
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyMeshCreateFromDM(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  PetscCall(PetscMemzero(mesh, sizeof(RDyMesh)));

  // save the number of refinements
  PetscCall(DMGetRefineLevel(dm, &mesh->refine_level));

  // Determine the number of cells in the mesh
  PetscInt c_start, c_end;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &c_start, &c_end));
  mesh->num_cells = c_end - c_start;

  // Determine the number of edges in the mesh
  PetscInt e_start, e_end;
  PetscCall(DMPlexGetDepthStratum(dm, 1, &e_start, &e_end));
  mesh->num_edges = e_end - e_start;

  // Determine the number of vertices in the mesh
  PetscInt v_start, v_end;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &v_start, &v_end));
  mesh->num_vertices = v_end - v_start;

  // Determine few max attributes per cell and per vertex
  PetscCall(DetermineMaxAttributesForCellsAndVertices(dm, mesh));

  // Create mesh elements from the DM
  PetscCall(RDyCellsCreateFromDM(dm, mesh->max_nvertices_per_cell, mesh->max_nedges_per_cell, &mesh->cells));
  PetscCall(RDyEdgesCreateFromDM(dm, &mesh->edges));
  PetscCall(RDyVerticesCreateFromDM(dm, mesh->max_ncells_per_vertex, mesh->max_nedges_per_vertex, &mesh->vertices, &mesh->num_vertices_global));
  PetscCall(ComputeAdditionalEdgeAttributes(dm, mesh));
  PetscCall(ComputeAdditionalCellAttributes(dm, mesh));

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
  PetscMPIInt rank;
  MPI_Comm_rank(comm, &rank);
  PetscCall(SaveNaturalCellIDs(dm, &mesh->cells, rank));

  PetscCall(MPI_Allreduce(&mesh->num_cells_local, &mesh->num_cells_global, 1, MPI_INTEGER, MPI_SUM, comm));

  if (!mesh->refine_level) {
    PetscCall(CreateCoordinatesVectorInNaturalOrder(comm, mesh));
    PetscCall(CreateCellConnectionVector(dm, mesh));
    PetscCall(CreateCellCentroidVectors(dm, mesh));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
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

  if (!mesh.refine_level) {
    PetscCall(VecDestroy(&mesh.output.vertices_xyz_norder));
    PetscCall(VecDestroy(&mesh.output.cell_conns_norder));
    PetscCall(VecDestroy(&mesh.output.xc));
    PetscCall(VecDestroy(&mesh.output.yc));
    PetscCall(VecDestroy(&mesh.output.zc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
