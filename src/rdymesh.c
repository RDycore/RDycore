#include <petscdmplex.h>
#include <private/rdymeshimpl.h>

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
  PetscCall(RDyFree(mesh.nG2A));
  PetscFunctionReturn(0);
}
