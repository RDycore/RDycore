#include <private/rdyforcingimpl.h>

//--- Spatial mapping functions for the forcing module

/// @brief Extracts cell centroids from the RDycore mesh.
/// @param [in]  rdy  RDy instance
/// @param [in]  n    Number of owned cells
/// @param [out] xc   X-coordinates of cell centroids (allocated)
/// @param [out] yc   Y-coordinates of cell centroids (allocated)
PetscErrorCode RDyForcingGetCellCentroidsFromMesh(RDy rdy, PetscInt n, PetscReal **xc, PetscReal **yc) {
  PetscFunctionBegin;

  PetscCalloc1(n, xc);
  PetscCalloc1(n, yc);

  PetscCall(RDyGetOwnedCellXCentroids(rdy, n, *xc));
  PetscCall(RDyGetOwnedCellYCentroids(rdy, n, *yc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Extracts boundary edge centroids from the RDycore mesh.
/// @param [in]  rdy    RDy instance
/// @param [in]  n      Number of boundary edges
/// @param [in]  bc_id  Boundary condition index
/// @param [out] xc     X-coordinates of boundary edge centroids (allocated)
/// @param [out] yc     Y-coordinates of boundary edge centroids (allocated)
PetscErrorCode RDyForcingGetBoundaryEdgeCentroidsFromMesh(RDy rdy, PetscInt n, PetscInt bc_id, PetscReal **xc, PetscReal **yc) {
  PetscFunctionBegin;

  PetscCalloc1(n, xc);
  PetscCalloc1(n, yc);

  PetscCall(RDyGetBoundaryEdgeXCentroids(rdy, bc_id, n, *xc));
  PetscCall(RDyGetBoundaryEdgeYCentroids(rdy, bc_id, n, *yc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reads coordinates from an unstructured dataset mesh file.
/// @param [in,out] data  Unstructured dataset (reads mesh_file, writes data_xc/data_yc/ndata)
PetscErrorCode RDyForcingReadUnstructuredDatasetCoordinates(RDyUnstructuredDataset *data) {
  PetscFunctionBegin;

  PetscViewer  viewer;
  Vec          vec;
  PetscScalar *vec_ptr;

  PetscCall(VecCreate(PETSC_COMM_SELF, &vec));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, data->mesh_file, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecGetArray(vec, &vec_ptr));

  data->ndata = vec_ptr[0];
  PetscCalloc1(data->ndata, &data->data_xc);
  PetscCalloc1(data->ndata, &data->data_yc);

  PetscInt stride = vec_ptr[1];
  PetscCheck(stride == 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "Stride (= %" PetscInt_FMT ") of unstructured dataset is unexpected.",
             (PetscInt)vec_ptr[1]);

  PetscInt offset = 2;
  for (PetscInt i = 0; i < data->ndata; i++) {
    data->data_xc[i] = vec_ptr[i * stride + offset];
    data->data_yc[i] = vec_ptr[i * stride + 1 + offset];
  }

  PetscCall(VecRestoreArray(vec, &vec_ptr));
  PetscCall(VecDestroy(&vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates nearest-neighbor mapping from unstructured dataset to mesh elements.
/// @param [in,out] data  Unstructured dataset (reads mesh_xc/mesh_yc/data_xc/data_yc, writes data2mesh_idx)
PetscErrorCode RDyForcingCreateUnstructuredDatasetMap(RDyUnstructuredDataset *data) {
  PetscFunctionBegin;

  PetscCalloc1(data->mesh_nelements, &data->data2mesh_idx);

  for (PetscInt icell = 0; icell < data->mesh_nelements; icell++) {
    PetscReal xc = data->mesh_xc[icell];
    PetscReal yc = data->mesh_yc[icell];

    PetscReal min_dist;
    for (PetscInt kk = 0; kk < data->ndata; kk++) {
      PetscReal dx   = xc - data->data_xc[kk];
      PetscReal dy   = yc - data->data_yc[kk];
      PetscReal dist = PetscPowReal(dx * dx + dy * dy, 0.5);

      if (kk == 0) {
        min_dist                   = dist;
        data->data2mesh_idx[icell] = kk;
      } else {
        if (dist < min_dist) {
          min_dist                   = dist;
          data->data2mesh_idx[icell] = kk;
        }
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates nearest-neighbor mapping from raster dataset to mesh cells.
/// @param [in]     rdy   RDy instance
/// @param [in,out] data  Raster dataset (reads mesh_xc/mesh_yc/data_xc/data_yc, writes data2mesh_idx)
PetscErrorCode RDyForcingCreateRasterDatasetMapping(RDy rdy, RDyRasterDataset *data) {
  PetscFunctionBegin;

  for (PetscInt icell = 0; icell < data->mesh_ncells_local; icell++) {
    PetscReal min_dist = (PetscMax(data->ncols, data->nrows) + 1) * data->cellsize;
    PetscReal xc       = data->mesh_xc[icell];
    PetscReal yc       = data->mesh_yc[icell];

    PetscInt idx = 0;
    for (PetscInt irow = 0; irow < data->nrows; irow++) {
      for (PetscInt icol = 0; icol < data->ncols; icol++) {
        PetscReal dx = xc - data->data_xc[idx];
        PetscReal dy = yc - data->data_yc[idx];

        PetscReal dist = PetscPowReal(dx * dx + dy * dy, 0.5);
        if (dist < min_dist) {
          min_dist                   = dist;
          data->data2mesh_idx[icell] = idx;
        }
        idx++;
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reads a pre-computed spatial map from a binary file.
/// @param [in]  rdy            RDy instance
/// @param [in]  filename       Path to binary file containing the map
/// @param [in]  ncells         Number of local cells
/// @param [out] data2mesh_idx  Allocated mapping array
PetscErrorCode RDyForcingReadSpatialMap(RDy rdy, const char filename[], PetscInt ncells, PetscInt **data2mesh_idx) {
  PetscFunctionBegin;

  PetscCalloc1(ncells, data2mesh_idx);

  Vec global;
  PetscCall(RDyReadOneDOFGlobalVecFromBinaryFile(rdy, filename, &global));

  PetscInt size;
  VecGetLocalSize(global, &size);
  PetscCheck(ncells == size, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The ncells (=%" PetscInt_FMT ") does not match the local size of global Vec (=%" PetscInt_FMT ")", ncells, size);

  PetscScalar *global_ptr;
  PetscCall(VecGetArray(global, &global_ptr));
  for (PetscInt ii = 0; ii < ncells; ii++) {
    (*data2mesh_idx)[ii] = global_ptr[ii];
  }
  PetscCall(VecRestoreArray(global, &global_ptr));

  PetscCall(VecDestroy(&global));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Writes the data-to-mesh mapping to a binary file using RDycore's global Vec I/O.
/// @param [in] rdy       RDy instance
/// @param [in] filename  Output file path
/// @param [in] ncells    Number of cells
/// @param [in] d2m       Mapping array
PetscErrorCode RDyForcingWriteMap(RDy rdy, char *filename, PetscInt ncells, PetscInt *d2m) {
  PetscFunctionBegin;
  Vec          global;
  PetscScalar *global_ptr;
  PetscCall(RDyCreateOneDOFGlobalVec(rdy, &global));

  PetscCall(VecGetArray(global, &global_ptr));
  for (PetscInt i = 0; i < ncells; i++) global_ptr[i] = d2m[i];
  PetscCall(VecRestoreArray(global, &global_ptr));

  PetscCall(RDyWriteOneDOFGlobalVecToBinaryFile(rdy, filename, &global));

  PetscCall(VecDestroy(&global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Writes a detailed mapping file for debugging (includes coordinates).
/// @param [in] filename  Output file path
/// @param [in] ncells    Number of cells
/// @param [in] d2m       Mapping array
/// @param [in] dxc       Dataset x-coordinates
/// @param [in] dyc       Dataset y-coordinates
/// @param [in] mxc       Mesh x-coordinates
/// @param [in] myc       Mesh y-coordinates
PetscErrorCode RDyForcingWriteMappingForDebugging(char *filename, PetscInt ncells, PetscInt *d2m, PetscReal *dxc, PetscReal *dyc, PetscReal *mxc,
                                                  PetscReal *myc) {
  PetscFunctionBegin;

  PetscInt stride = 6;
  Vec      vec;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ncells * stride, &vec));

  PetscScalar *vec_p;
  PetscCall(VecGetArray(vec, &vec_p));
  for (PetscInt i = 0; i < ncells; i++) {
    vec_p[i * stride + 0] = i;
    vec_p[i * stride + 1] = mxc[i];
    vec_p[i * stride + 2] = myc[i];
    vec_p[i * stride + 3] = d2m[i];
    vec_p[i * stride + 4] = dxc[d2m[i]];
    vec_p[i * stride + 5] = dyc[d2m[i]];
  }
  PetscCall(VecRestoreArray(vec, &vec_p));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, filename, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}
