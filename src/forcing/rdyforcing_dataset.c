#include <private/rdyforcingimpl.h>

//--- Dataset I/O functions for the forcing module

/// @brief Loads a PETSc Vec in binary format containing time-value pairs.
/// @param [in]  filename  Name of the PETSc Vec file
/// @param [out] data_vec  Pointer to PETSc Vec in which the file is loaded
/// @param [out] ndata     Number of temporal data values
PetscErrorCode RDyForcingOpenData(char *filename, Vec *data_vec, PetscInt *ndata) {
  PetscFunctionBegin;

  PetscViewer viewer;
  PetscCall(VecCreate(PETSC_COMM_SELF, data_vec));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, filename, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(*data_vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscInt size;
  PetscCall(VecGetSize(*data_vec, &size));
  *ndata = size / 2;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Finds the current data value at a given time (with optional temporal interpolation).
/// @param [in]     data_ptr               Pointer to interleaved time-value data
/// @param [in]     ndata                  Number of time-value pairs
/// @param [in]     cur_time               Current simulation time
/// @param [in]     temporally_interpolate Whether to interpolate between bracketing values
/// @param [in,out] cur_data_idx           Index of current time bracket (updated)
/// @param [out]    cur_data               Interpolated or stepped data value
PetscErrorCode RDyForcingGetCurrentData(PetscScalar *data_ptr, PetscInt ndata, PetscReal cur_time, PetscBool temporally_interpolate,
                                        PetscInt *cur_data_idx, PetscReal *cur_data) {
  PetscFunctionBegin;

  PetscBool found  = PETSC_FALSE;
  PetscInt  stride = 2;
  PetscReal time_up, time_dn;
  PetscReal data_up, data_dn;

  for (PetscInt itime = 0; itime < ndata - 1; itime++) {
    time_dn = data_ptr[itime * stride];
    data_dn = data_ptr[itime * stride + 1];

    time_up = data_ptr[itime * stride + 2];
    data_up = data_ptr[itime * stride + 3];

    if (cur_time >= time_dn && cur_time < time_up) {
      found         = PETSC_TRUE;
      *cur_data_idx = itime;
      break;
    }
  }

  if (!found) {
    *cur_data_idx = ndata - 1;
    *cur_data     = data_ptr[ndata * 2 - 1];
  } else {
    if (temporally_interpolate) {
      *cur_data = (cur_time - time_dn) / (time_up - time_dn) * (data_up - data_dn) + data_dn;
    } else {
      *cur_data = data_dn;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens a spatially-homogeneous, temporally-varying dataset.
/// @param [in,out] data  Homogeneous dataset structure
PetscErrorCode RDyForcingOpenHomogeneousDataset(RDyHomogeneousDataset *data) {
  PetscFunctionBegin;

  PetscCall(RDyForcingOpenData(data->filename, &data->data_vec, &data->ndata));

  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  data->cur_idx  = -1;
  data->prev_idx = -1;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Destroys a homogeneous dataset, releasing resources.
/// @param [in,out] data  Homogeneous dataset structure
PetscErrorCode RDyForcingDestroyHomogeneousDataset(RDyHomogeneousDataset *data) {
  PetscFunctionBegin;

  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens multiple homogeneous datasets.
/// @param [in,out] multi_data  Multi-homogeneous dataset structure
PetscErrorCode RDyForcingOpenMultiHomogeneousDataset(RDyMultiHomogeneousDataset *multi_data) {
  PetscFunctionBegin;

  for (PetscInt idata = 0; idata < multi_data->ndata; idata++) {
    PetscCall(RDyForcingOpenHomogeneousDataset(&multi_data->data[idata]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Generates a dated filename from a directory and date structure.
/// @param [in,out] current_date  Date structure (normalized via mktime)
/// @param [in]     dir           Directory path
/// @param [out]    file          Generated filename
PetscErrorCode RDyForcingDetermineDatasetFilename(struct tm *current_date, char *dir, char *file) {
  PetscFunctionBegin;

  mktime(current_date);
  snprintf(file, PETSC_MAX_PATH_LEN - 1, "%s/%4d-%02d-%02d:%02d-%02d.%s.bin", dir, current_date->tm_year + 1900, current_date->tm_mon + 1,
           current_date->tm_mday, current_date->tm_hour, current_date->tm_min, PETSC_ID_TYPE);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens a raster (gridded) dataset file.
/// @param [in,out] data  Raster dataset structure
PetscErrorCode RDyForcingOpenRasterDataset(RDyRasterDataset *data) {
  PetscFunctionBegin;

  PetscCall(RDyForcingDetermineDatasetFilename(&data->current_date, data->dir, data->file));
  PetscPrintf(PETSC_COMM_WORLD, "Opening %s \n", data->file);

  data->dtime_in_hour = 1.0;
  data->ndata_file    = 1;

  PetscInt tmp;
  PetscCall(RDyForcingOpenData(data->file, &data->data_vec, &tmp));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  data->header_offset = 5;

  data->ncols    = (PetscInt)data->data_ptr[0];
  data->nrows    = (PetscInt)data->data_ptr[1];
  data->xlc      = data->data_ptr[2];
  data->ylc      = data->data_ptr[3];
  data->cellsize = data->data_ptr[4];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Destroys a raster dataset, releasing resources.
/// @param [in,out] data  Raster dataset structure
PetscErrorCode RDyForcingDestroyRasterDataset(RDyRasterDataset *data) {
  PetscFunctionBegin;

  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  PetscCall(PetscFree(data->mesh_xc));
  PetscCall(PetscFree(data->mesh_yc));
  PetscCall(PetscFree(data->data_xc));
  PetscCall(PetscFree(data->data_yc));
  PetscCall(PetscFree(data->data2mesh_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens the next hourly raster dataset file.
/// @param [in,out] data  Raster dataset structure
PetscErrorCode RDyForcingOpenNextRasterDataset(RDyRasterDataset *data) {
  PetscFunctionBegin;

  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  struct tm *current_date = &data->current_date;
  current_date->tm_hour++;
  mktime(current_date);

  PetscCall(RDyForcingDetermineDatasetFilename(&data->current_date, data->dir, data->file));
  PetscPrintf(PETSC_COMM_WORLD, "Opening %s \n", data->file);

  PetscInt ndata;
  PetscCall(RDyForcingOpenData(data->file, &data->data_vec, &ndata));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  PetscCheck(data->ncols == (PetscInt)data->data_ptr[0], PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The number of columns in the previous and new rainfal do not match");
  PetscCheck(data->nrows == (PetscInt)data->data_ptr[1], PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The number of rows in the previous and new rainfal do not match");
  PetscCheck(data->xlc == data->data_ptr[2], PETSC_COMM_WORLD, PETSC_ERR_USER, "The xc of the previous and new rainfal do not match");
  PetscCheck(data->ylc == data->data_ptr[3], PETSC_COMM_WORLD, PETSC_ERR_USER, "The yc of the previous and new rainfal do not match");
  PetscCheck(data->cellsize == data->data_ptr[4], PETSC_COMM_WORLD, PETSC_ERR_USER, "The cellsize of the previous and new rainfal do not match");

  data->ndata_file++;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens an unstructured grid dataset file.
/// @param [in,out] data                 Unstructured dataset structure
/// @param [in]     expected_data_stride  Expected stride (number of values per element)
PetscErrorCode RDyForcingOpenUnstructuredDataset(RDyUnstructuredDataset *data, PetscInt expected_data_stride) {
  PetscFunctionBegin;

  PetscCall(RDyForcingDetermineDatasetFilename(&data->current_date, data->dir, data->file));

  data->dtime_in_hour = 1.0;
  data->ndata_file    = 1;

  PetscViewer viewer;
  PetscCall(VecCreate(PETSC_COMM_SELF, &data->data_vec));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, data->file, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(data->data_vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscInt size;
  PetscCall(VecGetSize(data->data_vec, &size));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  data->ndata  = data->data_ptr[0];
  data->stride = data->data_ptr[1];

  PetscCheck((size - 2) / data->stride == data->ndata, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The length (=%" PetscInt_FMT ") of loaded Vec does is not consistent with first (N = %" PetscInt_FMT
             ") and second (stride = %" PetscInt_FMT ") in the Vec",
             size, data->ndata, data->stride);

  PetscCheck(data->stride == expected_data_stride, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The data stride is %" PetscInt_FMT ", while the expected stride is %" PetscInt_FMT ".", data->stride, expected_data_stride);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Destroys an unstructured dataset, releasing resources.
/// @param [in,out] data  Unstructured dataset structure
PetscErrorCode RDyForcingDestroyUnstructuredDataset(RDyUnstructuredDataset *data) {
  PetscFunctionBegin;

  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  PetscCall(PetscFree(data->data_xc));
  PetscCall(PetscFree(data->data_yc));
  if (data->mesh_nelements) {
    PetscCall(PetscFree(data->mesh_xc));
    PetscCall(PetscFree(data->mesh_yc));
    PetscCall(PetscFree(data->data2mesh_idx));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens the next hourly unstructured dataset file.
/// @param [in,out] data  Unstructured dataset structure
PetscErrorCode RDyForcingOpenNextUnstructuredDataset(RDyUnstructuredDataset *data) {
  PetscFunctionBegin;

  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  struct tm *current_date = &data->current_date;
  current_date->tm_hour++;
  mktime(current_date);

  PetscCall(RDyForcingDetermineDatasetFilename(&data->current_date, data->dir, data->file));
  PetscPrintf(PETSC_COMM_WORLD, "Opening %s \n", data->file);

  PetscViewer viewer;
  PetscCall(VecCreate(PETSC_COMM_SELF, &data->data_vec));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, data->file, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(data->data_vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  data->ndata_file++;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sets constant rainfall values for all cells.
/// @param [in]  rain_rate  Rainfall rate
/// @param [in]  ncells     Number of cells
/// @param [out] rain       Output array filled with rain_rate
PetscErrorCode RDyForcingSetConstantRainfall(PetscReal rain_rate, PetscInt ncells, PetscReal *rain) {
  PetscFunctionBegin;
  for (PetscInt icell = 0; icell < ncells; icell++) {
    rain[icell] = rain_rate;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reads raster data and maps it to mesh cells.
/// @param [in,out] data      Raster dataset
/// @param [in]     cur_time  Current simulation time
/// @param [in]     ncells    Number of cells
/// @param [out]    rain      Mapped rainfall values (m/s)
PetscErrorCode RDyForcingSetRasterData(RDyRasterDataset *data, PetscReal cur_time, PetscInt ncells, PetscReal *rain) {
  PetscFunctionBegin;

  if (cur_time / 3600.0 >= (data->ndata_file) * data->dtime_in_hour) {
    RDyForcingOpenNextRasterDataset(data);
  }

  PetscInt  offset                = data->header_offset;
  PetscReal mm_per_hr_2_m_per_sec = 1.0 / (1000.0 * 3600.0);

  for (PetscInt icell = 0; icell < ncells; icell++) {
    PetscInt idx = data->data2mesh_idx[icell];
    rain[icell]  = data->data_ptr[idx + offset] * mm_per_hr_2_m_per_sec;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reads homogeneous temporal data and fills the output array.
/// @param [in,out] homogeneous_data  Homogeneous dataset
/// @param [in]     cur_time          Current simulation time
/// @param [in]     ncells            Number of cells
/// @param [out]    values            Output array filled with current value
PetscErrorCode RDyForcingSetHomogeneousData(RDyHomogeneousDataset *homogeneous_data, PetscReal cur_time, PetscInt ncells, PetscReal *values) {
  PetscFunctionBegin;

  PetscReal    cur_value;
  PetscScalar *data_ptr               = homogeneous_data->data_ptr;
  PetscInt     ndata                  = homogeneous_data->ndata;
  PetscBool    temporally_interpolate = homogeneous_data->temporally_interpolate;
  PetscInt    *cur_idx                = &homogeneous_data->cur_idx;
  PetscInt    *prev_idx               = &homogeneous_data->prev_idx;
  PetscCall(RDyForcingGetCurrentData(data_ptr, ndata, cur_time, temporally_interpolate, cur_idx, &cur_value));

  if (temporally_interpolate || *cur_idx != *prev_idx) {
    *prev_idx = *cur_idx;
    for (PetscInt icell = 0; icell < ncells; icell++) {
      values[icell] = cur_value;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reads unstructured data and maps to mesh elements.
/// @param [in,out] data         Unstructured dataset
/// @param [in]     cur_time     Current simulation time
/// @param [out]    data_values  Mapped output values
PetscErrorCode RDyForcingSetUnstructuredData(RDyUnstructuredDataset *data, PetscReal cur_time, PetscReal *data_values) {
  PetscFunctionBegin;

  if (cur_time / 3600.0 >= (data->ndata_file) * data->dtime_in_hour) {
    RDyForcingOpenNextUnstructuredDataset(data);
  }

  PetscInt offset = 2;
  PetscInt stride = data->stride;

  for (PetscInt icell = 0; icell < data->mesh_nelements; icell++) {
    PetscInt idx = data->data2mesh_idx[icell] * stride;

    for (PetscInt ii = 0; ii < stride; ii++) {
      data_values[icell * stride + ii] = data->data_ptr[idx + ii + offset];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Reads homogeneous boundary condition data and fills output with [h, hu, hv] triples.
/// @param [in,out] bc_data     Homogeneous dataset for BC
/// @param [in]     cur_time    Current simulation time
/// @param [in]     num_values  Number of boundary edges
/// @param [out]    bc_values   Output array of [h, 0, 0] triples
PetscErrorCode RDyForcingSetHomogeneousBoundary(RDyHomogeneousDataset *bc_data, PetscReal cur_time, PetscInt num_values, PetscReal *bc_values) {
  PetscFunctionBegin;

  PetscReal    cur_bc;
  PetscScalar *bc_ptr                 = bc_data->data_ptr;
  PetscInt     ndata                  = bc_data->ndata;
  PetscBool    temporally_interpolate = bc_data->temporally_interpolate;
  PetscInt    *cur_bc_idx             = &bc_data->cur_idx;
  PetscInt    *prev_bc_idx            = &bc_data->prev_idx;

  PetscCall(RDyForcingGetCurrentData(bc_ptr, ndata, cur_time, temporally_interpolate, cur_bc_idx, &cur_bc));

  if (temporally_interpolate || *cur_bc_idx != *prev_bc_idx) {
    *prev_bc_idx = *cur_bc_idx;
    for (PetscInt ii = 0; ii < num_values; ii++) {
      bc_values[ii * 3]     = cur_bc;
      bc_values[ii * 3 + 1] = 0.0;
      bc_values[ii * 3 + 2] = 0.0;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
