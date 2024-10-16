#include <petscsys.h>
#include <rdycore.h>
#include <time.h>

static const char *help_str =
    "rdycore - a standalone driver for RDycore\n"
    "usage: rdycore [options] <filename>\n";

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s <input.yaml>\n\n", exe_name);
}

typedef enum { UNSET = 0, CONSTANT, HOMOGENEOUS, RASTER, UNSTRUCTURED } DatasetType;

typedef struct {
  PetscReal rate;
} ConstantDataset;

typedef struct {
  char         filename[PETSC_MAX_PATH_LEN];
  Vec          data_vec;
  PetscInt     ndata;
  PetscScalar *data_ptr;
  PetscBool    temporally_interpolate;
  PetscInt     cur_idx, prev_idx;
} HomogeneousDataset;

typedef struct {
  char dir[PETSC_MAX_PATH_LEN];
  char file[PETSC_MAX_PATH_LEN];

  struct tm start_date, current_date;  // start and current date for rainfall dataset

  // binary data
  Vec          data_vec;
  PetscScalar *data_ptr;

  PetscInt header_offset;

  // temporal duration of the rainfall dataset
  PetscReal dtime_in_hour;
  PetscInt  ndata_file;

  // header of data
  PetscInt  ncols, nrows;  // number of columns and rows
  PetscReal xlc, ylc;      // cell centroid coordinates
  PetscReal cellsize;      // dx = dy of cell

  PetscInt   mesh_ncells_local;  // number of cells in RDycore meshes
  PetscInt  *data2mesh_idx;      // for each RDycore cell, the index of the data in the raster dataset
  PetscReal *data_xc, *data_yc;  // x and y coordinates of data in the raster dataset
  PetscReal *mesh_xc, *mesh_yc;  // x and y coordinates of RDycore cells

  PetscBool output_map;  // if true, writes the mapping between the RDycore cells and the dataset
} RasterDataset;

typedef struct {
  char dir[PETSC_MAX_PATH_LEN];
  char file[PETSC_MAX_PATH_LEN];
  char mesh_file[PETSC_MAX_PATH_LEN];

  PetscReal dtime_in_hour;
  PetscInt  ndata_file;

  struct tm start_date, current_date;  // start and current date for rainfall dataset

  // binary data
  Vec          data_vec;
  PetscScalar *data_ptr;

  PetscInt ndata;
  PetscInt stride;

  PetscInt   mesh_nelements;     // number of cells or boundary edges in RDycore mesh
  PetscInt  *data2mesh_idx;      // for each RDycore element (cells or boundary edges), the index of the data in the unstructured dataset
  PetscReal *data_xc, *data_yc;  // x and y coordinates of data
  PetscReal *mesh_xc, *mesh_yc;  // x and y coordinates of RDycore elments

  PetscBool output_map;  // if true, writes the mapping between the RDycore cells and the dataset

} UnstructuredDataset;

typedef struct {
  DatasetType         type;
  ConstantDataset     constant;      // spatio-temporally constant rainfall
  HomogeneousDataset  homogeneous;   // spatially-constnat, temporally-varying rainfall
  RasterDataset       raster;        // spatio-temporally varying rainfall in raster format
  UnstructuredDataset unstructured;  // spatio-temporally varying rainfall in unstructured grid format

  PetscInt   ndata;             // size of source-sink data for RDycore
  PetscReal *data_for_rdycore;  // values of source-sink for RDycore
} SourceSink;

typedef struct {
  DatasetType         type;
  HomogeneousDataset  homogeneous;   // spatially-homogeneous, temporally-varying BC
  UnstructuredDataset unstructured;  // spatio-temporally varying BC in unstructured grid format

  PetscInt   ndata;             // size of boundary condition data for RDycore
  PetscInt   dirichlet_bc_idx;  // ID of the dirichlet BC in RDycore
  PetscReal *data_for_rdycore;  // values of boundary condition for RDycore
} BoundaryCondition;

/// @brief Loads a PETSc Vec in binary format that contains the data in the
///        following format:
///
/// time_1 value_1
/// time_2 value_2
/// time_3 value_3
///
/// @param *filename [in]  Name of the PETSc Vec file
/// @param data_vec  [out] Poitner to PETSc Vec in which the file is loaded
/// @param ndata     [out] Number of temporal data values (= 3 in the above example)
/// @return PETSC_SUCESS on success
static PetscErrorCode OpenData(char *filename, Vec *data_vec, PetscInt *ndata) {
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

/// @brief Given the current time, finds the rainfall rate.
/// @param data_ptr               [in]  A pointer obtained from VecGetArray for the Vec that
///                                     has the spatially-homogenous, temporally variable rainfall dataset
/// @param ndata                  [in]  Number of temporal values in the rainfall dataset
/// @param cur_time               [in]  Current time
/// @param temporally_interpolate [in]  if TRUE, temporally average rainfall values between two data values
/// @param cur_data_idx           [out] Temporal index of the data selected
/// @param cur_data               [out] Rainfall value
/// @return PETSC_SUCESS on success
PetscErrorCode GetCurrentData(PetscScalar *data_ptr, PetscInt ndata, PetscReal cur_time, PetscBool temporally_interpolate, PetscInt *cur_data_idx,
                              PetscReal *cur_data) {
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

/// @brief Loads up a spatially homogeneous, temporally varying dataset, which is
///        a PETSc Vec in binary format.
/// @param *data [inout] Pointer to a HomogeneousDataset struct
/// @return PETSC_SUCESS on success
static PetscErrorCode OpenHomogeneousDataset(HomogeneousDataset *data) {
  PetscFunctionBegin;

  PetscCall(OpenData(data->filename, &data->data_vec, &data->ndata));

  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  // set initial settings
  data->cur_idx  = -1;
  data->prev_idx = -1;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Destroys the open PETSc Vec corresponding to a spatially homogeneous, temporally varying dataset
/// @param *data [inout] Pointer to the HomogeneousDataset
/// @return PETSC_SUCESS on success
static PetscErrorCode DestroyHomogeneousDataset(HomogeneousDataset *data) {
  PetscFunctionBegin;

  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sets a constant rainfall value for all grid cells
/// @param rain_rate [in]  Constant rainfall rate
/// @param ncells    [in]  Number of local cells
/// @param *rain     [out] Rainfall rate for local cells
/// @return PETSC_SUCESS on success
PetscErrorCode SetConstantRainfall(PetscReal rain_rate, PetscInt ncells, PetscReal *rain) {
  PetscFunctionBegin;
  for (PetscInt icell = 0; icell < ncells; icell++) {
    rain[icell] = rain_rate;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Given the current data, determins the file name
/// @param current_date [in]  A tm struct for current time
/// @param dir          [in]  Path to the directory containing the file
/// @param *file        [out] Filename (including the path to file)
/// @return PETSC_SUCESS on success
static PetscErrorCode DetermineDatasetFilename(struct tm *current_date, char *dir, char *file) {
  PetscFunctionBegin;

  mktime(current_date);
  snprintf(file, PETSC_MAX_PATH_LEN - 1, "%s/%4d-%02d-%02d:%02d-%02d.%s.bin", dir, current_date->tm_year + 1900, current_date->tm_mon + 1,
           current_date->tm_mday, current_date->tm_hour, current_date->tm_min, PETSC_ID_TYPE);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens a raster dataset that is a PETSc Vec in a binary format that contains the following
///        information:
///        - ncols    : number of columns in the rainfall dataset
///        - nrows    : number of rowns in the rainfall dataset
///        - xlc      : x coordinate of the lower left corner [m]
///        - ylc      : y coordinate of the lower left corner [m]
///        - cellsize : size of grid cells in the rainfall dataset [m]
///        - data     : data values for ncols * nrows cells
///
/// @param *data [inout] Pointer to a RasterDataset struct
/// @return PETSC_SUCESS on success
static PetscErrorCode OpenRasterDataset(RasterDataset *data) {
  PetscFunctionBegin;

  PetscCall(DetermineDatasetFilename(&data->current_date, data->dir, data->file));
  PetscPrintf(PETSC_COMM_WORLD, "Opening %s \n", data->file);

  data->dtime_in_hour = 1.0;  // assume an hourly dataset
  data->ndata_file    = 1;

  PetscInt tmp;
  PetscCall(OpenData(data->file, &data->data_vec, &tmp));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  data->header_offset = 5;

  data->ncols    = (PetscInt)data->data_ptr[0];
  data->nrows    = (PetscInt)data->data_ptr[1];
  data->xlc      = data->data_ptr[2];
  data->ylc      = data->data_ptr[3];
  data->cellsize = data->data_ptr[4];

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Closes the dataset currently open and frees up memory
/// @param data Pointer to a RasterDataset struct
/// @return PETSC_SUCESS on success
static PetscErrorCode DestroyRasterDataset(RasterDataset *data) {
  PetscFunctionBegin;

  // close the file
  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  // Free up memory
  PetscCall(PetscFree(data->mesh_xc));
  PetscCall(PetscFree(data->mesh_yc));
  PetscCall(PetscFree(data->data_xc));
  PetscCall(PetscFree(data->data_yc));
  PetscCall(PetscFree(data->data2mesh_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Opens an unstructured dataset that is a PETSc Vec in binary format that contains the following
///        information:
///
///        n                                      : Number of unstructured grid points
///        s                                      : Stride of the data
///        val_0_0, val_0_1, ... val_0_s-1        : s values for grid point with index = 0
///        val_1_0, val_1_1, ... val_1_s-1        : s values for grid point with index = 0
///        ...
///        ...
///        val_n-1_0, val_n-1_1, ... val_n-1_s-1  : s values for grid point with index = n-1
///
/// @param *data [inout] A UnstructuredDataset struct
/// @return PETSC_SUCESS on success
static PetscErrorCode OpenUnstructuredDataset(UnstructuredDataset *data) {
  PetscFunctionBegin;

  PetscCall(DetermineDatasetFilename(&data->current_date, data->dir, data->file));

  data->dtime_in_hour = 1.0;  // assume an hourly dataset
  data->ndata_file    = 1;

  // load the PETSc Vec
  PetscViewer viewer;
  PetscCall(VecCreate(PETSC_COMM_SELF, &data->data_vec));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, data->file, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(data->data_vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscInt size;
  PetscCall(VecGetSize(data->data_vec, &size));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  // set n and s
  data->ndata  = data->data_ptr[0];
  data->stride = data->data_ptr[1];

  // check the length of the PETSc Vec is consistent with n and s values
  PetscCheck((size - 2) / data->stride == data->ndata, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The length (=%" PetscInt_FMT ") of loaded Vec does is not consistent with first (N = %" PetscInt_FMT
             ") and second (stride = %" PetscInt_FMT ") in the Vec",
             size, data->ndata, data->stride);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Saves out the mapping of a dataset to RDycore as a binary Vec
/// @param filename [in] Name of the map file
/// @param ncells   [in] Number of RDycore's elements
/// @param d2m      [in] Index of data-to-mesh mapping
/// @param dxc      [in] x coordinate of the data
/// @param dyc      [in] y coordinate of the data
/// @param mxc      [in] x coordinate of RDycore's elements
/// @param myc      [in] y coordinate of RDycore's elements
/// @return PETSC_SUCESS on success
static PetscErrorCode WriteMappingForDebugging(char *filename, PetscInt ncells, PetscInt *d2m, PetscReal *dxc, PetscReal *dyc, PetscReal *mxc,
                                               PetscReal *myc) {
  PetscFunctionBegin;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  PetscInt stride = 6;
  Vec      vec;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ncells * stride, &vec));

  // fill the Vec
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

  // save the Vec
  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, filename, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(vec, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sets up the mapping between the unstructured grid dataset cells and the
///        local cells (for source-sink condition) or boundary edges (for boundary condition).
/// @param data Pointer to a UnstructuredDataset struct
/// @return PETSC_SUCESS on success
static PetscErrorCode CreateUnstructuredDatasetMapping(UnstructuredDataset *data) {
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

  PetscCalloc1(data->mesh_nelements, &data->data2mesh_idx);

  // for each boundary edge, find nearest dataset cell
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

/// @brief Closes the currently open Vec and loads in the next binary file in the Vec
/// @param data A pointer to UnstructuredDataset
/// @return PETSC_SUCESS on success
static PetscErrorCode OpenNextUnstructuredDataset(UnstructuredDataset *data) {
  PetscFunctionBegin;

  // close the existing file
  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  // increase the date
  struct tm *current_date = &data->current_date;
  current_date->tm_hour++;
  mktime(current_date);

  // determine the new file
  PetscCall(DetermineDatasetFilename(&data->current_date, data->dir, data->file));
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

/// @brief Closes the currently open Vec and free up memory used for mapping
/// @param data A pointer to UnstructuredDataset struct
/// @return PETSC_SUCESS on success
static PetscErrorCode DestroyUnstructuredDataset(UnstructuredDataset *data) {
  PetscFunctionBegin;

  // close the existing file
  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  PetscCall(PetscFree(data->data_xc));
  PetscCall(PetscFree(data->data_yc));
  PetscCall(PetscFree(data->mesh_xc));
  PetscCall(PetscFree(data->mesh_yc));
  PetscCall(PetscFree(data->data2mesh_idx));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Set spatially varying BC conditions from an unstructured dataset
/// @param data        [in]  Pointer to a UnstructuredDataset struct
/// @param cur_time    [in]  Current time
/// @param data_values [out] BC values for boundary cells
/// @return PETSC_SUCESS on success
PetscErrorCode SetUnstructuredData(UnstructuredDataset *data, PetscReal cur_time, PetscReal *data_values) {
  PetscFunctionBegin;

  // Is it time to open a new file?
  if (cur_time / 3600.0 >= (data->ndata_file) * data->dtime_in_hour) {
    OpenNextUnstructuredDataset(data);
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

/// @brief close the currently open raster dataset file and open a new file
/// @param data [inout] Pointer to a RasterDataset
/// @return PETSC_SUCESS on success
static PetscErrorCode OpenNextRasterDataset(RasterDataset *data) {
  PetscFunctionBegin;

  // close the existing file
  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  // increase the date
  struct tm *current_date = &data->current_date;
  current_date->tm_hour++;
  mktime(current_date);

  // determine the new file
  PetscCall(DetermineDatasetFilename(&data->current_date, data->dir, data->file));
  PetscPrintf(PETSC_COMM_WORLD, "Opening %s \n", data->file);

  PetscInt ndata;
  PetscCall(OpenData(data->file, &data->data_vec, &ndata));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  // check to ensure the size and header information of the new rainfall data is the same as the older one
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

/// @brief For each local RDycore grid cell, find the nearest neighbor cell in the raster dataset
/// @param rdy   [in]    A RDy struct
/// @param *data [inout] A pointer to a RasterDataset struct
/// @return PETSC_SUCESS on success
static PetscErrorCode CreateRasterDatasetMapping(RDy rdy, RasterDataset *data) {
  PetscFunctionBegin;

  PetscCall(RDyGetNumLocalCells(rdy, &data->mesh_ncells_local));
  PetscCalloc1(data->mesh_ncells_local, &data->mesh_xc);
  PetscCalloc1(data->mesh_ncells_local, &data->mesh_yc);
  PetscCalloc1(data->mesh_ncells_local, &data->data2mesh_idx);

  PetscCalloc1(data->ncols * data->nrows, &data->data_xc);
  PetscCalloc1(data->ncols * data->nrows, &data->data_yc);

  PetscInt idx = 0;
  for (PetscInt irow = 0; irow < data->nrows; irow++) {
    for (PetscInt icol = 0; icol < data->ncols; icol++) {
      data->data_xc[idx] = data->xlc + icol * data->cellsize + data->cellsize / 2.0;
      data->data_yc[idx] = data->ylc + (data->nrows - 1 - irow) * data->cellsize + data->cellsize / 2.0;
      idx++;
    }
  }

  PetscCall(RDyGetLocalCellXCentroids(rdy, data->mesh_ncells_local, data->mesh_xc));
  PetscCall(RDyGetLocalCellYCentroids(rdy, data->mesh_ncells_local, data->mesh_yc));

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

/// @brief Set rainfall rate for local cells from a spatially raster rainfall dataset. It is assumed that
///        rainfall rate is mm/hr.
/// @param data     [in]  A pointer to RasterDataset struct
/// @param cur_time [in]  Current time
/// @param ncells   [in]  Number of local cells
/// @param *rain    [out] Rainfall rate for local cells
/// @return PETSC_SUCESS on success
PetscErrorCode SetRasterData(RasterDataset *data, PetscReal cur_time, PetscInt ncells, PetscReal *rain) {
  PetscFunctionBegin;

  // Is it time to open a new file?
  if (cur_time / 3600.0 >= (data->ndata_file) * data->dtime_in_hour) {
    OpenNextRasterDataset(data);
  }

  PetscInt  offset                = data->header_offset;
  PetscReal mm_per_hr_2_m_per_sec = 1.0 / (1000.0 * 3600.0);

  for (PetscInt icell = 0; icell < ncells; icell++) {
    PetscInt idx = data->data2mesh_idx[icell];
    rain[icell]  = data->data_ptr[idx + offset] * mm_per_hr_2_m_per_sec;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sets rainfall value for local grid cells based on spatially homogenous, temporally-varying rainfall dataset
/// @param homogeneous_rain [in]  A pointer to HomogeneousDataset
/// @param cur_time         [in]  Current time
/// @param ncells           [in]  Number of local cells
/// @param *rain            [out] Rainfall rate for local cells
/// @return PETSC_SUCESS on success
PetscErrorCode SetHomogeneousData(HomogeneousDataset *homogeneous_rain, PetscReal cur_time, PetscInt ncells, PetscReal *rain) {
  PetscFunctionBegin;

  PetscReal    cur_rain;
  PetscScalar *rain_ptr               = homogeneous_rain->data_ptr;
  PetscInt     nrain                  = homogeneous_rain->ndata;
  PetscBool    temporally_interpolate = homogeneous_rain->temporally_interpolate;
  PetscInt    *cur_rain_idx           = &homogeneous_rain->cur_idx;
  PetscInt    *prev_rain_idx          = &homogeneous_rain->prev_idx;
  PetscCall(GetCurrentData(rain_ptr, nrain, cur_time, temporally_interpolate, cur_rain_idx, &cur_rain));

  if (temporally_interpolate || *cur_rain_idx != *prev_rain_idx) {  // is it time to update the source term?
    *prev_rain_idx = *cur_rain_idx;
    for (PetscInt icell = 0; icell < ncells; icell++) {
      rain[icell] = cur_rain;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Set saptially homogeneous water height boundary condition for all boundary cells
/// @param bc_data    [in]  Pointer to a HomogeneousDataset struct
/// @param cur_time   [in]  Current time
/// @param num_values [in]  Number of boundary cells
/// @param bc_values  [out] Values for boundary condition. Note: hu and hv are set to zero
/// @return PETSC_SUCESS on success
PetscErrorCode SetHomogeneousBoundary(HomogeneousDataset *bc_data, PetscReal cur_time, PetscInt num_values, PetscReal *bc_values) {
  PetscFunctionBegin;

  PetscReal    cur_bc;
  PetscScalar *bc_ptr                 = bc_data->data_ptr;
  PetscInt     ndata                  = bc_data->ndata;
  PetscBool    temporally_interpolate = bc_data->temporally_interpolate;
  PetscInt    *cur_bc_idx             = &bc_data->cur_idx;
  PetscInt    *prev_bc_idx            = &bc_data->prev_idx;

  PetscCall(GetCurrentData(bc_ptr, ndata, cur_time, temporally_interpolate, cur_bc_idx, &cur_bc));

  if (temporally_interpolate || *cur_bc_idx != *prev_bc_idx) {  // is it time to update the source term?
    *prev_bc_idx = *cur_bc_idx;
    for (PetscInt ii = 0; ii < num_values; ii++) {
      bc_values[ii * 3]     = cur_bc;
      bc_values[ii * 3 + 1] = 0.0;
      bc_values[ii * 3 + 2] = 0.0;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Save information about rainfall dataset from the command line options.
///        Supported rainfall datasets types include:
///        - contant rainfall
///        - spatially homogeneous, temporally varying rainfall
///        - spatially and temporing varying rainfall files in raster format
///
/// @param *rain_dataset [inout] Pointer to a SourceSink struct
/// @return PETSC_SUCESS on success
PetscErrorCode ParseRainfallDataOptions(SourceSink *rain_dataset) {
  PetscFunctionBegin;

  // set default settings
  rain_dataset->type                               = UNSET;
  rain_dataset->constant.rate                      = 0.0;
  rain_dataset->homogeneous.temporally_interpolate = PETSC_FALSE;
  rain_dataset->unstructured.ndata                 = 0;
  rain_dataset->unstructured.stride                = 0;
  rain_dataset->unstructured.mesh_nelements        = 0;
  rain_dataset->unstructured.output_map            = PETSC_FALSE;

  PetscCall(
      PetscOptionsGetBool(NULL, NULL, "-temporally_interpolate_spatially_homogeneous_rain", &rain_dataset->homogeneous.temporally_interpolate, NULL));

  PetscInt dataset_type_count = 0;  // number of number of datatypes specified via command line

  // spatiotemporally constant rainfall
  PetscBool constant_rain_flag;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-constant_rain_rate", &rain_dataset->constant.rate, &constant_rain_flag));
  if (constant_rain_flag) {
    dataset_type_count++;
    rain_dataset->type = CONSTANT;
  }

  // spatially-homogenous, temporally-varying rainfall dataset
  PetscBool homogeneous_rain_flag;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-homogeneous_rain_file", rain_dataset->homogeneous.filename,
                                  sizeof(rain_dataset->homogeneous.filename), &homogeneous_rain_flag));
  if (homogeneous_rain_flag) {
    dataset_type_count++;
    rain_dataset->type = HOMOGENEOUS;
  }

  // raster rainfall dataset
#define NUM_RASTER_RAIN_DATE_VALUES 5  // number of parameters expected for raster rain date
  PetscBool raster_start_date_flag;
  {
    PetscBool raster_dir_flag;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-raster_rain_dir", rain_dataset->raster.dir, sizeof(rain_dataset->raster.dir), &raster_dir_flag));

    PetscOptionsGetBool(NULL, NULL, "-raster_rain_output_map", &rain_dataset->raster.output_map, NULL);

    PetscInt date[NUM_RASTER_RAIN_DATE_VALUES];
    PetscInt ndate = NUM_RASTER_RAIN_DATE_VALUES;
    PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-raster_rain_start_date", date, &ndate, &raster_start_date_flag));
    if (raster_start_date_flag) {
      dataset_type_count++;

      PetscCheck(ndate == NUM_RASTER_RAIN_DATE_VALUES, PETSC_COMM_WORLD, PETSC_ERR_USER,
                 "Expect %d values when using -raster_rain_start_date YY,MO,DD,HH,MM", NUM_RASTER_RAIN_DATE_VALUES);
      PetscCheck(raster_dir_flag == PETSC_TRUE, PETSC_COMM_WORLD, PETSC_ERR_USER,
                 "Need to specify path to spatially raster rainfall via -raster_rain_dir <dir>");

      rain_dataset->type = RASTER;

      rain_dataset->raster.start_date = (struct tm){
          .tm_year  = date[0] - 1900,
          .tm_mon   = date[1] - 1,
          .tm_mday  = date[2],
          .tm_hour  = date[3],
          .tm_min   = date[4],
          .tm_isdst = -1,
      };

      rain_dataset->raster.current_date = (struct tm){
          .tm_year  = date[0] - 1900,
          .tm_mon   = date[1] - 1,
          .tm_mday  = date[2],
          .tm_hour  = date[3],
          .tm_min   = date[4],
          .tm_isdst = -1,
      };
    }
  }
#undef NUM_RASTER_RAIN_DATE_VALUES

  // unstructured rainfall dataset
#define NUM_UNSTRUCTURED_RAIN_DATE_VALUES 5  // number of parameters expected for unstructured rain date
  PetscBool unstructured_start_date_flag;
  {
    PetscBool unstructured_rain_dir_flag;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-unstructured_rain_dir", rain_dataset->unstructured.dir, sizeof(rain_dataset->unstructured.dir),
                                    &unstructured_rain_dir_flag));

    PetscOptionsGetBool(NULL, NULL, "-unstructured_rain_output_map", &rain_dataset->unstructured.output_map, NULL);

    PetscInt date[NUM_UNSTRUCTURED_RAIN_DATE_VALUES];
    PetscInt ndate = NUM_UNSTRUCTURED_RAIN_DATE_VALUES;
    PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-unstructured_rain_start_date", date, &ndate, &unstructured_start_date_flag));
    if (unstructured_start_date_flag) {
      dataset_type_count++;

      PetscCheck(ndate == NUM_UNSTRUCTURED_RAIN_DATE_VALUES, PETSC_COMM_WORLD, PETSC_ERR_USER,
                 "Expect %d values when using -unstructured_rain_start_date YY,MO,DD,HH,MM", NUM_UNSTRUCTURED_RAIN_DATE_VALUES);
      PetscCheck(unstructured_rain_dir_flag == PETSC_TRUE, PETSC_COMM_WORLD, PETSC_ERR_USER,
                 "Need to specify path to unstructured BC data via -unstructured_rain_dir <dir>");

      PetscBool flag;
      PetscCall(PetscOptionsGetString(NULL, NULL, "-unstructured_rain_mesh_file", rain_dataset->unstructured.mesh_file,
                                      sizeof(rain_dataset->unstructured.mesh_file), &flag));
      PetscCheck(flag, PETSC_COMM_WORLD, PETSC_ERR_USER, "Need to specify the mesh file -unstructured_rain_mesh_file <file>");

      rain_dataset->type = UNSTRUCTURED;

      rain_dataset->unstructured.start_date = (struct tm){
          .tm_year  = date[0] - 1900,
          .tm_mon   = date[1] - 1,
          .tm_mday  = date[2],
          .tm_hour  = date[3],
          .tm_min   = date[4],
          .tm_isdst = -1,
      };

      rain_dataset->unstructured.current_date = (struct tm){
          .tm_year  = date[0] - 1900,
          .tm_mon   = date[1] - 1,
          .tm_mday  = date[2],
          .tm_hour  = date[3],
          .tm_min   = date[4],
          .tm_isdst = -1,
      };
    }
  }
#undef NUM_UNSTRUCTURED_RAIN_DATE_VALUES

  PetscCheck(dataset_type_count <= 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "More than one rainfall cannot be specified. Rainfall types sepcified : Constat %d; Homogeneous %d; Raster %d; Unsturcutred %d",
             constant_rain_flag, homogeneous_rain_flag, raster_start_date_flag, unstructured_start_date_flag);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Save information about boundary condition dataset from the command line options.
///        Supported boundary condition dataset types include:
///        - spatially homogeneous, temporally varying boundary condition
///        - spatially and temporing varying boundary condition files in unstructured grid format
///
/// @param *bc [inout] Pointer to a BoundaryCondition struct
/// @return PETSC_SUCESS on success
PetscErrorCode ParseBoundaryDataOptions(BoundaryCondition *bc) {
  PetscFunctionBegin;

  // set default settings
  bc->type                               = UNSET;
  bc->ndata                              = 0;
  bc->dirichlet_bc_idx                   = -1;
  bc->homogeneous.temporally_interpolate = PETSC_FALSE;
  bc->unstructured.ndata                 = 0;
  bc->unstructured.stride                = 0;
  bc->unstructured.mesh_nelements        = 0;
  bc->unstructured.output_map            = PETSC_FALSE;

  PetscInt dataset_type_count = 0;  // number of number of datatypes specified via command line

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-temporally_interpolate_bc", &bc->homogeneous.temporally_interpolate, NULL));

  // spatially-homogeneous, temporally-varying boundary condition dataset
  PetscBool homogenous_bc_flag;
  PetscCall(
      PetscOptionsGetString(NULL, NULL, "-homogeneous_bc_file", bc->homogeneous.filename, sizeof(bc->homogeneous.filename), &homogenous_bc_flag));
  if (homogenous_bc_flag) {
    dataset_type_count++;
    bc->type = HOMOGENEOUS;
  }

  // unstructured boundary condition dataset
#define NUM_UNSTRUCTURED_BC_DATE_VALUES 5  // number of parameters expected for raster rain date
  PetscBool unstructured_bc_dir_flag;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-unstructured_bc_dir", bc->unstructured.dir, sizeof(bc->unstructured.dir), &unstructured_bc_dir_flag));

  PetscOptionsGetBool(NULL, NULL, "-unstructured_bc_output_map", &bc->unstructured.output_map, NULL);

  PetscInt  date[NUM_UNSTRUCTURED_BC_DATE_VALUES];
  PetscInt  ndate = NUM_UNSTRUCTURED_BC_DATE_VALUES;
  PetscBool unstructured_start_date_flag;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-unstructured_bc_start_date", date, &ndate, &unstructured_start_date_flag));
  if (unstructured_start_date_flag) {
    dataset_type_count++;

    PetscCheck(ndate == NUM_UNSTRUCTURED_BC_DATE_VALUES, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "Expect %d values when using -unstructured_bc_start_date YY,MO,DD,HH,MM", NUM_UNSTRUCTURED_BC_DATE_VALUES);
    PetscCheck(unstructured_bc_dir_flag == PETSC_TRUE, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "Need to specify path to unstructured BC data via -unstructured_bc_dir <dir>");

    PetscBool flag;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-unstructured_bc_mesh_file", bc->unstructured.mesh_file, sizeof(bc->unstructured.mesh_file), &flag));
    PetscCheck(flag, PETSC_COMM_WORLD, PETSC_ERR_USER, "Need to specify the mesh file -unstructured_bc_mesh_file <file>");

    bc->type = UNSTRUCTURED;

    bc->unstructured.start_date = (struct tm){
        .tm_year  = date[0] - 1900,
        .tm_mon   = date[1] - 1,
        .tm_mday  = date[2],
        .tm_hour  = date[3],
        .tm_min   = date[4],
        .tm_isdst = -1,
    };

    bc->unstructured.current_date = (struct tm){
        .tm_year  = date[0] - 1900,
        .tm_mon   = date[1] - 1,
        .tm_mday  = date[2],
        .tm_hour  = date[3],
        .tm_min   = date[4],
        .tm_isdst = -1,
    };
  }
#undef NUM_UNSTRUCTURED_BC_DATE_VALUES

  PetscCheck(dataset_type_count <= 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "More than one boundary condition cannot be specified. Rainfall types sepcified : Homogeneous %d; Raster %d ", homogenous_bc_flag,
             unstructured_start_date_flag);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sets up the rainfall source that is specified via command line
/// @param rdy           [in]  A RDy struct
/// @param n             [in]  Number of local cells on which rainfall source will be applied
/// @param *rain_dataset [out] Pointer to a Sourcesink struct
/// @return PETSC_SUCESS on success
PetscErrorCode CreateRainfallDataset(RDy rdy, PetscInt n, SourceSink *rain_dataset) {
  PetscFunctionBegin;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  static char debug_file[PETSC_MAX_PATH_LEN] = {0};

  // parse the command line arguments related to rainfall
  PetscCall(ParseRainfallDataOptions(rain_dataset));

  switch (rain_dataset->type) {
    case UNSET:
      break;
    case CONSTANT:
      rain_dataset->ndata = n;
      PetscCalloc1(n, &rain_dataset->data_for_rdycore);
      break;
    case HOMOGENEOUS:
      rain_dataset->ndata = n;
      PetscCalloc1(n, &rain_dataset->data_for_rdycore);
      PetscCall(OpenHomogeneousDataset(&rain_dataset->homogeneous));
      break;
    case RASTER:
      rain_dataset->ndata = n;
      PetscCalloc1(n, &rain_dataset->data_for_rdycore);
      PetscCall(OpenRasterDataset(&rain_dataset->raster));
      PetscCall(CreateRasterDatasetMapping(rdy, &rain_dataset->raster));

      if (rain_dataset->raster.output_map) {
        sprintf(debug_file, "map.source-sink.raster.rank_%d.bin", rank);
        PetscCall(WriteMappingForDebugging(debug_file, rain_dataset->raster.mesh_ncells_local, rain_dataset->raster.data2mesh_idx,
                                           rain_dataset->raster.data_xc, rain_dataset->raster.data_yc, rain_dataset->raster.mesh_xc,
                                           rain_dataset->raster.mesh_yc));
      }

      break;
    case UNSTRUCTURED:
      rain_dataset->ndata = n;

      PetscCalloc1(n, &rain_dataset->data_for_rdycore);
      PetscCall(OpenUnstructuredDataset(&rain_dataset->unstructured));

      PetscCheck((rain_dataset->unstructured.stride == 1), PETSC_COMM_WORLD, PETSC_ERR_USER, "The stride of rainfall dataset is not 1.");

      rain_dataset->unstructured.mesh_nelements = n;

      // allocate memory to save x/y coordinates of local cells
      PetscCalloc1(n, &rain_dataset->unstructured.mesh_xc);
      PetscCalloc1(n, &rain_dataset->unstructured.mesh_yc);

      // get x/y coordinates of local cells from RDycore
      PetscCall(RDyGetLocalCellXCentroids(rdy, n, rain_dataset->unstructured.mesh_xc));
      PetscCall(RDyGetLocalCellYCentroids(rdy, n, rain_dataset->unstructured.mesh_yc));

      // set up the mapping between the dataset and local cells
      PetscCall(CreateUnstructuredDatasetMapping(&rain_dataset->unstructured));

      if (rain_dataset->unstructured.output_map) {
        sprintf(debug_file, "map.source-sink.unstructured.rank_%d.bin", rank);
        PetscCall(WriteMappingForDebugging(debug_file, rain_dataset->unstructured.mesh_nelements, rain_dataset->unstructured.data2mesh_idx,
                                           rain_dataset->unstructured.data_xc, rain_dataset->unstructured.data_yc, rain_dataset->unstructured.mesh_xc,
                                           rain_dataset->unstructured.mesh_yc));
      }
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief First get the rainfall values from the rainfall dataset and then set those values in RDy
/// @param rdy           [inout] A RDy struct
/// @param time          [in]    Current time
/// @param *rain_dataset [inout] A pointer to a SourceSink dataset
/// @return PETSC_SUCESS on success
PetscErrorCode ApplyRainfallDataset(RDy rdy, PetscReal time, SourceSink *rain_dataset) {
  PetscFunctionBegin;

  switch (rain_dataset->type) {
    case UNSET:
      break;
    case CONSTANT:
      PetscCall(SetConstantRainfall(rain_dataset->constant.rate, rain_dataset->ndata, rain_dataset->data_for_rdycore));
      PetscCall(RDySetWaterSourceForLocalCells(rdy, rain_dataset->ndata, rain_dataset->data_for_rdycore));
      break;
    case HOMOGENEOUS:
      PetscCall(SetHomogeneousData(&rain_dataset->homogeneous, time, rain_dataset->ndata, rain_dataset->data_for_rdycore));
      PetscCall(RDySetWaterSourceForLocalCells(rdy, rain_dataset->ndata, rain_dataset->data_for_rdycore));
      break;
    case RASTER:
      PetscCall(SetRasterData(&rain_dataset->raster, time, rain_dataset->ndata, rain_dataset->data_for_rdycore));
      PetscCall(RDySetWaterSourceForLocalCells(rdy, rain_dataset->ndata, rain_dataset->data_for_rdycore));
      break;
    case UNSTRUCTURED:
      PetscCall(SetUnstructuredData(&rain_dataset->unstructured, time, rain_dataset->data_for_rdycore));
      PetscCall(RDySetWaterSourceForLocalCells(rdy, rain_dataset->ndata, rain_dataset->data_for_rdycore));
      break;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Close rainfall dataset and free up memory.
/// @param rain_dataset [inout] Pointer to a SourceSink struct
/// @return PETSC_SUCESS on success
PetscErrorCode DestroyRainfallDataset(SourceSink *rain_dataset) {
  PetscFunctionBegin;

  switch (rain_dataset->type) {
    case UNSET:
      break;
    case CONSTANT:
      PetscFree(rain_dataset->data_for_rdycore);
      break;
    case HOMOGENEOUS:
      PetscFree(rain_dataset->data_for_rdycore);
      PetscCall(DestroyHomogeneousDataset(&rain_dataset->homogeneous));
      break;
    case RASTER:
      PetscFree(rain_dataset->data_for_rdycore);
      PetscCall(DestroyRasterDataset(&rain_dataset->raster));
      break;
    case UNSTRUCTURED:
      PetscFree(rain_dataset->data_for_rdycore);
      PetscCall(DestroyUnstructuredDataset(&rain_dataset->unstructured));
      break;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Set up boundary condition based on command line options
/// @param rdy   [in]    A RDy struct
/// @param *data [inout] Pointer to a BoundaryCondition struct
/// @return PETSC_SUCESS on success
PetscErrorCode CreateBoundaryConditionDataset(RDy rdy, BoundaryCondition *bc_dataset) {
  PetscFunctionBegin;

  MPI_Comm comm = PETSC_COMM_WORLD;

  PetscMPIInt rank;
  MPI_Comm_rank(comm, &rank);
  static char debug_file[PETSC_MAX_PATH_LEN] = {0};

  PetscCall(ParseBoundaryDataOptions(bc_dataset));

  switch (bc_dataset->type) {
    case UNSET:
      break;
    case CONSTANT:
      break;
    case HOMOGENEOUS:
      PetscCall(OpenHomogeneousDataset(&bc_dataset->homogeneous));
      break;
    case RASTER:
      break;
    case UNSTRUCTURED:
      PetscCall(OpenUnstructuredDataset(&bc_dataset->unstructured));
      break;
  }

  if (bc_dataset->type != UNSET) {
    PetscInt nbcs, dirc_bc_idx = -1, num_edges_dirc_bc = 0;
    PetscCall(RDyGetNumBoundaryConditions(rdy, &nbcs));
    for (PetscInt ibc = 0; ibc < nbcs; ibc++) {
      PetscInt num_edges, bc_type;
      PetscCall(RDyGetNumBoundaryEdges(rdy, ibc, &num_edges));
      PetscCall(RDyGetBoundaryConditionFlowType(rdy, ibc, &bc_type));
      if (bc_type == CONDITION_DIRICHLET) {
        PetscCheck(dirc_bc_idx == -1, comm, PETSC_ERR_USER,
                   "When BC file specified via -homogeneous_bc_file argument, only one CONDITION_DIRICHLET can be present in the yaml");
        dirc_bc_idx       = ibc;
        num_edges_dirc_bc = num_edges;
      }
    }

    PetscMPIInt global_dirc_bc_idx = -1;
    MPI_Allreduce(&dirc_bc_idx, &global_dirc_bc_idx, 1, MPI_INT, MPI_MAX, comm);
    PetscCheck(global_dirc_bc_idx > -1, comm, PETSC_ERR_USER,
               "The BC file specified via -homogeneous_bc_file argument, but no CONDITION_DIRICHLET found in the yaml");

    bc_dataset->ndata            = num_edges_dirc_bc * 3;
    bc_dataset->dirichlet_bc_idx = global_dirc_bc_idx;
    PetscCalloc1(bc_dataset->ndata, &bc_dataset->data_for_rdycore);

    if ((bc_dataset->type == UNSTRUCTURED) & (num_edges_dirc_bc > 0)) {
      PetscCheck((bc_dataset->unstructured.stride == 3), PETSC_COMM_WORLD, PETSC_ERR_USER, "The stride of boundary condition dataset is not 3.");

      bc_dataset->unstructured.mesh_nelements = num_edges_dirc_bc;

      // allocate memory to save x/y coordinates of the boundary edges
      PetscCalloc1(bc_dataset->unstructured.mesh_nelements, &bc_dataset->unstructured.mesh_xc);
      PetscCalloc1(bc_dataset->unstructured.mesh_nelements, &bc_dataset->unstructured.mesh_yc);

      // get the x/y coordinates of the boundary edges from RDycore
      PetscCall(RDyGetBoundaryEdgeXCentroids(rdy, global_dirc_bc_idx, bc_dataset->unstructured.mesh_nelements, bc_dataset->unstructured.mesh_xc));
      PetscCall(RDyGetBoundaryEdgeYCentroids(rdy, global_dirc_bc_idx, bc_dataset->unstructured.mesh_nelements, bc_dataset->unstructured.mesh_yc));

      // set up the mapping between the dataset and boundary edges
      PetscCall(CreateUnstructuredDatasetMapping(&bc_dataset->unstructured));

      if (bc_dataset->unstructured.output_map) {
        sprintf(debug_file, "map.bc.unstructured.rank_%d.bin", rank);
        PetscCall(WriteMappingForDebugging(debug_file, bc_dataset->unstructured.mesh_nelements, bc_dataset->unstructured.data2mesh_idx,
                                           bc_dataset->unstructured.data_xc, bc_dataset->unstructured.data_yc, bc_dataset->unstructured.mesh_xc,
                                           bc_dataset->unstructured.mesh_yc));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief First get the boundary values from the boundary condition dataset and then set those values in RDy
/// @param rdy        [in]  A RDy struct
/// @param time       [in]  Current time
/// @param bc_dataset [out] Pointer to a BoundaryCondition
/// @return PETSC_SUCESS on success
PetscErrorCode ApplyBoundaryCondition(RDy rdy, PetscReal time, BoundaryCondition *bc_dataset) {
  PetscFunctionBegin;

  if (bc_dataset->ndata) {
    switch (bc_dataset->type) {
      case UNSET:
        break;
      case CONSTANT:
        break;
      case HOMOGENEOUS:
        PetscCall(SetHomogeneousBoundary(&bc_dataset->homogeneous, time, bc_dataset->ndata / 3, bc_dataset->data_for_rdycore));
        PetscCall(RDySetDirichletBoundaryValues(rdy, bc_dataset->dirichlet_bc_idx, bc_dataset->ndata / 3, 3, bc_dataset->data_for_rdycore));
        break;
      case RASTER:
        break;
      case UNSTRUCTURED:
        PetscCall(SetUnstructuredData(&bc_dataset->unstructured, time, bc_dataset->data_for_rdycore));
        PetscCall(RDySetDirichletBoundaryValues(rdy, bc_dataset->dirichlet_bc_idx, bc_dataset->ndata / 3, 3, bc_dataset->data_for_rdycore));
        break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Close the boundary condition dataset and free up memory
/// @param bc_dataset [inout] Pointer to a BoundaryCondition struct
/// @return PETSC_SUCESS on success
PetscErrorCode DestroyBoundaryConditionDataset(BoundaryCondition *bc_dataset) {
  PetscFunctionBegin;

  if (bc_dataset->ndata) {
    PetscCall(PetscFree(bc_dataset->data_for_rdycore));
  }

  switch (bc_dataset->type) {
    case UNSET:
      break;
    case CONSTANT:
      break;
    case HOMOGENEOUS:
      PetscCall(DestroyHomogeneousDataset(&bc_dataset->homogeneous));
      break;
    case RASTER:
      break;
    case UNSTRUCTURED:
      if (bc_dataset->ndata) {
        PetscCall(PetscFree(bc_dataset->unstructured.mesh_xc));
        PetscCall(PetscFree(bc_dataset->unstructured.mesh_yc));
      }
      PetscCall(DestroyUnstructuredDataset(&bc_dataset->unstructured));
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  // print usage info if no arguments given
  if (argc < 2) {
    usage(argv[0]);
    exit(-1);
  }

  // initialize subsystems
  PetscCall(RDyInit(argc, argv, help_str));

  if (strcmp(argv[1], "-help")) {  // if given a config file
    // create rdycore and set it up with the given file
    MPI_Comm comm = PETSC_COMM_WORLD;
    RDy      rdy;
    PetscCall(RDyCreate(comm, argv[1], &rdy));

    // Open datasets if specified via command line arguments.
    // Currently, these datasets need to be opened before calling
    // RDySetup to avoid an error. Possible issue for the error could
    // be that RDySetup is setting a default DM for all VecLoads.

    SourceSink        rain_dataset;
    BoundaryCondition bc_dataset;

    PetscCall(RDySetup(rdy));

    PetscInt n;
    PetscCall(RDyGetNumLocalCells(rdy, &n));

    // create rainfall and boundary condition datasets
    PetscCall(CreateRainfallDataset(rdy, n, &rain_dataset));
    PetscCall(CreateBoundaryConditionDataset(rdy, &bc_dataset));

    // allocate arrays for inspecting simulation data
    PetscReal *h, *hu, *hv;
    PetscCalloc1(n, &h);
    PetscCalloc1(n, &hu);
    PetscCalloc1(n, &hv);

    // run the simulation to completion using the time parameters in the
    // config file
    RDyTimeUnit time_unit;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscReal prev_time, coupling_interval;
    PetscCall(RDyGetTime(rdy, time_unit, &prev_time));
    PetscCall(RDyGetCouplingInterval(rdy, time_unit, &coupling_interval));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-coupling_interval", &coupling_interval, NULL));
    PetscCall(RDySetCouplingInterval(rdy, time_unit, coupling_interval));

    while (!RDyFinished(rdy)) {  // returns true based on stopping criteria

      PetscReal time, time_step;
      PetscCall(RDyGetTime(rdy, time_unit, &time));

      PetscCall(ApplyRainfallDataset(rdy, time, &rain_dataset));
      PetscCall(ApplyBoundaryCondition(rdy, time, &bc_dataset));

      // advance the solution by the coupling interval specified in the config file
      PetscCall(RDyAdvance(rdy));

      // the following just check that RDycore is doing the right thing

      PetscCall(RDyGetTime(rdy, time_unit, &time));
      PetscCall(RDyGetTimeStep(rdy, time_unit, &time_step));
      PetscCheck(time > prev_time, comm, PETSC_ERR_USER, "Non-increasing time!");
      PetscCheck(time_step > 0.0, comm, PETSC_ERR_USER, "Non-positive time step!");

      if (!RDyRestarted(rdy)) {
        PetscCheck(fabs(time - prev_time - coupling_interval) < 1e-12, comm, PETSC_ERR_USER, "RDyAdvance advanced time improperly (%g, %g, %g)!",
                   prev_time, time, fabs(time - prev_time + coupling_interval));
        prev_time += coupling_interval;
      } else {
        prev_time = time;
      }

      PetscInt step;
      PetscCall(RDyGetStep(rdy, &step));
      PetscCheck(step > 0, comm, PETSC_ERR_USER, "Non-positive step index!");

      PetscCall(RDyGetLocalCellHeights(rdy, n, h));
      PetscCall(RDyGetLocalCellXMomentums(rdy, n, hu));
      PetscCall(RDyGetLocalCellYMomentums(rdy, n, hv));
    }

    // clean up
    PetscCall(DestroyRainfallDataset(&rain_dataset));
    PetscCall(DestroyBoundaryConditionDataset(&bc_dataset));

    PetscCall(PetscFree(h));
    PetscCall(PetscFree(hu));
    PetscCall(PetscFree(hv));
    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
