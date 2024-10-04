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

typedef enum { CONSTANT = 0, HOMOGENEOUS, HETEROGENEOUS } DatasetType;

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

  PetscInt dataset_id_opened;

  // binary data
  Vec          data_vec;
  PetscScalar *data_ptr;

  PetscInt ndata;
  PetscInt header_offset;

  // temporal duration of the rainfall dataset
  PetscReal dtime_in_hour;
  PetscInt  ndata_file;

  // header of data
  PetscInt  ncols, nrows;  // number of columns and rows
  PetscReal xlc, ylc;      // cell centroid coordinates
  PetscReal cellsize;      // dx = dy of cell

  PetscInt  *data2mesh_idx;
  PetscReal *data_xc, *data_yc;
  PetscReal *mesh_xc, *mesh_yc;
  PetscInt   mesh_ncells_local;

} RasterDataset;

typedef struct {
  DatasetType        type;
  ConstantDataset    constant;
  HomogeneousDataset homogeneous;
  RasterDataset      heterogeneous;
} Rain;

// open a Vec that contains data in the following format:
//
// time_1 value_1
// time_2 value_2
// time_3 value_3
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

// For a given cur_time,
//   cur_data = value_1 if cur_time >= time_1 and cur_time < time_2
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

// loads the binary data in a Vec
static PetscErrorCode OpenHomogeneousDataset(HomogeneousDataset *data) {
  PetscFunctionBegin;

  PetscCall(OpenData(data->filename, &data->data_vec, &data->ndata));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));
  data->cur_idx                = -1;
  data->prev_idx               = -1;
  data->temporally_interpolate = PETSC_FALSE;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// close and destroys the Vec
static PetscErrorCode CloseHomogeneousDataset(HomogeneousDataset *data) {
  PetscFunctionBegin;

  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// set a constant rainfall for all grid cells
PetscErrorCode SetConstantRainfall(PetscReal rain_rate, PetscInt ncells, PetscReal *rain) {
  PetscFunctionBegin;
  for (PetscInt icell = 0; icell < ncells; icell++) {
    rain[icell] = rain_rate;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// compute the file name of the rainfall given the current data
static PetscErrorCode DetermineRasterDatasetFilename(RasterDataset *hetero_data) {
  PetscFunctionBegin;

  struct tm *current_date = &hetero_data->current_date;

  mktime(current_date);
  snprintf(hetero_data->file, PETSC_MAX_PATH_LEN - 1, "%s/%4d-%02d-%02d:%02d-%02d.%s.bin", hetero_data->dir, current_date->tm_year + 1900,
           current_date->tm_mon + 1, current_date->tm_mday, current_date->tm_hour, current_date->tm_min, PETSC_ID_TYPE);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// open the binary rainfall file that has following information:
// - ncols    : number of columns in the rainfall dataset
// - nrows    : number of rowns in the rainfall dataset
// - xlc      : x coordinate of the lower left corner [m]
// - ylc      : y coordinate of the lower left corner [m]
// - cellsize : size of grid cells in the rainfall dataset [m]
// - data     : rainfall rate for ncols * nrows cells [mm/hr]
static PetscErrorCode OpenRasterDataset(RasterDataset *data) {
  PetscFunctionBegin;

  PetscCall(DetermineRasterDatasetFilename(data));
  PetscPrintf(PETSC_COMM_WORLD, "Opening %s \n", data->file);

  data->dtime_in_hour = 1.0;  // assume an hourly dataset
  data->ndata_file    = 1;

  PetscCall(OpenData(data->file, &data->data_vec, &data->ndata));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  data->header_offset = 5;

  data->ncols    = (PetscInt)data->data_ptr[0];
  data->nrows    = (PetscInt)data->data_ptr[1];
  data->xlc      = data->data_ptr[2];
  data->ylc      = data->data_ptr[3];
  data->cellsize = data->data_ptr[4];

  if (0) {
    printf("ncols = %" PetscInt_FMT "\n", data->ncols);
    printf("nrows = %" PetscInt_FMT "\n", data->nrows);
    printf("xlc   = %f\n", data->xlc);
    printf("ylc   = %f\n", data->ylc);
    printf("size  = %f\n", data->cellsize);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// close the currently open rain dataset and open a new dataset file
static PetscErrorCode OpenANewRasterDataset(RasterDataset *data) {
  PetscFunctionBegin;

  // close the existing file
  PetscCall(VecRestoreArray(data->data_vec, &data->data_ptr));
  PetscCall(VecDestroy(&data->data_vec));

  // increase the date
  struct tm *current_date = &data->current_date;
  current_date->tm_hour++;
  mktime(current_date);

  // determine the new file
  PetscCall(DetermineRasterDatasetFilename(data));
  PetscPrintf(PETSC_COMM_WORLD, "Opening %s \n", data->file);

  PetscInt ndata;
  PetscCall(OpenData(data->file, &data->data_vec, &ndata));
  PetscCall(VecGetArray(data->data_vec, &data->data_ptr));

  // check to ensure the size and header information of the new rainfall data is the same as the older one
  PetscCheck(ndata == data->ndata, PETSC_COMM_WORLD, PETSC_ERR_USER, "The ndata of previous and new rainfal do not match");
  PetscCheck(data->ncols == (PetscInt)data->data_ptr[0], PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The number of columns in the previous and new rainfal do not match");
  PetscCheck(data->nrows == (PetscInt)data->data_ptr[1], PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The number of rows in the previous and new rainfal do not match");
  PetscCheck(data->xlc == data->data_ptr[2], PETSC_COMM_WORLD, PETSC_ERR_USER, "The xc of the previous and new rainfal do not match");
  PetscCheck(data->ylc == data->data_ptr[3], PETSC_COMM_WORLD, PETSC_ERR_USER, "The yc of the previous and new rainfal do not match");
  PetscCheck(data->cellsize == data->data_ptr[4], PETSC_COMM_WORLD, PETSC_ERR_USER,
             "The cellsize of the previous and new rainfal do not match");

  data->ndata_file++;

  PetscFunctionReturn(PETSC_SUCCESS);
}

// for each local RDycore grid cell, find the nearest neighbor cell in the rainfall dataset
static PetscErrorCode SetupRasterDatasetMapping(RDy rdy, RasterDataset *data) {
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
          min_dist                          = dist;
          data->data2mesh_idx[icell] = idx;
        }
        idx++;
      }
    }

    if (0) {
      PetscInt idx = data->data2mesh_idx[icell];
      printf("%04" PetscInt_FMT " %f %f %02" PetscInt_FMT " %f %f\n", icell, xc, yc, idx, data->data_xc[idx], data->data_yc[idx]);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// set spatially heterogeneous rainfall rate
PetscErrorCode SetRasterDataset(RasterDataset *data, PetscReal cur_time, PetscInt ncells, PetscReal *rain) {
  PetscFunctionBegin;

  // Is it time to open a new file?
  if (cur_time / 3600.0 >= (data->ndata_file) * data->dtime_in_hour) {
    OpenANewRasterDataset(data);
  }

  PetscInt  offset                = data->header_offset;
  PetscReal mm_per_hr_2_m_per_sec = 1.0 / (1000.0 * 3600.0);

  for (PetscInt icell = 0; icell < ncells; icell++) {
    PetscInt idx = data->data2mesh_idx[icell];
    rain[icell]  = data->data_ptr[idx + offset] * mm_per_hr_2_m_per_sec;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// set spatially homogeneous rainfall rate for all grid cells
PetscErrorCode SetHomogeneousRainfall(HomogeneousDataset *homogeneous_rain, PetscReal cur_time, PetscInt ncells, PetscReal *rain) {
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

// save information about rainfall dataset from command line options
PetscErrorCode ParseRainfallDataOptions(Rain *rain_dataset) {
  PetscFunctionBegin;

  PetscBool flag;

  // set default rainfall
  rain_dataset->type          = CONSTANT;
  rain_dataset->constant.rate = 0.0;

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-constant_rain_rate", &rain_dataset->constant.rate, NULL));

  PetscCall(
      PetscOptionsGetString(NULL, NULL, "-homogeneous_rain_file", rain_dataset->homogeneous.filename, sizeof(rain_dataset->homogeneous.filename), &flag));
  if (flag) {
    rain_dataset->type = HOMOGENEOUS;
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-temporally_interpolate_spatially_homogeneous_rain", &rain_dataset->homogeneous.temporally_interpolate, NULL));

  PetscBool sp_hetero_dir_flag;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-heterogeneous_rain_dir", rain_dataset->heterogeneous.dir, sizeof(rain_dataset->heterogeneous.dir),
                                  &sp_hetero_dir_flag));

#define NUM_HETEROGENEOUS_RAIN_DATE_VALUES 5  // number of parameters expected for heterogeneous rain date
  PetscInt date[NUM_HETEROGENEOUS_RAIN_DATE_VALUES];
  PetscInt ndate = NUM_HETEROGENEOUS_RAIN_DATE_VALUES;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-heterogeneous_rain_start_date", date, &ndate, &flag));
  if (flag) {
    PetscCheck(ndate == NUM_HETEROGENEOUS_RAIN_DATE_VALUES, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "Expect %d values when using -heterogeneous_rain_start_date YY,MO,DD,HH,MM", NUM_HETEROGENEOUS_RAIN_DATE_VALUES);
    PetscCheck(rain_dataset->type != HOMOGENEOUS, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "Can only specify homogeneous or heterogeneous rainfall datasets.");
    PetscCheck(sp_hetero_dir_flag == PETSC_TRUE, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "Need to specify path to spatially heterogeneous rainfall via -heterogeneous_rain_dir <dir>");

    rain_dataset->type = HETEROGENEOUS;

    rain_dataset->heterogeneous.start_date = (struct tm){
        .tm_year  = date[0] - 1900,
        .tm_mon   = date[1] - 1,
        .tm_mday  = date[2],
        .tm_hour  = date[3],
        .tm_min   = date[4],
        .tm_isdst = -1,
    };

    rain_dataset->heterogeneous.current_date = (struct tm){
        .tm_year  = date[0] - 1900,
        .tm_mon   = date[1] - 1,
        .tm_mday  = date[2],
        .tm_hour  = date[3],
        .tm_min   = date[4],
        .tm_isdst = -1,
    };

    rain_dataset->heterogeneous.ndata = 0;
  }
#undef NUM_HETEROGENEOUS_RAIN_DATE_VALUES

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
    PetscBool bc_specified;

    Rain      rain_dataset;
    char      bcfile[PETSC_MAX_PATH_LEN];
    PetscBool interpolate_bc = PETSC_FALSE;

    PetscCall(ParseRainfallDataOptions(&rain_dataset));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-homogeneous_bc_file", bcfile, sizeof(bcfile), &bc_specified));
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-temporally_interpolate_bc", &interpolate_bc, NULL));

    Vec          bc_vec = NULL;
    PetscScalar *bc_ptr = NULL;
    PetscInt     nbc;

    if (bc_specified) {
      PetscCall(OpenData(bcfile, &bc_vec, &nbc));
      PetscCall(VecGetArray(bc_vec, &bc_ptr));
    }

    PetscCall(RDySetup(rdy));

    switch (rain_dataset.type) {
      case CONSTANT:
        break;
      case HOMOGENEOUS:
        PetscCall(OpenHomogeneousDataset(&rain_dataset.homogeneous));
        break;
      case HETEROGENEOUS:
        PetscCall(OpenRasterDataset(&rain_dataset.heterogeneous));
        PetscCall(SetupRasterDatasetMapping(rdy, &rain_dataset.heterogeneous));
        break;
    }

    // allocate arrays for inspecting simulation data
    PetscInt n;
    PetscCall(RDyGetNumLocalCells(rdy, &n));
    PetscReal *h, *hu, *hv, *rain;
    PetscCalloc1(n, &h);
    PetscCalloc1(n, &hu);
    PetscCalloc1(n, &hv);
    PetscCalloc1(n, &rain);

    // get information about boundary conditions
    PetscInt nbcs, dirc_bc_idx = -1, num_edges_dirc_bc = 0;
    PetscCall(RDyGetNumBoundaryConditions(rdy, &nbcs));
    for (PetscInt ibc = 0; ibc < nbcs; ibc++) {
      PetscInt num_edges, bc_type;
      PetscCall(RDyGetNumBoundaryEdges(rdy, ibc, &num_edges));
      PetscCall(RDyGetBoundaryConditionFlowType(rdy, ibc, &bc_type));
      if (bc_type == CONDITION_DIRICHLET) {
        if (bc_specified) {
          PetscCheck(dirc_bc_idx == -1, comm, PETSC_ERR_USER,
                     "When BC file specified via -homogeneous_bc_file argument, only one CONDITION_DIRICHLET can be present in the yaml");
        }
        dirc_bc_idx       = ibc;
        num_edges_dirc_bc = num_edges;
      }
    }

    if (bc_specified) {
      PetscMPIInt global_dirc_bc_idx = -1;
      MPI_Allreduce(&dirc_bc_idx, &global_dirc_bc_idx, 1, MPI_INT, MPI_MAX, comm);
      PetscCheck(global_dirc_bc_idx > -1, comm, PETSC_ERR_USER,
                 "The BC file specified via -homogeneous_bc_file argument, but no CONDITION_DIRICHLET found in the yaml");
    }
    PetscReal *bc_values;
    PetscCalloc1(num_edges_dirc_bc * 3, &bc_values);

    // run the simulation to completion using the time parameters in the
    // config file
    RDyTimeUnit time_unit;
    PetscCall(RDyGetTimeUnit(rdy, &time_unit));
    PetscReal prev_time, coupling_interval;
    PetscCall(RDyGetTime(rdy, time_unit, &prev_time));
    PetscCall(RDyGetCouplingInterval(rdy, time_unit, &coupling_interval));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-coupling_interval", &coupling_interval, NULL));
    PetscCall(RDySetCouplingInterval(rdy, time_unit, coupling_interval));

    PetscInt cur_bc_idx = -1, prev_bc_idx = -1;

    while (!RDyFinished(rdy)) {  // returns true based on stopping criteria

      PetscReal time, time_step;
      PetscCall(RDyGetTime(rdy, time_unit, &time));

      switch (rain_dataset.type) {
        case CONSTANT:
          PetscCall(SetConstantRainfall(rain_dataset.constant.rate, n, rain));
          break;
        case HOMOGENEOUS:
          PetscCall(SetHomogeneousRainfall(&rain_dataset.homogeneous, time, n, rain));
          break;
        case HETEROGENEOUS:
          PetscCall(SetRasterDataset(&rain_dataset.heterogeneous, time, n, rain));
          break;
      }

      PetscCall(RDySetWaterSourceForLocalCells(rdy, n, rain));

      if (bc_specified && num_edges_dirc_bc > 0) {
        PetscReal cur_bc;
        PetscCall(GetCurrentData(bc_ptr, nbc, time, interpolate_bc, &cur_bc_idx, &cur_bc));
        if (interpolate_bc || cur_bc_idx != prev_bc_idx) {  // is it time to update the bc?
          prev_bc_idx = cur_bc_idx;
          for (PetscInt iedge = 0; iedge < num_edges_dirc_bc; iedge++) {
            bc_values[iedge * 3]     = cur_bc;
            bc_values[iedge * 3 + 1] = 0.0;
            bc_values[iedge * 3 + 2] = 0.0;
          }
          PetscCall(RDySetDirichletBoundaryValues(rdy, dirc_bc_idx, num_edges_dirc_bc, 3, bc_values));
        }
      }

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
    switch (rain_dataset.type) {
      case CONSTANT:
        break;
      case HOMOGENEOUS:
        PetscCall(CloseHomogeneousDataset(&rain_dataset.homogeneous));
        break;
      case HETEROGENEOUS:
        break;
    }

    if (bc_specified) {
      PetscCall(VecRestoreArray(bc_vec, &bc_ptr));
      PetscCall(VecDestroy(&bc_vec));
    }

    PetscFree(h);
    PetscFree(hu);
    PetscFree(hv);
    PetscFree(rain);
    PetscFree(bc_values);
    PetscCall(RDyDestroy(&rdy));
  }

  PetscCall(RDyFinalize());
  return 0;
}
