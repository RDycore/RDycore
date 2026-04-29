#include <private/rdyforcingimpl.h>

//--- Public API and CLI parsing for the forcing module

//--------------------------------------------------------------------
// Internal helpers
//--------------------------------------------------------------------

/// @brief Searches the boundary conditions of an RDy instance for Dirichlet BCs.
///        Reports the local and global BC index, the number of local edges, and
///        whether more than one Dirichlet BC is present.
/// @param [in]  rdy                        RDy instance
/// @param [out] dirc_bc_idx                local index of the first Dirichlet BC found (-1 if none)
/// @param [out] num_edges_dirc_bc          number of local boundary edges on that BC
/// @param [out] global_dirc_bc_idx         global index of the Dirichlet BC across all MPI ranks (-1 if none)
/// @param [out] multiple_dirc_bcs_present  set to PETSC_TRUE if more than one Dirichlet BC exists
/// @return PetscErrorCode
static PetscErrorCode FindDirichletBCID(RDy rdy, PetscInt *dirc_bc_idx, PetscInt *num_edges_dirc_bc, PetscInt *global_dirc_bc_idx,
                                        PetscBool *multiple_dirc_bcs_present) {
  PetscFunctionBegin;

  MPI_Comm comm = PETSC_COMM_WORLD;

  *dirc_bc_idx               = -1;
  *global_dirc_bc_idx        = -1;
  *num_edges_dirc_bc         = 0;
  *multiple_dirc_bcs_present = PETSC_FALSE;

  PetscInt nbcs;
  PetscCall(RDyGetNumBoundaryConditions(rdy, &nbcs));

  for (PetscInt ibc = 0; ibc < nbcs; ibc++) {
    PetscInt num_edges, bc_type;
    PetscCall(RDyGetNumBoundaryEdges(rdy, ibc, &num_edges));
    PetscCall(RDyGetBoundaryConditionFlowType(rdy, ibc, &bc_type));

    if (bc_type == CONDITION_DIRICHLET) {
      if (*dirc_bc_idx != -1) *multiple_dirc_bcs_present = PETSC_TRUE;
      *dirc_bc_idx       = ibc;
      *num_edges_dirc_bc = num_edges;
    }
  }

  MPI_Allreduce(dirc_bc_idx, global_dirc_bc_idx, 1, MPIU_INT, MPI_MAX, comm);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Post-processes a spatially-homogeneous boundary condition dataset after parsing:
///        locates the single Dirichlet BC, allocates the value array, and stores the BC index.
/// @param [in]     rdy         RDy instance
/// @param [in,out] bc_dataset  boundary condition dataset to post-process
/// @return PetscErrorCode
static PetscErrorCode DoPostprocessForBoundaryHomogeneousDataset(RDy rdy, RDyForcingBoundaryCondition *bc_dataset) {
  PetscFunctionBegin;

  MPI_Comm  comm        = PETSC_COMM_WORLD;
  PetscInt  dirc_bc_idx = 0, num_edges_dirc_bc = 0;
  PetscInt  global_dirc_bc_idx = -1;
  PetscBool multiple_dirc_bcs_present;

  PetscCall(FindDirichletBCID(rdy, &dirc_bc_idx, &num_edges_dirc_bc, &global_dirc_bc_idx, &multiple_dirc_bcs_present));

  PetscCheck(multiple_dirc_bcs_present == PETSC_FALSE, comm, PETSC_ERR_USER,
             "When BC file specified via -homogeneous_bc_file argument, only one CONDITION_DIRICHLET can be present in the yaml");
  PetscCheck(global_dirc_bc_idx > -1, comm, PETSC_ERR_USER,
             "The BC file specified via -homogeneous_bc_file argument, but no CONDITION_DIRICHLET found in the yaml");

  bc_dataset->ndata            = num_edges_dirc_bc * 3;
  bc_dataset->dirichlet_bc_idx = global_dirc_bc_idx;
  PetscCalloc1(bc_dataset->ndata, &bc_dataset->data_for_rdycore);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Post-processes an unstructured source/sink dataset after opening:
///        retrieves mesh cell centroids, reads dataset coordinates, and builds
///        (or reads) the spatial mapping from dataset points to mesh cells.
/// @param [in]     rdy   RDy instance
/// @param [in,out] data  unstructured dataset to post-process
/// @return PetscErrorCode
static PetscErrorCode DoPostprocessForSourceUnstructuredDataset(RDy rdy, RDyUnstructuredDataset *data) {
  PetscFunctionBegin;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  static char debug_file[PETSC_MAX_PATH_LEN] = {0};

  PetscCall(RDyGetNumOwnedCells(rdy, &data->mesh_nelements));
  PetscCall(RDyForcingGetCellCentroidsFromMesh(rdy, data->mesh_nelements, &data->mesh_xc, &data->mesh_yc));

  PetscCall(RDyForcingReadUnstructuredDatasetCoordinates(data));

  if (data->read_map) {
    PetscCall(RDyForcingReadSpatialMap(rdy, data->map_file, data->mesh_nelements, &data->data2mesh_idx));
  } else {
    PetscCall(RDyForcingCreateUnstructuredDatasetMap(data));
  }

  if (data->write_map_for_debugging) {
    snprintf(debug_file, PETSC_MAX_PATH_LEN, "map.source-sink.unstructured.rank_%d.bin", rank);
    PetscCall(
        RDyForcingWriteMappingForDebugging(debug_file, data->mesh_nelements, data->data2mesh_idx, data->data_xc, data->data_yc, data->mesh_xc, data->mesh_yc));
  }

  if (data->write_map) {
    PetscCall(RDyForcingWriteMap(rdy, data->map_file, data->mesh_nelements, data->data2mesh_idx));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Post-processes an unstructured boundary condition dataset after opening:
///        locates the single Dirichlet BC, allocates the value array, retrieves boundary
///        edge centroids, reads dataset coordinates, and builds the spatial mapping.
/// @param [in]     rdy         RDy instance
/// @param [in,out] bc_dataset  boundary condition dataset to post-process
/// @return PetscErrorCode
static PetscErrorCode DoPostprocessForBoundaryUnstructuredDataset(RDy rdy, RDyForcingBoundaryCondition *bc_dataset) {
  PetscFunctionBegin;

  MPI_Comm  comm        = PETSC_COMM_WORLD;
  PetscInt  dirc_bc_idx = 0, num_edges_dirc_bc = 0;
  PetscInt  global_dirc_bc_idx = -1;
  PetscBool multiple_dirc_bcs_present;

  PetscCall(FindDirichletBCID(rdy, &dirc_bc_idx, &num_edges_dirc_bc, &global_dirc_bc_idx, &multiple_dirc_bcs_present));

  PetscCheck(multiple_dirc_bcs_present == PETSC_FALSE, comm, PETSC_ERR_USER,
             "When BC file specified via -unstructured_bc_dir argument, only one CONDITION_DIRICHLET can be present in the yaml");
  PetscCheck(global_dirc_bc_idx > -1, comm, PETSC_ERR_USER,
             "The BC file specified via -unstructured_bc_dir argument, but no CONDITION_DIRICHLET found in the yaml");

  bc_dataset->ndata            = num_edges_dirc_bc * 3;
  bc_dataset->dirichlet_bc_idx = global_dirc_bc_idx;
  PetscCalloc1(bc_dataset->ndata, &bc_dataset->data_for_rdycore);

  if ((num_edges_dirc_bc > 0)) {
    bc_dataset->unstructured.mesh_nelements = num_edges_dirc_bc;

    PetscCall(RDyForcingGetBoundaryEdgeCentroidsFromMesh(rdy, num_edges_dirc_bc, global_dirc_bc_idx, &bc_dataset->unstructured.mesh_xc,
                                                         &bc_dataset->unstructured.mesh_yc));

    PetscCall(RDyForcingReadUnstructuredDatasetCoordinates(&bc_dataset->unstructured));

    PetscCall(RDyForcingCreateUnstructuredDatasetMap(&bc_dataset->unstructured));

    if (bc_dataset->unstructured.write_map_for_debugging) {
      PetscMPIInt rank;
      MPI_Comm_rank(comm, &rank);
      static char debug_file[PETSC_MAX_PATH_LEN] = {0};
      snprintf(debug_file, PETSC_MAX_PATH_LEN, "map.bc.unstructured.rank_%d.bin", rank);

      PetscCall(RDyForcingWriteMappingForDebugging(debug_file, bc_dataset->unstructured.mesh_nelements, bc_dataset->unstructured.data2mesh_idx,
                                                   bc_dataset->unstructured.data_xc, bc_dataset->unstructured.data_yc,
                                                   bc_dataset->unstructured.mesh_xc, bc_dataset->unstructured.mesh_yc));
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Post-processes a multi-homogeneous boundary condition dataset after opening:
///        enumerates all Dirichlet BCs, matches each to a dataset by region ID,
///        and allocates per-BC value arrays.
/// @param [in]     rdy         RDy instance
/// @param [in,out] bc_dataset  boundary condition dataset to post-process
/// @return PetscErrorCode
static PetscErrorCode DoPostprocessForBoundaryMultiHomogeneousDataset(RDy rdy, RDyForcingBoundaryCondition *bc_dataset) {
  PetscFunctionBegin;

  MPI_Comm comm = PETSC_COMM_WORLD;

  RDyMultiHomogeneousDataset *multihomogeneous = &bc_dataset->multihomogeneous;
  multihomogeneous->ndirichlet_bcs             = 0;

  PetscInt nbcs;
  PetscCall(RDyGetNumBoundaryConditions(rdy, &nbcs));

  if (nbcs > 0) {
    PetscInt *boundary_id, *boundary_nedges, *boundary_type;

    PetscCalloc1(nbcs, &boundary_id);
    PetscCalloc1(nbcs, &boundary_nedges);
    PetscCalloc1(nbcs, &boundary_type);

    for (PetscInt ibc = 0; ibc < nbcs; ibc++) {
      PetscCall(RDyGetNumBoundaryEdges(rdy, ibc, &boundary_nedges[ibc]));
      PetscCall(RDyGetBoundaryConditionFlowType(rdy, ibc, &boundary_type[ibc]));
      PetscCall(RDyGetBoundaryID(rdy, ibc, &boundary_id[ibc]));
      if (boundary_type[ibc] == CONDITION_DIRICHLET) multihomogeneous->ndirichlet_bcs++;
    }

    if (multihomogeneous->ndirichlet_bcs > 0) {
      PetscCalloc1(multihomogeneous->ndirichlet_bcs, &multihomogeneous->dirichlet_bc_idx);
      PetscCalloc1(multihomogeneous->ndirichlet_bcs, &multihomogeneous->dirichlet_bc_to_data_idx);
      PetscCalloc1(multihomogeneous->ndirichlet_bcs, &multihomogeneous->ndata_for_rdycore);

      PetscCall(PetscMalloc1(multihomogeneous->ndirichlet_bcs, &multihomogeneous->data_for_rdycore));

      PetscInt count = 0;
      for (PetscInt i = 0; i < nbcs; i++) {
        if (boundary_type[i] == CONDITION_DIRICHLET) {
          multihomogeneous->dirichlet_bc_idx[count]         = -1;
          multihomogeneous->dirichlet_bc_to_data_idx[count] = -1;
          PetscBool found                                   = PETSC_FALSE;

          for (PetscInt j = 0; j < multihomogeneous->ndata; j++) {
            if (boundary_id[i] == multihomogeneous->region_ids[j]) {
              found = PETSC_TRUE;

              multihomogeneous->dirichlet_bc_idx[count]         = i;
              multihomogeneous->dirichlet_bc_to_data_idx[count] = j;
              multihomogeneous->ndata_for_rdycore[count]        = boundary_nedges[i] * 3;

              PetscCall(PetscMalloc1(multihomogeneous->ndata_for_rdycore[count], &multihomogeneous->data_for_rdycore[count]));
              break;
            }
          }

          PetscCheck(found, comm, PETSC_ERR_USER, "A dirichlet BC is not mapped onto the -homogeneous_bc_region_ids");
          count++;
        }
      }

      PetscFree(boundary_id);
      PetscFree(boundary_nedges);
      PetscFree(boundary_type);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Post-processes a raster source/sink dataset after opening:
///        computes data cell centroids from the raster header, retrieves mesh cell
///        centroids, and builds (or reads) the spatial mapping from raster cells to mesh cells.
/// @param [in]     rdy   RDy instance
/// @param [in,out] data  raster dataset to post-process
/// @return PetscErrorCode
static PetscErrorCode DoPostprocessForSourceRasterDataset(RDy rdy, RDyRasterDataset *data) {
  PetscFunctionBegin;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  static char debug_file[PETSC_MAX_PATH_LEN] = {0};

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

  PetscCall(RDyGetNumOwnedCells(rdy, &data->mesh_ncells_local));
  PetscCalloc1(data->mesh_ncells_local, &data->mesh_xc);
  PetscCalloc1(data->mesh_ncells_local, &data->mesh_yc);
  PetscCalloc1(data->mesh_ncells_local, &data->data2mesh_idx);

  PetscCall(RDyForcingGetCellCentroidsFromMesh(rdy, data->mesh_ncells_local, &data->mesh_xc, &data->mesh_yc));

  if (data->read_map) {
    PetscCall(RDyForcingReadSpatialMap(rdy, data->map_file, data->mesh_ncells_local, &data->data2mesh_idx));
  } else {
    PetscCall(RDyForcingCreateRasterDatasetMapping(rdy, data));
  }

  if (data->write_map_for_debugging) {
    snprintf(debug_file, PETSC_MAX_PATH_LEN, "map.source-sink.raster.rank_%d.bin", rank);
    PetscCall(RDyForcingWriteMappingForDebugging(debug_file, data->mesh_ncells_local, data->data2mesh_idx, data->data_xc, data->data_yc,
                                                 data->mesh_xc, data->mesh_yc));
  }

  if (data->write_map) {
    PetscCall(RDyForcingWriteMap(rdy, data->map_file, data->mesh_ncells_local, data->data2mesh_idx));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//--------------------------------------------------------------------
// CLI option parsing
//--------------------------------------------------------------------

/// @brief Parses PETSc CLI options to configure the source/sink (rainfall) dataset.
///        Exactly one dataset type may be active at a time; an error is raised if
///        more than one type is specified.
/// @param [out] rain_dataset  source/sink dataset to populate from CLI options
/// @return PetscErrorCode
PetscErrorCode RDyForcingParseRainfallDataOptions(RDyForcingSourceSink *rain_dataset) {
  PetscFunctionBegin;

  rain_dataset->type                                 = FORCING_DATASET_UNSET;
  rain_dataset->constant.rate                        = 0.0;
  rain_dataset->homogeneous.temporally_interpolate   = PETSC_FALSE;
  rain_dataset->unstructured.ndata                   = 0;
  rain_dataset->unstructured.stride                  = 0;
  rain_dataset->unstructured.mesh_nelements          = 0;
  rain_dataset->unstructured.write_map_for_debugging = PETSC_FALSE;
  rain_dataset->unstructured.write_map               = PETSC_FALSE;

  PetscCall(
      PetscOptionsGetBool(NULL, NULL, "-temporally_interpolate_spatially_homogeneous_rain", &rain_dataset->homogeneous.temporally_interpolate, NULL));

  PetscInt dataset_type_count = 0;

  // spatiotemporally constant rainfall
  PetscBool constant_rain_flag;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-constant_rain_rate", &rain_dataset->constant.rate, &constant_rain_flag));
  if (constant_rain_flag) {
    dataset_type_count++;
    rain_dataset->type = FORCING_DATASET_CONSTANT;
  }

  // spatially-homogeneous, temporally-varying rainfall dataset
  PetscBool homogeneous_rain_flag;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-homogeneous_rain_file", rain_dataset->homogeneous.filename,
                                  sizeof(rain_dataset->homogeneous.filename), &homogeneous_rain_flag));
  if (homogeneous_rain_flag) {
    dataset_type_count++;
    rain_dataset->type = FORCING_DATASET_HOMOGENEOUS;
  }

  // raster rainfall dataset
#define NUM_RASTER_RAIN_DATE_VALUES 5
  PetscBool raster_start_date_flag;
  {
    PetscBool raster_dir_flag;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-raster_rain_dir", rain_dataset->raster.dir, sizeof(rain_dataset->raster.dir), &raster_dir_flag));

    PetscOptionsGetBool(NULL, NULL, "-raster_rain_write_map_for_debugging", &rain_dataset->raster.write_map_for_debugging, NULL);
    PetscOptionsGetString(NULL, NULL, "-raster_rain_write_map_file", rain_dataset->raster.map_file, sizeof(rain_dataset->raster.map_file),
                          &rain_dataset->raster.write_map);
    PetscOptionsGetString(NULL, NULL, "-raster_rain_read_map_file", rain_dataset->raster.map_file, sizeof(rain_dataset->raster.map_file),
                          &rain_dataset->raster.read_map);

    PetscInt date[NUM_RASTER_RAIN_DATE_VALUES];
    PetscInt ndate = NUM_RASTER_RAIN_DATE_VALUES;
    PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-raster_rain_start_date", date, &ndate, &raster_start_date_flag));
    if (raster_start_date_flag) {
      dataset_type_count++;

      PetscCheck(ndate == NUM_RASTER_RAIN_DATE_VALUES, PETSC_COMM_WORLD, PETSC_ERR_USER,
                 "Expect %d values when using -raster_rain_start_date YY,MO,DD,HH,MM", NUM_RASTER_RAIN_DATE_VALUES);
      PetscCheck(raster_dir_flag == PETSC_TRUE, PETSC_COMM_WORLD, PETSC_ERR_USER,
                 "Need to specify path to spatially raster rainfall via -raster_rain_dir <dir>");

      rain_dataset->type = FORCING_DATASET_RASTER;

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
#define NUM_UNSTRUCTURED_RAIN_DATE_VALUES 5
  PetscBool unstructured_start_date_flag;
  {
    PetscBool unstructured_rain_dir_flag;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-unstructured_rain_dir", rain_dataset->unstructured.dir, sizeof(rain_dataset->unstructured.dir),
                                    &unstructured_rain_dir_flag));

    PetscOptionsGetBool(NULL, NULL, "-unstructured_rain_write_map_for_debugging", &rain_dataset->unstructured.write_map_for_debugging, NULL);
    PetscOptionsGetString(NULL, NULL, "-unstructured_rain_write_map_file", rain_dataset->unstructured.map_file,
                          sizeof(rain_dataset->unstructured.map_file), &rain_dataset->unstructured.write_map);
    PetscOptionsGetString(NULL, NULL, "-unstructured_rain_read_map_file", rain_dataset->unstructured.map_file,
                          sizeof(rain_dataset->unstructured.map_file), &rain_dataset->unstructured.read_map);

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

      rain_dataset->type = FORCING_DATASET_UNSTRUCTURED;

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

#define MAX_DATASETS 20

  char     *filenames[PETSC_MAX_PATH_LEN];
  PetscInt  nfiles = MAX_DATASETS;
  PetscBool multi_files_flag;

  PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-homogeneous_rain_files", filenames, &nfiles, &multi_files_flag));

  PetscInt  region_ids[MAX_DATASETS];
  PetscInt  nsources = MAX_DATASETS;
  PetscBool multi_rainfall_flag;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-homogeneous_rain_region_ids", region_ids, &nsources, &multi_rainfall_flag));

  PetscCheck(multi_files_flag == multi_rainfall_flag, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "Both -homogeneous_rain_files and -homogeneous_rain_region_ids need to specified");
  if (multi_files_flag) {
    PetscCheck(nfiles == nsources, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "The number of -homogeneous_rain_files and -homogeneous_rain_region_ids are not the same");
    rain_dataset->type = FORCING_DATASET_MULTI_HOMOGENEOUS;
    dataset_type_count++;

    rain_dataset->multihomogeneous.ndata = nfiles;

    PetscCall(PetscMalloc1(rain_dataset->multihomogeneous.ndata, &rain_dataset->multihomogeneous.data));
    PetscCalloc1(rain_dataset->multihomogeneous.ndata, &rain_dataset->multihomogeneous.region_ids);

    for (PetscInt ifile = 0; ifile < rain_dataset->multihomogeneous.ndata; ifile++) {
      PetscCall(PetscStrcpy(rain_dataset->multihomogeneous.data[ifile].filename, filenames[ifile]));
      rain_dataset->multihomogeneous.region_ids[ifile] = region_ids[ifile];
    }
  }
#undef MAX_DATASETS

  PetscCheck(dataset_type_count <= 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "More than one rainfall cannot be specified. Rainfall types sepcified : Constat %d; Homogeneous %d; Raster %d; Unsturcutred %d",
             constant_rain_flag, homogeneous_rain_flag, raster_start_date_flag, unstructured_start_date_flag);

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Parses PETSc CLI options to configure the boundary condition dataset.
///        Exactly one dataset type may be active at a time; an error is raised if
///        more than one type is specified.
/// @param [out] bc  boundary condition dataset to populate from CLI options
/// @return PetscErrorCode
PetscErrorCode RDyForcingParseBoundaryDataOptions(RDyForcingBoundaryCondition *bc) {
  PetscFunctionBegin;

  bc->type                                 = FORCING_DATASET_UNSET;
  bc->ndata                                = 0;
  bc->dirichlet_bc_idx                     = -1;
  bc->homogeneous.temporally_interpolate   = PETSC_FALSE;
  bc->unstructured.ndata                   = 0;
  bc->unstructured.stride                  = 0;
  bc->unstructured.mesh_nelements          = 0;
  bc->unstructured.write_map_for_debugging = PETSC_FALSE;

  PetscInt dataset_type_count = 0;

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-temporally_interpolate_bc", &bc->homogeneous.temporally_interpolate, NULL));

  // spatially-homogeneous, temporally-varying boundary condition dataset
  PetscBool homogenous_bc_flag;
  PetscCall(
      PetscOptionsGetString(NULL, NULL, "-homogeneous_bc_file", bc->homogeneous.filename, sizeof(bc->homogeneous.filename), &homogenous_bc_flag));
  if (homogenous_bc_flag) {
    dataset_type_count++;
    bc->type = FORCING_DATASET_HOMOGENEOUS;
  }

  // unstructured boundary condition dataset
#define NUM_UNSTRUCTURED_BC_DATE_VALUES 5
  PetscBool unstructured_bc_dir_flag;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-unstructured_bc_dir", bc->unstructured.dir, sizeof(bc->unstructured.dir), &unstructured_bc_dir_flag));

  PetscOptionsGetBool(NULL, NULL, "-unstructured_bc_write_map_for_debugging", &bc->unstructured.write_map_for_debugging, NULL);
  PetscOptionsGetString(NULL, NULL, "-unstructured_bc_write_map_file", bc->unstructured.map_file, sizeof(bc->unstructured.map_file),
                        &bc->unstructured.write_map);
  PetscOptionsGetString(NULL, NULL, "-unstructured_bc_read_map_file", bc->unstructured.map_file, sizeof(bc->unstructured.map_file),
                        &bc->unstructured.read_map);

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

    bc->type = FORCING_DATASET_UNSTRUCTURED;

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

#define MAX_DATASETS 20

  char     *filenames[PETSC_MAX_PATH_LEN];
  PetscInt  nfiles = MAX_DATASETS;
  PetscBool multi_files_flag;

  PetscCall(PetscOptionsGetStringArray(NULL, NULL, "-homogeneous_bc_files", filenames, &nfiles, &multi_files_flag));

  PetscInt  region_ids[MAX_DATASETS];
  PetscInt  nbcs = MAX_DATASETS;
  PetscBool multi_bc_flag;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-homogeneous_bc_region_ids", region_ids, &nbcs, &multi_bc_flag));

  PetscCheck(multi_files_flag == multi_bc_flag, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "Both -homogeneous_bc_files and -homogeneous_bc_region_ids need to specified");
  if (multi_files_flag) {
    PetscCheck(nfiles == nbcs, PETSC_COMM_WORLD, PETSC_ERR_USER,
               "The number of -homogeneous_bc_files and -homogeneous_bc_region_ids are not the same");
    bc->type = FORCING_DATASET_MULTI_HOMOGENEOUS;
    dataset_type_count++;

    bc->multihomogeneous.ndata = nfiles;

    PetscCall(PetscMalloc1(bc->multihomogeneous.ndata, &bc->multihomogeneous.data));
    PetscCalloc1(bc->multihomogeneous.ndata, &bc->multihomogeneous.region_ids);

    for (PetscInt ifile = 0; ifile < bc->multihomogeneous.ndata; ifile++) {
      PetscCall(PetscStrcpy(bc->multihomogeneous.data[ifile].filename, filenames[ifile]));
      bc->multihomogeneous.region_ids[ifile] = region_ids[ifile];
    }
  }
#undef MAX_DATASETS

  PetscCheck(dataset_type_count <= 1, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "More than one boundary condition cannot be specified. Rainfall types specified : Homogeneous %u; Raster %u; Multi-Homogeneous %u",
             homogenous_bc_flag, unstructured_start_date_flag, multi_files_flag);

  PetscFunctionReturn(PETSC_SUCCESS);
}

//--------------------------------------------------------------------
// Public API
//--------------------------------------------------------------------

/// @brief Creates a forcing object by parsing CLI options and opening datasets.
/// @param [in]  rdy      RDy instance
/// @param [out] forcing  Pointer to the created RDyForcing object
/// @return PetscErrorCode
PetscErrorCode RDyCreateForcing(RDy rdy, RDyForcing *forcing) {
  PetscFunctionBegin;

  struct _p_RDyForcing *f;
  PetscCall(PetscNew(&f));

  PetscInt n;
  PetscCall(RDyGetNumOwnedCells(rdy, &n));

  // Parse CLI options for source/sink (rainfall)
  PetscCall(RDyForcingParseRainfallDataOptions(&f->source));

  PetscInt expected_data_stride;
  switch (f->source.type) {
    case FORCING_DATASET_UNSET:
      break;
    case FORCING_DATASET_CONSTANT:
      f->source.ndata = n;
      PetscCalloc1(n, &f->source.data_for_rdycore);
      break;
    case FORCING_DATASET_HOMOGENEOUS:
      f->source.ndata = n;
      PetscCalloc1(n, &f->source.data_for_rdycore);
      PetscCall(RDyForcingOpenHomogeneousDataset(&f->source.homogeneous));
      break;
    case FORCING_DATASET_RASTER:
      f->source.ndata = n;
      PetscCalloc1(n, &f->source.data_for_rdycore);
      PetscCall(RDyForcingOpenRasterDataset(&f->source.raster));
      PetscCall(DoPostprocessForSourceRasterDataset(rdy, &f->source.raster));
      break;
    case FORCING_DATASET_UNSTRUCTURED:
      expected_data_stride = 1;
      f->source.ndata      = n;
      PetscCalloc1(n, &f->source.data_for_rdycore);
      PetscCall(RDyForcingOpenUnstructuredDataset(&f->source.unstructured, expected_data_stride));
      PetscCall(DoPostprocessForSourceUnstructuredDataset(rdy, &f->source.unstructured));
      break;
    case FORCING_DATASET_MULTI_HOMOGENEOUS:
      PetscCall(RDyForcingOpenMultiHomogeneousDataset(&f->source.multihomogeneous));
      break;
  }

  // Parse CLI options for boundary conditions
  PetscCall(RDyForcingParseBoundaryDataOptions(&f->boundary));

  switch (f->boundary.type) {
    case FORCING_DATASET_UNSET:
      break;
    case FORCING_DATASET_CONSTANT:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend RDyCreateForcing for boundary condition of type CONSTANT");
      break;
    case FORCING_DATASET_HOMOGENEOUS:
      PetscCall(RDyForcingOpenHomogeneousDataset(&f->boundary.homogeneous));
      PetscCall(DoPostprocessForBoundaryHomogeneousDataset(rdy, &f->boundary));
      break;
    case FORCING_DATASET_RASTER:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend RDyCreateForcing for boundary condition of type RASTER");
      break;
    case FORCING_DATASET_UNSTRUCTURED:
      expected_data_stride = 3;
      PetscCall(RDyForcingOpenUnstructuredDataset(&f->boundary.unstructured, expected_data_stride));
      PetscCall(DoPostprocessForBoundaryUnstructuredDataset(rdy, &f->boundary));
      break;
    case FORCING_DATASET_MULTI_HOMOGENEOUS:
      PetscCall(RDyForcingOpenMultiHomogeneousDataset(&f->boundary.multihomogeneous));
      PetscCall(DoPostprocessForBoundaryMultiHomogeneousDataset(rdy, &f->boundary));
      break;
  }

  *forcing = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Sets up forcing (currently a no-op; reserved for future use).
/// @param [in] forcing  RDyForcing object
/// @return PetscErrorCode
PetscErrorCode RDySetupForcing(RDyForcing forcing) {
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Applies forcing data at the given time: reads datasets and dispatches to RDycore source/BC setters.
/// @param [in] forcing  RDyForcing object
/// @param [in] time     Current simulation time
/// @return PetscErrorCode
PetscErrorCode RDyApplyForcing(RDy rdy, RDyForcing forcing, PetscReal time) {
  PetscFunctionBegin;

  // Apply source/sink (rainfall)
  switch (forcing->source.type) {
    case FORCING_DATASET_UNSET:
      break;
    case FORCING_DATASET_CONSTANT:
      if (forcing->source.ndata) {
        PetscCall(RDyForcingSetConstantRainfall(forcing->source.constant.rate, forcing->source.ndata, forcing->source.data_for_rdycore));
        PetscCall(RDySetRegionalWaterSource(rdy, 1, forcing->source.ndata, forcing->source.data_for_rdycore));
      }
      break;
    case FORCING_DATASET_HOMOGENEOUS:
      if (forcing->source.ndata) {
        PetscCall(RDyForcingSetHomogeneousData(&forcing->source.homogeneous, time, forcing->source.ndata, forcing->source.data_for_rdycore));
        PetscCall(RDySetRegionalWaterSource(rdy, 1, forcing->source.ndata, forcing->source.data_for_rdycore));
      }
      break;
    case FORCING_DATASET_RASTER:
      if (forcing->source.ndata) {
        PetscCall(RDyForcingSetRasterData(&forcing->source.raster, time, forcing->source.ndata, forcing->source.data_for_rdycore));
        PetscCall(RDySetRegionalWaterSource(rdy, 1, forcing->source.ndata, forcing->source.data_for_rdycore));
      }
      break;
    case FORCING_DATASET_UNSTRUCTURED:
      if (forcing->source.ndata) {
        PetscCall(RDyForcingSetUnstructuredData(&forcing->source.unstructured, time, forcing->source.data_for_rdycore));
        PetscCall(RDySetRegionalWaterSource(rdy, 1, forcing->source.ndata, forcing->source.data_for_rdycore));
      }
      break;
    case FORCING_DATASET_MULTI_HOMOGENEOUS:
      for (PetscInt idata = 0; idata < forcing->source.multihomogeneous.ndata; idata++) {
        PetscInt  size = 1;
        PetscReal data;
        PetscCall(RDyForcingSetHomogeneousData(&forcing->source.multihomogeneous.data[idata], time, size, &data));
        PetscCall(RDySetHomogeneousRegionalWaterSource(rdy, forcing->source.multihomogeneous.region_ids[idata] - 1, data));
      }
      break;
  }

  // Apply boundary conditions
  switch (forcing->boundary.type) {
    case FORCING_DATASET_UNSET:
      break;
    case FORCING_DATASET_CONSTANT:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend RDyApplyForcing for boundary condition of type CONSTANT");
      break;
    case FORCING_DATASET_HOMOGENEOUS:
      if (forcing->boundary.ndata) {
        PetscCall(RDyForcingSetHomogeneousBoundary(&forcing->boundary.homogeneous, time, forcing->boundary.ndata / 3,
                                                   forcing->boundary.data_for_rdycore));
        PetscCall(RDySetFlowDirichletBoundaryValues(rdy, forcing->boundary.dirichlet_bc_idx, forcing->boundary.ndata / 3, 3,
                                                    forcing->boundary.data_for_rdycore));
      }
      break;
    case FORCING_DATASET_RASTER:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Extend RDyApplyForcing for boundary condition of type RASTER");
      break;
    case FORCING_DATASET_UNSTRUCTURED:
      if (forcing->boundary.ndata) {
        PetscCall(RDyForcingSetUnstructuredData(&forcing->boundary.unstructured, time, forcing->boundary.data_for_rdycore));
        PetscCall(RDySetFlowDirichletBoundaryValues(rdy, forcing->boundary.dirichlet_bc_idx, forcing->boundary.ndata / 3, 3,
                                                    forcing->boundary.data_for_rdycore));
      }
      break;
    case FORCING_DATASET_MULTI_HOMOGENEOUS:
      for (PetscInt ibc = 0; ibc < forcing->boundary.multihomogeneous.ndirichlet_bcs; ibc++) {
        PetscInt data_idx = forcing->boundary.multihomogeneous.dirichlet_bc_to_data_idx[ibc];
        PetscInt bc_idx   = forcing->boundary.multihomogeneous.dirichlet_bc_idx[ibc];
        PetscInt nedges   = forcing->boundary.multihomogeneous.ndata_for_rdycore[ibc] / 3;

        if (nedges) {
          PetscCall(RDyForcingSetHomogeneousBoundary(&forcing->boundary.multihomogeneous.data[data_idx], time, nedges,
                                                     forcing->boundary.multihomogeneous.data_for_rdycore[ibc]));
          PetscCall(RDySetFlowDirichletBoundaryValues(rdy, bc_idx, nedges, 3, forcing->boundary.multihomogeneous.data_for_rdycore[ibc]));
        }
      }
      break;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Destroys a forcing object, releasing all resources.
/// @param [in,out] forcing  Pointer to the RDyForcing object (set to NULL on return)
/// @return PetscErrorCode
PetscErrorCode RDyDestroyForcing(RDyForcing *forcing) {
  PetscFunctionBegin;

  struct _p_RDyForcing *f = *forcing;

  // Destroy source/sink datasets
  switch (f->source.type) {
    case FORCING_DATASET_UNSET:
      break;
    case FORCING_DATASET_CONSTANT:
      PetscFree(f->source.data_for_rdycore);
      break;
    case FORCING_DATASET_HOMOGENEOUS:
      PetscFree(f->source.data_for_rdycore);
      PetscCall(RDyForcingDestroyHomogeneousDataset(&f->source.homogeneous));
      break;
    case FORCING_DATASET_RASTER:
      PetscFree(f->source.data_for_rdycore);
      PetscCall(RDyForcingDestroyRasterDataset(&f->source.raster));
      break;
    case FORCING_DATASET_UNSTRUCTURED:
      PetscFree(f->source.data_for_rdycore);
      PetscCall(RDyForcingDestroyUnstructuredDataset(&f->source.unstructured));
      break;
    case FORCING_DATASET_MULTI_HOMOGENEOUS:
      for (PetscInt idata = 0; idata < f->source.multihomogeneous.ndata; idata++) {
        PetscCall(RDyForcingDestroyHomogeneousDataset(&f->source.multihomogeneous.data[idata]));
      }
      PetscCall(PetscFree(f->source.multihomogeneous.data));
      break;
  }

  // Destroy boundary condition datasets
  if (f->boundary.ndata) {
    PetscCall(PetscFree(f->boundary.data_for_rdycore));
  }

  switch (f->boundary.type) {
    case FORCING_DATASET_UNSET:
      break;
    case FORCING_DATASET_CONSTANT:
      break;
    case FORCING_DATASET_HOMOGENEOUS:
      PetscCall(RDyForcingDestroyHomogeneousDataset(&f->boundary.homogeneous));
      break;
    case FORCING_DATASET_RASTER:
      break;
    case FORCING_DATASET_UNSTRUCTURED:
      PetscCall(RDyForcingDestroyUnstructuredDataset(&f->boundary.unstructured));
      break;
    case FORCING_DATASET_MULTI_HOMOGENEOUS:
      for (PetscInt idata = 0; idata < f->boundary.multihomogeneous.ndata; idata++) {
        PetscCall(RDyForcingDestroyHomogeneousDataset(&f->boundary.multihomogeneous.data[idata]));
      }
      PetscCall(PetscFree(f->boundary.multihomogeneous.data));

      if (f->boundary.multihomogeneous.ndirichlet_bcs) {
        PetscCall(PetscFree(f->boundary.multihomogeneous.dirichlet_bc_idx));
        PetscCall(PetscFree(f->boundary.multihomogeneous.dirichlet_bc_to_data_idx));
        PetscCall(PetscFree(f->boundary.multihomogeneous.ndata_for_rdycore));

        for (PetscInt i = 0; i < f->boundary.multihomogeneous.ndirichlet_bcs; i++) {
          PetscCall(PetscFree(f->boundary.multihomogeneous.data_for_rdycore[i]));
        }
        PetscCall(PetscFree(f->boundary.multihomogeneous.data_for_rdycore));
      }
      break;
  }

  PetscCall(PetscFree(f));
  *forcing = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}
