#ifndef RDYFORCINGIMPL_H
#define RDYFORCINGIMPL_H

#include <petscsys.h>
#include <petscvec.h>
#include <rdycore.h>
#include <time.h>

//--- Dataset type enum
typedef enum { FORCING_DATASET_UNSET = 0, FORCING_DATASET_CONSTANT, FORCING_DATASET_HOMOGENEOUS, FORCING_DATASET_RASTER, FORCING_DATASET_UNSTRUCTURED, FORCING_DATASET_MULTI_HOMOGENEOUS } RDyForcingDatasetType;

//--- Dataset structures

typedef struct {
  PetscReal rate;  // constant source/sink rate [m/s]
} RDyConstantDataset;

typedef struct {
  char         filename[PETSC_MAX_PATH_LEN];  // path to the binary PETSc Vec data file
  Vec          data_vec;                      // PETSc Vec loaded from the data file
  PetscInt     ndata;                         // number of temporal data values
  PetscScalar *data_ptr;                      // raw pointer into data_vec array
  PetscBool    temporally_interpolate;        // if true, interpolate between adjacent time values
  PetscInt     cur_idx, prev_idx;             // current and previous temporal indices
} RDyHomogeneousDataset;

typedef struct {
  char dir[PETSC_MAX_PATH_LEN];      // directory containing the dataset files
  char file[PETSC_MAX_PATH_LEN];     // base filename of the dataset
  char map_file[PETSC_MAX_PATH_LEN]; // path to the spatial mapping file

  struct tm start_date, current_date;  // start and current date for the dataset

  // binary data
  Vec          data_vec;  // PETSc Vec loaded from the current data file
  PetscScalar *data_ptr;  // raw pointer into data_vec array

  PetscInt header_offset;  // number of values in the file header before the data records start

  PetscReal dtime_in_hour;  // temporal duration of each dataset file
  PetscInt  ndata_file;     // number of time records per file

  PetscInt  ncols, nrows;  // number of columns and rows
  PetscReal xlc, ylc;      // cell centroid coordinates of lower-left corner
  PetscReal cellsize;      // dx = dy of cell

  PetscInt   mesh_ncells_local;  // number of cells in RDycore mesh
  PetscInt  *data2mesh_idx;      // for each RDycore cell, the index of the data in the raster dataset
  PetscReal *data_xc, *data_yc;  // x and y coordinates of data in the raster dataset
  PetscReal *mesh_xc, *mesh_yc;  // x and y coordinates of RDycore cells

  PetscBool write_map_for_debugging;  // if true, write the mapping for debugging
  PetscBool write_map;                // if true, write the map between RDycore cells and the dataset
  PetscBool read_map;                 // if true, read the map between RDycore cells and the dataset
} RDyRasterDataset;

typedef struct {
  char dir[PETSC_MAX_PATH_LEN];       // directory containing the dataset files
  char file[PETSC_MAX_PATH_LEN];      // base filename of the dataset
  char mesh_file[PETSC_MAX_PATH_LEN]; // path to the mesh file on which the dataset is defined
  char map_file[PETSC_MAX_PATH_LEN];  // path to the spatial mapping file

  PetscReal dtime_in_hour;  // temporal duration of each dataset file
  PetscInt  ndata_file;     // number of time records per file

  struct tm start_date, current_date;  // start and current date for the dataset

  // binary data
  Vec          data_vec;  // PETSc Vec loaded from the current data file
  PetscScalar *data_ptr;  // raw pointer into data_vec array

  PetscInt ndata;   // number of spatial data points
  PetscInt stride;  // number of values per data point

  PetscInt   mesh_nelements;     // number of cells or boundary edges in RDycore mesh
  PetscInt  *data2mesh_idx;      // for each RDycore element (cell or boundary edge), the index of the data in the unstructured dataset
  PetscReal *data_xc, *data_yc;  // x and y coordinates of data
  PetscReal *mesh_xc, *mesh_yc;  // x and y coordinates of RDycore elements

  PetscBool write_map_for_debugging;  // if true, write the mapping for debugging
  PetscBool write_map;                // if true, write the map between RDycore elements and the dataset
  PetscBool read_map;                 // if true, read the map between RDycore elements and the dataset
} RDyUnstructuredDataset;

typedef struct {
  RDyHomogeneousDataset *data;   // array of spatially-homogeneous, temporally-varying datasets
  PetscInt               ndata;  // number of datasets
  PetscInt              *region_ids;  // mesh region ID for each dataset

  PetscInt  ndirichlet_bcs;              // number of Dirichlet BCs
  PetscInt *dirichlet_bc_idx;            // RDycore indices of the Dirichlet BCs
  PetscInt *dirichlet_bc_to_data_idx;    // maps each Dirichlet BC to a dataset index

  PetscReal **data_for_rdycore;  // per-BC value arrays passed to RDycore
  PetscReal  *ndata_for_rdycore; // sizes of per-BC value arrays
} RDyMultiHomogeneousDataset;

//--- Source/sink and boundary condition structures

typedef struct {
  RDyForcingDatasetType        type;             // active dataset type (selects which union member is used)
  RDyConstantDataset           constant;         // spatio-temporally constant source/sink
  RDyHomogeneousDataset        homogeneous;       // spatially-constant, temporally-varying source/sink
  RDyRasterDataset             raster;            // spatio-temporally varying source/sink in raster format
  RDyUnstructuredDataset       unstructured;      // spatio-temporally varying source/sink in unstructured grid format
  RDyMultiHomogeneousDataset   multihomogeneous;  // multiple spatially-constant, temporally-varying source/sinks

  PetscInt   ndata;             // size of source/sink data array passed to RDycore
  PetscReal *data_for_rdycore;  // source/sink values passed to RDycore
} RDyForcingSourceSink;

typedef struct {
  RDyForcingDatasetType        type;             // active dataset type (selects which union member is used)
  RDyHomogeneousDataset        homogeneous;       // spatially-homogeneous, temporally-varying BC
  RDyUnstructuredDataset       unstructured;      // spatio-temporally varying BC in unstructured grid format
  RDyMultiHomogeneousDataset   multihomogeneous;  // multiple spatially-constant, temporally-varying BCs

  PetscInt   ndata;             // size of boundary condition data array passed to RDycore
  PetscInt   dirichlet_bc_idx;  // RDycore index of the Dirichlet BC
  PetscReal *data_for_rdycore;  // boundary condition values passed to RDycore
} RDyForcingBoundaryCondition;

//--- Top-level forcing object

typedef struct _p_RDyForcing *RDyForcing;

struct _p_RDyForcing {
  RDyForcingSourceSink         source;
  RDyForcingBoundaryCondition  boundary;
};

//--- Internal function declarations (rdyforcing_dataset.c)
PETSC_INTERN PetscErrorCode RDyForcingOpenData(char *filename, Vec *data_vec, PetscInt *ndata);
PETSC_INTERN PetscErrorCode RDyForcingGetCurrentData(PetscScalar *data_ptr, PetscInt ndata, PetscReal cur_time, PetscBool temporally_interpolate,
                                                     PetscInt *cur_data_idx, PetscReal *cur_data);
PETSC_INTERN PetscErrorCode RDyForcingOpenHomogeneousDataset(RDyHomogeneousDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingDestroyHomogeneousDataset(RDyHomogeneousDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingOpenMultiHomogeneousDataset(RDyMultiHomogeneousDataset *multi_data);
PETSC_INTERN PetscErrorCode RDyForcingDetermineDatasetFilename(struct tm *current_date, char *dir, char *file);
PETSC_INTERN PetscErrorCode RDyForcingOpenRasterDataset(RDyRasterDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingDestroyRasterDataset(RDyRasterDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingOpenNextRasterDataset(RDyRasterDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingOpenUnstructuredDataset(RDyUnstructuredDataset *data, PetscInt expected_data_stride);
PETSC_INTERN PetscErrorCode RDyForcingDestroyUnstructuredDataset(RDyUnstructuredDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingOpenNextUnstructuredDataset(RDyUnstructuredDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingSetConstantRainfall(PetscReal rain_rate, PetscInt ncells, PetscReal *rain);
PETSC_INTERN PetscErrorCode RDyForcingSetRasterData(RDyRasterDataset *data, PetscReal cur_time, PetscInt ncells, PetscReal *rain);
PETSC_INTERN PetscErrorCode RDyForcingSetHomogeneousData(RDyHomogeneousDataset *homogeneous_data, PetscReal cur_time, PetscInt ncells,
                                                         PetscReal *values);
PETSC_INTERN PetscErrorCode RDyForcingSetUnstructuredData(RDyUnstructuredDataset *data, PetscReal cur_time, PetscReal *data_values);
PETSC_INTERN PetscErrorCode RDyForcingSetHomogeneousBoundary(RDyHomogeneousDataset *bc_data, PetscReal cur_time, PetscInt num_values,
                                                             PetscReal *bc_values);

//--- Internal function declarations (rdyforcing_map.c)
PETSC_INTERN PetscErrorCode RDyForcingGetCellCentroidsFromMesh(RDy rdy, PetscInt n, PetscReal **xc, PetscReal **yc);
PETSC_INTERN PetscErrorCode RDyForcingGetBoundaryEdgeCentroidsFromMesh(RDy rdy, PetscInt n, PetscInt bc_id, PetscReal **xc, PetscReal **yc);
PETSC_INTERN PetscErrorCode RDyForcingReadUnstructuredDatasetCoordinates(RDyUnstructuredDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingCreateUnstructuredDatasetMap(RDyUnstructuredDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingCreateRasterDatasetMapping(RDy rdy, RDyRasterDataset *data);
PETSC_INTERN PetscErrorCode RDyForcingReadSpatialMap(RDy rdy, const char filename[], PetscInt ncells, PetscInt **data2mesh_idx);
PETSC_INTERN PetscErrorCode RDyForcingWriteMap(RDy rdy, char *filename, PetscInt ncells, PetscInt *d2m);
PETSC_INTERN PetscErrorCode RDyForcingWriteMappingForDebugging(char *filename, PetscInt ncells, PetscInt *d2m, PetscReal *dxc, PetscReal *dyc,
                                                               PetscReal *mxc, PetscReal *myc);

//--- Internal function declarations (rdyforcing.c)
PETSC_INTERN PetscErrorCode RDyForcingParseRainfallDataOptions(RDyForcingSourceSink *rain_dataset);
PETSC_INTERN PetscErrorCode RDyForcingParseBoundaryDataOptions(RDyForcingBoundaryCondition *bc);
PETSC_INTERN PetscErrorCode RDyCreateForcing(RDy rdy, RDyForcing *forcing);
PETSC_INTERN PetscErrorCode RDySetupForcing(RDyForcing forcing);
PETSC_INTERN PetscErrorCode RDyApplyForcing(RDy rdy, RDyForcing forcing, PetscReal time);
PETSC_INTERN PetscErrorCode RDyDestroyForcing(RDyForcing *forcing);

#endif  // RDYFORCINGIMPL_H
