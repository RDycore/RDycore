#include <errno.h>
#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>
#include <rdycore.h>
#include <string.h>

// maximum number of grid element blocks / topologies
#define MAX_NUM_TOPOLOGIES 32

// name of the HDF5 group containing mesh geometry written by PETSc
static const char *h5_geom_group = "/geometry";

// name of the HDF5 group containing mesh topologies written by PETSc
static const char *h5_topo_group = "/viz/topology";

// Writes a XDMF "heavy data" to an HDF5 file. The time is expressed in the
// units given in the configuration file.
static PetscErrorCode WriteXDMFHDF5Data(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  PetscViewer viewer;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", fname));
  const char *units = TimeUnitAsString(rdy->config.time.unit);
  RDyLogDetail(rdy, "Step %d: writing XDMF HDF5 output at t = %g %s to %s", step, time, units, fname);

  // write the grid if we're the first step in a batch.
  if (step % rdy->config.output.batch_size == 0) {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
    PetscCall(DMView(rdy->dm, viewer));
  } else {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_APPEND, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  }

  // write solution data to a new GROUP with components in separate datasets
  char group_name[1025];
  snprintf(group_name, 1024, "%d %E %s", step, time, units);
  PetscCall(PetscViewerHDF5PushGroup(viewer, group_name));

  // create and populate a multi-component natural vector
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
  PetscCall(DMPlexGlobalToNaturalBegin(rdy->dm, rdy->X, natural));
  PetscCall(DMPlexGlobalToNaturalEnd(rdy->dm, rdy->X, natural));

  // extract each component into a separate vector and write it to the group
  // FIXME: This setup is specific to the shallow water equations. We can
  // FIXME: generalize it later.
  const char *comp_names[3] = {
      "Height",
      "MomentumX",
      "MomentumY",
  };
  Vec      comp;  // single-component natural vector
  PetscInt n, N, bs;
  PetscCall(VecGetLocalSize(natural, &n));
  PetscCall(VecGetSize(natural, &N));
  PetscCall(VecGetBlockSize(natural, &bs));
  PetscCall(VecCreateMPI(rdy->comm, n / bs, N / bs, &comp));
  PetscReal *Xi;  // multi-component natural vector data
  PetscCall(VecGetArray(natural, &Xi));
  for (PetscInt c = 0; c < bs; ++c) {
    PetscObjectSetName((PetscObject)comp, comp_names[c]);
    PetscReal *Xci;  // single-component natural vector data
    PetscCall(VecGetArray(comp, &Xci));
    for (PetscInt i = 0; i < n / bs; ++i) Xci[i] = Xi[bs * i + c];
    PetscCall(VecRestoreArray(comp, &Xci));
    PetscCall(VecView(comp, viewer));
  }
  PetscCall(VecRestoreArray(natural, &Xi));

  // clean up
  PetscCall(VecDestroy(&comp));
  PetscCall(VecDestroy(&natural));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
}

//--------------------------------
// HDF5 topology extraction logic
//--------------------------------

// this type stores information about a grid topology
typedef struct {
  // topology group path within the topology group
  char path[PETSC_MAX_PATH_LEN];
  // number of cells in the topology
  int num_cells;
  // number of corners per cell
  int num_corners;
} GridTopology;

// extracts grid topology information from a specific entry in an HDF5 file
static herr_t GetGridTopology(hid_t topo_group, const char *name, const H5L_info2_t *info, void *op_data) {
  GridTopology *topologies = op_data;

  // find the next available topology, which should be the one with no path
  PetscInt i = 0;
  while (strlen(topologies[i].path) > 0) ++i;

  // access the cells dataset to fetch the relevant topology info
  hid_t cells_id = H5Dopen2(topo_group, "cells", H5P_DEFAULT);
  if (cells_id == H5I_INVALID_HID) return -1;

  // access the data space
  hid_t cell_space_id = H5Dget_space(cells_id);
  if (cell_space_id == H5I_INVALID_HID) return -1;

  // get the number of cells and their number of corners
  hsize_t dims[2], max_dims[2];
  int     dim = H5Sget_simple_extent_dims(cell_space_id, dims, max_dims);
  if (dim != 2) return -1;
  H5Dclose(cells_id);

  // write everything to the ith topology
  topologies[i].num_cells   = dims[0];
  topologies[i].num_corners = dims[1];
  snprintf(topologies[i].path, PETSC_MAX_PATH_LEN - 1, "%s/%s", h5_topo_group, name);

  return 0;
}

// extracts grid topology data from the given HDF5 file
static PetscErrorCode ExtractGridTopologies(MPI_Comm comm, hid_t h5_file, GridTopology *topologies, PetscInt *num_topologies) {
  PetscFunctionBegin;

  // zero out all topology data
  memset(topologies, 0, sizeof(GridTopology) * MAX_NUM_TOPOLOGIES);

  // meshes with more than one element block have multiple topologies, so we
  // must query the file to find their pathes. All topologies are stored as
  // groups within the topology group.

  hid_t topo_id = H5Gopen(h5_file, h5_topo_group, H5P_DEFAULT);
  PetscCheck(topo_id != H5I_INVALID_HID, comm, PETSC_ERR_USER, "Could not open %s group in HDF5 output file", h5_topo_group);
  hsize_t index = 0;
  herr_t  err   = H5Literate2(topo_id, H5_INDEX_NAME, H5_ITER_INC, &index, GetGridTopology, topologies);
  PetscCheck(err == 0, comm, PETSC_ERR_USER, "Iteration over %s HDF5 group failed", h5_topo_group);
  H5Gclose(topo_id);

  *num_topologies = index;

  PetscFunctionReturn(0);
}

//------------------------------------
// End HDF5 topology extraction logic
//------------------------------------

/// Generates an XMDF "light data" file (.xmf) for the given step and time. The
/// time is expressed in the units specified in the .yaml input file.
static PetscErrorCode WriteXDMFXMFData(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  GridTopology topologies[MAX_NUM_TOPOLOGIES];

  // mesh metadata
  PetscInt num_vertices = rdy->mesh.num_vertices;
  PetscInt coord_dim;
  PetscCall(DMGetCoordinateDim(rdy->dm, &coord_dim));
  const char *geom_types[4] = {NULL, NULL, "XY", "XYZ"};  // use geom_types[coord_dim]

  char h5_name[PETSC_MAX_PATH_LEN], xmf_name[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", h5_name));
  PetscCall(DetermineOutputFile(rdy, step, time, "xmf", xmf_name));
  const char *units = TimeUnitAsString(rdy->config.time.unit);
  RDyLogDetail(rdy, "Step %d: writing XDMF XMF output at t = %g %s to %s", step, time, units, xmf_name);

  FILE *fp;
  PetscCall(PetscFOpen(rdy->comm, xmf_name, "w", &fp));

  // open the HDF5 file so we can extract some information from it
  hid_t h5_file = H5Fopen(h5_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  PetscCheck(h5_file != H5I_INVALID_HID, rdy->comm, PETSC_ERR_USER, "Could not open HDF5 output file %s", h5_name);

  // extract the grid topologies from our HDF5 file
  PetscInt num_topologies;
  PetscCall(ExtractGridTopologies(rdy->comm, h5_file, topologies, &num_topologies));

  H5Fclose(h5_file);

  // determine the "base name" for the HDF5 file (which excludes its directory),
  // since the XMF and H5 files should be in the same directory
  char       *last_slash  = strrchr(h5_name, '/');
  const char *h5_basename = last_slash ? &last_slash[1] : h5_name;

  // write the header
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "<?xml version=\"1.0\" ?>\n"
                         "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
                         "<Xdmf>\n  <Domain Name=\"domain\">\n"));

  // construct the group name containing the time-specific solution data
  char time_group[1025];
  snprintf(time_group, 1024, "%d %E %s", step, time, units);

  // write time-specific field data
  for (PetscInt i = 0; i < num_topologies; ++i) {
    // write space grid header
    PetscInt    num_corners  = topologies[i].num_corners;
    PetscInt    num_cells    = topologies[i].num_cells;
    const char *topo_type[5] = {"Invalid", "Invalid", "Invalid", "Triangle", "Quadrilateral"};
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "    <Grid Name=\"domain\" GridType=\"Uniform\">\n"
                           "      <Time Value=\"%E\" />\n",
                           time));
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "      <Topology Type=\"%s\" NumberOfElements=\"%d\">\n"
                           "        <DataItem Format=\"HDF\" DataType=\"int\" Dimensions=\"%d %d\">\n"
                           "          %s:%s/cells\n"
                           "        </DataItem>\n"
                           "      </Topology>\n",
                           topo_type[num_corners], num_cells, num_cells, num_corners, h5_basename, h5_topo_group));
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "      <Geometry GeometryType=\"%s\">\n"
                           "        <DataItem Format=\"HDF\" Dimensions=\"%d %d\">\n"
                           "          %s:%s/vertices\n"
                           "        </DataItem>\n"
                           "      </Geometry>\n",
                           geom_types[coord_dim], num_vertices, coord_dim, h5_basename, h5_geom_group));

    // write vertex field metadata
    // (none so far!)

    // write cell field metadata
    const char *cell_field_names[3] = {"Height", "MomentumX", "MomentumY"};
    for (int f = 0; f < 3; ++f) {
      PetscCall(PetscFPrintf(rdy->comm, fp,
                             "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                             "      <DataItem Dimensions=\"%d\" Format=\"HDF\">\n"
                             "        %s:/%s/%s\n"
                             "      </DataItem>\n"
                             "      </Attribute>\n",
                             cell_field_names[f], num_cells, h5_basename, time_group, cell_field_names[f]));
    }
    // write space grid footer
    PetscCall(PetscFPrintf(rdy->comm, fp, "    </Grid>\n"));
  }

  // write footer and close the file
  PetscCall(PetscFPrintf(rdy->comm, fp, "  </Domain>\n</Xdmf>\n"));
  PetscCall(PetscFClose(rdy->comm, fp));
  PetscFunctionReturn(0);
}

PetscErrorCode WriteXDMFOutput(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy rdy = ctx;
  if (step % rdy->config.output.frequency == 0) {
    PetscReal t = ConvertTimeFromSeconds(time, rdy->config.time.unit);
    if (rdy->config.output.format == OUTPUT_XDMF) {
      PetscCall(WriteXDMFHDF5Data(rdy, step, t));
      PetscCall(WriteXDMFXMFData(rdy, step, t));
    }
  }
  PetscFunctionReturn(0);
}
