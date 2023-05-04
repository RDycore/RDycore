#include <errno.h>
#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>

// maximum number of grid element blocks / topologies
#define MAX_NUM_TOPOLOGIES 32

// writes a XDMF "heavy data" to an HDF5 file
PetscErrorCode WriteXDMFHeavyData(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  PetscViewer viewer;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", fname));
  RDyLogDetail(rdy, "Step %d: writing XDMF HDF5 output to %s", step, fname);

  // write the grid if we're on the first step
  if (step == 0) {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
    PetscCall(DMView(rdy->dm, viewer));
  } else {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_APPEND, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  }

  // write solution data to a new GROUP with components in separate datasets
  PetscReal time_in_days = time / (24 * 3600);  // seconds -> days
  char      groupName[1025];
  snprintf(groupName, 1024, "%d %E d", step, time_in_days);
  PetscCall(PetscViewerHDF5PushGroup(viewer, groupName));

  // create and populate a multi-component natural vector
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
  PetscCall(DMPlexGlobalToNaturalBegin(rdy->dm, rdy->X, natural));
  PetscCall(DMPlexGlobalToNaturalEnd(rdy->dm, rdy->X, natural));

  // extract each component into a separate vector and write it to the group
  // FIXME: This setup is specific to the shallow water equations. We can
  // FIXME: generalize it later.
  const char *comp_names[3] = {
      "Water_Height",
      "X_Momentum",
      "Y_Momentum",
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
  // topology group path within the /viz group
  char path[PETSC_MAX_PATH_LEN];
  // number of cells in the topology
  int num_cells;
  // number of corners per cell
  int num_corners;
} GridTopology;

static herr_t GetGridTopology(hid_t viz_group, const char *name, const H5L_info2_t *info, void *op_data) {
  GridTopology *topologies = op_data;

  // find the next available topology, which should be the one with no path
  PetscInt i = 0;
  while (strlen(topologies[i].path) > 0) ++i;

  // access the cells dataset to fetch the relevant topology info
  hid_t cells_id = H5Dopen2(viz_group, "cells", H5P_DATASET_ACCESS);
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
  snprintf(topologies[i].path, PETSC_MAX_PATH_LEN - 1, "/viz/%s", name);

  return 0;
}

static PetscErrorCode ExtractGridTopologies(MPI_Comm comm, const char *h5_name, GridTopology *topologies, PetscInt *num_topologies) {
  PetscFunctionBegin;

  // zero out all topology data
  memset(topologies, 0, sizeof(GridTopology) * MAX_NUM_TOPOLOGIES);

  // meshes with more than one element block have multiple topologies, so we
  // must query the file to find their pathes. All topologies are stored as
  // groups within the "viz" group.
  hid_t h5_file = H5Fopen(h5_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  PetscCheck(h5_file != H5I_INVALID_HID, comm, PETSC_ERR_USER, "Could not open HDF5 output file %s", h5_name);

  hid_t viz_id = H5Gopen(h5_file, "/viz", H5P_GROUP_ACCESS);
  PetscCheck(viz_id != H5I_INVALID_HID, comm, PETSC_ERR_USER, "Could not open /viz group in file %s", h5_name);
  hsize_t index = 0;
  herr_t  err   = H5Literate2(viz_id, H5_INDEX_NAME, H5_ITER_INC, &index, GetGridTopology, topologies);
  PetscCheck(err == 0, comm, PETSC_ERR_USER, "Iteration over /viz HDF5 group failed");
  H5Gclose(viz_id);
  H5Fclose(h5_file);

  *num_topologies = index;

  PetscFunctionReturn(0);
}

//------------------------------------
// End HDF5 topology extraction logic
//------------------------------------

// generates an XMDF "light data" file (.xmf)
PetscErrorCode WriteXDMFLightData(RDy rdy) {
  PetscFunctionBegin;

  // data (HDF5) paths
  const char  *geom_path = "/geometry";
  GridTopology topologies[MAX_NUM_TOPOLOGIES];

  // time metadata
  PetscReal final_time = ConvertTimeToSeconds(rdy->config.final_time, rdy->config.time_unit);
  PetscInt  num_times  = (int)(final_time / rdy->dt);

  // mesh metadata
  PetscInt num_vertices = rdy->mesh.num_vertices;
  PetscInt space_dim    = 2;

  char h5_name[PETSC_MAX_PATH_LEN], xmf_name[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, 0, 0.0, "h5", h5_name));
  PetscCall(DetermineOutputFile(rdy, 0, 0.0, "xmf", xmf_name));
  RDyLogDetail(rdy, "Writing XDMF light data to %s...", xmf_name);
  FILE *fp;
  PetscCall(PetscFOpen(rdy->comm, xmf_name, "w", &fp));

  // extract the grid topologies from our HDF5 file.
  PetscInt num_topologies;
  PetscCall(ExtractGridTopologies(rdy->comm, h5_name, topologies, &num_topologies));

  // write header
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "<?xml version=\"1.0\" ?>\n"
                         "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" [\n"
                         "<!ENTITY HeavyData \"%s\">\n"
                         "]>\n\n"
                         "<Xdmf>\n  <Domain Name=\"domain\">\n",
                         h5_name));

  // write cell topology metadata
  for (PetscInt i = 0; i < num_topologies; ++i) {
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "    <DataItem Name=\"cells\"\n"
                           "              ItemType=\"Uniform\"\n"
                           "              Format=\"HDF\"\n"
                           "              NumberType=\"Float\" Precision=\"8\"\n"
                           "              Dimensions=\"%d %d\">\n"
                           "      &HeavyData;:/%s/cells\n"
                           "    </DataItem>\n",
                           topologies[i].num_cells, topologies[i].num_corners, topologies[i].path));
  }

  // write vertex metadata
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "    <DataItem Name=\"vertices\"\n"
                         "              Format=\"HDF\"\n"
                         "              Dimensions=\"%d %d\">\n"
                         "      &HeavyData;:/%s/vertices\n"
                         "    </DataItem>\n",
                         num_vertices, space_dim, geom_path));

  // write time grid header
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n"
                         "      <Time TimeType=\"List\">\n"
                         "        <DataItem Format=\"XML\" NumberType=\"Float\" Dimensions=\"%d\">\n"
                         "          ",
                         num_times));
  for (int n = 0; n < num_times; ++n) {
    PetscReal tn = n * rdy->dt;
    PetscCall(PetscFPrintf(rdy->comm, fp, "%g ", tn));
  }
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "        </DataItem>\n"
                         "      </Time>\n"));

  // write field data for each time
  for (int n = 0; n < num_times; ++n) {
    for (PetscInt i = 0; i < num_topologies; ++i) {
      // write space grid header
      PetscInt    num_corners  = topologies[i].num_corners;
      PetscInt    num_cells    = topologies[i].num_cells;
      const char *topo_type[5] = {"Invalid", "Invalid", "Invalid", "Triangle", "Quadrilateral"};
      PetscCall(PetscFPrintf(rdy->comm, fp,
                             "      <Grid Name=\"domain\" GridType=\"Uniform\">\n"
                             "        <Topology\n"
                             "           TopologyType=\"%s\"\n"
                             "           NumberOfElements=\"%d\">\n"
                             "          <DataItem Reference=\"XML\">\n"
                             "            /Xdmf/Domain/DataItem[@Name=\"cells\"]\n"
                             "          </DataItem>\n"
                             "        </Topology>\n"
                             "        <Geometry GeometryType=\"XY\">\n"
                             "          <DataItem Reference=\"XML\">\n"
                             "            /Xdmf/Domain/DataItem[@Name=\"vertices\"]\n"
                             "          </DataItem>\n"
                             "        </Geometry>\n",
                             topo_type[num_corners], num_cells));

      // write vertex field metadata
      // (none so far!)

      // write cell field metadata
      PetscReal   tn                  = n * rdy->dt;
      const char *cell_field_names[3] = {"Water_Height", "X_Momentum", "Y_Momentum"};
      for (int f = 0; f < 3; ++f) {
        PetscCall(PetscFPrintf(rdy->comm, fp,
                               "        <Attribute Name=\"%s\" Center=\"Cell\">\n"
                               "          <DataItem DataType=\"Float\" Precision=\"8\" Dimensions=\"%d\" Format=\"HDF\">\n"
                               "            &HeavyData;:/%d %.7E d/%s\n"
                               "          </DataItem>\n"
                               "        </Attribute>\n",
                               cell_field_names[f], num_cells, n, tn, cell_field_names[f]));

        // write space grid footer
        PetscCall(PetscFPrintf(rdy->comm, fp, "      </Grid>\n"));
      }
    }
  }

  // write time grid footer
  PetscCall(PetscFPrintf(rdy->comm, fp, "    </Grid>\n"));

  // write footer
  PetscCall(PetscFPrintf(rdy->comm, fp, "  </Domain>\n</Xdmf>\n"));

  PetscCall(PetscFClose(rdy->comm, fp));

  PetscFunctionReturn(0);
}
