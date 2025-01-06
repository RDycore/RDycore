#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>

static PetscErrorCode WriteFieldData(MPI_Comm comm, DM dm, Vec global_vec, char file_name[PETSC_MAX_PATH_LEN], char group_name[PETSC_MAX_PATH_LEN],
                                     PetscFileMode file_mode) {
  PetscFunctionBegin;

  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(comm, file_name, file_mode, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  PetscCall(PetscViewerHDF5SetCollective(viewer, PETSC_TRUE));  // enable collective MPI-IO transfers
  PetscCall(PetscViewerHDF5PushGroup(viewer, group_name));

  // create and populate a multi-component vector
  Vec       data_vec;
  PetscBool use_natural;
  PetscCall(DMGetUseNatural(dm, &use_natural));
  if (use_natural) {
    PetscCall(DMPlexCreateNaturalVector(dm, &data_vec));
    PetscCall(DMPlexGlobalToNaturalBegin(dm, global_vec, data_vec));
    PetscCall(DMPlexGlobalToNaturalEnd(dm, global_vec, data_vec));
  } else {
    data_vec = global_vec;
    PetscCall(PetscObjectReference((PetscObject)data_vec));
  }

  // fetch our section to extract names of components
  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));

  // the section should contain a single field with a number of components
  // equal to the block size of the data vector
  PetscInt num_fields, num_comp, bs;
  PetscCall(PetscSectionGetNumFields(section, &num_fields));
  PetscCheck(num_fields == 1, comm, PETSC_ERR_USER, "Primary DM section has %" PetscInt_FMT " fields (should contain 1)", num_fields);
  PetscCall(PetscSectionGetFieldComponents(section, 0, &num_comp));
  PetscCall(VecGetBlockSize(data_vec, &bs));
  PetscCheck(num_comp == bs, comm, PETSC_ERR_USER,
             "Vector block size (%" PetscInt_FMT ") is not equal to number of field components (%" PetscInt_FMT ")", bs, num_comp);

  // extract each component into a separate vector and write it to the group
  Vec     *comp;  // array of single-component vectors
  PetscInt n, N;
  PetscCall(VecGetLocalSize(data_vec, &n));
  PetscCall(VecGetSize(data_vec, &N));
  PetscCall(PetscMalloc1(bs, &comp));
  for (PetscInt c = 0; c < bs; ++c) {
    PetscCall(VecCreateMPI(comm, n / bs, N / bs, &comp[c]));
    const char *comp_name;
    PetscCall(PetscSectionGetComponentName(section, 0, c, &comp_name));
    PetscCall(PetscObjectSetName((PetscObject)comp[c], comp_name));
  }
  PetscCall(VecStrideGatherAll(data_vec, comp, INSERT_VALUES));
  for (PetscInt c = 0; c < bs; ++c) {
    PetscCall(VecView(comp[c], viewer));
    PetscCall(VecDestroy(&comp[c]));
  }
  PetscCall(PetscFree(comp));

  // clean up
  if (use_natural) PetscCall(VecDestroy(&data_vec));  // FIXME: does this leave an extra ref in non-natural case?
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteGrid(MPI_Comm comm, RDyMesh *mesh, char file_name[PETSC_MAX_PATH_LEN]) {
  PetscFunctionBegin;

  PetscViewer viewer;
  char        group_name[1025];
  snprintf(group_name, 1024, "Domain");
  PetscCall(PetscViewerHDF5Open(comm, file_name, FILE_MODE_APPEND, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  PetscCall(PetscViewerHDF5PushGroup(viewer, group_name));

  PetscCall(VecView(mesh->output.vertices_xyz_norder, viewer));
  PetscCall(VecView(mesh->output.cell_conns_norder, viewer));
  PetscCall(VecView(mesh->output.xc, viewer));
  PetscCall(VecView(mesh->output.yc, viewer));
  PetscCall(VecView(mesh->output.zc, viewer));

  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Writes a XDMF "heavy data" to an HDF5 file. The time is expressed in the
// units given in the configuration file.
static PetscErrorCode WriteXDMFHDF5Data(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  // Determine the output file name.
  char file_name[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", file_name));
  const char *units = TimeUnitAsString(rdy->config.time.unit);

  // construct a group name that encodes the step, time, units
  char group_name[PETSC_MAX_PATH_LEN];
  snprintf(group_name, PETSC_MAX_PATH_LEN, "%" PetscInt_FMT " %E %s", step, time, units);

  // create or append to a file depending on whether this step is the first in a dataset
  PetscInt      dataset   = step / rdy->config.output.step_interval;
  PetscFileMode file_mode = (dataset % rdy->config.output.batch_size == 0) ? FILE_MODE_WRITE : FILE_MODE_APPEND;

  RDyLogDetail(rdy, "Step %" PetscInt_FMT ": writing XDMF HDF5 output at t = %g %s to %s", step, time, units, file_name);

  // write solution data for the primary DM
  PetscCall(WriteFieldData(rdy->comm, rdy->dm, rdy->u_global, file_name, group_name, file_mode));

  // write the grid if we're the first step in a batch.
  if (dataset % rdy->config.output.batch_size == 0) {
    PetscCall(WriteGrid(rdy->comm, &rdy->mesh, file_name));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Generates an XMDF "light data" file (.xmf) for the given step and time. The
/// time is expressed in the units specified in the .yaml input file.
static PetscErrorCode WriteXDMFXMFData(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  // mesh metadata
  PetscInt num_vertices = rdy->mesh.num_vertices_global;
  PetscInt coord_dim;
  PetscCall(DMGetCoordinateDim(rdy->dm, &coord_dim));

  char h5_name[PETSC_MAX_PATH_LEN], xmf_name[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", h5_name));
  PetscCall(DetermineOutputFile(rdy, step, time, "xmf", xmf_name));
  const char *units = TimeUnitAsString(rdy->config.time.unit);
  RDyLogDetail(rdy, "Step %" PetscInt_FMT ": writing XDMF XMF output at t = %g %s to %s", step, time, units, xmf_name);

  FILE *fp;
  PetscCall(PetscFOpen(rdy->comm, xmf_name, "w", &fp));

  // open the HDF5 file so we can extract some information from it
  hid_t h5_file = H5Fopen(h5_name, H5F_ACC_RDONLY, H5P_DEFAULT);
  PetscCheck(h5_file != H5I_INVALID_HID, rdy->comm, PETSC_ERR_USER, "Could not open HDF5 output file %s", h5_name);

  H5Fclose(h5_file);

  // determine the "base name" for the HDF5 file (which excludes its directory),
  // since the XMF and H5 files should be in the same directory
  char       *last_slash  = strrchr(h5_name, '/');
  const char *h5_basename = last_slash ? &last_slash[1] : h5_name;

  // write the header
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "<?xml version=\"1.0\" ?>\n"
                         "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
                         "<Xdmf>\n  <Domain>\n"));

  // construct the group name containing the time-specific solution data
  char time_group[1025];
  snprintf(time_group, 1024, "%" PetscInt_FMT " %E %s", step, time, units);

  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "    <Grid Name=\"domain\">\n"
                         "      <Time Value=\"%E\" />\n",
                         time));

  RDyMesh *mesh = &rdy->mesh;
  PetscInt size;
  VecGetSize(mesh->output.cell_conns_norder, &size);
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "      <Topology Type=\"Mixed\" NumberOfElements=\"%" PetscInt_FMT "\">\n"
                         "        <DataItem Format=\"HDF\" DataType=\"int\" Dimensions=\"%" PetscInt_FMT "\">\n"
                         "          %s:/Domain/Cells\n"
                         "        </DataItem>\n"
                         "      </Topology>\n",
                         mesh->num_cells_global, size, h5_basename));
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "      <Geometry GeometryType=\"XYZ\">\n"
                         "        <DataItem Format=\"HDF\" Dimensions=\"%" PetscInt_FMT " 3\">\n"
                         "          %s:Domain/Vertices\n"
                         "        </DataItem>\n"
                         "      </Geometry>\n",
                         num_vertices, h5_basename));

  const char *geometric_cell_field_names[3] = {"XC", "YC", "ZC"};
  for (int f = 0; f < 3; ++f) {
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                           "        <DataItem Dimensions=\"%" PetscInt_FMT "\" Format=\"HDF\">\n"
                           "          %s:/Domain/%s\n"
                           "        </DataItem>\n"
                           "      </Attribute>\n",
                           geometric_cell_field_names[f], mesh->num_cells_global, h5_basename, geometric_cell_field_names[f]));
  }

  // write cell field metadata
  PetscSection section;
  PetscInt     num_comp;
  PetscCall(DMGetLocalSection(rdy->dm, &section));
  PetscCall(PetscSectionGetFieldComponents(section, 0, &num_comp));
  for (int f = 0; f < 3; ++f) {
    const char *comp_name;
    PetscCall(PetscSectionGetComponentName(section, 0, f, &comp_name));
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                           "        <DataItem Dimensions=\"%" PetscInt_FMT "\" Format=\"HDF\">\n"
                           "          %s:/%s/%s\n"
                           "        </DataItem>\n"
                           "      </Attribute>\n",
                           comp_name, mesh->num_cells_global, h5_basename, time_group, comp_name));
  }

  PetscCall(PetscFPrintf(rdy->comm, fp, "    </Grid>\n"));

  // write footer and close the file
  PetscCall(PetscFPrintf(rdy->comm, fp, "  </Domain>\n</Xdmf>\n"));
  PetscCall(PetscFClose(rdy->comm, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WriteXDMFOutput(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy               rdy    = ctx;
  RDyOutputSection *output = &rdy->config.output;

  if (output->enable && (time != output->prev_output_time)) {
    PetscBool write_output = PETSC_FALSE;

    // check if it is time to output based on temporal interval
    if (output->time_interval > 0) {
      PetscReal dt   = ConvertTimeToSeconds(output->time_interval * 1.0, output->time_unit);
      PetscReal t    = time;
      PetscReal tmp  = fmod(t, dt);
      PetscReal diff = (tmp - dt);
      write_output   = (PetscAbsReal(tmp) < 10.0 * DBL_EPSILON || PetscAbsReal(diff) < 10.0 * DBL_EPSILON);
    }

    // check if it is time to output based on step interval
    if (output->step_interval > 0 && !write_output) {
      if (step % output->step_interval == 0) write_output = PETSC_TRUE;
    }

    // write output
    if (write_output) {
      // save the time output was written
      output->prev_output_time = time;

      PetscReal t = ConvertTimeFromSeconds(time, rdy->config.time.unit);
      if (output->format == OUTPUT_XDMF) {
        PetscCall(WriteXDMFHDF5Data(rdy, step, t));
        PetscCall(WriteXDMFXMFData(rdy, step, t));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
