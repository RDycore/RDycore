#include <errno.h>
#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>

// Writes a XDMF "heavy data" to an HDF5 file. The time is expressed in the
// units given in the configuration file.
static PetscErrorCode WriteXDMFHDF5Data(RDy rdy, PetscInt step, PetscReal time) {
  PetscFunctionBegin;

  PetscViewer viewer;

  // Determine the output file name.
  char fname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", fname));
  const char *units = TimeUnitAsString(rdy->config.time.unit);
  RDyLogDetail(rdy, "Step %" PetscInt_FMT ": writing XDMF HDF5 output at t = %g %s to %s", step, time, units, fname);

  // write the grid if we're the first step in a batch.
  PetscInt dataset = step / rdy->config.output.step_interval;
  if (dataset % rdy->config.output.batch_size == 0) {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  } else {
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_APPEND, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  }
  // turn on collective MPI-IO transfers
  PetscCall(PetscViewerHDF5SetCollective(viewer, PETSC_TRUE));

  // write solution data to a new GROUP with components in separate datasets
  char group_name[1025];
  snprintf(group_name, 1024, "%" PetscInt_FMT " %E %s", step, time, units);
  PetscCall(PetscViewerHDF5PushGroup(viewer, group_name));

  // create and populate a multi-component natural vector
  Vec       natural;
  PetscBool use_natural;
  PetscCall(DMGetUseNatural(rdy->dm, &use_natural));
  if (use_natural) {
    PetscCall(DMPlexCreateNaturalVector(rdy->dm, &natural));
    PetscCall(DMPlexGlobalToNaturalBegin(rdy->dm, rdy->u_global, natural));
    PetscCall(DMPlexGlobalToNaturalEnd(rdy->dm, rdy->u_global, natural));
  } else {
    natural = rdy->u_global;
    PetscCall(PetscObjectReference((PetscObject)natural));
  }

  // extract each component into a separate vector and write it to the group
  // FIXME: This setup is specific to the shallow water equations. We can
  // FIXME: generalize it later.
  const char *comp_names[3] = {
      "Height",
      "MomentumX",
      "MomentumY",
  };
  Vec     *comp;  // single-component natural vector
  PetscInt n, N, bs;
  PetscCall(VecGetLocalSize(natural, &n));
  PetscCall(VecGetSize(natural, &N));
  PetscCall(VecGetBlockSize(natural, &bs));
  PetscCall(PetscMalloc1(bs, &comp));
  for (PetscInt c = 0; c < bs; ++c) {
    PetscCall(VecCreateMPI(rdy->comm, n / bs, N / bs, &comp[c]));
    PetscCall(PetscObjectSetName((PetscObject)comp[c], comp_names[c]));
  }
  PetscCall(VecStrideGatherAll(natural, comp, INSERT_VALUES));
  for (PetscInt c = 0; c < bs; ++c) {
    PetscCall(VecView(comp[c], viewer));
    PetscCall(VecDestroy(&comp[c]));
  }
  PetscCall(PetscFree(comp));

  // clean up
  PetscCall(VecDestroy(&natural));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  if (dataset % rdy->config.output.batch_size == 0) {
    char group_name[1025];
    snprintf(group_name, 1024, "Domain");
    PetscCall(PetscViewerHDF5Open(rdy->comm, fname, FILE_MODE_APPEND, &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
    PetscCall(PetscViewerHDF5PushGroup(viewer, group_name));

    RDyMesh *mesh = &rdy->mesh;
    PetscCall(VecView(mesh->output.vertices_xyz_norder, viewer));
    PetscCall(VecView(mesh->output.cell_conns_norder, viewer));
    PetscCall(VecView(mesh->output.xc, viewer));
    PetscCall(VecView(mesh->output.yc, viewer));
    PetscCall(VecView(mesh->output.zc, viewer));

    PetscCall(PetscViewerHDF5PopGroup(viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
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
  const char *cell_field_names[3] = {"Height", "MomentumX", "MomentumY"};
  for (int f = 0; f < 3; ++f) {
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                           "        <DataItem Dimensions=\"%" PetscInt_FMT "\" Format=\"HDF\">\n"
                           "          %s:/%s/%s\n"
                           "        </DataItem>\n"
                           "      </Attribute>\n",
                           cell_field_names[f], mesh->num_cells_global, h5_basename, time_group, cell_field_names[f]));
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
