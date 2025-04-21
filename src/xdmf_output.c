#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>

// writes output fields that match the given set, or a default set
static PetscErrorCode WriteFieldData(DM dm, Vec global_vec, char **output_fields, PetscInt n_output_fields, PetscViewer viewer,
                                     PetscInt num_refinements) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  // create and populate a multi-component vector
  Vec data_vec;

  if (!num_refinements) {  // FIXME: can't we get this info from the DM itself?
    PetscCall(DMPlexCreateNaturalVector(dm, &data_vec));
    PetscCall(DMPlexGlobalToNaturalBegin(dm, global_vec, data_vec));
    PetscCall(DMPlexGlobalToNaturalEnd(dm, global_vec, data_vec));
  } else {
    // a refined DM doesn't have a meaningful natural ordering, so we dump a
    // global vector instead
    PetscCall(VecDuplicate(global_vec, &data_vec));
    PetscCall(VecCopy(global_vec, data_vec));
    PetscCall(PetscObjectReference((PetscObject)data_vec));
  }

  // fetch our section to extract names of components
  PetscSection section;
  PetscCall(DMGetLocalSection(dm, &section));

  // the block size of a vector is total number of components in all its fields
  PetscInt num_fields, num_comp, bs, tot_num_comp = 0;
  PetscCall(PetscSectionGetNumFields(section, &num_fields));
  for (PetscInt f = 0; f < num_fields; ++f) {
    PetscCall(PetscSectionGetFieldComponents(section, f, &num_comp));
    tot_num_comp += num_comp;
  }
  PetscCall(VecGetBlockSize(data_vec, &bs));
  PetscCheck(tot_num_comp == bs, comm, PETSC_ERR_USER,
             "Vector block size (%" PetscInt_FMT ") is not equal to number of field components (%" PetscInt_FMT ")", bs, tot_num_comp);
  for (PetscInt f = 0; f < num_fields; ++f) {
    PetscCall(PetscSectionGetFieldComponents(section, f, &num_comp));

    // extract each component into a separate vector and write it
    Vec     *comp;
    PetscInt n, N;
    PetscCall(VecGetLocalSize(data_vec, &n));
    PetscCall(VecGetSize(data_vec, &N));
    PetscCall(PetscMalloc1(bs, &comp));  // NOLINT
    for (PetscInt c = 0; c < bs; ++c) {
      PetscCall(VecCreateMPI(comm, n / bs, N / bs, &comp[c]));
      const char *name;
      if (num_comp == 1) {
        // for single-component fields, use the field name
        PetscCall(PetscSectionGetFieldName(section, f, &name));
      } else {
        PetscCall(PetscSectionGetComponentName(section, f, c, &name));
      }
      PetscCall(PetscObjectSetName((PetscObject)comp[c], name));
    }
    PetscCall(VecStrideGatherAll(data_vec, comp, INSERT_VALUES));
    for (PetscInt c = 0; c < bs; ++c) {
      if (n_output_fields > 0) {
        const char *name;
        PetscCall(PetscObjectGetName((PetscObject)comp[c], &name));
        for (PetscInt i = 0; i < n_output_fields; ++i) {
          if (!strcmp(output_fields[i], name)) {
            PetscCall(VecView(comp[c], viewer));
            PetscCall(VecDestroy(&comp[c]));
            break;
          }
        }
      } else {  // no output fields specified -- write out all data
        PetscCall(VecView(comp[c], viewer));
        PetscCall(VecDestroy(&comp[c]));
      }
    }
    PetscCall(PetscFree(comp));
  }

  PetscCall(VecDestroy(&data_vec));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DetermineGridFile(RDy rdy, PetscInt step, PetscReal time, char *filename) {
  PetscFunctionBegin;

  if (rdy->config.output.separate_grid_file) {
    // the grid is stored in its own file
    char prefix[PETSC_MAX_PATH_LEN], output_dir[PETSC_MAX_PATH_LEN];
    PetscCall(DetermineConfigPrefix(rdy, prefix));
    PetscCall(GetOutputDirectory(rdy, output_dir));
    snprintf(filename, PETSC_MAX_PATH_LEN, "%s/%s-grid.h5", output_dir, prefix);
  } else {
    // the grid is stored in the base HDF5 file
    DetermineOutputFile(rdy, step, time, "h5", filename);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteGrid(MPI_Comm comm, RDyMesh *mesh, PetscViewer viewer) {
  PetscFunctionBegin;

  PetscCall(PetscViewerHDF5PushGroup(viewer, "Domain"));

  PetscCall(VecView(mesh->output.vertices_xyz_norder, viewer));
  PetscCall(VecView(mesh->output.cell_conns_norder, viewer));

  // NOTE: for some reason, these get deposited into the "fields" group(!)
  PetscCall(VecView(mesh->output.xc, viewer));
  PetscCall(VecView(mesh->output.yc, viewer));
  PetscCall(VecView(mesh->output.zc, viewer));

  PetscCall(PetscViewerHDF5PopGroup(viewer));

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

  // create or append to a file depending on whether this step is the first in a dataset
  PetscInt      dataset   = step / rdy->config.output.step_interval;
  PetscFileMode file_mode = (dataset % rdy->config.output.batch_size == 0) ? FILE_MODE_WRITE : FILE_MODE_APPEND;

  RDyLogDetail(rdy, "Step %" PetscInt_FMT ": writing XDMF HDF5 output at t = %g %s to %s", step, time, units, file_name);

  // write solution data for the primary and auxiliary DMs
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(rdy->comm, file_name, file_mode, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  PetscCall(PetscViewerHDF5SetCollective(viewer, PETSC_TRUE));  // enable collective MPI-IO transfers

  // write time-dependent solution and diagnostic fields
  char   **output_fields   = rdy->config.output.fields;
  PetscInt n_output_fields = rdy->config.output.fields_count;
  char     group_name[PETSC_MAX_PATH_LEN];
  snprintf(group_name, PETSC_MAX_PATH_LEN, "%" PetscInt_FMT " %E %s", step, time, units);
  PetscCall(PetscViewerHDF5PushGroup(viewer, group_name));
  PetscCall(WriteFieldData(rdy->dm, rdy->u_global, output_fields, n_output_fields, viewer, rdy->num_refinements));
  if (n_output_fields > 0) {  // diagnostics only written out if specified
    PetscCall(WriteFieldData(rdy->aux_dm, rdy->diags_vec, output_fields, n_output_fields, viewer, rdy->num_refinements));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  // write the grid
  if (rdy->config.output.separate_grid_file) {
    // on the first step ONLY, write the grid to its own file
    if (step == 0) {
      char h5_gridname[PETSC_MAX_PATH_LEN];
      PetscCall(DetermineGridFile(rdy, step, time, h5_gridname));
      RDyLogDetail(rdy, "Step 0: writing XDMF HDF5 grid to %s", h5_gridname);

      PetscViewer grid_viewer;
      PetscCall(PetscViewerHDF5Open(rdy->comm, h5_gridname, FILE_MODE_WRITE, &grid_viewer));
      PetscCall(PetscViewerPushFormat(grid_viewer, PETSC_VIEWER_HDF5_XDMF));
      PetscCall(PetscViewerHDF5SetCollective(grid_viewer, PETSC_TRUE));
      PetscCall(WriteGrid(rdy->comm, &rdy->mesh, grid_viewer));
      PetscCall(PetscViewerPopFormat(grid_viewer));
      PetscCall(PetscViewerDestroy(&grid_viewer));
    }
  } else {
    // write the grid with the rest of the HDF5 data if we're the first step in a batch.
    if (dataset % rdy->config.output.batch_size == 0) {
      PetscCall(WriteGrid(rdy->comm, &rdy->mesh, viewer));
    }
  }

  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteFieldMetadata(MPI_Comm comm, FILE *fp, const char *h5_basename, const char *group_name, DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  // write cell field metadata
  PetscSection section;
  PetscInt     num_fields, num_comp;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetNumFields(section, &num_fields));
  for (PetscInt f = 0; f < num_fields; ++f) {
    PetscCall(PetscSectionGetFieldComponents(section, f, &num_comp));
    for (PetscInt c = 0; c < num_comp; ++c) {
      const char *name;
      if (num_comp == 1) {
        PetscCall(PetscSectionGetFieldName(section, f, &name));
      } else {
        PetscCall(PetscSectionGetComponentName(section, f, c, &name));
      }
      PetscCall(PetscFPrintf(comm, fp,
                             "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                             "        <DataItem Dimensions=\"%" PetscInt_FMT "\" Format=\"HDF\">\n"
                             "          %s:/%s/%s\n"
                             "        </DataItem>\n"
                             "      </Attribute>\n",
                             name, mesh->num_cells_global, h5_basename, group_name, name));
    }
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

  char gridname[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineGridFile(rdy, step, time, gridname));
  last_slash              = strrchr(gridname, '/');
  const char *h5_gridname = last_slash ? &last_slash[1] : gridname;

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
                         mesh->num_cells_global, size, h5_gridname));
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "      <Geometry GeometryType=\"XYZ\">\n"
                         "        <DataItem Format=\"HDF\" Dimensions=\"%" PetscInt_FMT " 3\">\n"
                         "          %s:/Domain/Vertices\n"
                         "        </DataItem>\n"
                         "      </Geometry>\n",
                         num_vertices, h5_gridname));

  // write out mesh coordinates (data placed in "fields" group for some reason)
  const char *grid_coord_names[3] = {"XC", "YC", "ZC"};
  for (int f = 0; f < 3; ++f) {
    PetscCall(PetscFPrintf(rdy->comm, fp,
                           "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                           "        <DataItem Dimensions=\"%" PetscInt_FMT "\" Format=\"HDF\">\n"
                           "          %s:/fields/%s\n"
                           "        </DataItem>\n"
                           "      </Attribute>\n",
                           grid_coord_names[f], mesh->num_cells_global, h5_gridname, grid_coord_names[f]));
  }

  PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, rdy->dm, &rdy->mesh));
  PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, rdy->aux_dm, &rdy->mesh));

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
