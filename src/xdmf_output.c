#include <petscdmceed.h>
#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>

// Returns PETSC_TRUE if at least one solution *_Mean field (Height_Mean, MomentumX_Mean, etc.) is requested.
static PetscBool AnySolnMeanRequested(const RDyOutputSection *output, const SectionFieldSpec *soln_avg_fields) {
  for (PetscInt i = 0; i < output->fields_count; ++i) {
    for (PetscInt c = 0; c < soln_avg_fields->num_field_components[0]; ++c) {
      if (!strcmp(output->fields[i], soln_avg_fields->field_component_names[0][c])) return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

// Returns PETSC_TRUE if at least one *_Mean primitive variable field is requested
// (excludes Height_Mean, which is owned by solution time-average output).
static PetscBool AnyMeanPrimVarRequested(const RDyOutputSection *output, const SectionFieldSpec *prim_vars_fields) {
  // skip component 0 (Height_Mean), which is now owned by solution time-average output
  for (PetscInt i = 0; i < output->fields_count; ++i) {
    for (PetscInt c = 1; c < prim_vars_fields->num_field_components[0]; ++c) {
      if (!strcmp(output->fields[i], prim_vars_fields->field_component_names[0][c])) return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

// Returns PETSC_TRUE if at least one instantaneous primitive variable field is requested.
// (VelocityX, VelocityY, Concentration%i — excludes Height which comes from u_global)
static PetscBool AnyInstPrimVarRequested(const RDyOutputSection *output, const SectionFieldSpec *inst_spec) {
  for (PetscInt i = 0; i < output->fields_count; ++i) {
    // skip component 0 (Height), which is intentionally excluded from instantaneous prim var output
    for (PetscInt c = 1; c < inst_spec->num_field_components[0]; ++c) {
      if (!strcmp(output->fields[i], inst_spec->field_component_names[0][c])) return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

// Returns PETSC_TRUE if at least one instantaneous source field is requested.
static PetscBool AnySrcInstRequested(const RDyOutputSection *output, const SectionFieldSpec *src_inst_fields) {
  if (!src_inst_fields->num_fields) return PETSC_FALSE;
  for (PetscInt i = 0; i < output->fields_count; ++i) {
    for (PetscInt c = 0; c < src_inst_fields->num_field_components[0]; ++c) {
      if (!strcmp(output->fields[i], src_inst_fields->field_component_names[0][c])) return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

// Returns PETSC_TRUE if at least one time-averaged source field is requested.
static PetscBool AnySrcMeanRequested(const RDyOutputSection *output, const SectionFieldSpec *src_avg_fields) {
  if (!src_avg_fields->num_fields) return PETSC_FALSE;
  for (PetscInt i = 0; i < output->fields_count; ++i) {
    for (PetscInt c = 0; c < src_avg_fields->num_field_components[0]; ++c) {
      if (!strcmp(output->fields[i], src_avg_fields->field_component_names[0][c])) return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

// Writes components of global_vec whose names (from spec) match output.fields.
// Components with index < component_offset are skipped unconditionally.
// If output.fields_count == 0, no components are written.
static PetscErrorCode WriteFilteredFieldData(DM dm, Vec global_vec, PetscInt component_offset, const SectionFieldSpec *spec, RDyOutputSection output,
                                             PetscViewer viewer, PetscInt num_refinements) {
  PetscFunctionBegin;

  if (output.fields_count == 0) PetscFunctionReturn(PETSC_SUCCESS);

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));

  Vec data_vec;
  if (!num_refinements) {
    PetscCall(DMPlexCreateNaturalVector(dm, &data_vec));
    PetscCall(DMPlexGlobalToNaturalBegin(dm, global_vec, data_vec));
    PetscCall(DMPlexGlobalToNaturalEnd(dm, global_vec, data_vec));
  } else {
    PetscCall(VecDuplicate(global_vec, &data_vec));
    PetscCall(VecCopy(global_vec, data_vec));
  }

  PetscInt bs;
  PetscCall(VecGetBlockSize(data_vec, &bs));
  PetscInt n, N;
  PetscCall(VecGetLocalSize(data_vec, &n));
  PetscCall(VecGetSize(data_vec, &N));

  PetscInt c_global = 0;
  for (PetscInt f = 0; f < spec->num_fields; ++f) {
    for (PetscInt c = 0; c < spec->num_field_components[f]; ++c, ++c_global) {
      if (c_global < component_offset) continue;
      const char *name  = spec->field_component_names[f][c];
      PetscBool   write = PETSC_FALSE;
      for (PetscInt i = 0; i < output.fields_count; ++i) {
        if (!strcmp(output.fields[i], name)) {
          write = PETSC_TRUE;
          break;
        }
      }
      if (write) {
        Vec comp;
        PetscCall(VecCreateMPI(comm, n / bs, N / bs, &comp));
        PetscCall(PetscObjectSetName((PetscObject)comp, name));
        PetscCall(VecStrideGather(data_vec, c_global, comp, INSERT_VALUES));
        PetscCall(VecView(comp, viewer));
        PetscCall(VecDestroy(&comp));
      }
    }
  }

  PetscCall(VecDestroy(&data_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Writes XMF metadata for components of spec whose names match output.fields.
// Components with index < component_offset are skipped unconditionally.
// If output.fields_count == 0, no metadata is written.
static PetscErrorCode WriteFieldMetadata(MPI_Comm comm, FILE *fp, const char *h5_basename, const char *group_name, const SectionFieldSpec *spec,
                                         RDyOutputSection output, PetscInt component_offset, PetscInt num_cells_global) {
  PetscFunctionBegin;

  if (output.fields_count == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt c_global = 0;
  for (PetscInt f = 0; f < spec->num_fields; ++f) {
    for (PetscInt c = 0; c < spec->num_field_components[f]; ++c, ++c_global) {
      if (c_global < component_offset) continue;
      const char *name  = spec->field_component_names[f][c];
      PetscBool   write = PETSC_FALSE;
      for (PetscInt i = 0; i < output.fields_count; ++i) {
        if (!strcmp(output.fields[i], name)) {
          write = PETSC_TRUE;
          break;
        }
      }
      if (write) {
        PetscCall(PetscFPrintf(comm, fp,
                               "      <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                               "        <DataItem Dimensions=\"%" PetscInt_FMT "\" Format=\"HDF\">\n"
                               "          %s:/%s/%s\n"
                               "        </DataItem>\n"
                               "      </Attribute>\n",
                               name, num_cells_global, h5_basename, group_name, name));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// writes output fields that match the given set; writes all fields if none are given
// NOTE: This function is retained as a static helper but is no longer called directly.
// All output now goes through WriteFilteredFieldData.

static PetscErrorCode DetermineGridFile(RDy rdy, char *filename) {
  PetscFunctionBegin;

  // the grid is stored in its own file
  char prefix[PETSC_MAX_PATH_LEN], output_dir[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineConfigPrefix(rdy, prefix));
  PetscCall(GetOutputDirectory(rdy, output_dir));
  if (!rdy->amr.is_refinement_on) {
    snprintf(filename, PETSC_MAX_PATH_LEN, "%s/%s-grid.h5", output_dir, prefix);
  } else {
    snprintf(filename, PETSC_MAX_PATH_LEN, "%s/%s-grid.r%d.h5", output_dir, prefix, rdy->amr.num_refinements);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteGrid(MPI_Comm comm, RDyMesh *mesh, PetscViewer viewer) {
  PetscFunctionBegin;

  PetscCall(PetscViewerHDF5PushGroup(viewer, "Domain"));
  PetscCall(VecView(mesh->output.vertices_xyz_norder, viewer));
  PetscCall(VecView(mesh->output.cell_conns_norder, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(PetscViewerHDF5PushGroup(viewer, "fields"));
  PetscCall(VecView(mesh->output.xc, viewer));
  PetscCall(VecView(mesh->output.yc, viewer));
  PetscCall(VecView(mesh->output.zc, viewer));
  PetscCall(VecView(mesh->output.area, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Accumulates solution variables for time-averaged output.
static PetscErrorCode AccumulateSolutionVariables(RDy rdy) {
  PetscFunctionBegin;
  PetscReal dt = rdy->dt;
  rdy->soln_accumulated_time += dt;
  PetscCall(VecAXPY(rdy->vec_soln_accum, dt, rdy->u_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Computes vec_soln_avg = vec_soln_accum / soln_accumulated_time.
static PetscErrorCode UpdateSolutionMean(RDy rdy) {
  PetscFunctionBegin;
  PetscCall(VecCopy(rdy->vec_soln_accum, rdy->vec_soln_avg));
  PetscCall(VecScale(rdy->vec_soln_avg, 1.0 / rdy->soln_accumulated_time));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Resets solution accumulation state after a write.
static PetscErrorCode ResetSolutionAccum(RDy rdy) {
  PetscFunctionBegin;
  rdy->soln_accumulated_time = 0.0;
  PetscCall(VecZeroEntries(rdy->vec_soln_accum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Copies the current instantaneous source snapshot into vec_src_inst.
static PetscErrorCode CopyInstantaneousSourceVariables(RDy rdy) {
  PetscFunctionBegin;
  Operator *op = rdy->operator;
  if (CeedEnabled()) {
    const CeedScalar *src_data;
    PetscCallCEED(CeedVectorGetArrayRead(op->ceed.ceed_src_inst, CEED_MEM_HOST, &src_data));
    PetscScalar *inst_data;
    PetscCall(VecGetArrayWrite(rdy->vec_src_inst, &inst_data));
    PetscInt n = rdy->mesh.num_owned_cells * op->num_components;
    for (PetscInt i = 0; i < n; ++i) inst_data[i] = src_data[i];
    PetscCall(VecRestoreArrayWrite(rdy->vec_src_inst, &inst_data));
    PetscCallCEED(CeedVectorRestoreArrayRead(op->ceed.ceed_src_inst, &src_data));
  } else {
    PetscCall(VecCopy(op->src_inst, rdy->vec_src_inst));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Accumulates source variables for time-averaged output.
static PetscErrorCode AccumulateSourceVariables(RDy rdy) {
  PetscFunctionBegin;
  PetscReal dt = rdy->dt;
  rdy->src_accumulated_time += dt;
  Operator *op = rdy->operator;
  if (CeedEnabled()) {
    PetscCallCEED(CeedVectorAXPY(op->ceed.ceed_src_accum, dt, op->ceed.ceed_src_inst));
  } else {
    PetscCall(VecAXPY(op->src_accum, dt, op->src_inst));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Computes vec_src_avg from the accumulated source data.
static PetscErrorCode UpdateSourceMean(RDy rdy) {
  PetscFunctionBegin;
  Operator *op              = rdy->operator;
  PetscReal           scale = 1.0 / rdy->src_accumulated_time;
  if (CeedEnabled()) {
    const CeedScalar *accum_data;
    PetscCallCEED(CeedVectorGetArrayRead(op->ceed.ceed_src_accum, CEED_MEM_HOST, &accum_data));
    PetscScalar *avg_data;
    PetscCall(VecGetArrayWrite(rdy->vec_src_avg, &avg_data));
    PetscInt n = rdy->mesh.num_owned_cells * op->num_components;
    for (PetscInt i = 0; i < n; ++i) avg_data[i] = accum_data[i] * scale;
    PetscCall(VecRestoreArrayWrite(rdy->vec_src_avg, &avg_data));
    PetscCallCEED(CeedVectorRestoreArrayRead(op->ceed.ceed_src_accum, &accum_data));
  } else {
    PetscCall(VecCopy(op->src_accum, rdy->vec_src_avg));
    PetscCall(VecScale(rdy->vec_src_avg, scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Resets source accumulation state after a write.
static PetscErrorCode ResetSourceAccum(RDy rdy) {
  PetscFunctionBegin;
  rdy->src_accumulated_time = 0.0;
  Operator *op              = rdy->operator;
  if (CeedEnabled()) {
    PetscCallCEED(CeedVectorSetValue(op->ceed.ceed_src_accum, 0.0));
  } else {
    PetscCall(VecZeroEntries(op->src_accum));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteOutputHDF5Metadata(RDy rdy, PetscViewer viewer) {
  PetscFunctionBegin;
  PetscCall(PetscViewerHDF5WriteGroup(viewer, "/metadata"));
  PetscCall(PetscViewerHDF5PushGroup(viewer, "/metadata"));
  PetscCall(PetscViewerHDF5WriteAttribute(viewer, NULL, "rdycore_version", PETSC_STRING, RDYCORE_GIT_HASH));
  PetscCall(PetscViewerHDF5PopGroup(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Writes a XDMF "heavy data" to an HDF5 file. The time is expressed in the
// units given in the configuration file.
static PetscErrorCode WriteXDMFHDF5Data(RDy rdy, PetscInt step, PetscReal time, char *h5_gridname) {
  PetscFunctionBegin;

  // Determine the output file name.
  char file_name[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineOutputFile(rdy, step, time, "h5", file_name));
  const char *units = TimeUnitAsString(rdy->config.time.unit);

  // create or append to a file depending on whether this step is the first in a dataset
  PetscInt      dataset   = step / rdy->config.output.output_interval;
  PetscFileMode file_mode = (dataset % rdy->config.output.batch_size == 0) ? FILE_MODE_WRITE : FILE_MODE_APPEND;

  RDyLogDetail(rdy, "Step %" PetscInt_FMT ": writing XDMF HDF5 output at t = %g %s to %s", step, time, units, file_name);

  // write solution data for the primary and auxiliary DMs
  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(rdy->comm, file_name, file_mode, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_XDMF));
  PetscCall(PetscViewerHDF5SetCollective(viewer, PETSC_TRUE));  // enable collective MPI-IO transfers

  if (file_mode == FILE_MODE_WRITE) {
    PetscCall(WriteOutputHDF5Metadata(rdy, viewer));
  }

  // write time-dependent solution and source fields
  char group_name[PETSC_MAX_PATH_LEN];
  snprintf(group_name, PETSC_MAX_PATH_LEN, "%" PetscInt_FMT " %E %s", step, time, units);
  PetscCall(PetscViewerHDF5PushGroup(viewer, group_name));
  // solution instantaneous (component_offset=0)
  PetscCall(WriteFilteredFieldData(rdy->dm, rdy->u_global, 0, &rdy->soln_fields, rdy->config.output, viewer, rdy->amr.num_refinements));
  // solution time-averaged (component_offset=0)
  PetscCall(WriteFilteredFieldData(rdy->dm, rdy->vec_soln_avg, 0, &rdy->soln_avg_fields, rdy->config.output, viewer, rdy->amr.num_refinements));
  // sources instantaneous (component_offset=0)
  PetscCall(WriteFilteredFieldData(rdy->dm, rdy->vec_src_inst, 0, &rdy->src_inst_fields, rdy->config.output, viewer, rdy->amr.num_refinements));
  // sources time-averaged (component_offset=0)
  PetscCall(WriteFilteredFieldData(rdy->dm, rdy->vec_src_avg, 0, &rdy->src_avg_fields, rdy->config.output, viewer, rdy->amr.num_refinements));
  // mean primitive variables (component_offset=1: skip Height_Mean)
  PetscCall(WriteFilteredFieldData(rdy->dm, rdy->vec_prim_vars_avg, 1, &rdy->prim_vars_fields, rdy->config.output, viewer, rdy->amr.num_refinements));
  // instantaneous primitive variables (component_offset=1: skip Height at component 0)
  PetscCall(
      WriteFilteredFieldData(rdy->dm, rdy->vec_prim_vars_inst, 1, &rdy->prim_vars_inst_fields, rdy->config.output, viewer, rdy->amr.num_refinements));
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  // on the first step ONLY, write the grid to its own file
  PetscBool write_grid = PETSC_FALSE;
  if (!rdy->amr.is_refinement_on) {
    if (step == 0) write_grid = PETSC_TRUE;
  } else {
    if (rdy->amr.last_refinement_level_outputted < rdy->amr.num_refinements) {
      write_grid                               = PETSC_TRUE;
      rdy->amr.last_refinement_level_outputted = rdy->amr.num_refinements;
    }
  }

  if (write_grid) {
    RDyLogDetail(rdy, "Step %" PetscInt_FMT ": writing XDMF HDF5 grid to %s", step, h5_gridname);

    PetscViewer grid_viewer;
    PetscCall(PetscViewerHDF5Open(rdy->comm, h5_gridname, FILE_MODE_WRITE, &grid_viewer));
    PetscCall(PetscViewerPushFormat(grid_viewer, PETSC_VIEWER_HDF5_XDMF));
    PetscCall(PetscViewerHDF5SetCollective(grid_viewer, PETSC_TRUE));
    PetscCall(WriteGrid(rdy->comm, &rdy->mesh, grid_viewer));
    PetscCall(PetscViewerPopFormat(grid_viewer));
    PetscCall(PetscViewerDestroy(&grid_viewer));
  }

  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Generates an XMDF "light data" file (.xmf) for the given step and time. The
/// time is expressed in the units specified in the .yaml input file.
static PetscErrorCode WriteXDMFXMFData(RDy rdy, PetscInt step, PetscReal time, char *gridname) {
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

  // write out mesh coordinates (stored in the "fields" group by WriteGrid)
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
  PetscCall(PetscFPrintf(rdy->comm, fp,
                         "      <Attribute Name=\"Area\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
                         "        <DataItem Dimensions=\"%" PetscInt_FMT "\" Format=\"HDF\">\n"
                         "          %s:/fields/Area\n"
                         "        </DataItem>\n"
                         "      </Attribute>\n",
                         mesh->num_cells_global, h5_gridname));

  // solution instantaneous (component_offset=0)
  PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &rdy->soln_fields, rdy->config.output, 0, rdy->mesh.num_cells_global));
  // solution time-averaged (component_offset=0)
  PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &rdy->soln_avg_fields, rdy->config.output, 0, rdy->mesh.num_cells_global));
  // sources instantaneous (component_offset=0)
  PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &rdy->src_inst_fields, rdy->config.output, 0, rdy->mesh.num_cells_global));
  // sources time-averaged (component_offset=0)
  PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &rdy->src_avg_fields, rdy->config.output, 0, rdy->mesh.num_cells_global));
  // mean primitive variables (component_offset=1: skip Height_Mean)
  PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &rdy->prim_vars_fields, rdy->config.output, 1, rdy->mesh.num_cells_global));
  // instantaneous primitive variables (component_offset=1: skip Height at component 0)
  PetscCall(
      WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &rdy->prim_vars_inst_fields, rdy->config.output, 1, rdy->mesh.num_cells_global));

  PetscCall(PetscFPrintf(rdy->comm, fp, "    </Grid>\n"));

  // write footer and close the file
  PetscCall(PetscFPrintf(rdy->comm, fp, "  </Domain>\n</Xdmf>\n"));
  PetscCall(PetscFClose(rdy->comm, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Copies the current operator primitive variables into vec_prim_vars_inst for output.
static PetscErrorCode CopyInstantaneousPrimitiveVariables(RDy rdy) {
  PetscFunctionBegin;
  Operator *op = rdy->operator;
  if (CeedEnabled()) {
    const CeedScalar *pv_data;
    PetscCallCEED(CeedVectorGetArrayRead(op->ceed.primitive_variables, CEED_MEM_HOST, &pv_data));
    PetscScalar *inst_data;
    PetscCall(VecGetArrayWrite(rdy->vec_prim_vars_inst, &inst_data));
    PetscInt n = rdy->mesh.num_owned_cells * op->num_components;
    for (PetscInt i = 0; i < n; ++i) inst_data[i] = pv_data[i];
    PetscCall(VecRestoreArrayWrite(rdy->vec_prim_vars_inst, &inst_data));
    PetscCallCEED(CeedVectorRestoreArrayRead(op->ceed.primitive_variables, &pv_data));
  } else {
    PetscCall(VecCopy(op->primitive_variables, rdy->vec_prim_vars_inst));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Accumulates dt-weighted primitive variables (h, u, v) into the running sum.
static PetscErrorCode AccumulatePrimitiveVariables(RDy rdy) {
  PetscFunctionBegin;

  PetscReal dt = rdy->dt;
  rdy->prim_vars_accumulated_time += dt;

  Operator *op = rdy->operator;
  if (CeedEnabled()) {
    PetscCallCEED(CeedVectorAXPY(op->ceed.primitive_variables_accum, dt, op->ceed.primitive_variables));
  } else {
    PetscCall(VecAXPY(op->primitive_variables_accum, dt, op->primitive_variables));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Resets the primitive variables accumulation state after a write.
static PetscErrorCode ResetPrimitiveVariablesAccum(RDy rdy) {
  PetscFunctionBegin;

  rdy->prim_vars_accumulated_time = 0.0;
  Operator *op                    = rdy->operator;
  if (CeedEnabled()) {
    PetscCallCEED(CeedVectorSetValue(op->ceed.primitive_variables_accum, 0.0));
  } else {
    PetscCall(VecZeroEntries(op->primitive_variables_accum));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Computes vec_prim_vars_avg = primitive_variables_accum / prim_vars_accumulated_time.
static PetscErrorCode UpdatePrimitiveVariablesMean(RDy rdy) {
  PetscFunctionBegin;

  Operator *op              = rdy->operator;
  PetscReal           scale = 1.0 / rdy->prim_vars_accumulated_time;

  if (CeedEnabled()) {
    const CeedScalar *accum_data;
    PetscCallCEED(CeedVectorGetArrayRead(op->ceed.primitive_variables_accum, CEED_MEM_HOST, &accum_data));
    PetscScalar *avg_data;
    PetscCall(VecGetArrayWrite(rdy->vec_prim_vars_avg, &avg_data));
    PetscInt n = rdy->mesh.num_owned_cells * op->num_components;
    for (PetscInt i = 0; i < n; ++i) avg_data[i] = accum_data[i] * scale;
    PetscCall(VecRestoreArrayWrite(rdy->vec_prim_vars_avg, &avg_data));
    PetscCallCEED(CeedVectorRestoreArrayRead(op->ceed.primitive_variables_accum, &accum_data));
  } else {
    PetscCall(VecCopy(op->primitive_variables_accum, rdy->vec_prim_vars_avg));
    PetscCall(VecScale(rdy->vec_prim_vars_avg, scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode WriteXDMFOutput(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy               rdy    = ctx;
  RDyOutputSection *output = &rdy->config.output;

  if (output->enable && (time != output->prev_output_time)) {
    PetscBool write_output = PETSC_FALSE;

    // accumulate solution if any solution *_Mean field is requested
    PetscBool need_soln_mean = AnySolnMeanRequested(output, &rdy->soln_avg_fields);
    if (need_soln_mean) {
      PetscCall(AccumulateSolutionVariables(rdy));
    }

    // accumulate primitive variables only if at least one prim var *_Mean field is requested
    PetscBool need_mean = AnyMeanPrimVarRequested(output, &rdy->prim_vars_fields);
    if (need_mean) {
      PetscCall(AccumulatePrimitiveVariables(rdy));
    }

    // accumulate source variables if any source *_Mean field is requested
    PetscBool need_src_mean = AnySrcMeanRequested(output, &rdy->src_avg_fields);
    if (need_src_mean) {
      PetscCall(AccumulateSourceVariables(rdy));
    }

    // check if it is time to output based on temporal interval
    if (output->time_interval > 0) {
      PetscReal dt   = ConvertTimeToSeconds(output->time_interval * 1.0, output->time_unit);
      PetscReal t    = time;
      PetscReal tmp  = fmod(t, dt);
      PetscReal diff = (tmp - dt);
      write_output   = (PetscAbsReal(tmp) < 10.0 * DBL_EPSILON || PetscAbsReal(diff) < 10.0 * DBL_EPSILON);
    }

    // check if it is time to output based on step interval
    if (output->output_interval > 0 && !write_output) {
      if (step % output->output_interval == 0) write_output = PETSC_TRUE;
    }

    // write output
    if (write_output) {
      // compute time-averaged solution for this output interval
      if (need_soln_mean) {
        PetscCall(UpdateSolutionMean(rdy));
      }

      // compute time-averaged primitive variables for this output interval
      if (need_mean) {
        PetscCall(UpdatePrimitiveVariablesMean(rdy));
      }

      // copy instantaneous source variables if any instantaneous source field is requested
      PetscBool need_src_inst = AnySrcInstRequested(output, &rdy->src_inst_fields);
      if (need_src_inst) {
        PetscCall(CopyInstantaneousSourceVariables(rdy));
      }

      // compute time-averaged source variables for this output interval
      if (need_src_mean) {
        PetscCall(UpdateSourceMean(rdy));
      }

      // copy instantaneous primitive variables if any instantaneous prim var field is requested
      PetscBool need_inst = AnyInstPrimVarRequested(output, &rdy->prim_vars_inst_fields);
      if (need_inst) {
        PetscCall(CopyInstantaneousPrimitiveVariables(rdy));
      }

      // save the time output was written
      output->prev_output_time = time;

      PetscReal t = ConvertTimeFromSeconds(time, rdy->config.time.unit);
      if (output->format == OUTPUT_XDMF) {
        char h5_gridname[PETSC_MAX_PATH_LEN];
        PetscCall(DetermineGridFile(rdy, h5_gridname));
        PetscCall(WriteXDMFHDF5Data(rdy, step, t, h5_gridname));
        PetscCall(WriteXDMFXMFData(rdy, step, t, h5_gridname));
      }

      // reset accumulation state for the next output interval
      if (need_soln_mean) {
        PetscCall(ResetSolutionAccum(rdy));
      }
      if (need_mean) {
        PetscCall(ResetPrimitiveVariablesAccum(rdy));
      }
      if (need_src_mean) {
        PetscCall(ResetSourceAccum(rdy));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
