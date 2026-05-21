#include <petscdmceed.h>
#include <petscviewerhdf5.h>  // note: includes hdf5.h
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>

// Returns PETSC_TRUE if at least one component of spec appears in output->fields.
// When skip_first_component is PETSC_TRUE, component 0 is excluded (used for prim vars
// where component 0 is Height, already written from u_global/soln_fields).
static PetscBool AnyFieldRequested(const RDyOutputSection *output, const SectionFieldSpec *spec, PetscBool skip_first_component) {
  if (!spec->num_fields) return PETSC_FALSE;
  PetscInt start = skip_first_component ? 1 : 0;
  for (PetscInt i = 0; i < output->fields_count; ++i) {
    for (PetscInt c = start; c < spec->num_field_components[0]; ++c) {
      if (!strcmp(output->fields[i], spec->field_component_names[0][c])) return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

// Writes components of global_vec whose names (from spec) match output.fields.
// When skip_first_component is PETSC_TRUE, component 0 is skipped unconditionally.
// If output.fields_count == 0, no components are written.
static PetscErrorCode WriteRequestedFields(DM dm, Vec global_vec, PetscBool skip_first_component, const SectionFieldSpec *spec,
                                           RDyOutputSection output, PetscViewer viewer, PetscInt num_refinements) {
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

  PetscInt component_offset = (PetscInt)skip_first_component;

  // count total components described by the spec (excluding the offset)
  PetscInt total_spec_comp = 0;
  for (PetscInt f = 0; f < spec->num_fields; ++f) total_spec_comp += spec->num_field_components[f];

  PetscCheck(bs > 0, comm, PETSC_ERR_USER, "WriteRequestedFields: vector block size must be positive (got %" PetscInt_FMT ")", bs);
  PetscCheck(n % bs == 0, comm, PETSC_ERR_USER,
             "WriteRequestedFields: local vector size (%" PetscInt_FMT ") is not divisible by block size (%" PetscInt_FMT ")", n, bs);
  PetscCheck(N % bs == 0, comm, PETSC_ERR_USER,
             "WriteRequestedFields: global vector size (%" PetscInt_FMT ") is not divisible by block size (%" PetscInt_FMT ")", N, bs);
  PetscCheck(total_spec_comp - component_offset <= bs, comm, PETSC_ERR_USER,
             "WriteRequestedFields: spec describes %" PetscInt_FMT " components beyond offset %" PetscInt_FMT
             ", exceeding vector block size %" PetscInt_FMT,
             total_spec_comp - component_offset, component_offset, bs);

  // allocate one component Vec per block slot and gather all components in a
  // single pass, then view only the requested ones (avoids repeated stride-gather
  // passes when multiple fields are requested)
  Vec *comp_vecs;
  PetscCall(PetscMalloc1(bs, &comp_vecs));
  for (PetscInt j = 0; j < bs; ++j) {
    PetscCall(VecCreateMPI(comm, n / bs, N / bs, &comp_vecs[j]));
  }
  PetscCall(VecStrideGatherAll(data_vec, comp_vecs, INSERT_VALUES));

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
        PetscCall(PetscObjectSetName((PetscObject)comp_vecs[c_global], name));
        PetscCall(VecView(comp_vecs[c_global], viewer));
      }
    }
  }

  for (PetscInt j = 0; j < bs; ++j) PetscCall(VecDestroy(&comp_vecs[j]));
  PetscCall(PetscFree(comp_vecs));

  PetscCall(VecDestroy(&data_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Writes XMF metadata for components of spec whose names match output.fields.
// When skip_first_component is PETSC_TRUE, component 0 is skipped unconditionally.
// If output.fields_count == 0, no metadata is written.
static PetscErrorCode WriteFieldMetadata(MPI_Comm comm, FILE *fp, const char *h5_basename, const char *group_name, const SectionFieldSpec *spec,
                                         RDyOutputSection output, PetscBool skip_first_component, PetscInt num_cells_global) {
  PetscFunctionBegin;

  if (output.fields_count == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt component_offset = (PetscInt)skip_first_component;
  PetscInt c_global         = 0;
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

// writes output fields that match the given set; writes no fields when fields_count == 0

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

/// Stages CEED instantaneous data into petsc_inst for HDF5 output.
/// No-op in PETSc-only mode (data is already in petsc_inst).
static PetscErrorCode StageInstantaneous(OutputVar *out) {
  PetscFunctionBegin;
  if (out->ceed_inst) {
    PetscInt n;
    PetscCall(VecGetLocalSize(out->petsc_inst, &n));
    const CeedScalar *src_data;
    PetscCallCEED(CeedVectorGetArrayRead(out->ceed_inst, CEED_MEM_HOST, &src_data));
    PetscScalar *dst_data;
    PetscCall(VecGetArrayWrite(out->petsc_inst, &dst_data));
    for (PetscInt i = 0; i < n; ++i) dst_data[i] = src_data[i];
    PetscCall(VecRestoreArrayWrite(out->petsc_inst, &dst_data));
    PetscCallCEED(CeedVectorRestoreArrayRead(out->ceed_inst, &src_data));
  }
  // PETSc path: petsc_inst already holds the current data — nothing to do.
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Accumulates a dt-weighted contribution into the accumulator.
static PetscErrorCode AccumulateOutputVar(OutputVar *out, PetscReal dt) {
  PetscFunctionBegin;
  out->accumulated_time += dt;
  if (out->ceed_accum) {
    PetscCallCEED(CeedVectorAXPY(out->ceed_accum, dt, out->ceed_inst));
  } else {
    PetscCall(VecAXPY(out->petsc_accum, dt, out->petsc_inst));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Scales petsc_accum by 1/accumulated_time, making it the time-averaged output.
/// CEED path: copies ceed_accum → petsc_accum with scaling.
/// PETSc path: scales petsc_accum in place.
static PetscErrorCode ComputeMean(OutputVar *out, MPI_Comm comm) {
  PetscFunctionBegin;
  PetscCheck(out->accumulated_time > 0.0, comm, PETSC_ERR_USER,
             "Cannot compute mean: accumulated time is zero (no timesteps accumulated before output)");
  PetscReal scale = 1.0 / out->accumulated_time;
  if (out->ceed_accum) {
    PetscInt n;
    PetscCall(VecGetLocalSize(out->petsc_accum, &n));
    const CeedScalar *accum_data;
    PetscCallCEED(CeedVectorGetArrayRead(out->ceed_accum, CEED_MEM_HOST, &accum_data));
    PetscScalar *avg_data;
    PetscCall(VecGetArrayWrite(out->petsc_accum, &avg_data));
    for (PetscInt i = 0; i < n; ++i) avg_data[i] = accum_data[i] * scale;
    PetscCall(VecRestoreArrayWrite(out->petsc_accum, &avg_data));
    PetscCallCEED(CeedVectorRestoreArrayRead(out->ceed_accum, &accum_data));
  } else {
    PetscCall(VecScale(out->petsc_accum, scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Zeros the accumulator(s) and resets accumulated_time.
static PetscErrorCode ResetOutputVar(OutputVar *out) {
  PetscFunctionBegin;
  out->accumulated_time = 0.0;
  if (out->ceed_accum) {
    PetscCallCEED(CeedVectorSetValue(out->ceed_accum, 0.0));
  }
  PetscCall(VecZeroEntries(out->petsc_accum));
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
  // solution instantaneous (uses soln_fields + u_global directly, component_offset=0)
  PetscCall(WriteRequestedFields(rdy->dm, rdy->u_global, PETSC_FALSE, &rdy->soln_fields, rdy->config.output, viewer, rdy->amr.num_refinements));
  // all OutputVar instances: avg (from petsc_accum, already scaled by ComputeMean) + inst (from petsc_inst)
  OutputVar *outputs[]   = {&rdy->soln_output, &rdy->prim_vars_output, &rdy->src_output};
  PetscInt   num_outputs = (PetscInt)(sizeof(outputs) / sizeof(outputs[0]));
  for (PetscInt oi = 0; oi < num_outputs; ++oi) {
    OutputVar *outvar = outputs[oi];
    PetscCall(WriteRequestedFields(rdy->dm, outvar->petsc_accum, outvar->skip_first_component, &outvar->avg_fields, rdy->config.output, viewer,
                                   rdy->amr.num_refinements));
    if (outvar->inst_fields.num_fields) {
      PetscCall(WriteRequestedFields(rdy->dm, outvar->petsc_inst, outvar->skip_first_component, &outvar->inst_fields, rdy->config.output, viewer,
                                     rdy->amr.num_refinements));
    }
  }
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
  PetscCall(
      WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &rdy->soln_fields, rdy->config.output, PETSC_FALSE, rdy->mesh.num_cells_global));
  // all OutputVar instances: avg (from petsc_accum) + inst (from petsc_inst)
  OutputVar *outputs[]   = {&rdy->soln_output, &rdy->prim_vars_output, &rdy->src_output};
  PetscInt   num_outputs = (PetscInt)(sizeof(outputs) / sizeof(outputs[0]));
  for (PetscInt oi = 0; oi < num_outputs; ++oi) {
    OutputVar *outvar = outputs[oi];
    PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &outvar->avg_fields, rdy->config.output, outvar->skip_first_component,
                                 rdy->mesh.num_cells_global));
    if (outvar->inst_fields.num_fields) {
      PetscCall(WriteFieldMetadata(rdy->comm, fp, h5_basename, time_group, &outvar->inst_fields, rdy->config.output, outvar->skip_first_component,
                                   rdy->mesh.num_cells_global));
    }
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

    PetscInt step;
    PetscCall(TSGetStepNumber(rdy->ts, &step));

    // accumulate all output vars if their corresponding mean fields are requested
    PetscBool need_soln_mean = AnyFieldRequested(output, &rdy->soln_output.avg_fields, rdy->soln_output.skip_first_component);
    PetscBool need_pv_mean   = AnyFieldRequested(output, &rdy->prim_vars_output.avg_fields, rdy->prim_vars_output.skip_first_component);
    PetscBool need_src_mean  = AnyFieldRequested(output, &rdy->src_output.avg_fields, rdy->src_output.skip_first_component);

    if (rdy->last_accumulated_step != step) {
      if (need_soln_mean) PetscCall(AccumulateOutputVar(&rdy->soln_output, rdy->dt));
      if (need_pv_mean) PetscCall(AccumulateOutputVar(&rdy->prim_vars_output, rdy->dt));
      if (need_src_mean) PetscCall(AccumulateOutputVar(&rdy->src_output, rdy->dt));
      rdy->last_accumulated_step = step;
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
      // compute time-averaged fields
      if (need_soln_mean) PetscCall(ComputeMean(&rdy->soln_output, rdy->comm));
      if (need_pv_mean) PetscCall(ComputeMean(&rdy->prim_vars_output, rdy->comm));
      if (need_src_mean) PetscCall(ComputeMean(&rdy->src_output, rdy->comm));

      // stage instantaneous data for prim vars and sources
      PetscBool need_pv_inst  = AnyFieldRequested(output, &rdy->prim_vars_output.inst_fields, rdy->prim_vars_output.skip_first_component);
      PetscBool need_src_inst = AnyFieldRequested(output, &rdy->src_output.inst_fields, rdy->src_output.skip_first_component);
      if (need_pv_inst) PetscCall(StageInstantaneous(&rdy->prim_vars_output));
      if (need_src_inst) PetscCall(StageInstantaneous(&rdy->src_output));

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
      if (need_soln_mean) PetscCall(ResetOutputVar(&rdy->soln_output));
      if (need_pv_mean) PetscCall(ResetOutputVar(&rdy->prim_vars_output));
      if (need_src_mean) PetscCall(ResetOutputVar(&rdy->src_output));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
