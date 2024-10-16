#include <errno.h>
#include <private/rdycoreimpl.h>
#include <private/rdysweimpl.h>
#include <rdycore.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

extern PetscReal ConvertTimeToSeconds(PetscReal time, RDyTimeUnit time_unit);
extern PetscReal ConvertTimeFromSeconds(PetscReal time, RDyTimeUnit time_unit);

/// Returns the name of the output directory:
/// * "output"                        (single-run mode)
/// * "output/<ensemble-member-name>" (ensemble mode)
PetscErrorCode GetOutputDirectory(RDy rdy, char dir[PETSC_MAX_PATH_LEN]) {
  PetscFunctionBegin;
  static char output_dir[PETSC_MAX_PATH_LEN] = {0};
  if (!output_dir[0]) {
    if (rdy->config.ensemble.size > 1) {
      sprintf(output_dir, "output/%s", rdy->config.ensemble.members[rdy->ensemble_member_index].name);
    } else {
      strcpy(output_dir, "output");
    }
  }
  strncpy(dir, output_dir, PETSC_MAX_PATH_LEN - 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// creates the directory on rank 0 of the provided communicator, broadcasting
// results to the rest of the communicator's processes
PetscErrorCode CreateDirectory(MPI_Comm comm, const char *directory) {
  PetscFunctionBegin;
  PetscMPIInt rank, result_and_errno[2];
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    result_and_errno[0] = mkdir(directory, 0755);
    result_and_errno[1] = errno;
  }
  MPI_Bcast(&result_and_errno, 2, MPI_INT, 0, comm);
  int result = result_and_errno[0];
  int err_no = result_and_errno[1];
  PetscCheck((result == 0) || (err_no == EEXIST), comm, PETSC_ERR_USER, "Could not create directory: %s (errno = %d)", directory, err_no);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// creates the output directory if it doesn't exist
static PetscErrorCode CreateOutputDirectory(RDy rdy) {
  PetscFunctionBegin;

  char output_dir[PETSC_MAX_PATH_LEN];
  PetscCall(GetOutputDirectory(rdy, output_dir));
  RDyLogDebug(rdy, "Creating output directory %s...", output_dir);

  // create the output/ directory on global rank 0
  if (rdy->config.ensemble.size > 1) {
    PetscCall(CreateDirectory(rdy->global_comm, "output"));
  }
  PetscCall(CreateDirectory(rdy->comm, output_dir));
  MPI_Barrier(rdy->global_comm);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// generates a filename in the given directory
// * with the given prefix
// * with the given zero-padded index (padded to the given max index value)
// * and the given suffix
PetscErrorCode GenerateIndexedFilename(const char *directory, const char *prefix, PetscInt index, PetscInt max_index_val, const char *suffix,
                                       char *filename) {
  PetscFunctionBegin;
  int  num_digits = (int)(log10((double)max_index_val)) + 1;
  char fmt[16]    = {0};
  snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
  char ending[PETSC_MAX_PATH_LEN];
  snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, index, suffix);
  snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s%s", directory, prefix, ending);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Determines the appropriate output file name based on
/// * the desired file format as specified by rdy->config.output_format
/// * the time step index and simulation time
/// * the specified suffix
PetscErrorCode DetermineOutputFile(RDy rdy, PetscInt step, PetscReal time, const char *suffix, char *filename) {
  PetscFunctionBegin;

  char prefix[PETSC_MAX_PATH_LEN];
  PetscCall(DetermineConfigPrefix(rdy, prefix));

  // encode specific information into the filename based on its format
  char output_dir[PETSC_MAX_PATH_LEN];
  PetscCall(GetOutputDirectory(rdy, output_dir));
  if (rdy->config.output.format == OUTPUT_BINARY) {  // PETSc native binary format
    PetscCall(GenerateIndexedFilename(output_dir, prefix, step, rdy->config.time.max_step, suffix, filename));
  } else if (rdy->config.output.format == OUTPUT_XDMF) {
    if (!strcasecmp(suffix, "h5")) {  // XDMF "heavy" data
      if (rdy->config.output.batch_size == 1) {
        // output from each step gets its own HDF5 file
        snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s-%" PetscInt_FMT ".%s", output_dir, prefix, step, suffix);
      } else {
        // output data is grouped into batches of a fixed number of time steps
        PetscInt batch_size    = rdy->config.output.batch_size;
        PetscInt step_interval = rdy->config.output.step_interval;
        PetscInt batch         = step / step_interval / batch_size;
        PetscInt max_batch     = rdy->config.time.max_step / step_interval / batch_size;
        if (max_batch < 1) max_batch = 1;
        PetscCall(GenerateIndexedFilename(output_dir, prefix, batch, max_batch, suffix, filename));
      }
    } else if (!strcasecmp(suffix, "xmf")) {  // XDMF "light" data
      PetscCheck(!strcasecmp(suffix, "xmf"), rdy->comm, PETSC_ERR_USER, "Invalid suffix for XDMF output: %s", suffix);
      // encode the step into the filename with zero-padding based on the
      // maximum step number
      PetscCall(GenerateIndexedFilename(output_dir, prefix, step, rdy->config.time.max_step, suffix, filename));
    } else {
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported file suffix: %s", suffix);
    }
  } else if (rdy->config.output.format == OUTPUT_CGNS) {
    // the CGNS viewer handles its own batching and only needs a format string
    snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s-%%" PetscInt_FMT ".%s", output_dir, prefix, suffix);
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported output format specified.");
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyOutputViewer(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->output_vf) PetscCall(PetscViewerAndFormatDestroy(&rdy->output_vf));
  if (rdy->output_viewer) PetscCall(PetscViewerDestroy(&rdy->output_viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// this writes a log message for output at the proper interval
PetscErrorCode WriteOutputLogMessage(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy rdy = ctx;
  if (step % rdy->config.output.step_interval == 0) {
    static const char *formats[3] = {"binary", "XDMF", "CGNS"};
    const char        *format     = formats[rdy->config.output.format];
    const char        *units      = TimeUnitAsString(rdy->config.time.unit);
    RDyLogDetail(rdy, "Step %" PetscInt_FMT ": writing %s output at t = %g %s", step, format, time, units);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Creates a Viewer for visualization and sets up visualization monitoring.
// This function handles a lot of the disparities between what's available in
// PETSc's command line options and what exists in the API. We need this to be
// able to encode all of our settings in a single input file for reproducibility.
static PetscErrorCode CreateOutputViewer(RDy rdy) {
  PetscFunctionBegin;

  PetscViewerFormat format = PETSC_VIEWER_DEFAULT;
  if (rdy->config.output.enable) {
    if (rdy->config.output.step_interval) {
      RDyLogDebug(rdy, "Writing output every %" PetscInt_FMT " timestep(s)", rdy->config.output.step_interval);
    }

    if (rdy->config.output.time_interval) {
      const char *units = TimeUnitAsString(rdy->config.output.time_unit);
      RDyLogDebug(rdy, "Writing output every %" PetscInt_FMT " %s", rdy->config.output.time_interval, units);
    }

    switch (rdy->config.output.format) {
      case OUTPUT_NONE:
        // nothing to do here
        break;
      case OUTPUT_CGNS:
        // we've already configured this viewer in SetAdditionalOptions (see read_config_file.c)
        break;
      case OUTPUT_XDMF:
        // we don't actually use this viewer, so maybe this doesn't matter?
        PetscCall(PetscViewerCreate(rdy->comm, &rdy->output_viewer));
        PetscCall(PetscViewerSetType(rdy->output_viewer, PETSCVIEWERHDF5));
        format = PETSC_VIEWER_HDF5_XDMF;
        break;
      case OUTPUT_BINARY:
        PetscCall(PetscViewerCreate(rdy->comm, &rdy->output_viewer));
        PetscCall(PetscViewerSetType(rdy->output_viewer, PETSCVIEWERBINARY));
    }

    // apply any command-line option overrides
    if (rdy->output_viewer) {
      PetscCall(PetscViewerSetFromOptions(rdy->output_viewer));
      PetscCall(PetscViewerAndFormatCreate(rdy->output_viewer, format, &rdy->output_vf));
    }

    // set up solution monitoring
    if (rdy->config.output.format == OUTPUT_XDMF) {
      // we do our own special thing for XDMF
      PetscCall(TSMonitorSet(rdy->ts, WriteXDMFOutput, rdy, NULL));
    } else {
      // enable DETAIL logging for non-XDMF output
      if (rdy->config.logging.level >= LOG_DETAIL) {
        PetscCall(TSMonitorSet(rdy->ts, WriteOutputLogMessage, rdy, NULL));
      }
      // CGNS output is handled via the Options database. We need to set monitoring
      // for all formats that aren't XDMF or CGNS.
      if (rdy->config.output.format != OUTPUT_CGNS) {
        PetscCall(TSMonitorSet(rdy->ts, (PetscErrorCode(*)(TS, PetscInt, PetscReal, Vec, void *))TSMonitorSolution, rdy->output_vf, NULL));
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// this is called when -preload is set so we can get more accurate timings
// for the solver we're using
static PetscErrorCode CalibrateSolverTimers(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDebug(rdy, "Performing preload calibration...");

  // create a "preload" solution so we can advance one step
  Vec u_preload;
  PetscCall(VecDuplicate(rdy->u_global, &u_preload));
  PetscCall(VecCopy(rdy->u_global, u_preload));

  // take a single internal step to warm up the cache
  PetscCall(TSSetSolution(rdy->ts, u_preload));
  PetscCall(TSStep(rdy->ts));

  // clean up
  PetscCall(VecDestroy(&u_preload));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Advances the solution by the coupling interval specified in the input
/// configuration.
PetscErrorCode RDyAdvance(RDy rdy) {
  PetscFunctionBegin;

  // if we're at the start of the simulation, set up monitoring
  PetscInt step;
  PetscCall(TSGetStepNumber(rdy->ts, &step));
  if (step == 0) {
    PetscCall(CreateOutputDirectory(rdy));

    // create a viewer with the proper format for visualization output
    PetscCall(CreateOutputViewer(rdy));

    // initialize time series data
    PetscCall(InitTimeSeries(rdy));

    RDyLogDebug(rdy, "Running simulation...");
  }

  PetscReal time;
  PetscCall(TSGetTime(rdy->ts, &time));

  PetscReal interval           = ConvertTimeToSeconds(rdy->config.time.coupling_interval, rdy->config.time.unit);
  PetscReal next_coupling_time = time + interval;

  RDyLogDetail(rdy, "Advancing from t = %g to %g...", ConvertTimeFromSeconds(time, rdy->config.time.unit),
               ConvertTimeFromSeconds(next_coupling_time, rdy->config.time.unit));

  // if adaptive time is enabled, try to increase the dt
  RDyTimeAdaptiveSection *time_adap = &rdy->config.time.adaptive;
  if (time_adap->enable) {
    CourantNumberDiagnostics *cnum_diags = &rdy->courant_num_diags;

    // if previous courant number is valid
    if (cnum_diags->is_set) {
      // get current timestep
      PetscReal dt = rdy->dt;

      PetscReal factor = 0.0;
      if (cnum_diags->max_courant_num < time_adap->target_courant_number) {
        // timestep can be increased, so find the factor by which timestep can be increased
        factor = PetscMin(time_adap->target_courant_number / cnum_diags->max_courant_num, time_adap->max_increase_factor);

        // increase the timestep
        dt *= factor;

        // ensure the increase in timestep is less than the coupling timestep
        if (dt > interval) dt = interval;

      } else {
        // decrease the timestep
        factor = time_adap->target_courant_number / cnum_diags->max_courant_num;

        // decrease the timestep
        dt *= factor;
      }

      // if needed, log the Courant number
      if (rdy->config.logging.level >= LOG_DEBUG) {
        const char *units  = TimeUnitAsString(rdy->config.time.unit);
        PetscReal   dt_old = ConvertTimeFromSeconds(rdy->dt, rdy->config.time.unit);
        PetscReal   dt_new = ConvertTimeFromSeconds(dt, rdy->config.time.unit);
        RDyLogDebug(rdy, "Increasing dt from %f [%s] to %f [%s]", dt_old, units, dt_new, units);
      }

      // update the timestep
      rdy->dt = dt;
    }
  }

  PetscCall(TSSetMaxTime(rdy->ts, next_coupling_time));
  PetscCall(TSSetExactFinalTime(rdy->ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));
  PetscCall(TSSetSolution(rdy->ts, rdy->u_global));

  CourantNumberDiagnostics *courant_num_diags = &rdy->courant_num_diags;
  courant_num_diags->is_set                   = PETSC_FALSE;

  // advance the solution to the specified time (handling preloading if requested)
  PetscPreLoadBegin(PETSC_FALSE, "RDyAdvance solve");
  if (PetscPreLoadingOn) {
    PetscCall(CalibrateSolverTimers(rdy));
    PetscCall(TSSetTime(rdy->ts, time));
    PetscCall(TSSetStepNumber(rdy->ts, step));
  } else {
    PetscCall(TSSolve(rdy->ts, rdy->u_global));
  }
  PetscPreLoadEnd();

  if (time_adap->enable & !courant_num_diags->is_set) {
    PetscCall(SWEFindMaxCourantNumber(rdy));
  }

  // are we finished?
  PetscCall(TSGetTime(rdy->ts, &time));
  PetscReal final_time = ConvertTimeToSeconds(rdy->config.time.final_time, rdy->config.time.unit);
  if (time >= final_time) {
    // clean up
    PetscCall(DestroyOutputViewer(rdy));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Returns true if RDycore has satisfied its simulation termination criteria
/// (i.e. the simulation time exceeds the requeѕted final time or the maximum
///  number of time steps has been reached), false otherwise.
PetscBool RDyFinished(RDy rdy) {
  PetscFunctionBegin;
  PetscBool finished = PETSC_FALSE;
  PetscReal time;
  PetscCall(TSGetTime(rdy->ts, &time));
  PetscReal time_in_unit = ConvertTimeFromSeconds(time, rdy->config.time.unit);
  PetscInt  step;
  PetscCall(TSGetStepNumber(rdy->ts, &step));
  if ((time_in_unit >= rdy->config.time.final_time)) {
    finished = PETSC_TRUE;
  }
  PetscFunctionReturn(finished);
}

/// Retrieves the time units specified in the config file
PetscErrorCode RDyGetTimeUnit(RDy rdy, RDyTimeUnit *unit) {
  PetscFunctionBegin;
  *unit = rdy->config.time.unit;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves the simulation time in the desired units
PetscErrorCode RDyGetTime(RDy rdy, RDyTimeUnit unit, PetscReal *time) {
  PetscFunctionBegin;
  PetscReal t;
  PetscCall(TSGetTime(rdy->ts, &t));
  *time = ConvertTimeFromSeconds(t, unit);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves the internal time step size in the desired units
PetscErrorCode RDyGetTimeStep(RDy rdy, RDyTimeUnit unit, PetscReal *time_step) {
  PetscFunctionBegin;
  *time_step = ConvertTimeFromSeconds(rdy->dt, unit);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Converts the time t_from, expressed in units unit_from, to the time t_to,
/// expressed in units unit_to.
PetscErrorCode RDyConvertTime(RDyTimeUnit unit_from, PetscReal t_from, RDyTimeUnit unit_to, PetscReal *t_to) {
  PetscFunctionBegin;
  *t_to = ConvertTimeToSeconds(t_from, unit_from);
  *t_to = ConvertTimeFromSeconds(*t_to, unit_to);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Stores the step index in step.
PetscErrorCode RDyGetStep(RDy rdy, PetscInt *step) {
  PetscFunctionBegin;
  PetscCall(TSGetStepNumber(rdy->ts, step));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Retrieves the coupling interval in the desired units
PetscErrorCode RDyGetCouplingInterval(RDy rdy, RDyTimeUnit unit, PetscReal *interval) {
  PetscFunctionBegin;
  // convert the coupling interval from config file units to desired units
  PetscCall(RDyConvertTime(rdy->config.time.unit, rdy->config.time.coupling_interval, unit, interval));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets the coupling interval in the desired units
PetscErrorCode RDySetCouplingInterval(RDy rdy, RDyTimeUnit unit, PetscReal interval) {
  PetscFunctionBegin;
  // we store the coupling interval in the units specified in the config file
  rdy->config.time.coupling_interval = ConvertTimeToSeconds(interval, unit);
  rdy->config.time.coupling_interval = ConvertTimeFromSeconds(rdy->config.time.coupling_interval, rdy->config.time.unit);
  PetscFunctionReturn(PETSC_SUCCESS);
}
