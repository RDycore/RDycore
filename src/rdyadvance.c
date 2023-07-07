#include <errno.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// output directory name (relative to current working directory)
static const char *output_dir = "output";

extern PetscReal ConvertTimeToSeconds(PetscReal time, RDyTimeUnit time_unit);
extern PetscReal ConvertTimeFromSeconds(PetscReal time, RDyTimeUnit time_unit);

/// Creates the output directory if it doesn't exist.
PetscErrorCode CreateOutputDir(RDy rdy) {
  PetscFunctionBegin;

  RDyLogDebug(rdy, "Creating output directory %s...", output_dir);

  int result_and_errno[2];
  if (rdy->rank == 0) {
    result_and_errno[0] = mkdir(output_dir, 0755);
    result_and_errno[1] = errno;
  }
  MPI_Bcast(&result_and_errno, 2, MPI_INT, 0, rdy->comm);
  int result = result_and_errno[0];
  int err_no = result_and_errno[1];
  PetscCheck((result == 0) || (err_no == EEXIST), rdy->comm, PETSC_ERR_USER, "Could not create output directory: %s (errno = %d)", output_dir,
             err_no);

  PetscFunctionReturn(0);
}

// generates a filename in the output directory
// * with the given prefix
// * with the given zero-padded index (padded to the given max index value)
// * and the given suffix
static PetscErrorCode GenerateIndexedFilename(const char *prefix, PetscInt index, PetscInt max_index_val, const char *suffix, char *filename) {
  PetscFunctionBegin;
  int  num_digits = (int)(log10((double)max_index_val)) + 1;
  char fmt[16]    = {0};
  snprintf(fmt, 15, "-%%0%dd.%%s", num_digits);
  char ending[PETSC_MAX_PATH_LEN];
  snprintf(ending, PETSC_MAX_PATH_LEN - 1, fmt, index, suffix);
  snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s%s", output_dir, prefix, ending);
  PetscFunctionReturn(0);
}

/// Determines the appropriate output file name based on
/// * the desired file format as specified by rdy->config.output_format
/// * the time step index and simulation time
/// * the specified suffix
PetscErrorCode DetermineOutputFile(RDy rdy, PetscInt step, PetscReal time, const char *suffix, char *filename) {
  PetscFunctionBegin;

  size_t config_len = strlen(rdy->config_file);
  char   prefix[config_len + 1];
  memset(prefix, 0, sizeof(char) * (config_len + 1));
  char *p = strstr(rdy->config_file, ".yaml");
  if (!p) {  // could be .yml, I suppose (Windows habits die hard!)
    p = strstr(rdy->config_file, ".yml");
  }
  if (p) {
    size_t prefix_len = p - rdy->config_file;
    strncpy(prefix, rdy->config_file, prefix_len);
  } else {
    strcpy(prefix, rdy->config_file);
  }

  // encode specific information into the filename based on its format
  if (rdy->config.output.format == OUTPUT_BINARY) {  // PETSc native binary format
    PetscCall(GenerateIndexedFilename(prefix, step, rdy->config.time.max_step, suffix, filename));
  } else if (rdy->config.output.format == OUTPUT_XDMF) {
    if (!strcasecmp(suffix, "h5")) {  // XDMF "heavy" data
      if (rdy->config.output.batch_size == 1) {
        // output from each step gets its own HDF5 file
        snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s-%d.%s", output_dir, prefix, step, suffix);
      } else {
        // output data is grouped into batches of a fixed number of time steps
        PetscInt batch_size = rdy->config.output.batch_size;
        PetscInt freq       = rdy->config.output.frequency;
        PetscInt batch      = step / freq / batch_size;
        PetscInt max_batch  = rdy->config.time.max_step / freq / batch_size;
        if (max_batch < 1) max_batch = 1;
        PetscCall(GenerateIndexedFilename(prefix, batch, max_batch, suffix, filename));
      }
    } else if (!strcasecmp(suffix, "xmf")) {  // XDMF "light" data
      PetscCheck(!strcasecmp(suffix, "xmf"), rdy->comm, PETSC_ERR_USER, "Invalid suffix for XDMF output: %s", suffix);
      // encode the step into the filename with zero-padding based on the
      // maximum step number
      PetscCall(GenerateIndexedFilename(prefix, step, rdy->config.time.max_step, suffix, filename));
    } else {
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported file suffix: %s", suffix);
    }
  } else if (rdy->config.output.format == OUTPUT_CGNS) {
    // the CGNS viewer handles its own batching and only needs a format string
    snprintf(filename, PETSC_MAX_PATH_LEN - 1, "%s/%s-%%d.%s", output_dir, prefix, suffix);
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Unsupported output format specified.");
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyOutputViewer(RDy rdy) {
  PetscFunctionBegin;
  if (rdy->output_vf) PetscCall(PetscViewerAndFormatDestroy(&rdy->output_vf));
  if (rdy->output_viewer) PetscCall(PetscViewerDestroy(&rdy->output_viewer));
  PetscFunctionReturn(0);
}

// this writes a log message for output at the proper frequency
PetscErrorCode WriteOutputLogMessage(TS ts, PetscInt step, PetscReal time, Vec X, void *ctx) {
  PetscFunctionBegin;
  RDy rdy = ctx;
  if (step % rdy->config.output.frequency == 0) {
    static const char *formats[3] = {"binary", "XDMF", "CGNS"};
    const char        *format     = formats[rdy->config.output.format];
    const char        *units      = TimeUnitAsString(rdy->config.time.unit);
    RDyLogDetail(rdy, "Step %d: writing %s output at t = %g %s", step, format, time, units);
  }
  PetscFunctionReturn(0);
}

// Creates a Viewer for visualization and sets up visualization monitoring.
// This function handles a lot of the disparities between what's available in
// PETSc's command line options and what exists in the API. We need this to be
// able to encode all of our settings in a single input file for reproducibility.
static PetscErrorCode CreateOutputViewer(RDy rdy) {
  PetscFunctionBegin;

  PetscViewerFormat format = PETSC_VIEWER_DEFAULT;
  if (rdy->config.output.frequency) {
    RDyLogDebug(rdy, "Writing output every %d timestep(s)", rdy->config.output.frequency);
    switch (rdy->config.output.format) {
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

  PetscFunctionReturn(0);
}

// this is called when -preload is set so we can get more accurate timings
// for the solver we're using
static PetscErrorCode CalibrateSolverTimers(RDy rdy) {
  PetscFunctionBegin;

  // create a "preload" solution so we can advance one step
  Vec X_preload;
  PetscCall(VecDuplicate(rdy->X, &X_preload));
  PetscCall(VecCopy(rdy->X, X_preload));

  // set tolerances to make the calibration step cheaper
  SNES      snes;
  PetscReal r_tol;
  PetscCall(TSGetSNES(rdy->ts, &snes));
  PetscCall(SNESGetTolerances(snes, NULL, &r_tol, NULL, NULL, NULL));
  PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, .99, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  // take the step
  PetscCall(TSSetSolution(rdy->ts, X_preload));
  PetscCall(TSStep(rdy->ts));

  // reset the tolerances and clean up
  PetscCall(SNESSetTolerances(snes, PETSC_DEFAULT, r_tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(VecDestroy(&X_preload));

  PetscFunctionReturn(0);
}

/// Advances the solution by the coupling interval specified in the input
/// configuration.
PetscErrorCode RDyAdvance(RDy rdy) {
  PetscFunctionBegin;

  // if we're at the start of the simulation, set up monitoring
  if (rdy->step == 0) {
    PetscCall(CreateOutputDir(rdy));

    // set up monitoring functions for handling restarts and outputs
    // if (rdy->config.restart_frequency) {
    //   PetscCall(TSMonitorSet(rdy->ts, WriteRestartFiles, rdy, NULL));
    // }

    // create a viewer with the proper format for visualization output
    PetscCall(CreateOutputViewer(rdy));

    RDyLogDebug(rdy, "Running simulation...");
  }

  // advance the solution to the specified time
  PetscPreLoadBegin(PETSC_FALSE, "RDyAdvance solve");  // <-- enable with -preload true
  PetscCall(TSSetTime(rdy->ts, rdy->t));
  PetscReal interval = ConvertTimeToSeconds(rdy->config.time.coupling_interval, rdy->config.time.unit);
  RDyLogDetail(rdy, "Advancing from t = %g to %g...", ConvertTimeFromSeconds(rdy->t, rdy->config.time.unit),
               ConvertTimeFromSeconds(rdy->t + interval, rdy->config.time.unit));
  PetscCall(TSSetMaxTime(rdy->ts, rdy->t + interval));
  PetscCall(TSSetStepNumber(rdy->ts, rdy->step));
  PetscCall(TSSetTimeStep(rdy->ts, rdy->dt));
  PetscCall(TSSetSolution(rdy->ts, rdy->X));
  if (PetscPreLoadingOn) {
    PetscCall(CalibrateSolverTimers(rdy));
  } else {
    PetscCall(TSSolve(rdy->ts, rdy->X));
  }
  PetscPreLoadEnd();

  PetscCall(TSGetTime(rdy->ts, &rdy->t));
  PetscCall(TSGetStepNumber(rdy->ts, &rdy->step));

  // Are we finished?
  PetscReal final_time = ConvertTimeToSeconds(rdy->config.time.final_time, rdy->config.time.unit);
  if (rdy->t >= final_time) {
    // if we've overstepped the final time, interpolate backward
    if (rdy->t > final_time) {
      PetscCall(TSInterpolate(rdy->ts, final_time, rdy->X));
      PetscCall(TSSetTime(rdy->ts, final_time));
      PetscCall(TSMonitor(rdy->ts, -1, final_time, rdy->X));
    }

    // clean up
    PetscCall(DestroyOutputViewer(rdy));
  }

  PetscFunctionReturn(0);
}

/// Returns true if RDycore has satisfied its simulation termination criteria
/// (i.e. the simulation time exceeds the requeѕted final time or the maximum
///  number of time steps has been reached), false otherwise.
PetscBool RDyFinished(RDy rdy) {
  PetscFunctionBegin;
  PetscBool finished = PETSC_FALSE;
  PetscReal t        = ConvertTimeFromSeconds(rdy->t, rdy->config.time.unit);
  if ((t >= rdy->config.time.final_time) || (rdy->step >= rdy->config.time.max_step)) {
    finished = PETSC_TRUE;
  }
  PetscFunctionReturn(finished);
}

/// Stores the simulation time (in config-specified units) in time.
PetscErrorCode RDyGetTime(RDy rdy, PetscReal *time) {
  PetscFunctionBegin;
  *time = ConvertTimeFromSeconds(rdy->t, rdy->config.time.unit);
  PetscFunctionReturn(0);
}

/// Stores the internal time step size (in config-specified units) in time_step.
PetscErrorCode RDyGetTimeStep(RDy rdy, PetscReal *time_step) {
  PetscFunctionBegin;
  *time_step = ConvertTimeFromSeconds(rdy->dt, rdy->config.time.unit);
  PetscFunctionReturn(0);
}

/// Stores the step index in step.
PetscErrorCode RDyGetStep(RDy rdy, PetscInt *step) {
  PetscFunctionBegin;
  *step = rdy->step;
  PetscFunctionReturn(0);
}

/// Stores the coupling interval (in config-specified units) in interval.
PetscErrorCode RDyGetCouplingInterval(RDy rdy, PetscReal *interval) {
  PetscFunctionBegin;
  *interval = rdy->config.time.coupling_interval;
  PetscFunctionReturn(0);
}

/// Sets the coupling interval (in config-specified units).
PetscErrorCode RDySetCouplingInterval(RDy rdy, PetscReal interval) {
  PetscFunctionBegin;
  rdy->config.time.coupling_interval = interval;
  PetscFunctionReturn(0);
}
