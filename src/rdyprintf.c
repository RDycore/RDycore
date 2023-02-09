#include <private/rdycoreimpl.h>
#include <rdycore.h>

static const char* FlagString(PetscBool flag) {
  return flag ? "enabled" : "disabled";
}

static const char* BedFrictionString(RDyBedFriction model) {
  static const char *strings[3] = {
    "disabled",
    "Chezy",
    "Manning"
  };
  return strings[model];
}

static void PrintPhysics(FILE *file, RDy rdy) {
  fprintf(file, "-------\n");
  fprintf(file, "Physics\n");
  fprintf(file, "-------\n\n");
  fprintf(file, "Sediment model: %s\n", FlagString(rdy->sediment));
  fprintf(file, "Salinity model: %s\n", FlagString(rdy->salinity));
  fprintf(file, "Bed friction model: %s\n", BedFrictionString(rdy->bed_friction));
  fprintf(file, "\n");
}

static const char* SpatialString(RDySpatial method) {
  static const char *strings[2] = {
    "finite volume (FV)",
    "finite element (FE)"
  };
  return strings[method];
}

static const char* TemporalString(RDyTemporal method) {
  static const char *strings[3] = {
    "forward euler",
    "4th-order Runge-Kutta",
    "backward euler"
  };
  return strings[method];
}

static const char* RiemannString(RDyRiemann solver) {
  static const char *strings[2] = {
    "roe",
    "hllc"
  };
  return strings[solver];
}

static void PrintNumerics(FILE *file, RDy rdy) {
  fprintf(file, "--------\n");
  fprintf(file, "Numerics\n");
  fprintf(file, "--------\n\n");
  fprintf(file, "Spatial discretization: %s\n", SpatialString(rdy->spatial));
  fprintf(file, "Temporal discretization: %s\n", TemporalString(rdy->temporal));
  fprintf(file, "Riemann solver: %s\n", RiemannString(rdy->riemann));
  fprintf(file, "\n");
}

static const char* TimeUnitString(RDyTimeUnit unit) {
  static const char *strings[5] = {
    "minutes",
    "hours",
    "days",
    "months",
    "years"
  };
  return strings[unit];
}

static void PrintTime(FILE *file, RDy rdy) {
  fprintf(file, "----\n");
  fprintf(file, "Time\n");
  fprintf(file, "----\n\n");
  fprintf(file, "Final time: %g %s\n", rdy->final_time, TimeUnitString(rdy->time_unit));
  fprintf(file, "\n");
}

static void PrintRestart(FILE *file, RDy rdy) {
  fprintf(file, "-------\n");
  fprintf(file, "Restart\n");
  fprintf(file, "-------\n\n");
  if (rdy->restart_frequency > 0) {
    fprintf(file, "Restart file format: %s\n", rdy->restart_format);
    fprintf(file, "Restart frequency: %d\n", rdy->restart_frequency);
  } else {
    fprintf(file, "(disabled)\n");
  }
  fprintf(file, "\n");
}

static void PrintLogging(FILE *file, RDy rdy) {
  fprintf(file, "-------\n");
  fprintf(file, "Logging\n");
  fprintf(file, "-------\n\n");
  if (strlen(rdy->log_file)) {
    fprintf(file, "Primary log file: %s\n", rdy->log_file);
  } else {
    fprintf(file, "Primary log file: <stdout>\n");
  }
  fprintf(file, "\n");
}

/// Writes configuration information to the given FILE.
static PetscErrorCode RDyFprintf(FILE *file, RDy rdy) {
  PetscFunctionBegin;

  fprintf(file, "==========================================================\n");
  fprintf(file, "RDycore (input read from %s)\n", rdy->filename);
  fprintf(file, "==========================================================\n\n");

  PrintPhysics(file, rdy);
  PrintNumerics(file, rdy);
  PrintRestart(file, rdy);
  PrintLogging(file, rdy);

  PetscFunctionReturn(0);
}

PetscErrorCode RDyPrintf(RDy rdy) {
  PetscFunctionBegin;

  if (rdy->rank == 0) {
    FILE *file;
    if (strlen(rdy->log_file)) {
      file = fopen(rdy->log_file, "r");
      PetscCheck(file, rdy->comm, PETSC_ERR_FILE_OPEN, "Invalid log file: %s",
        rdy->log_file);
    } else { // no file set--use stdout
      file = stdout;
    }
    PetscCall(RDyFprintf(file, rdy));
    if (file != stdout) {
      fclose(file);
    }
  }

  PetscFunctionReturn(0);
}

