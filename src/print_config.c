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

static PetscErrorCode PrintPhysics(RDy rdy, RDyLogLevel level) {
  PetscFunctionBegin;
  RDyLog(rdy, level, "-------\n");
  RDyLog(rdy, level, "Physics\n");
  RDyLog(rdy, level, "-------\n\n");
  RDyLog(rdy, level, "Sediment model: %s\n", FlagString(rdy->sediment));
  RDyLog(rdy, level, "Salinity model: %s\n", FlagString(rdy->salinity));
  RDyLog(rdy, level, "Bed friction model: %s\n\n", BedFrictionString(rdy->bed_friction));
  PetscFunctionReturn(0);
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

static PetscErrorCode PrintNumerics(RDy rdy, RDyLogLevel level) {
  PetscFunctionBegin;
  RDyLog(rdy, level, "--------\n");
  RDyLog(rdy, level, "Numerics\n");
  RDyLog(rdy, level, "--------\n\n");
  RDyLog(rdy, level, "Spatial discretization: %s\n", SpatialString(rdy->spatial));
  RDyLog(rdy, level, "Temporal discretization: %s\n", TemporalString(rdy->temporal));
  RDyLog(rdy, level, "Riemann solver: %s\n\n", RiemannString(rdy->riemann));
  PetscFunctionReturn(0);
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

static PetscErrorCode PrintTime(RDy rdy, RDyLogLevel level) {
  PetscFunctionBegin;
  RDyLog(rdy, level, "----\n");
  RDyLog(rdy, level, "Time\n");
  RDyLog(rdy, level, "----\n\n");
  RDyLog(rdy, level, "Final time: %g %s\n", rdy->final_time, TimeUnitString(rdy->time_unit));
  RDyLog(rdy, level, "\n");
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintRestart(RDy rdy, RDyLogLevel level) {
  PetscFunctionBegin;
  RDyLog(rdy, level, "-------\n");
  RDyLog(rdy, level, "Restart\n");
  RDyLog(rdy, level, "-------\n\n");
  if (rdy->restart_frequency > 0) {
    RDyLog(rdy, level, "Restart file format: %s\n", rdy->restart_format);
    RDyLog(rdy, level, "Restart frequency: %d\n\n", rdy->restart_frequency);
  } else {
    RDyLog(rdy, level, "(disabled)\n\n");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintLogging(RDy rdy, RDyLogLevel level) {
  PetscFunctionBegin;
  RDyLog(rdy, level, "-------\n");
  RDyLog(rdy, level, "Logging\n");
  RDyLog(rdy, level, "-------\n\n");
  if (strlen(rdy->log_file)) {
    RDyLog(rdy, level, "Primary log file: %s\n\n", rdy->log_file);
  } else {
    RDyLog(rdy, level, "Primary log file: <stdout>\n\n");
  }
  PetscFunctionReturn(0);
}

// prints config information at the requested log level
PetscErrorCode PrintConfig(RDy rdy, RDyLogLevel level) {
  PetscFunctionBegin;

  RDyLog(rdy, level, "==========================================================\n");
  RDyLog(rdy, level, "RDycore (input read from %s)\n", rdy->config_file);
  RDyLog(rdy, level, "==========================================================\n\n");

  PetscCall(PrintPhysics(rdy, level));
  PetscCall(PrintNumerics(rdy, level));
  PetscCall(PrintRestart(rdy, level));
  PetscCall(PrintLogging(rdy, level));

  PetscFunctionReturn(0);
}

