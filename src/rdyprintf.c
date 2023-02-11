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

static PetscErrorCode PrintPhysics(RDy rdy) {
  PetscFunctionBegin;
  RDyLogInfo(rdy, "-------\n");
  RDyLogInfo(rdy, "Physics\n");
  RDyLogInfo(rdy, "-------\n\n");
  RDyLogInfo(rdy, "Sediment model: %s\n", FlagString(rdy->sediment));
  RDyLogInfo(rdy, "Salinity model: %s\n", FlagString(rdy->salinity));
  RDyLogInfo(rdy, "Bed friction model: %s\n\n", BedFrictionString(rdy->bed_friction));
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

static PetscErrorCode PrintNumerics(RDy rdy) {
  PetscFunctionBegin;
  RDyLogInfo(rdy, "--------\n");
  RDyLogInfo(rdy, "Numerics\n");
  RDyLogInfo(rdy, "--------\n\n");
  RDyLogInfo(rdy, "Spatial discretization: %s\n", SpatialString(rdy->spatial));
  RDyLogInfo(rdy, "Temporal discretization: %s\n", TemporalString(rdy->temporal));
  RDyLogInfo(rdy, "Riemann solver: %s\n\n", RiemannString(rdy->riemann));
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

static PetscErrorCode PrintTime(RDy rdy) {
  PetscFunctionBegin;
  RDyLogInfo(rdy, "----\n");
  RDyLogInfo(rdy, "Time\n");
  RDyLogInfo(rdy, "----\n\n");
  RDyLogInfo(rdy, "Final time: %g %s\n", rdy->final_time, TimeUnitString(rdy->time_unit));
  RDyLogInfo(rdy, "\n");
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintRestart(RDy rdy) {
  PetscFunctionBegin;
  RDyLogInfo(rdy, "-------\n");
  RDyLogInfo(rdy, "Restart\n");
  RDyLogInfo(rdy, "-------\n\n");
  if (rdy->restart_frequency > 0) {
    RDyLogInfo(rdy, "Restart file format: %s\n", rdy->restart_format);
    RDyLogInfo(rdy, "Restart frequency: %d\n\n", rdy->restart_frequency);
  } else {
    RDyLogInfo(rdy, "(disabled)\n\n");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintLogging(RDy rdy) {
  PetscFunctionBegin;
  RDyLogInfo(rdy, "-------\n");
  RDyLogInfo(rdy, "Logging\n");
  RDyLogInfo(rdy, "-------\n\n");
  if (strlen(rdy->log_file)) {
    RDyLogInfo(rdy, "Primary log file: %s\n\n", rdy->log_file);
  } else {
    RDyLogInfo(rdy, "Primary log file: <stdout>\n\n");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RDyPrintf(RDy rdy) {
  PetscFunctionBegin;

  RDyLogInfo(rdy, "==========================================================\n");
  RDyLogInfo(rdy, "RDycore (input read from %s)\n", rdy->config_file);
  RDyLogInfo(rdy, "==========================================================\n\n");

  PetscCall(PrintPhysics(rdy));
  PetscCall(PrintNumerics(rdy));
  PetscCall(PrintRestart(rdy));
  PetscCall(PrintLogging(rdy));

  PetscFunctionReturn(0);
}

