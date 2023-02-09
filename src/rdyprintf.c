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
  RDyLog(rdy, "-------\n");
  RDyLog(rdy, "Physics\n");
  RDyLog(rdy, "-------\n\n");
  RDyLog(rdy, "Sediment model: %s\n", FlagString(rdy->sediment));
  RDyLog(rdy, "Salinity model: %s\n", FlagString(rdy->salinity));
  RDyLog(rdy, "Bed friction model: %s\n\n", BedFrictionString(rdy->bed_friction));
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
  RDyLog(rdy, "--------\n");
  RDyLog(rdy, "Numerics\n");
  RDyLog(rdy, "--------\n\n");
  RDyLog(rdy, "Spatial discretization: %s\n", SpatialString(rdy->spatial));
  RDyLog(rdy, "Temporal discretization: %s\n", TemporalString(rdy->temporal));
  RDyLog(rdy, "Riemann solver: %s\n\n", RiemannString(rdy->riemann));
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
  RDyLog(rdy, "----\n");
  RDyLog(rdy, "Time\n");
  RDyLog(rdy, "----\n\n");
  RDyLog(rdy, "Final time: %g %s\n", rdy->final_time, TimeUnitString(rdy->time_unit));
  RDyLog(rdy, "\n");
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintRestart(RDy rdy) {
  PetscFunctionBegin;
  RDyLog(rdy, "-------\n");
  RDyLog(rdy, "Restart\n");
  RDyLog(rdy, "-------\n\n");
  if (rdy->restart_frequency > 0) {
    RDyLog(rdy, "Restart file format: %s\n", rdy->restart_format);
    RDyLog(rdy, "Restart frequency: %d\n\n", rdy->restart_frequency);
  } else {
    RDyLog(rdy, "(disabled)\n\n");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintLogging(RDy rdy) {
  PetscFunctionBegin;
  RDyLog(rdy, "-------\n");
  RDyLog(rdy, "Logging\n");
  RDyLog(rdy, "-------\n\n");
  if (strlen(rdy->log_file)) {
    RDyLog(rdy, "Primary log file: %s\n\n", rdy->log_file);
  } else {
    RDyLog(rdy, "Primary log file: <stdout>\n\n");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RDyPrintf(RDy rdy) {
  PetscFunctionBegin;

  RDyLog(rdy, "==========================================================\n");
  RDyLog(rdy, "RDycore (input read from %s)\n", rdy->filename);
  RDyLog(rdy, "==========================================================\n\n");

  PetscCall(PrintPhysics(rdy));
  PetscCall(PrintNumerics(rdy));
  PetscCall(PrintRestart(rdy));
  PetscCall(PrintLogging(rdy));

  PetscFunctionReturn(0);
}

