#include <private/rdycoreimpl.h>
#include <rdycore.h>

static const char* FlagString(PetscBool flag) { return flag ? "enabled" : "disabled"; }

static const char* BedFrictionString(RDyBedFriction model) {
  static const char* strings[3] = {"disabled", "Chezy", "Manning"};
  return strings[model];
}

static PetscErrorCode PrintPhysics(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Physics:");
  RDyLogDetail(rdy, "  Sediment model: %s", FlagString(rdy->config.sediment));
  RDyLogDetail(rdy, "  Salinity model: %s", FlagString(rdy->config.salinity));
  RDyLogDetail(rdy, "  Bed friction model: %s", BedFrictionString(rdy->config.bed_friction));
  PetscFunctionReturn(0);
}

static const char* SpatialString(RDySpatial method) {
  static const char* strings[2] = {"finite volume (FV)", "finite element (FE)"};
  return strings[method];
}

static const char* TemporalString(RDyTemporal method) {
  static const char* strings[3] = {"forward euler", "4th-order Runge-Kutta", "backward euler"};
  return strings[method];
}

static const char* RiemannString(RDyRiemann solver) {
  static const char* strings[2] = {"roe", "hllc"};
  return strings[solver];
}

static PetscErrorCode PrintNumerics(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Numerics:");
  RDyLogDetail(rdy, "  Spatial discretization: %s", SpatialString(rdy->config.spatial));
  RDyLogDetail(rdy, "  Temporal discretization: %s", TemporalString(rdy->config.temporal));
  RDyLogDetail(rdy, "  Riemann solver: %s", RiemannString(rdy->config.riemann));
  PetscFunctionReturn(0);
}

static const char* TimeUnitString(RDyTimeUnit unit) {
  static const char* strings[5] = {"seconds","minutes", "hours", "days", "months", "years"};
  return strings[unit];
}

static PetscErrorCode PrintTime(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Time:");
  RDyLogDetail(rdy, "  Final time: %g %s", rdy->config.final_time, TimeUnitString(rdy->config.time_unit));
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintRestart(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Restart:");
  if (rdy->config.restart_frequency > 0) {
    RDyLogDetail(rdy, "  File format: %s", rdy->config.restart_format);
    RDyLogDetail(rdy, "  Frequency: %d", rdy->config.restart_frequency);
  } else {
    RDyLogDetail(rdy, "  (disabled)");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PrintLogging(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Logging:");
  if (strlen(rdy->config.log_file)) {
    RDyLogDetail(rdy, "  Primary log file: %s", rdy->config.log_file);
  } else {
    RDyLogDetail(rdy, "  Primary log file: <stdout>");
  }
  PetscFunctionReturn(0);
}

// prints config information at the requested log level
PetscErrorCode PrintConfig(RDy rdy) {
  PetscFunctionBegin;

  RDyLogDetail(rdy, "==========================================================");
  RDyLogDetail(rdy, "RDycore (input read from %s)", rdy->config_file);
  RDyLogDetail(rdy, "==========================================================");

  PetscCall(PrintPhysics(rdy));
  PetscCall(PrintNumerics(rdy));
  PetscCall(PrintTime(rdy));
  PetscCall(PrintLogging(rdy));
  PetscCall(PrintRestart(rdy));

  PetscFunctionReturn(0);
}
