#include <private/rdycoreimpl.h>
#include <rdycore.h>

static const char* FlagString(PetscBool flag) { return flag ? "enabled" : "disabled"; }

static PetscErrorCode PrintPhysics(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Physics:");
  RDyLogDetail(rdy, "  Flow:");
  RDyLogDetail(rdy, "    Bed friction: %s", FlagString(rdy->config.physics.flow.bed_friction));
  RDyLogDetail(rdy, "  Sediment model: %s", FlagString(rdy->config.physics.sediment));
  RDyLogDetail(rdy, "  Salinity model: %s", FlagString(rdy->config.physics.salinity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char* SpatialString(RDyNumericsSpatial method) {
  static const char* strings[2] = {"finite volume (FV)", "finite element (FE)"};
  return strings[method];
}

static const char* TemporalString(RDyNumericsTemporal method) {
  static const char* strings[3] = {"forward euler", "4th-order Runge-Kutta", "backward euler"};
  return strings[method];
}

static const char* RiemannString(RDyNumericsRiemann solver) {
  static const char* strings[2] = {"roe", "hllc"};
  return strings[solver];
}

static PetscErrorCode PrintNumerics(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Numerics:");
  RDyLogDetail(rdy, "  Spatial discretization: %s", SpatialString(rdy->config.numerics.spatial));
  RDyLogDetail(rdy, "  Temporal discretization: %s", TemporalString(rdy->config.numerics.temporal));
  RDyLogDetail(rdy, "  Riemann solver: %s", RiemannString(rdy->config.numerics.riemann));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char* TimeUnitString(RDyTimeUnit unit) {
  static const char* strings[6] = {"seconds", "minutes", "hours", "days", "months", "years"};
  return strings[unit];
}

static PetscErrorCode PrintTime(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Time:");
  RDyLogDetail(rdy, "  Final time: %g %s", rdy->config.time.final_time, TimeUnitString(rdy->config.time.unit));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintRestart(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Restart:");
  if (rdy->config.restart.interval > 0) {
    char format[12];
    if (rdy->config.restart.format == PETSC_VIEWER_NATIVE) {
      strcpy(format, "binary");
    } else {
      strcpy(format, "hdf5");
    }
    RDyLogDetail(rdy, "  File format: %s", format);
    RDyLogDetail(rdy, "  interval: %" PetscInt_FMT " ", rdy->config.restart.interval);
  } else {
    RDyLogDetail(rdy, "  (disabled)");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintLogging(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Logging:");
  if (strlen(rdy->config.logging.file)) {
    RDyLogDetail(rdy, "  Primary log file: %s", rdy->config.logging.file);
  } else {
    RDyLogDetail(rdy, "  Primary log file: <stdout>");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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

  PetscFunctionReturn(PETSC_SUCCESS);
}
