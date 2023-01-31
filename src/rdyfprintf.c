#include <private/rdycoreimpl.h>
#include <rdycore.h>

static const char* FlagString(PetscBool flag) {
  return flag ? "enabled" : "disabled";
}

static const char* BedFrictionString(RDyBedFriction model) {
  if (model == BED_FRICTION_NONE) {
    return "disabled";
  } else if (model == BED_FRICTION_CHEZY) {
    return "Chezy";
  } else {
    return "Manning";
  }
}

static void PrintPhysics(FILE *file, RDy rdy) {
  fprintf(file, "---------\n");
  fprintf(file, " Physics\n");
  fprintf(file, "---------\n\n");
  fprintf(file, "Sediment model: %s\n", FlagString(rdy->sediment));
  fprintf(file, "Salinity model: %s\n", FlagString(rdy->salinity));
  fprintf(file, "Bed friction model: %s\n", BedFrictionString(rdy->bed_friction));
  fprintf(file, "\n");
}

static const char* SpatialString(RDySpatial method) {
  if (method == SPATIAL_FV) {
    return "finite volume method (FV)";
  } else { // (method == SPATIAL_FE)
    return "finite element method (FE)";
  }
}

static const char* TemporalString(RDyTemporal method) {
  if (method == TEMPORAL_EULER) {
    return "forward euler (EULER)";
  } else if (method == TEMPORAL_RK4) {
    return "4th-order Runge-Kutta (RK4)";
  } else { // (method == TEMPORAL_BEULER)
    return "backward euler (BEULER)";
  }
}

static const char* RiemannString(RDyRiemann solver) {
  if (solver == RIEMANN_ROE) {
    return "roe (ROE)";
  } else { // (solver == RIEMANN_HLL)
    return "hll (HLL)";
  }
}

static void PrintNumerics(FILE *file, RDy rdy) {
  fprintf(file, "----------\n");
  fprintf(file, " Numerics\n");
  fprintf(file, "----------\n\n");
  fprintf(file, "Spatial discretization: %s\n", SpatialString(rdy->spatial));
  fprintf(file, "Temporal discretization: %s\n", TemporalString(rdy->temporal));
  fprintf(file, "Riemann solver: %s\n", RiemannString(rdy->riemann));
  fprintf(file, "\n");
}

/// Writes configuration information to the given FILE.
PetscErrorCode RDyFprintf(FILE *file, RDy rdy) {
  PetscFunctionBegin;

  fprintf(file, "==========================================================\n");
  fprintf(file, "RDycore (input read from %s)\n", rdy->filename);
  fprintf(file, "==========================================================\n\n");

  PrintPhysics(file, rdy);
  PrintNumerics(file, rdy);

  PetscFunctionReturn(0);
}

