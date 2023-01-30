#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>

#include <float.h>

static PetscErrorCode ReadBoundaryConditions(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadFlow(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadGrid(RDy rdy) {
  PetscFunctionBegin;

  // grid:
  //   file: <path-to-file/mesh.exo> or <path-to-file/mesh.h5>
  //
  // OR
  //
  // grid:
  //   nx: <nx>
  //   ny: <ny>
  //   xmin: <xmin>
  //   xmax: <xmax>
  //   ymin: <ymin>
  //   ymax: <ymax>
  //   inactive: <path-to-raster-file/inactive.pbm>
  //
  // (see https://netpbm.sourceforge.net/doc/pbm.html for details)

  // Are we given a grid file?
  char      grid_file[PETSC_MAX_PATH_LEN];
  PetscBool have_grid_file;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-grid_file", grid_file, sizeof(grid_file), &have_grid_file));
  if (have_grid_file) {
    // Read the grid from the file.
    PetscCall(DMPlexCreateFromFile(rdy->comm, grid_file, "grid", PETSC_TRUE, &rdy->dm));
  } else {
    // Look for uniform grid parameters.
    PetscBool present;
    PetscInt  nx, ny;
    PetscReal xmin, xmax, ymin, ymax;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-grid_nx", &nx, &present));
    PetscCheck(present, rdy->comm, PETSC_ERR_USER, "grid.nx not provided!");
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-grid_ny", &ny, &present));
    PetscCheck(present, rdy->comm, PETSC_ERR_USER, "grid.ny not provided!");
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-grid_xmin", &xmin, &present));
    PetscCheck(present, rdy->comm, PETSC_ERR_USER, "grid.xmin not provided!");
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-grid_xmax", &xmax, &present));
    PetscCheck(present, rdy->comm, PETSC_ERR_USER, "grid.xmax not provided!");
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-grid_ymin", &ymin, &present));
    PetscCheck(present, rdy->comm, PETSC_ERR_USER, "grid.ymin not provided!");
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-grid_ymax", &ymax, &present));
    PetscCheck(present, rdy->comm, PETSC_ERR_USER, "grid.ymax not provided!");

    PetscCheck((xmax > xmin), rdy->comm, PETSC_ERR_USER, "grid.xmax <= grid.xmin!");
    PetscCheck((ymax > ymin), rdy->comm, PETSC_ERR_USER, "grid.ymax <= grid.ymin!");

    // Create a uniform grid with the given information.
    PetscReal lower[2] = {xmin, ymin};
    PetscReal upper[2] = {xmax, ymax};
    PetscCall(DMPlexCreateBoxMesh(rdy->comm, 2, PETSC_FALSE, NULL, lower, upper, NULL, PETSC_TRUE, &rdy->dm));

    // Look for (optional) raster information to punch out inactive cells.
    char file[PETSC_MAX_PATH_LEN];
    PetscBool have_file;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-grid_inactive", file,
      sizeof(file), &have_file));
    if (have_file) {
      // FIXME: implement inactive cell logic here.
    }

    // Now look for (optional) region information.
    // FIXME: Have to query Petsc options database for this.

    // Look for (optional) surface information.
    // FIXME: Have to query Petsc options database for this.

  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ReadInitialConditions(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadNumerics(RDy rdy) {
  PetscFunctionBegin;

  // numerics:
  //   spatial: <fv|fe>
  //   temporal: <euler|rk4|beuler>
  //   riemann: <roe|hll>

  char method[32];
  PetscBool present;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-numerics_spatial", method,
    sizeof(method), &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "numerics.spatial not provided!");
  if (!strcasecmp(method, "fv")) {
    rdy->spatial = FV;
  } else if (!strcasecmp(method, "fe")) {
    rdy->spatial = FE;
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "invalid numerics.spatial: %s", method);
  }

  PetscCall(PetscOptionsGetString(NULL, NULL, "-numerics_temporal", method,
    sizeof(method), &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "numerics.temporal not provided!");
  if (!strcasecmp(method, "euler")) {
    rdy->temporal = EULER;
  } else if (!strcasecmp(method, "rk4")) {
    rdy->temporal = RK4;
  } else if (!strcasecmp(method, "beuler")) {
    rdy->temporal = BEULER;
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "invalid numerics.temporal: %s", method);
  }

  PetscCall(PetscOptionsGetString(NULL, NULL, "-numerics_riemann", method,
    sizeof(method), &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "numerics.riemann not provided!");
  if (!strcasecmp(method, "roe")) {
    rdy->riemann = ROE;
  } else if (!strcasecmp(method, "hll")) {
    rdy->riemann = HLL;
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "invalid numerics.riemann: %s", method);
  }

  // Currently, only FV, EULER, and ROE are implemented.
  if (rdy->spatial != FV) {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
      "Only the finite volume spatial method (FV) is currently implemented.");
  }
  if (rdy->temporal != EULER) {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
      "Only the forward euler temporal method (EULER) is currently implemented.");
  }
  if (rdy->riemann != ROE) {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
      "Only the roe riemann solver (ROE) is currently implemented.");
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ReadPhysics(RDy rdy) {
  PetscFunctionBegin;

  // physics:
  //   sediment: <true|false>
  //   salinity: <true|false>
  //   bed_friction: <chezy|manning>

  PetscBool present;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-physics_sediment", &rdy->sediment,
    &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "physics.sediment not provided!");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-physics_salinity", &rdy->salinity,
    &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "physics.salinity not provided!");

  char model[32];
  PetscCall(PetscOptionsGetString(NULL, NULL, "-physics_bed_friction", model,
    sizeof(model), &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "physics.bed_friction not provided!");
  if (!strcasecmp(model, "chezy")) {
    rdy->bed_friction = CHEZY;
  } else if (!strcasecmp(model, "manning")) {
    rdy->bed_friction = MANNING;
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "invalid physics.bed_model: %s", model);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ReadRestart(RDy rdy) {
  PetscFunctionBegin;

  // restart:
  //   format: <bin|h5>
  //   frequency: <value>
  //   unit: <nsteps|nminutes|nhours|ndays|nmonths|nyears>

  PetscFunctionReturn(0);
}

static PetscErrorCode ReadSediments(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadSourcesAndSinks(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadTime(RDy rdy) {
  PetscFunctionBegin;

  // time:
  //   final_time: <value>
  //   unit: <nsteps|nminutes|nhours|ndays|nmonths|nyears>
  //   max_step: <value>

  // fetch values
  PetscBool present;
  PetscReal final_time;
  char      unit[32];
  PetscInt max_step;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-time_final_time", &final_time, &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "time.final_time not provided!");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-time_unit", unit, sizeof(unit), &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "time.unit not provided!");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-time_max_step", &max_step, &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "time.max_step not provided!");

  // validate values
  PetscCheck((final_time > 0.0), rdy->comm, PETSC_ERR_USER,
    "invalid final_time: %g\n", final_time);
  PetscCheck((max_step >= 0), rdy->comm, PETSC_ERR_USER,
    "invalid max_step: %d\n", max_step);
  PetscBool unit_valid = PETSC_TRUE;
  if (!strcasecmp(unit, "steps")) {
    rdy->max_step = (int)final_time;
    rdy->final_time = DBL_MAX;
  } else {
    rdy->max_step = max_step;
    PetscReal factor = -1.0;
    if (!strcasecmp(unit, "minutes")) {
      factor = 1.0;
    } else if (!strcasecmp(unit, "hours")) {
      factor = 60.0;
    } else if (!strcasecmp(unit, "days")) {
      factor = 24 * 60;
    } else if (!strcasecmp(unit, "months")) {
      factor = 30 * 24 * 60;
    } else if (!strcasecmp(unit, "years")) {
      factor = 365.25 * 24 * 60;
    } else {
      unit_valid = PETSC_FALSE;
    }
    // Times are stored internally in minutes.
    rdy->final_time = factor * final_time;
  }
  PetscCheck(unit_valid, rdy->comm, PETSC_ERR_USER, "invalid unit: %s", unit);

  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy after it has been configured.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  // read all relevant YAML "blocks"

  PetscCall(ReadBoundaryConditions(rdy));
  PetscCall(ReadFlow(rdy));
  PetscCall(ReadGrid(rdy));
  PetscCall(ReadInitialConditions(rdy));
  PetscCall(ReadNumerics(rdy));
  PetscCall(ReadPhysics(rdy));
  PetscCall(ReadRestart(rdy));
  PetscCall(ReadSediments(rdy));
  PetscCall(ReadSourcesAndSinks(rdy));
  PetscCall(ReadTime(rdy));

  PetscFunctionReturn(0);
}
