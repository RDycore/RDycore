#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>

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
    char      raster_file[PETSC_MAX_PATH_LEN];
    PetscBool have_raster_file;
    PetscCall(PetscOptionsGetString(NULL, NULL, "-grid_raster", raster_file, sizeof(raster_file), &have_raster_file));
    if (have_raster_file) {
      // FIXME: implement inactive cell logic here.
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ReadInitialConditions(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadNumerics(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadPhysics(RDy rdy) {
  PetscFunctionBegin;
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
  //   final_time_unit: <nsteps|nminutes|nhours|ndays|nmonths|nyears>
  //   max_step: <value>
  //   max_step_unit: <nminutes|nhours|ndays|nmonths|nyears>

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
