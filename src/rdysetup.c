#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <rdycore.h>

#include <yaml.h>

#include <float.h>

//======================================================
// Traversing YAML mappings in PETSc's options database
//======================================================
//
// The design of the options database in PETSc does not currently allow for one
// to traverse items in YAML mappings. Instead, one must know the exact name of
// every item one wants to retrieve. This puts a large restriction on the types
// of YAML files that can be parsed. Here, we use libyaml (which is bundled with
// PETSc) directly to parse our configuration file.

// This is the maximum length of an identifier in the YAML file.
#define YAML_MAX_LEN 1024

// This enum defines the different sections in our configuration file.
typedef enum {
  NO_SECTION = 0,
  PHYSICS_SECTION,
  NUMERICS_SECTION,
  TIME_SECTION,
  RESTART_SECTION,
  GRID_SECTION,
  GRID_REGIONS_SECTION,
  GRID_SURFACES_SECTION,
  INITIAL_CONDITIONS_SECTION,
  BOUNDARY_CONDITIONS_SECTION,
  SOURCES_SECTION,
  FLOW_CONDITIONS_SECTION,
  SEDIMENT_CONDITIONS_SECTION,
  SALINITY_CONDITIONS_SECTION
} YamlSection;

// This type defines the state of our YAML parser, which allows us to determine
// where we are in the configuration file.
typedef struct {
  // section being currently parsed
  YamlSection section;
  // is the above section "open" (has its mapping start event been encountered)?
  PetscBool section_open;
  // name of subsection being currently parsed (if any, else blank string)
  char subsection[YAML_MAX_LEN];
  // name of parameter being current parsed (if any, else blank string)
  char parameter[YAML_MAX_LEN];
} YamlParserState;

// this converts a YAML string to a boolean
static PetscErrorCode ConvertToBool(MPI_Comm comm, const char *param,
                                    const char *str,
                                    PetscBool *val) {
  PetscFunctionBegin;

  PetscCheck(!strcmp(str, "true") || !strcmp(str, "false"), comm,
    PETSC_ERR_USER, "Invalid value for %s (bool expected, got '%s'",
    param, str);
  *val = !strcmp(str, "true");
  PetscFunctionReturn(0);
}

// this converts a YAML string to a real number
static PetscErrorCode ConvertToReal(MPI_Comm comm, const char *param,
                                    const char *str, PetscReal *val) {
  PetscFunctionBegin;

  char *endp;
  *val = strtod(str, &endp);
  PetscCheck(endp != str, comm, PETSC_ERR_USER,
    "Invalid real value for %s: %s", param, str);
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseTopLevel(yaml_event_t    *event,
                                    YamlParserState *state,
                                    RDy              rdy) {
  PetscFunctionBegin;

  // At the top level, we have only section names.
  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-section type encountered at top level!");

  // Set the current section based on the encountered value.
  const char *value = (const char*)(event->data.scalar.value);
  if (!strcmp(value, "physics")) {
    state->section = PHYSICS_SECTION;
  } else if (!strcmp(value, "numerics")) {
    state->section = NUMERICS_SECTION;
  } else if (!strcmp(value, "time")) {
    state->section = TIME_SECTION;
  } else if (!strcmp(value, "restart")) {
    state->section = RESTART_SECTION;
  } else if (!strcmp(value, "grid")) {
    state->section = GRID_SECTION;
  } else if (!strcmp(value, "initial_conditions")) {
    state->section = INITIAL_CONDITIONS_SECTION;
  } else if (!strcmp(value, "boundary_conditions")) {
    state->section = BOUNDARY_CONDITIONS_SECTION;
  } else if (!strcmp(value, "sources")) {
    state->section = SOURCES_SECTION;
  } else if (!strcmp(value, "flow_conditions")) {
    state->section = FLOW_CONDITIONS_SECTION;
  } else if (!strcmp(value, "sediment_conditions")) {
    state->section = SEDIMENT_CONDITIONS_SECTION;
  } else if (!strcmp(value, "salinity_conditions")) {
    state->section = SALINITY_CONDITIONS_SECTION;
  } else { // unrecognized section!
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
        "Unrecognized section in YAML config file: %s", value);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysics(yaml_event_t    *event,
                                   YamlParserState *state,
                                   RDy              rdy) {
  PetscFunctionBegin;

  // physics:
  //   sediment: <true|false>
  //   salinity: <true|false>
  //   bed_friction: <chezy|manning>

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in physics section!");

  const char *value = (const char*)(event->data.scalar.value);
  if (!strlen(state->subsection)) { // get the subsection name
    PetscCheck(!strcmp(value, "sediment") ||
               !strcmp(value, "salinity") ||
               !strcmp(value, "bed_friction"), rdy->comm, PETSC_ERR_USER,
      "Invalid subsection in physics: %s", value);
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else if (!strcmp(state->subsection, "sediment")) {
    PetscCheck(!strcmp(value, "enable"), rdy->comm, PETSC_ERR_USER,
      "Invalid parameter in physics.sediment: %s", value);
    if (!strlen(state->parameter)) { // parameter not set
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else { // parameter set, parse value
      if (!strcmp(state->parameter, "enable")) {
        ConvertToBool(rdy->comm, state->parameter, value, &rdy->sediment);
      }
    }
  } else if (!strcmp(state->subsection, "salinity")) {
    PetscCheck(!strcmp(value, "enable"), rdy->comm, PETSC_ERR_USER,
      "Invalid parameter in physics.salinity: %s", value);
    if (!strlen(state->parameter)) { // parameter not set
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else { // parameter set, parse value
      if (!strcmp(state->parameter, "enable")) {
        ConvertToBool(rdy->comm, state->parameter, value, &rdy->salinity);
      }
    }
  } else { // bed_friction
    if (!strlen(state->parameter)) { // parameter not set
      PetscCheck(!strcmp(value, "enable") ||
                 !strcmp(value, "model") ||
                 !strcmp(value, "coefficient"), rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in physics.bed_friction: %s", value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else { // parameter set, parse value
      if (!strcmp(state->parameter, "enable")) {
        PetscBool enable;
        ConvertToBool(rdy->comm, state->parameter, value, &enable);
        if (!enable) {
          rdy->bed_friction = BED_FRICTION_NONE; // disabled!
        }
      } else if (!strcmp(state->parameter, "model") &&
                 (rdy->bed_friction != BED_FRICTION_NONE)) {
        PetscCheck(!strcmp(value, "chezy") || !strcmp(value, "manning"),
                   rdy->comm, PETSC_ERR_USER, "Invalid bed_friction model: %s",
                   value);
        if (!strcmp(value, "chezy")) {
          rdy->bed_friction = BED_FRICTION_CHEZY;
        } else {
          rdy->bed_friction = BED_FRICTION_MANNING;
        }
      } else { // coefficient
        ConvertToReal(rdy->comm, state->parameter, value,
                      &rdy->bed_friction_coef);
      }
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseNumerics(yaml_event_t    *event,
                                    YamlParserState *state,
                                    RDy              rdy) {
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
    rdy->spatial = SPATIAL_FV;
  } else if (!strcasecmp(method, "fe")) {
    rdy->spatial = SPATIAL_FE;
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "invalid numerics.spatial: %s", method);
  }

  PetscCall(PetscOptionsGetString(NULL, NULL, "-numerics_temporal", method,
    sizeof(method), &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "numerics.temporal not provided!");
  if (!strcasecmp(method, "euler")) {
    rdy->temporal = TEMPORAL_EULER;
  } else if (!strcasecmp(method, "rk4")) {
    rdy->temporal = TEMPORAL_RK4;
  } else if (!strcasecmp(method, "beuler")) {
    rdy->temporal = TEMPORAL_BEULER;
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "invalid numerics.temporal: %s", method);
  }

  PetscCall(PetscOptionsGetString(NULL, NULL, "-numerics_riemann", method,
    sizeof(method), &present));
  PetscCheck(present, rdy->comm, PETSC_ERR_USER, "numerics.riemann not provided!");
  if (!strcasecmp(method, "roe")) {
    rdy->riemann = RIEMANN_ROE;
  } else if (!strcasecmp(method, "hll")) {
    rdy->riemann = RIEMANN_HLL;
  } else {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "invalid numerics.riemann: %s", method);
  }

  // Currently, only FV, EULER, and ROE are implemented.
  if (rdy->spatial != SPATIAL_FV) {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
      "Only the finite volume spatial method (FV) is currently implemented.");
  }
  if (rdy->temporal != TEMPORAL_EULER) {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
      "Only the forward euler temporal method (EULER) is currently implemented.");
  }
  if (rdy->riemann != RIEMANN_ROE) {
    PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
      "Only the roe riemann solver (ROE) is currently implemented.");
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseTime(yaml_event_t    *event,
                                YamlParserState *state,
                                RDy              rdy) {
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

static PetscErrorCode ParseRestart(yaml_event_t    *event,
                                   YamlParserState *state,
                                   RDy              rdy) {
  PetscFunctionBegin;

  // restart:
  //   format: <bin|h5>
  //   frequency: <value>
  //   unit: <nsteps|nminutes|nhours|ndays|nmonths|nyears>

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseGrid(yaml_event_t    *event,
                                YamlParserState *state,
                                RDy              rdy) {
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
  char grid_file[PETSC_MAX_PATH_LEN];
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
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseInitialConditions(yaml_event_t    *event,
                                             YamlParserState *state,
                                             RDy              rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseBoundaryConditions(yaml_event_t    *event,
                                              YamlParserState *state,
                                              RDy              rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSources(yaml_event_t    *event,
                                   YamlParserState *state,
                                   RDy              rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseFlowConditions(yaml_event_t    *event,
                                          YamlParserState *state,
                                          RDy              rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSedimentConditions(yaml_event_t    *event,
                                              YamlParserState *state,
                                              RDy              rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSalinityConditions(yaml_event_t    *event,
                                              YamlParserState *state,
                                              RDy              rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

// Handles a YAML event, populating the appropriate config info within rdy.
static PetscErrorCode HandleYamlEvent(yaml_event_t *event,
                                      YamlParserState *state,
                                      RDy rdy) {
  PetscFunctionBegin;

  if ((state->section != NO_SECTION) && !state->section_open &&
      (event->type == YAML_MAPPING_START_EVENT)) {
    // If we're inside a section that hasn't been opened and we encounter the
    // start of a mapping, open the section.
    state->section_open = PETSC_TRUE;
  } else if ((state->section != NO_SECTION) && !strlen(state->subsection) &&
      !strlen(state->parameter) && (event->type == YAML_MAPPING_END_EVENT)) {
    // If we're inside a opened section and we're not parsing a value or a
    // subsection, and we encounter the end of a mapping, close the section and
    // set it to NO_SECTION.
    state->section_open = PETSC_FALSE;
    state->section = NO_SECTION;
  } else {
    // Otherwise, dispatch the parser to the indicated section.
    switch (state->section) {
      case NO_SECTION:
        PetscCall(ParseTopLevel(event, state, rdy));
        break;
      case PHYSICS_SECTION:
        PetscCall(ParsePhysics(event, state, rdy));
        break;
      case NUMERICS_SECTION:
        PetscCall(ParseNumerics(event, state, rdy));
        break;
      case TIME_SECTION:
        PetscCall(ParseTime(event, state, rdy));
        break;
      case RESTART_SECTION:
        PetscCall(ParseRestart(event, state, rdy));
        break;
      case GRID_SECTION:
      case GRID_REGIONS_SECTION:
      case GRID_SURFACES_SECTION:
        PetscCall(ParseGrid(event, state, rdy));
        break;
      case INITIAL_CONDITIONS_SECTION:
        PetscCall(ParseInitialConditions(event, state, rdy));
        break;
      case BOUNDARY_CONDITIONS_SECTION:
        PetscCall(ParseBoundaryConditions(event, state, rdy));
        break;
      case SOURCES_SECTION:
        PetscCall(ParseSources(event, state, rdy));
        break;
      case FLOW_CONDITIONS_SECTION:
        PetscCall(ParseFlowConditions(event, state, rdy));
        break;
      case SEDIMENT_CONDITIONS_SECTION:
        PetscCall(ParseSedimentConditions(event, state, rdy));
        break;
      case SALINITY_CONDITIONS_SECTION:
        PetscCall(ParseSalinityConditions(event, state, rdy));
        break;
      default:
        // we ignore everything else in the file.
    }
  }

  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy, reading from the specified configuration
/// file.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  // Open the config file and set up a YAML parser for it.
  FILE *file;
  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_file(&parser, file);

  // Parse the file, handling each YAML "event" based on the parser state.
  YamlParserState state = {0};
  yaml_event_type_t event_type;
  do {
    yaml_event_t event;

    // Parse the next YAML "event" and handle any errors encountered.
    yaml_parser_parse(&parser, &event);
    if (parser.error != YAML_NO_ERROR) {
      char error_msg[1025];
      strncpy(error_msg, parser.problem, 1024);
      yaml_event_delete(&event);
      yaml_parser_delete(&parser);
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "%s", error_msg);
    }

    // Process the event, using it to populate our YAML data, and handle
    // any errors resulting from properly-formed YAML that doesn't conform
    // to our spec.
    PetscCall(HandleYamlEvent(&event, &state, rdy));

    // Discard the event and move on.
    event_type = event.type;
    yaml_event_delete(&event);
  } while (event_type != YAML_STREAM_END_EVENT);
  yaml_parser_delete(&parser);

  PetscFunctionReturn(0);
}
