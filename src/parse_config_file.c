#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>

#include <yaml.h>

#include <float.h>

// ======================
//  Two-Pass YAML Parser
// ======================
//
// The design of the options database in PETSc does not currently allow for one
// to traverse items in YAML mappings--one must know the exact name of every
// item one wants to retrieve. This puts a large restriction on the types of
// YAML files that can be parsed, so we've rolled our own.
//
// Here we've implemented a two-pass parser (using libyaml) to handle the
// RDycore configuration file whose specification can be found at
//
// https://rdycore.atlassian.net/wiki/spaces/PD/pages/24576001/RDycore+configuration+file
//
// We use two passes because items in some sections refer to items in others.
// Sections without such references are scanned on the first pass, and then
// the rest are scanned on the second pass. This ensures that the ordering of
// sections in the config file does not matter, which adheres to the Principle
// of Least Astonishment. Projects that ignore this principle tend to waste an
// astonishing amount of time.

// the maximum length of an identifier or value in the YAML file
#define YAML_MAX_LEN 1024

// This enum defines the different sections in our configuration file.
typedef enum {
  NO_SECTION = 0,
  PHYSICS_SECTION,
  PHYSICS_FLOW_SECTION,
  PHYSICS_SEDIMENT_SECTION,
  PHYSICS_SALINITY_SECTION,
  NUMERICS_SECTION,
  TIME_SECTION,
  RESTART_SECTION,
  LOGGING_SECTION,
  GRID_SECTION,
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
  // section currently being parsed
  YamlSection section;
  // are we inside the above section (has a mapping start event occurred)?
  PetscBool inside_section;
  // subsection currently being parsed (if any, else blank string)
  char subsection[YAML_MAX_LEN];
  // are we inside a subsection (mapping start event)?
  PetscBool inside_subsection;
  // name of parameter currently being parsed (if any, else blank string)
  char parameter[YAML_MAX_LEN];
} YamlParserState;

// This sets the index of a selection variable by matching the given string to
// one of a set of case-sensitive items. If the string doesn't match any of
// the items, the selection is set to -1.
static PetscErrorCode SelectItem(const char *str,
                                 PetscInt    num_items,
                                 const char *items[num_items],
                                 PetscInt    item_values[num_items],
                                 PetscInt   *selection) {
  PetscFunctionBegin;

  *selection = -1;
  for (PetscInt i = 0; i < num_items; ++i) {
    if (!strcmp(str, items[i])) {
      *selection = item_values[i];
      break;
    }
  }

  PetscFunctionReturn(0);
}

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
  *val = (PetscReal)(strtod(str, &endp));
  PetscCheck(endp != NULL, comm, PETSC_ERR_USER,
    "Invalid real value for %s: %s", param, str);
  PetscFunctionReturn(0);
}

// this converts a YAML string to an integer
static PetscErrorCode ConvertToInt(MPI_Comm comm, const char *param,
                                   const char *str, PetscInt *val) {
  PetscFunctionBegin;

  char *endp;
  *val = (PetscInt)(strtol(str, &endp, 10));
  PetscCheck(endp != NULL, comm, PETSC_ERR_USER,
    "Invalid integer value for %s: %s", param, str);

  PetscFunctionReturn(0);
}

// retrieves the index of a flow condition using its name
static PetscErrorCode FindFlowCondition(RDy rdy, const char *name,
                                        PetscInt *index) {
  PetscFunctionBegin;

  // Currently, we do a linear search on the name of the condition, which is O(N)
  // for N regions. If this is too slow, we can sort the conditions by name and
  // use binary search, which is O(log2 N).
  for (PetscInt i = 0; i < rdy->num_flow_conditions; ++i) {
    if (!strcmp(rdy->flow_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(0);
}

// retrieves the index of a sediment condition using its name
static PetscErrorCode FindSedimentCondition(RDy rdy, const char *name,
                                            PetscInt *index) {
  PetscFunctionBegin;

  // Currently, we do a linear search on the name of the condition, which is O(N)
  // for N regions. If this is too slow, we can sort the conditions by name and
  // use binary search, which is O(log2 N).
  for (PetscInt i = 0; i < rdy->num_sediment_conditions; ++i) {
    if (!strcmp(rdy->sediment_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(0);
}

// retrieves the index of a salinity condition using its name
static PetscErrorCode FindSalinityCondition(RDy rdy, const char *name,
                                            PetscInt *index) {
  PetscFunctionBegin;

  // Currently, we do a linear search on the name of the condition, which is O(N)
  // for N regions. If this is too slow, we can sort the conditions by name and
  // use binary search, which is O(log2 N).
  for (PetscInt i = 0; i < rdy->num_salinity_conditions; ++i) {
    if (!strcmp(rdy->salinity_conditions[i].name, name)) {
      *index = i;
      break;
    }
  }

  PetscFunctionReturn(0);
}

// opens a YAML section
static PetscErrorCode OpenSection(YamlParserState *state,
                                  RDy              rdy) {
  PetscFunctionBegin;
  if (state->section != NO_SECTION) {
    if (!state->inside_section) {
      state->inside_section = PETSC_TRUE;
    } else if (strlen(state->subsection) && !state->inside_subsection) {
      state->inside_subsection = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(0);
}

// closes a YAML section based on which one we're in
static PetscErrorCode CloseSection(YamlParserState *state,
                                   RDy              rdy) {
  PetscFunctionBegin;
  if (state->inside_subsection) { // exiting a subsection?
    if (state->section == FLOW_CONDITIONS_SECTION) {
      rdy->num_flow_conditions++;
      PetscCheck(rdy->num_flow_conditions <= MAX_NUM_CONDITIONS, rdy->comm,
        PETSC_ERR_USER, "Maximum number of flow conditions (%d) exceeded (increase MAX_NUM_CONDITIONS)",
        MAX_NUM_CONDITIONS);
    } else if (state->section == SEDIMENT_CONDITIONS_SECTION) {
      rdy->num_sediment_conditions++;
      PetscCheck(rdy->num_sediment_conditions <= MAX_NUM_CONDITIONS,
        rdy->comm, PETSC_ERR_USER,
        "Maximum number of sediment conditions (%d) exceeded (increase (MAX_NUM_CONDITIONS)",
        MAX_NUM_CONDITIONS);
    } else if (state->section == SALINITY_CONDITIONS_SECTION) {
      rdy->num_salinity_conditions++;
      PetscCheck(rdy->num_salinity_conditions <= MAX_NUM_CONDITIONS,
        rdy->comm, PETSC_ERR_USER,
        "Maximum number of salinity conditions (%d) exceeded (increase MAX_NUM_CONDITIONS)",
        MAX_NUM_CONDITIONS);
    }
    state->inside_subsection = PETSC_FALSE;
    state->subsection[0] = 0;
  } else if (state->inside_section) { // exiting a section?
    state->inside_section = PETSC_FALSE;
    switch (state->section) { // move up one section
      case PHYSICS_SEDIMENT_SECTION:
      case PHYSICS_SALINITY_SECTION:
      case PHYSICS_FLOW_SECTION:
        state->section = PHYSICS_SECTION;
        break;
      default:
        state->section = NO_SECTION;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseTopLevel(yaml_event_t    *event,
                                    YamlParserState *state,
                                    RDy              rdy) {
  PetscFunctionBegin;

  // At the top level, we have only section names.
  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-section type encountered at top level)!");

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
  } else if (!strcmp(value, "logging")) {
    state->section = LOGGING_SECTION;
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
  PetscInt selection;
  SelectItem(value, 3,
    (const char*[3]){"flow", "sediment", "salinity"},
    (PetscInt[3]){PHYSICS_FLOW_SECTION,
                  PHYSICS_SEDIMENT_SECTION,
                  PHYSICS_SALINITY_SECTION},
    &selection);
  PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
    "Invalid subsection in physics: %s", value);
  state->section = selection;
  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysicsFlow(yaml_event_t    *event,
                                       YamlParserState *state,
                                       RDy              rdy) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in physics.flow section!");
  const char *value = (const char*)(event->data.scalar.value);

  // check whether we are to descend into a subsection
  if (!strlen(state->subsection)) {
    if (!strlen(state->parameter) && !strcmp(value, "mode")) {
      // nope -- we are setting the flow mode
      strcpy(state->parameter, "mode");
    } else if (!strcmp(state->parameter, "mode")) {
      PetscInt selection;
      SelectItem(value, 2, (const char*[2]){"swe", "diffusion"},
        (PetscInt[2]){FLOW_SWE, FLOW_DIFFUSION}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid physics.flow.mode: %s", value);
      rdy->flow_mode = selection;
    } else {
      // okay, we are moving on to a subsection
      strncpy(state->subsection, value, YAML_MAX_LEN);
    }
  } else {
    // we should be inside a subsection that identifies a region.
    PetscCheck(state->inside_subsection, rdy->comm, PETSC_ERR_USER,
      "Invalid YAML in physics.%s", state->subsection);

    // currently, the only physics subsection we allow is bed_friction.
    PetscCheck(!strcmp(state->subsection, "bed_friction"), rdy->comm,
      PETSC_ERR_USER, "Invalid physics subsection: %s", state->subsection);

    if (!strlen(state->parameter)) { // parameter not set
      PetscCheck(!strcmp(value, "enable") ||
                 !strcmp(value, "model") ||
                 !strcmp(value, "coefficient"), rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in physics.bed_friction: %s", value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else { // parameter set, parse value
      if (!strcmp(state->parameter, "enable")) {
        PetscBool enable;
        PetscCall(ConvertToBool(rdy->comm, state->parameter, value, &enable));
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
        PetscCall(ConvertToReal(rdy->comm, state->parameter, value,
          &rdy->bed_friction_coef));
      }
      state->parameter[0] = 0; // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysicsSediment(yaml_event_t    *event,
                                           YamlParserState *state,
                                           RDy              rdy) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in physics.sediment section!");
  const char *value = (const char*)(event->data.scalar.value);

  PetscCheck(!strcmp(value, "enable"), rdy->comm, PETSC_ERR_USER,
    "Invalid parameter in physics.sediment: %s", value);
  if (!strlen(state->parameter)) { // parameter not set
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else { // parameter set, parse value
    if (!strcmp(state->parameter, "enable")) {
      PetscCall(ConvertToBool(rdy->comm, state->parameter, value,
        &rdy->sediment));
    }
    state->parameter[0] = 0; // clear parameter name
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysicsSalinity(yaml_event_t    *event,
                                           YamlParserState *state,
                                           RDy              rdy) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in physics.salinity section!");
  const char *value = (const char*)(event->data.scalar.value);

  PetscCheck(!strcmp(value, "enable"), rdy->comm, PETSC_ERR_USER,
    "Invalid parameter in physics.salinity: %s", value);
  if (!strlen(state->parameter)) { // parameter not set
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else { // parameter set, parse value
    if (!strcmp(state->parameter, "enable")) {
      PetscCall(ConvertToBool(rdy->comm, state->parameter, value,
        &rdy->salinity));
    }
    state->parameter[0] = 0; // clear parameter name
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

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in numerics section!");
  const char *value = (const char*)(event->data.scalar.value);

  if (!strlen(state->parameter)) { // parameter not set
    PetscInt selection;
    SelectItem(value, 3, (const char*[3]){"spatial", "temporal", "riemann"},
      (PetscInt[3]){0, 1, 2}, &selection);
    PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
      "Invalid parameter in numerics: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else { // parameter set, get value
    PetscInt selection;
    if (!strcmp(state->parameter, "spatial")) {
      SelectItem(value, 2, (const char*[2]){"fv", "fe"},
        (PetscInt[2]){SPATIAL_FV, SPATIAL_FE}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid numerics.spatial: %s", value);
      rdy->spatial = selection;
    } else if (!strcmp(state->parameter, "temporal")) {
      SelectItem(value, 3, (const char*[3]){"euler", "rk4", "beuler"},
        (PetscInt[3]){TEMPORAL_EULER, TEMPORAL_RK4, TEMPORAL_BEULER}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid numerics.temporal: %s", value);
    } else { // riemann
      SelectItem(value, 2, (const char*[2]){"roe", "hllc"},
        (PetscInt[2]){RIEMANN_ROE, RIEMANN_HLLC}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid numerics.riemann: %s", value);
      rdy->riemann = selection;
    }
    state->parameter[0] = 0; // clear parameter name
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

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in time section!");
  const char *value = (const char*)(event->data.scalar.value);

  if (!strlen(state->parameter)) { // parameter not set
    PetscInt selection;
    SelectItem(value, 3, (const char*[3]){"final_time", "unit", "max_step"},
      (PetscInt[3]){0, 1, 2}, &selection);
    PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
      "Invalid parameter in time: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else { // parameter set, get value
    if (!strcmp(state->parameter, "final_time")) {
      PetscCall(ConvertToReal(rdy->comm, state->parameter, value,
        &rdy->final_time));
      PetscCheck((rdy->final_time > 0.0), rdy->comm, PETSC_ERR_USER,
        "invalid time.final_time: %g\n", rdy->final_time);
    } else if (!strcmp(state->parameter, "unit")) {
      PetscInt selection;
      SelectItem(value, 5, (const char*[5]){"minutes", "hours", "days",
        "months", "years"},
        (PetscInt[5]){TIME_MINUTES, TIME_HOURS, TIME_DAYS,
                      TIME_MONTHS, TIME_YEARS}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid time.unit: %s", value);
      rdy->time_unit = selection;
    } else { // max_step
      PetscCall(ConvertToInt(rdy->comm, state->parameter, value,
                &rdy->max_step));
      PetscCheck((rdy->max_step >= 0), rdy->comm, PETSC_ERR_USER,
        "invalid time.max_step: %d\n", rdy->max_step);
    }
    state->parameter[0] = 0; // clear parameter name
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseRestart(yaml_event_t    *event,
                                   YamlParserState *state,
                                   RDy              rdy) {
  PetscFunctionBegin;

  // restart:
  //   format: <bin|h5>
  //   frequency: <value>

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in restart section!");
  const char *value = (const char*)(event->data.scalar.value);

  if (!strlen(state->parameter)) { // parameter not set
    PetscInt selection;
    SelectItem(value, 2, (const char*[2]){"format", "frequency"},
      (PetscInt[2]){0, 1}, &selection);
    PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
      "Invalid parameter in restart: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else { // parameter set, get value
    if (!strcmp(state->parameter, "format")) {
      PetscInt selection;
      SelectItem(value, 2, (const char*[2]){"bin", "h5"},
        (PetscInt[2]){0, 1}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid restart.format: %s", value);
      strncpy(rdy->restart_format, value, sizeof(rdy->restart_format));
    } else { // frequency
      PetscCall(ConvertToInt(rdy->comm, state->parameter, value,
        &rdy->restart_frequency));
      PetscCheck((rdy->restart_frequency > 0), rdy->comm, PETSC_ERR_USER,
        "Invalid restart.frequency: %d\n", rdy->restart_frequency);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseLogging(yaml_event_t    *event,
                                   YamlParserState *state,
                                   RDy              rdy) {
  PetscFunctionBegin;

  // logging:
  //   file: <path-to-log-file>
  //   level: <none|warning|info|detail|debug>

  const char *value = (const char*)(event->data.scalar.value);

  if (!strlen(state->parameter)) { // parameter not set
    if (!strcmp(value, "file") || !strcmp(value, "level")) {
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER,
        "invalid logging parameter: %s", value);
    }
  } else {
    if (!strcmp(state->parameter, "file")) {
      strncpy(rdy->log_file, value, PETSC_MAX_PATH_LEN);
    } else { // level
      PetscInt selection;
      SelectItem(value, 5,
        (const char*[5]){"none", "warning", "info", "detail", "debug"},
        (PetscInt[5]){LOG_NONE, LOG_WARNING, LOG_INFO, LOG_DETAIL, LOG_DEBUG},
        &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in logging.level: %s", value);
      rdy->log_level = selection;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseGrid(yaml_event_t    *event,
                                YamlParserState *state,
                                RDy              rdy) {
  PetscFunctionBegin;

  // grid:
  //   file: <path-to-msh-file>

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in grid section!");
  const char *value = (const char*)(event->data.scalar.value);

  if (!strlen(state->parameter)) { // parameter not set
    PetscCheck(!strcmp(value, "file"), rdy->comm, PETSC_ERR_USER,
      "Invalid grid parameter: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else {
    // Easy peasy--we just record the mesh file and leave.
    strncpy(rdy->mesh_file, value, PETSC_MAX_PATH_LEN);
    state->parameter[0] = 0; // clear parameter name
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseInitialConditions(yaml_event_t    *event,
                                             YamlParserState *state,
                                             RDy              rdy) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in initial_conditions section!");
  const char *value = (const char*)(event->data.scalar.value);

  // if we're not in a subsection, our parameter could be a single filename or
  // the name of a subsection
  if (!strlen(state->subsection)) {
    if (!strlen(state->parameter) && !strcmp(value, "file")) {
      strcpy(state->parameter, "file");
    } else if (!strcmp(state->parameter, "file")) {
      strncpy(rdy->initial_conditions_file, value, YAML_MAX_LEN);
    } else {
      // if we get here, we should NOT have an initial conditions file set.
      PetscCheck(!strlen(rdy->initial_conditions_file), rdy->comm, PETSC_ERR_USER,
        "Invalid initial_conditions! A file was specified, so no further "
        "parameters are allowed.");
      strncpy(state->subsection, value, YAML_MAX_LEN);
    }
  } else {
    // we should be inside a subsection that identifies a region.
    PetscCheck(state->inside_subsection, rdy->comm, PETSC_ERR_USER,
      "Invalid YAML in initial_conditions.%s", state->subsection);

    // What is the region's index?
    PetscInt region_index;
    PetscCall(ConvertToInt(rdy->comm, "region", state->subsection, &region_index));
    PetscCheck(region_index != -1, rdy->comm, PETSC_ERR_USER,
      "Invalid region in initial_conditions: %s", state->subsection);

    RDyCondition* ic = &rdy->initial_conditions[region_index];
    if (!strlen(state->parameter)) { // parameter name not set
      PetscInt selection;
      SelectItem(value, 3, (const char*[3]){"flow", "sediment", "salinity"},
        (PetscInt[3]){0, 1, 2}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in initial_conditions.%s: %s", state->subsection, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      PetscInt index;
      if (!strcmp(state->parameter, "flow")) {
        PetscCall(FindFlowCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid flow condition in initial_conditions.%s: %s",
          state->subsection, value);
        RDyFlowCondition *flow_cond = &rdy->flow_conditions[index];
        PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm,
          PETSC_ERR_USER,
          "initial flow condition %s for region %s is not of dirichlet type!",
          flow_cond->name, state->subsection);
        ic->flow = flow_cond;
      } else if (!strcmp(state->parameter, "sediment")) {
        PetscCall(FindSedimentCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid sediment condition in initial_conditions.%s: %s",
          state->subsection, value);
        RDySedimentCondition *sed_cond = &rdy->sediment_conditions[index];
        PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm,
          PETSC_ERR_USER,
          "initial sediment condition %s for region %s is not of dirichlet type!",
          sed_cond->name, state->subsection);
        ic->sediment = sed_cond;
      } else { // salinity
        PetscCall(FindSalinityCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid salinity condition in initial_conditions.%s: %s",
          state->subsection, value);
        RDySalinityCondition *sal_cond = &rdy->salinity_conditions[index];
        PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm,
          PETSC_ERR_USER,
          "initial salinity condition %s for region %s is not of dirichlet type!",
          sal_cond->name, state->subsection);
        ic->salinity = sal_cond;
      }
      state->parameter[0] = 0; // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseBoundaryConditions(yaml_event_t    *event,
                                              YamlParserState *state,
                                              RDy              rdy) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in boundary_conditions section!");
  const char *value = (const char*)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection that identifies a surface.
    PetscCheck(state->inside_subsection, rdy->comm, PETSC_ERR_USER,
      "Invalid YAML in boundary_conditions.%s", state->subsection);

    // What is the surface's index?
    PetscInt surface_index;
    PetscCall(ConvertToInt(rdy->comm, "surface", state->subsection, &surface_index));
    PetscCheck(surface_index != -1, rdy->comm, PETSC_ERR_USER,
      "Invalid surface in boundary_conditions: %s", state->subsection);

    RDyCondition* bc = &rdy->boundary_conditions[surface_index];
    if (!strlen(state->parameter)) { // parameter name not set
      PetscInt selection;
      SelectItem(value, 3, (const char*[3]){"flow", "sediment", "salinity"},
        (PetscInt[3]){0, 1, 2}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in boundary_conditions.%s: %s", state->subsection, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      PetscInt index;
      if (!strcmp(state->parameter, "flow")) {
        PetscCall(FindFlowCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid flow condition in boundary_conditions.%s: %s",
          state->subsection, value);
        bc->flow = &rdy->flow_conditions[index];
      } else if (!strcmp(state->parameter, "sediment")) {
        PetscCall(FindSedimentCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid sediment condition in boundary_conditions.%s: %s",
          state->subsection, value);
        bc->sediment = &rdy->sediment_conditions[index];
      } else { // salinity
        PetscCall(FindSalinityCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid salinity condition in boundary_conditions.%s: %s",
          state->subsection, value);
        bc->salinity = &rdy->salinity_conditions[index];
      }
      state->parameter[0] = 0; // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSources(yaml_event_t    *event,
                                   YamlParserState *state,
                                   RDy              rdy) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, rdy->comm, PETSC_ERR_USER,
    "Invalid YAML (non-scalar value encountered in sources section!");
  const char *value = (const char*)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection that identifies a region.
    PetscCheck(state->inside_subsection, rdy->comm, PETSC_ERR_USER,
      "Invalid YAML in sources.%s", state->subsection);

    // What is the region's index?
    PetscInt region_index;
    PetscCall(ConvertToInt(rdy->comm, "region", state->subsection, &region_index));
    PetscCheck(region_index != -1, rdy->comm, PETSC_ERR_USER,
      "Invalid region in sources: %s", state->subsection);

    RDyCondition* source = &rdy->sources[region_index];
    if (!strlen(state->parameter)) { // parameter name not set
      PetscInt selection;
      SelectItem(value, 3, (const char*[3]){"flow", "sediment", "salinity"},
        (PetscInt[3]){0, 1, 2}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in initial_conditions.%s: %s", state->subsection, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      PetscInt index;
      if (!strcmp(state->parameter, "flow")) {
        PetscCall(FindFlowCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid flow condition in sources.%s: %s", state->subsection, value);
        RDyFlowCondition *flow_cond = &rdy->flow_conditions[index];
        PetscCheck(flow_cond->type == CONDITION_DIRICHLET, rdy->comm,
          PETSC_ERR_USER,
          "source flow condition %s for region %s is not of dirichlet type!",
          flow_cond->name, state->subsection);
        source->flow = flow_cond;
      } else if (!strcmp(state->parameter, "sediment")) {
        PetscCall(FindSedimentCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid sediment condition in sources…%s: %s", state->subsection,
          value);
        RDySedimentCondition *sed_cond = &rdy->sediment_conditions[index];
        PetscCheck(sed_cond->type == CONDITION_DIRICHLET, rdy->comm,
          PETSC_ERR_USER,
          "source sediment condition %s for region %s is not of dirichlet type!",
          sed_cond->name, state->subsection);
        source->sediment = sed_cond;
      } else { // salinity
        PetscCall(FindSalinityCondition(rdy, value, &index));
        PetscCheck(index != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid salinity condition in sources.%s: %s", state->subsection,
          value);
        RDySalinityCondition *sal_cond = &rdy->salinity_conditions[index];
        PetscCheck(sal_cond->type == CONDITION_DIRICHLET, rdy->comm,
          PETSC_ERR_USER,
          "source salinity condition %s for region %s is not of dirichlet type!",
          sal_cond->name, state->subsection);
        source->salinity = sal_cond;
      }
      state->parameter[0] = 0; // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseFlowConditions(yaml_event_t    *event,
                                          YamlParserState *state,
                                          RDy              rdy) {
  PetscFunctionBegin;

  const char *value = (const char*)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection
    PetscCheck(state->inside_subsection, rdy->comm, PETSC_ERR_USER,
      "Invalid YAML in flow_conditions.%s", state->subsection);

    RDyFlowCondition* flow_cond = &rdy->flow_conditions[rdy->num_flow_conditions];
    if (!strlen(flow_cond->name)) { // condition name not set
      RDyAlloc(char, strlen(state->subsection), &flow_cond->name);
      strcpy((char*)flow_cond->name, state->subsection);
    } else if (!strlen(state->parameter)) { // parameter name not set
      PetscInt selection;
      SelectItem(value, 2, (const char*[2]){"type", "water_flux"},
        (PetscInt[2]){0, 1}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in flow condition %s: %s", flow_cond->name, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "type")) {
        PetscInt selection;
        SelectItem(value, 2, (const char*[2]){"dirichlet", "neumann"},
          (PetscInt[2]){CONDITION_DIRICHLET, CONDITION_NEUMANN}, &selection);
        PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid flow condition %s.type: %s", flow_cond->name, value);
        flow_cond->type = selection;
      } else { // water_flux
        PetscCall(ConvertToReal(rdy->comm, state->parameter, value,
          &flow_cond->water_flux));
      }
      state->parameter[0] = 0; // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSedimentConditions(yaml_event_t    *event,
                                              YamlParserState *state,
                                              RDy              rdy) {
  PetscFunctionBegin;

  const char *value = (const char*)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection
    PetscCheck(state->inside_subsection, rdy->comm, PETSC_ERR_USER,
      "Invalid YAML in sediment_conditions.%s", state->subsection);

    RDySedimentCondition* sed_cond = &rdy->sediment_conditions[rdy->num_sediment_conditions];
    if (!strlen(sed_cond->name)) { // condition name not set
      RDyAlloc(char, strlen(state->subsection), &sed_cond->name);
      strcpy((char*)sed_cond->name, state->subsection);
    } else if (!strlen(state->parameter)) { // parameter name not set
      PetscInt selection;
      SelectItem(value, 2, (const char*[2]){"type", "concentration"},
        (PetscInt[2]){0, 1}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in sediment condition %s: %s", sed_cond->name, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "type")) {
        PetscInt selection;
        SelectItem(value, 2, (const char*[2]){"dirichlet", "neumann"},
          (PetscInt[2]){CONDITION_DIRICHLET, CONDITION_NEUMANN}, &selection);
        PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid sediment condition %s.type: %s", sed_cond->name, value);
        sed_cond->type = selection;
      } else { // water_flux
        PetscCall(ConvertToReal(rdy->comm, state->parameter, value,
          &sed_cond->concentration));
      }
      state->parameter[0] = 0; // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSalinityConditions(yaml_event_t    *event,
                                              YamlParserState *state,
                                              RDy              rdy) {
  PetscFunctionBegin;

  const char *value = (const char*)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection
    PetscCheck(state->inside_subsection, rdy->comm, PETSC_ERR_USER,
      "Invalid YAML in salinity_conditions.%s", state->subsection);

    RDySalinityCondition* sal_cond = &rdy->salinity_conditions[rdy->num_salinity_conditions];
    if (!strlen(sal_cond->name)) { // condition name not set
      RDyAlloc(char, strlen(state->subsection), &sal_cond->name);
      strcpy((char*)sal_cond->name, state->subsection);
    } else if (!strlen(state->parameter)) { // parameter name not set
      PetscInt selection;
      SelectItem(value, 2, (const char*[2]){"type", "concentration"},
        (PetscInt[2]){0, 1}, &selection);
      PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
        "Invalid parameter in salinity condition %s: %s", sal_cond->name, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "type")) {
        PetscInt selection;
        SelectItem(value, 2, (const char*[2]){"dirichlet", "neumann"},
          (PetscInt[2]){CONDITION_DIRICHLET, CONDITION_NEUMANN}, &selection);
        PetscCheck(selection != -1, rdy->comm, PETSC_ERR_USER,
          "Invalid salinity condition %s.type: %s", sal_cond->name, value);
        sal_cond->type = selection;
      } else { // water_flux
        PetscCall(ConvertToReal(rdy->comm, state->parameter, value,
          &sal_cond->concentration));
      }
      state->parameter[0] = 0; // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

// This defines which pass the parser is working on.
typedef enum {
  FIRST_PASS,
  SECOND_PASS
} WhichPass;

// Handles a YAML event, populating the appropriate config info within rdy.
static PetscErrorCode HandleYamlEvent(WhichPass pass,
                                      yaml_event_t *event,
                                      YamlParserState *state,
                                      RDy rdy) {
  PetscFunctionBegin;

  // we don't need to do anything special at the beginning or the end of the
  // document
  if ((event->type == YAML_STREAM_START_EVENT) ||
      (event->type == YAML_DOCUMENT_START_EVENT) ||
      (event->type == YAML_DOCUMENT_END_EVENT) ||
      (event->type == YAML_STREAM_END_EVENT)) {
    PetscFunctionReturn(0);
  }

  // navigate sections via mapping starts and ends
  if (event->type == YAML_MAPPING_START_EVENT) {
    PetscCall(OpenSection(state, rdy));
  } else if (event->type == YAML_MAPPING_END_EVENT) {
    PetscCall(CloseSection(state, rdy));
  } else { // parse parameters in sections
    // Otherwise, dispatch the parser to the indicated section, based on which
    // pass we're performing.
    if (pass == FIRST_PASS) {
      // in the first pass, we scan everything with no dependencies
      switch (state->section) {
        case NO_SECTION:
          PetscCall(ParseTopLevel(event, state, rdy));
          break;
        case PHYSICS_SECTION:
          PetscCall(ParsePhysics(event, state, rdy));
          break;
        case PHYSICS_FLOW_SECTION:
          PetscCall(ParsePhysicsFlow(event, state, rdy));
          break;
        case PHYSICS_SEDIMENT_SECTION:
          PetscCall(ParsePhysicsSediment(event, state, rdy));
          break;
        case PHYSICS_SALINITY_SECTION:
          PetscCall(ParsePhysicsSediment(event, state, rdy));
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
        case LOGGING_SECTION:
          PetscCall(ParseLogging(event, state, rdy));
          break;
        case GRID_SECTION:
          PetscCall(ParseGrid(event, state, rdy));
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
          // we ignore everything else for now
      }
    } else { // second pass
      // in the second pass, we parse secions that refer to other sections
      switch (state->section) {
        case INITIAL_CONDITIONS_SECTION:
          PetscCall(ParseInitialConditions(event, state, rdy));
          break;
        case BOUNDARY_CONDITIONS_SECTION:
          PetscCall(ParseBoundaryConditions(event, state, rdy));
          break;
        case SOURCES_SECTION:
          PetscCall(ParseSources(event, state, rdy));
          break;
        default:
      }
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateQuadGrid(RDy rdy) {
  PetscFunctionBegin;

  PetscCheck((rdy->quadmesh.xmax > rdy->quadmesh.xmin), rdy->comm,
             PETSC_ERR_USER, "grid.xmax <= grid.xmin!");
  PetscCheck((rdy->quadmesh.ymax > rdy->quadmesh.ymin), rdy->comm,
             PETSC_ERR_USER, "grid.ymax <= grid.ymin!");

  // Create a uniform grid with the given information.
  PetscReal lower[2] = {rdy->quadmesh.xmin, rdy->quadmesh.ymin};
  PetscReal upper[2] = {rdy->quadmesh.xmax, rdy->quadmesh.ymax};
  PetscCall(DMPlexCreateBoxMesh(rdy->comm, 2, PETSC_FALSE, NULL, lower, upper,
                                NULL, PETSC_TRUE, &rdy->dm));

  // Look for (optional) raster information to punch out inactive cells.
  if (strlen(rdy->quadmesh.inactive_file)) {
    // FIXME: implement inactive cell logic here.
  }

  PetscFunctionReturn(0);
}

// initializes mesh region/surface data
static PetscErrorCode InitRegionsAndSurfaces(RDy rdy) {
  PetscFunctionBegin;

  // Get regions.
  DMLabel label;
  PetscCall(DMGetLabel(rdy->dm, "Cell Sets", &label));
  for (PetscInt region_id = 0; region_id < MAX_NUM_REGIONS; ++region_id) {
    RDyRegion *region = &rdy->regions[region_id];
    IS cell_is; // cell index space
    PetscCall(DMLabelGetStratumIS(label, region_id, &cell_is));
    if (cell_is) {
      PetscInt num_cells;
      PetscCall(ISGetLocalSize(cell_is, &num_cells));
      if (num_cells > 0) {
        rdy->region_ids[rdy->num_regions] = region_id;
        ++rdy->num_regions;
        region->num_cells = num_cells;
        PetscCall(RDyAlloc(PetscInt, region->num_cells, &region->cell_ids));
      }
      const PetscInt *cell_ids;
      PetscCall(ISGetIndices(cell_is, &cell_ids));
      memcpy(region->cell_ids, cell_ids, sizeof(PetscInt) * num_cells);
      PetscCall(ISRestoreIndices(cell_is, &cell_ids));
      PetscCall(ISDestroy(&cell_is));
    }
  }

  // Get surfaces.
  PetscCall(DMGetLabel(rdy->dm, "Face Sets", &label));
  for (PetscInt surface_id = 0; surface_id < MAX_NUM_SURFACES; ++surface_id) {
    RDySurface *surface = &rdy->surfaces[surface_id];
    IS edge_is; // edge index space
    PetscCall(DMLabelGetStratumIS(label, surface_id, &edge_is));
    if (edge_is) {
      PetscInt num_edges;
      PetscCall(ISGetLocalSize(edge_is, &num_edges));
      if (num_edges > 0) {
        rdy->surface_ids[rdy->num_surfaces] = surface_id;
        ++rdy->num_surfaces;
        surface->num_edges = num_edges;
        PetscCall(RDyAlloc(PetscInt, surface->num_edges, &surface->edge_ids));
      }
      const PetscInt *edge_ids;
      PetscCall(ISGetIndices(edge_is, &edge_ids));
      memcpy(surface->edge_ids, edge_ids, sizeof(PetscInt) * num_edges);
      PetscCall(ISRestoreIndices(edge_is, &edge_ids));
      PetscCall(ISDestroy(&edge_is));
    }
  }

  // make sure we have at least one region and surface
  PetscCheck(rdy->num_regions > 0, rdy->comm, PETSC_ERR_USER,
    "No regions were found in the grid!");
  PetscCheck(rdy->num_surfaces > 0, rdy->comm, PETSC_ERR_USER,
    "No surfaces were found in the grid!");

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseYaml(FILE *file, WhichPass pass, RDy rdy) {
  PetscFunctionBegin;

  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_file(&parser, file);

  // parse the file, handling each YAML "event" based on the parser state
  YamlParserState state = {0};
  yaml_event_type_t event_type;
  do {
    yaml_event_t event;

    // parse the next YAML "event" and handle any errors encountered
    yaml_parser_parse(&parser, &event);
    if (parser.error != YAML_NO_ERROR) {
      char error_msg[1025];
      strncpy(error_msg, parser.problem, 1024);
      yaml_event_delete(&event);
      yaml_parser_delete(&parser);
      PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "%s", error_msg);
    }

    // process the event, using it to populate our YAML data, and handle
    // any errors resulting from properly-formed YAML that doesn't conform
    // to our spec
    PetscCall(HandleYamlEvent(pass, &event, &state, rdy));

    // discard the event and move on
    event_type = event.type;
    yaml_event_delete(&event);
  } while (event_type != YAML_STREAM_END_EVENT);
  yaml_parser_delete(&parser);

  PetscFunctionReturn(0);
}

PetscErrorCode ParseConfigFile(FILE *file, RDy rdy) {
  PetscFunctionBegin;

  // do the first pass
  PetscCall(ParseYaml(file, FIRST_PASS, rdy));

  // create the grid from our specification
  if (strlen(rdy->mesh_file)) { // we are given a file
    PetscCall(DMPlexCreateFromFile(rdy->comm, rdy->mesh_file, "grid",
                                   PETSC_TRUE, &rdy->dm));
  } else { // we are asked to create a quad grid
    PetscCall(CreateQuadGrid(rdy));
  }

  // set up mesh regions and surfaces, reading them from our DMPlex object
  PetscCall(InitRegionsAndSurfaces(rdy));

  // now do the second pass and close the file
  rewind(file);
  ParseYaml(file, SECOND_PASS, rdy);

  PetscFunctionReturn(0);
}