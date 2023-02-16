#include <float.h>
#include <petscdmplex.h>
#include <private/rdycoreimpl.h>
#include <private/rdymemoryimpl.h>
#include <yaml.h>

// =============
//  YAML Parser
// =============
//
// The design of the options database in PETSc does not currently allow for one
// to traverse items in YAML mappings--one must know the exact name of every
// item one wants to retrieve. This puts a large restriction on the types of
// YAML files that can be parsed, so we've rolled our own.
//
// Here we've implemented a parser (using libyaml) to handle the RDycore
// configuration file whose specification can be found at
//
// https://rdycore.atlassian.net/wiki/spaces/PD/pages/24576001/RDycore+configuration+file
//
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
  // MPI communicator for processes parsing file
  MPI_Comm comm;
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
static PetscErrorCode SelectItem(const char *str, PetscInt num_items, const char *items[num_items], PetscInt item_values[num_items],
                                 PetscInt *selection) {
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
static PetscErrorCode ConvertToBool(MPI_Comm comm, const char *param, const char *str, PetscBool *val) {
  PetscFunctionBegin;

  PetscCheck(!strcmp(str, "true") || !strcmp(str, "false"), comm, PETSC_ERR_USER, "Invalid value for %s (bool expected, got '%s'", param, str);
  *val = !strcmp(str, "true");
  PetscFunctionReturn(0);
}

// this converts a YAML string to a real number
static PetscErrorCode ConvertToReal(MPI_Comm comm, const char *param, const char *str, PetscReal *val) {
  PetscFunctionBegin;

  char *endp;
  *val = (PetscReal)(strtod(str, &endp));
  PetscCheck(endp != NULL, comm, PETSC_ERR_USER, "Invalid real value for %s: %s", param, str);
  PetscFunctionReturn(0);
}

// this converts a YAML string to an integer
static PetscErrorCode ConvertToInt(MPI_Comm comm, const char *param, const char *str, PetscInt *val) {
  PetscFunctionBegin;

  char *endp;
  *val = (PetscInt)(strtol(str, &endp, 10));
  PetscCheck(endp != NULL, comm, PETSC_ERR_USER, "Invalid integer value for %s: %s", param, str);

  PetscFunctionReturn(0);
}

// opens a YAML section
static PetscErrorCode OpenSection(YamlParserState *state, RDyConfig *config) {
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
static PetscErrorCode CloseSection(YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;
  if (state->inside_subsection) {  // exiting a subsection?
    // handle some special cases in conditions sections first
    if (state->section == FLOW_CONDITIONS_SECTION) {
      config->num_flow_conditions++;
      PetscCheck(config->num_flow_conditions <= MAX_CONDITION_ID, state->comm, PETSC_ERR_USER,
                 "Maximum flow condition ID (%d) exceeded (increase MAX_CONDITION_ID)", MAX_CONDITION_ID);
    } else if (state->section == SEDIMENT_CONDITIONS_SECTION) {
      config->num_sediment_conditions++;
      PetscCheck(config->num_sediment_conditions <= MAX_CONDITION_ID, state->comm, PETSC_ERR_USER,
                 "Maximum number of sediment condition ID (%d) exceeded (increase (MAX_CONDITION_ID)", MAX_CONDITION_ID);
    } else if (state->section == SALINITY_CONDITIONS_SECTION) {
      config->num_salinity_conditions++;
      PetscCheck(config->num_salinity_conditions <= MAX_CONDITION_ID, state->comm, PETSC_ERR_USER,
                 "Maximum number of salinity condition ID (%d) exceeded (increase MAX_CONDITION_ID)", MAX_CONDITION_ID);
    }
    state->inside_subsection = PETSC_FALSE;
    state->subsection[0]     = 0;
  } else if (state->inside_section) {  // exiting a section?
    switch (state->section) {          // move up one section
      case PHYSICS_FLOW_SECTION:
      case PHYSICS_SEDIMENT_SECTION:
      case PHYSICS_SALINITY_SECTION:
        state->section = PHYSICS_SECTION;
        break;
      default:
        state->section        = NO_SECTION;
        state->inside_section = PETSC_FALSE;
        break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseTopLevel(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // At the top level, we have only section names.
  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-section type encountered at top level)!");

  // Set the current section based on the encountered value.
  const char *value = (const char *)(event->data.scalar.value);
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
  } else {  // unrecognized section!
    PetscCheck(PETSC_FALSE, state->comm, PETSC_ERR_USER, "Unrecognized section in YAML config file: %s", value);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysics(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // physics:
  //   sediment: <true|false>
  //   salinity: <true|false>
  //   bed_friction: <chezy|manning>

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-scalar value encountered in physics section!");

  const char *value = (const char *)(event->data.scalar.value);
  PetscInt    selection;
  SelectItem(value, 3, (const char *[3]){"flow", "sediment", "salinity"},
             (PetscInt[3]){PHYSICS_FLOW_SECTION, PHYSICS_SEDIMENT_SECTION, PHYSICS_SALINITY_SECTION}, &selection);
  PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid subsection in physics: %s", value);
  state->section = selection;
  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysicsFlow(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-scalar value encountered in physics.flow section!");
  const char *value = (const char *)(event->data.scalar.value);

  // check whether we should descend into a subsection
  if (!state->inside_subsection) {
    if (!strlen(state->parameter)) {
      if (!strcmp(value, "bed_friction")) {
        strncpy(state->subsection, value, YAML_MAX_LEN);
      } else {
        PetscCheck(!strcmp(value, "mode"), state->comm, PETSC_ERR_USER, "Invalid physics.flow parameter: %s", value);
        strcpy(state->parameter, "mode");
      }
    } else if (!strcmp(state->parameter, "mode")) {
      PetscInt selection;
      SelectItem(value, 2, (const char *[2]){"swe", "diffusion"}, (PetscInt[2]){FLOW_SWE, FLOW_DIFFUSION}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid physics.flow.mode: %s", value);
      config->flow_mode   = selection;
      state->parameter[0] = 0;  // clear parameter name
    }
  } else {
    // we should be inside a subsection that identifies a region.
    PetscCheck(state->inside_subsection, state->comm, PETSC_ERR_USER, "Invalid YAML in physics.flow.%s", state->subsection);

    // currently, the only physics flow subsection we allow is bed_friction.
    PetscCheck(!strcmp(state->subsection, "bed_friction"), state->comm, PETSC_ERR_USER, "Invalid physics.flow subsection: %s", state->subsection);

    if (!strlen(state->parameter)) {  // parameter not set
      PetscInt selection;
      SelectItem(value, 3, (const char *[3]){"enable", "model", "coefficient"}, (PetscInt[3]){0, 1, 2}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in physics.flow.bed_friction: %s", value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {  // parameter set, parse value
      if (!strcmp(state->parameter, "enable")) {
        PetscBool enable;
        PetscCall(ConvertToBool(state->comm, state->parameter, value, &enable));
        if (!enable) {
          config->bed_friction = BED_FRICTION_NONE;  // disabled!
        }
      } else if (!strcmp(state->parameter, "model") && (config->bed_friction != BED_FRICTION_NONE)) {
        PetscInt selection;
        SelectItem(value, 2, (const char *[2]){"chezy", "manning"}, (PetscInt[2]){BED_FRICTION_CHEZY, BED_FRICTION_MANNING}, &selection);
        PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid bed_friction model: %s", value);
      } else {  // coefficient
        PetscCall(ConvertToReal(state->comm, state->parameter, value, &config->bed_friction_coef));
      }
      state->parameter[0] = 0;  // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysicsSediment(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER,
             "Invalid YAML (non-scalar value encountered in physics.sediment section!");
  const char *value = (const char *)(event->data.scalar.value);

  if (!strlen(state->parameter)) {  // parameter not set
    PetscInt selection;
    SelectItem(value, 2, (const char *[2]){"enable", "d50"}, (PetscInt[2]){0, 1}, &selection);
    PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in physics.sediment: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else {  // parameter set, parse value
    if (!strcmp(state->parameter, "enable")) {
      PetscCall(ConvertToBool(state->comm, state->parameter, value, &config->sediment));
    } else {  // d50
      // FIXME
    }
    state->parameter[0] = 0;  // clear parameter name
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParsePhysicsSalinity(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER,
             "Invalid YAML (non-scalar value encountered in physics.salinity section!");
  const char *value = (const char *)(event->data.scalar.value);

  if (!strlen(state->parameter)) {  // parameter not set
    PetscCheck(!strcmp(value, "enable"), state->comm, PETSC_ERR_USER, "Invalid parameter in physics.salinity: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else {  // parameter set, parse value
    if (!strcmp(state->parameter, "enable")) {
      PetscCall(ConvertToBool(state->comm, state->parameter, value, &config->salinity));
    }
    state->parameter[0] = 0;  // clear parameter name
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseNumerics(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // numerics:
  //   spatial: <fv|fe>
  //   temporal: <euler|rk4|beuler>
  //   riemann: <roe|hll>

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-scalar value encountered in numerics section!");
  const char *value = (const char *)(event->data.scalar.value);

  if (!strlen(state->parameter)) {  // parameter not set
    PetscInt selection;
    SelectItem(value, 3, (const char *[3]){"spatial", "temporal", "riemann"}, (PetscInt[3]){0, 1, 2}, &selection);
    PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in numerics: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else {  // parameter set, get value
    PetscInt selection;
    if (!strcmp(state->parameter, "spatial")) {
      SelectItem(value, 2, (const char *[2]){"fv", "fe"}, (PetscInt[2]){SPATIAL_FV, SPATIAL_FE}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid numerics.spatial: %s", value);
      config->spatial = selection;
    } else if (!strcmp(state->parameter, "temporal")) {
      SelectItem(value, 3, (const char *[3]){"euler", "rk4", "beuler"}, (PetscInt[3]){TEMPORAL_EULER, TEMPORAL_RK4, TEMPORAL_BEULER}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid numerics.temporal: %s", value);
      config->temporal = selection;
    } else {  // riemann
      SelectItem(value, 2, (const char *[2]){"roe", "hllc"}, (PetscInt[2]){RIEMANN_ROE, RIEMANN_HLLC}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid numerics.riemann: %s", value);
      config->riemann = selection;
    }
    state->parameter[0] = 0;  // clear parameter name
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseTime(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // time:
  //   final_time: <value>
  //   unit: <nsteps|nminutes|nhours|ndays|nmonths|nyears>
  //   max_step: <value>

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-scalar value encountered in time section!");
  const char *value = (const char *)(event->data.scalar.value);

  if (!strlen(state->parameter)) {  // parameter not set
    PetscInt selection;
    SelectItem(value, 3, (const char *[3]){"final_time", "unit", "max_step"}, (PetscInt[3]){0, 1, 2}, &selection);
    PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in time: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else {  // parameter set, get value
    if (!strcmp(state->parameter, "final_time")) {
      PetscCall(ConvertToReal(state->comm, state->parameter, value, &config->final_time));
      PetscCheck((config->final_time > 0.0), state->comm, PETSC_ERR_USER, "invalid time.final_time: %g\n", config->final_time);
    } else if (!strcmp(state->parameter, "unit")) {
      PetscInt selection;
      SelectItem(value, 5, (const char *[5]){"minutes", "hours", "days", "months", "years"},
                 (PetscInt[5]){TIME_MINUTES, TIME_HOURS, TIME_DAYS, TIME_MONTHS, TIME_YEARS}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid time.unit: %s", value);
      config->time_unit = selection;
    } else {  // max_step
      PetscCall(ConvertToInt(state->comm, state->parameter, value, &config->max_step));
      PetscCheck((config->max_step >= 0), state->comm, PETSC_ERR_USER, "invalid time.max_step: %d\n", config->max_step);
    }
    state->parameter[0] = 0;  // clear parameter name
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseRestart(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // restart:
  //   format: <bin|h5>
  //   frequency: <value>

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-scalar value encountered in restart section!");
  const char *value = (const char *)(event->data.scalar.value);

  if (!strlen(state->parameter)) {  // parameter not set
    PetscInt selection;
    SelectItem(value, 2, (const char *[2]){"format", "frequency"}, (PetscInt[2]){0, 1}, &selection);
    PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in restart: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else {  // parameter set, get value
    if (!strcmp(state->parameter, "format")) {
      PetscInt selection;
      SelectItem(value, 2, (const char *[2]){"bin", "h5"}, (PetscInt[2]){0, 1}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid restart.format: %s", value);
      strncpy(config->restart_format, value, MAX_NAME_LEN);
    } else {  // frequency
      PetscCall(ConvertToInt(state->comm, state->parameter, value, &config->restart_frequency));
      PetscCheck((config->restart_frequency > 0), state->comm, PETSC_ERR_USER, "Invalid restart.frequency: %d\n", config->restart_frequency);
    }
    state->parameter[0] = 0;  // clear parameter name
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseLogging(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // logging:
  //   file: <path-to-log-file>
  //   level: <none|warning|info|detail|debug>

  const char *value = (const char *)(event->data.scalar.value);

  if (!strlen(state->parameter)) {  // parameter not set
    if (!strcmp(value, "file") || !strcmp(value, "level")) {
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      PetscCheck(PETSC_FALSE, state->comm, PETSC_ERR_USER, "invalid logging parameter: %s", value);
    }
  } else {
    if (!strcmp(state->parameter, "file")) {
      strncpy(config->log_file, value, PETSC_MAX_PATH_LEN);
    } else {  // level
      PetscInt selection;
      SelectItem(value, 5, (const char *[5]){"none", "warning", "info", "detail", "debug"},
                 (PetscInt[5]){LOG_NONE, LOG_WARNING, LOG_INFO, LOG_DETAIL, LOG_DEBUG}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in logging.level: %s", value);
      config->log_level = selection;
    }
    state->parameter[0] = 0;  // clear parameter name
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ParseGrid(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // grid:
  //   file: <path-to-msh-file>

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-scalar value encountered in grid section!");
  const char *value = (const char *)(event->data.scalar.value);

  if (!strlen(state->parameter)) {  // parameter not set
    PetscCheck(!strcmp(value, "file"), state->comm, PETSC_ERR_USER, "Invalid grid parameter: %s", value);
    strncpy(state->parameter, value, YAML_MAX_LEN);
  } else {
    // Easy peasy--we just record the mesh file and leave.
    strncpy(config->mesh_file, value, PETSC_MAX_PATH_LEN);
    state->parameter[0] = 0;  // clear parameter name
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseInitialConditions(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER,
             "Invalid YAML (non-scalar value encountered in initial_conditions section!");
  const char *value = (const char *)(event->data.scalar.value);

  // if we're not in a subsection, our parameter could be a single filename or
  // the name of a subsection
  if (!strlen(state->subsection)) {
    if (strlen(state->parameter)) {
      PetscCheck(!strcmp(value, "file"), state->comm, PETSC_ERR_USER, "Invalid initial_conditions parameter: %s", state->parameter);
      strcpy(state->parameter, value);
    } else if (!strcmp(state->parameter, "file")) {
      strncpy(config->initial_conditions_file, value, YAML_MAX_LEN);
      state->parameter[0] = 0;  // clear parameter name
    } else {                    // proceed to initial conditions subsection
      strncpy(state->subsection, value, YAML_MAX_LEN);
    }
  } else {
    // we should be inside a subsection that identifies a region.
    PetscCheck(state->inside_subsection, state->comm, PETSC_ERR_USER, "Invalid YAML in initial_conditions.%s", state->subsection);

    // What is the region's index?
    PetscInt region_index;
    PetscCall(ConvertToInt(state->comm, "region", state->subsection, &region_index));
    PetscCheck(region_index != -1, state->comm, PETSC_ERR_USER, "Invalid region in initial_conditions: %s", state->subsection);

    RDyConditionSpec *ic_spec = &config->initial_conditions[region_index];
    if (!strlen(state->parameter)) {  // parameter name not set
      PetscInt selection;
      SelectItem(value, 3, (const char *[3]){"flow", "sediment", "salinity"}, (PetscInt[3]){0, 1, 2}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in initial_conditions.%s: %s", state->subsection, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "flow")) {
        strncpy(ic_spec->flow_name, value, MAX_NAME_LEN);
        ++config->num_initial_conditions;
      } else if (!strcmp(state->parameter, "sediment")) {
        strncpy(ic_spec->sediment_name, value, MAX_NAME_LEN);
      } else {  // salinity
        strncpy(ic_spec->salinity_name, value, MAX_NAME_LEN);
      }
      state->parameter[0] = 0;  // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseBoundaryConditions(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER,
             "Invalid YAML (non-scalar value encountered in boundary_conditions section!");
  const char *value = (const char *)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection that identifies a surface.
    PetscCheck(state->inside_subsection, state->comm, PETSC_ERR_USER, "Invalid YAML in boundary_conditions.%s", state->subsection);

    // What is the surface's index?
    PetscInt surface_index;
    PetscCall(ConvertToInt(state->comm, "surface", state->subsection, &surface_index));
    PetscCheck(surface_index != -1, state->comm, PETSC_ERR_USER, "Invalid surface in boundary_conditions: %s", state->subsection);

    RDyConditionSpec *bc_spec = &config->boundary_conditions[surface_index];
    if (!strlen(state->parameter)) {  // parameter name not set
      PetscInt selection;
      SelectItem(value, 3, (const char *[3]){"flow", "sediment", "salinity"}, (PetscInt[3]){0, 1, 2}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in boundary_conditions.%s: %s", state->subsection, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "flow")) {
        strncpy(bc_spec->flow_name, value, MAX_NAME_LEN);
        ++config->num_boundary_conditions;
      } else if (!strcmp(state->parameter, "sediment")) {
        strncpy(bc_spec->sediment_name, value, MAX_NAME_LEN);
      } else {  // salinity
        strncpy(bc_spec->salinity_name, value, MAX_NAME_LEN);
      }
      state->parameter[0] = 0;  // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSources(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  PetscCheck(event->type == YAML_SCALAR_EVENT, state->comm, PETSC_ERR_USER, "Invalid YAML (non-scalar value encountered in sources section!");
  const char *value = (const char *)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection that identifies a region.
    PetscCheck(state->inside_subsection, state->comm, PETSC_ERR_USER, "Invalid YAML in sources.%s", state->subsection);

    // What is the region's index?
    PetscInt region_index;
    PetscCall(ConvertToInt(state->comm, "region", state->subsection, &region_index));
    PetscCheck(region_index != -1, state->comm, PETSC_ERR_USER, "Invalid region in sources: %s", state->subsection);

    RDyConditionSpec *src_spec = &config->sources[region_index];
    if (!strlen(state->parameter)) {  // parameter name not set
      PetscInt selection;
      SelectItem(value, 3, (const char *[3]){"flow", "sediment", "salinity"}, (PetscInt[3]){0, 1, 2}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in initial_conditions.%s: %s", state->subsection, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "flow")) {
        strncpy(src_spec->flow_name, value, MAX_NAME_LEN);
        ++config->num_sources;
      } else if (!strcmp(state->parameter, "sediment")) {
        strncpy(src_spec->sediment_name, value, MAX_NAME_LEN);
      } else {  // salinity
        strncpy(src_spec->salinity_name, value, MAX_NAME_LEN);
      }
      state->parameter[0] = 0;  // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseFlowConditions(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  if (event->type == YAML_SCALAR_EVENT) {
    const char *value = (const char *)(event->data.scalar.value);

    // if we're not in a subsection, our parameter is the name of the subsection
    if (!strlen(state->subsection)) {
      strncpy(state->subsection, value, YAML_MAX_LEN);
    } else {
      // we should be inside a subsection
      PetscCheck(state->inside_subsection, state->comm, PETSC_ERR_USER, "Invalid YAML in flow_conditions.%s", state->subsection);

      RDyFlowCondition *flow_cond = &config->flow_conditions[config->num_flow_conditions];
      if (!flow_cond->name) {  // condition name not set
        strncpy((char *)flow_cond->name, state->subsection, MAX_NAME_LEN);
      }
      if (!strlen(state->parameter)) {  // parameter name not set
        PetscInt selection;
        SelectItem(value, 3, (const char *[3]){"type", "height", "momentum"}, (PetscInt[3]){0, 1}, &selection);
        PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in flow condition '%s': %s", flow_cond->name, value);
        strncpy(state->parameter, value, YAML_MAX_LEN);
      } else {
        if (!strcmp(state->parameter, "type")) {
          PetscInt selection;
          SelectItem(value, 2, (const char *[2]){"dirichlet", "neumann"}, (PetscInt[2]){CONDITION_DIRICHLET, CONDITION_NEUMANN}, &selection);
          PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid flow condition %s.type: %s", flow_cond->name, value);
          flow_cond->type     = selection;
          state->parameter[0] = 0;  // clear parameter name
        } else if (!strcmp(state->parameter, "height")) {
          PetscCall(ConvertToReal(state->comm, state->parameter, value, &flow_cond->height));
          state->parameter[0] = 0;                   // clear parameter name
        } else {                                     // momentum
          if (flow_cond->momentum[0] == -FLT_MAX) {  // px not set
            PetscCall(ConvertToReal(state->comm, state->parameter, value, &flow_cond->momentum[0]));
          } else if (flow_cond->momentum[1] == -FLT_MAX) {  // py not yet
            PetscCall(ConvertToReal(state->comm, state->parameter, value, &flow_cond->momentum[1]));
            state->parameter[0] = 0;  // clear parameter name
          } else {                    // too many momentum components!
            PetscCheck(PETSC_FALSE, state->comm, PETSC_ERR_USER, "Too many components in flow_conditions.%s!", state->parameter);
          }
        }
      }
    }
  } else {
    RDyFlowCondition *flow_cond = &config->flow_conditions[config->num_flow_conditions];
    if (event->type == YAML_SEQUENCE_START_EVENT) {
      PetscCheck(!strcmp(state->parameter, "momentum"), state->comm, PETSC_ERR_USER, "Invalid sequence encountered in flow_conditions.%s",
                 state->parameter);
      // Set momentum components to magic numbers.
      flow_cond->momentum[0] = flow_cond->momentum[1] = -FLT_MAX;
    } else if (event->type == YAML_SEQUENCE_END_EVENT) {
      PetscCheck((flow_cond->momentum[0] != -FLT_MAX) && (flow_cond->momentum[1] != -FLT_MAX), state->comm, PETSC_ERR_USER,
                 "Incomplete momentum specification for flow_conditions.%s", state->parameter);
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSedimentConditions(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  const char *value = (const char *)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection
    PetscCheck(state->inside_subsection, state->comm, PETSC_ERR_USER, "Invalid YAML in sediment_conditions.%s", state->subsection);

    RDySedimentCondition *sed_cond = &config->sediment_conditions[config->num_sediment_conditions];
    if (!sed_cond->name) {  // condition name not set
      strncpy((char *)sed_cond->name, state->subsection, MAX_NAME_LEN);
    }
    if (!strlen(state->parameter)) {  // parameter name not set
      PetscInt selection;
      SelectItem(value, 2, (const char *[2]){"type", "concentration"}, (PetscInt[2]){0, 1}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in sediment condition '%s': %s", sed_cond->name, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "type")) {
        PetscInt selection;
        SelectItem(value, 2, (const char *[2]){"dirichlet", "neumann"}, (PetscInt[2]){CONDITION_DIRICHLET, CONDITION_NEUMANN}, &selection);
        PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid sediment condition %s.type: %s", sed_cond->name, value);
        sed_cond->type = selection;
      } else {  // water_flux
        PetscCall(ConvertToReal(state->comm, state->parameter, value, &sed_cond->concentration));
      }
      state->parameter[0] = 0;  // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode ParseSalinityConditions(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  const char *value = (const char *)(event->data.scalar.value);

  // if we're not in a subsection, our parameter is the name of the subsection
  if (!strlen(state->subsection)) {
    strncpy(state->subsection, value, YAML_MAX_LEN);
  } else {
    // we should be inside a subsection
    PetscCheck(state->inside_subsection, state->comm, PETSC_ERR_USER, "Invalid YAML in salinity_conditions.%s", state->subsection);

    RDySalinityCondition *sal_cond = &config->salinity_conditions[config->num_salinity_conditions];
    if (!strlen(sal_cond->name)) {  // condition name not set
      strncpy((char *)sal_cond->name, state->subsection, MAX_NAME_LEN);
    }
    if (!strlen(state->parameter)) {  // parameter name not set
      PetscInt selection;
      SelectItem(value, 2, (const char *[2]){"type", "concentration"}, (PetscInt[2]){0, 1}, &selection);
      PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid parameter in salinity condition '%s': %s", sal_cond->name, value);
      strncpy(state->parameter, value, YAML_MAX_LEN);
    } else {
      if (!strcmp(state->parameter, "type")) {
        PetscInt selection;
        SelectItem(value, 2, (const char *[2]){"dirichlet", "neumann"}, (PetscInt[2]){CONDITION_DIRICHLET, CONDITION_NEUMANN}, &selection);
        PetscCheck(selection != -1, state->comm, PETSC_ERR_USER, "Invalid salinity condition %s.type: %s", sal_cond->name, value);
        sal_cond->type = selection;
      } else {  // water_flux
        PetscCall(ConvertToReal(state->comm, state->parameter, value, &sal_cond->concentration));
      }
      state->parameter[0] = 0;  // clear parameter name
    }
  }

  PetscFunctionReturn(0);
}

// Handles a YAML event, populating the appropriate config info within rdy.
static PetscErrorCode HandleYamlEvent(yaml_event_t *event, YamlParserState *state, RDyConfig *config) {
  PetscFunctionBegin;

  // we don't need to do anything special at the beginning or the end of the
  // document
  if ((event->type == YAML_STREAM_START_EVENT) || (event->type == YAML_DOCUMENT_START_EVENT) || (event->type == YAML_DOCUMENT_END_EVENT) ||
      (event->type == YAML_STREAM_END_EVENT)) {
    PetscFunctionReturn(0);
  }

  // navigate sections via mapping starts and ends
  if (event->type == YAML_MAPPING_START_EVENT) {
    PetscCall(OpenSection(state, config));
  } else if (event->type == YAML_MAPPING_END_EVENT) {
    PetscCall(CloseSection(state, config));
  } else {  // parse parameters in sections
    // otherwise, dispatch the parser to the indicated section
    switch (state->section) {
      case NO_SECTION:
        PetscCall(ParseTopLevel(event, state, config));
        break;
      case PHYSICS_SECTION:
        PetscCall(ParsePhysics(event, state, config));
        break;
      case PHYSICS_FLOW_SECTION:
        PetscCall(ParsePhysicsFlow(event, state, config));
        break;
      case PHYSICS_SEDIMENT_SECTION:
        PetscCall(ParsePhysicsSediment(event, state, config));
        break;
      case PHYSICS_SALINITY_SECTION:
        PetscCall(ParsePhysicsSediment(event, state, config));
        break;
      case NUMERICS_SECTION:
        PetscCall(ParseNumerics(event, state, config));
        break;
      case TIME_SECTION:
        PetscCall(ParseTime(event, state, config));
        break;
      case RESTART_SECTION:
        PetscCall(ParseRestart(event, state, config));
        break;
      case LOGGING_SECTION:
        PetscCall(ParseLogging(event, state, config));
        break;
      case GRID_SECTION:
        PetscCall(ParseGrid(event, state, config));
        break;
      case INITIAL_CONDITIONS_SECTION:
        PetscCall(ParseInitialConditions(event, state, config));
        break;
      case BOUNDARY_CONDITIONS_SECTION:
        PetscCall(ParseBoundaryConditions(event, state, config));
        break;
      case SOURCES_SECTION:
        PetscCall(ParseSources(event, state, config));
        break;
      case FLOW_CONDITIONS_SECTION:
        PetscCall(ParseFlowConditions(event, state, config));
        break;
      case SEDIMENT_CONDITIONS_SECTION:
        PetscCall(ParseSedimentConditions(event, state, config));
        break;
      case SALINITY_CONDITIONS_SECTION:
        PetscCall(ParseSalinityConditions(event, state, config));
        break;
      default:
        // we ignore everything else for now
        break;
    }
  }

  PetscFunctionReturn(0);
}

// Parses the given YAML string into the given config representation
static PetscErrorCode ParseYaml(MPI_Comm comm, const char *yaml_str, RDyConfig *config) {
  PetscFunctionBegin;

  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_string(&parser, yaml_str, strlen(yaml_str));

  // parse the file, handling each YAML "event" based on the parser state
  YamlParserState   state = {.comm = comm};
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
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "%s", error_msg);
    }

    // process the event, using it to populate our YAML data, and handle
    // any errors resulting from properly-formed YAML that doesn't conform
    // to our spec
    PetscCall(HandleYamlEvent(&event, &state, config));

    // discard the event and move on
    event_type = event.type;
    yaml_event_delete(&event);
  } while (event_type != YAML_STREAM_END_EVENT);
  yaml_parser_delete(&parser);

  PetscFunctionReturn(0);
}

// checks config for any invalid or omitted parameters
PetscErrorCode ValidateConfig(MPI_Comm comm, const RDyConfig *config) {
  PetscFunctionBegin;

  // currently, only certain methods are implemented
  if (config->spatial != SPATIAL_FV) {
    PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only the finite volume spatial method (FV) is currently implemented.");
  }
  if (config->temporal != TEMPORAL_EULER) {
    PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only the forward euler temporal method (EULER) is currently implemented.");
  }
  if (config->riemann != RIEMANN_ROE) {
    PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only the roe riemann solver (ROE) is currently implemented.");
  }

  PetscCheck(strlen(config->mesh_file), comm, PETSC_ERR_USER, "grid.file not specified!");

  // We can accept an initial conditions file OR a set of initial conditions,
  // but not both.
  PetscCheck((strlen(config->initial_conditions_file) && !config->num_initial_conditions) ||
                 (!strlen(config->initial_conditions_file) && config->num_initial_conditions),
             comm, PETSC_ERR_USER,
             "Invalid initial_conditions! A file was specified, so no further "
             "parameters are allowed.");
  PetscFunctionReturn(0);
}

// reads the config file on process 0, broadcasts it as a string to all other
// processes, and parses the string into rdy->config
PetscErrorCode ReadConfigFile(RDy rdy) {
  PetscFunctionBegin;

  // open the config file on process 0, determine its size, and broadcast its
  // contents to all other processes.
  long  config_size;
  char *config_str;
  if (rdy->rank == 0) {
    // process 0: read the file
    FILE *file = NULL;
    PetscCall(PetscFOpen(rdy->comm, rdy->config_file, "r", &file));

    // determine the file's size and broadcast it
    fseek(file, 0, SEEK_END);
    config_size = ftell(file);
    MPI_Bcast(&config_size, 1, MPI_LONG, 0, rdy->comm);

    // create a content string and broadcast it
    PetscCall(RDyAlloc(char, config_size, &config_str));
    rewind(file);
    fread(config_str, sizeof(char), config_size, file);
    PetscCall(PetscFClose(rdy->comm, file));
    MPI_Bcast(config_str, config_size, MPI_CHAR, 0, rdy->comm);
  } else {
    // other processes: read the size of the content
    MPI_Bcast(&config_size, 1, MPI_LONG, 0, rdy->comm);

    // recreate the configuration string.
    PetscCall(RDyAlloc(char, config_size, &config_str));
    MPI_Bcast(config_str, config_size, MPI_CHAR, 0, rdy->comm);
  }

  // Parse the YAML config file into our config struct and validate it.
  PetscCall(ParseYaml(rdy->comm, config_str, &rdy->config));
  PetscCall(ValidateConfig(rdy->comm, &rdy->config));

  // Clean up.
  RDyFree(config_str);

  PetscFunctionReturn(0);
}
