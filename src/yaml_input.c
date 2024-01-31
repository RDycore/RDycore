#include <cyaml/cyaml.h>
#include <float.h>
#include <petscdmplex.h>
#include <private/rdycoreimpl.h>

// =============
//  YAML Parser
// =============
//
// The design of the options database in PETSc does not currently allow for one
// to traverse items in YAML mappings--one must know the exact name of every
// item one wants to retrieve. This puts a large restriction on the types of
// YAML files that can be parsed, so we've rolled our own.
//
// Here we've implemented a parser (using libcyaml) to handle the RDycore
// configuration file whose specification can be found at
//
// https://rdycore.github.io/RDycore/user/input.html
//
// In the style of libcyaml (https://github.com/tlsa/libcyaml/blob/main/docs/guide.md),
// this parser defines a schema for each section and populates the appropriate struct(s)
// within RDyConfig (include/private/rdyconfigimpl.h) accordingly. The schema for each
// section appears below and must remain consistent with the data structures in rdyconfigimpl.h.

// ====================
//  Schema definitions
// ====================

// Below, you'll see a bunch of schemas describing YAML sections and subsections
// with fields defined using macros like CYAML_FIELD_ENUM, CYAML_FIELD_INT,
// CYAML_FIELD_INT, etc.
//
// To specify an optional field with a default value, you must use the CYAML_FIELD
// macro, which looks a bit different. In particular:
// * the macro is called with the desired TYPE (in caps) as its first argument
// * the last argument to the macro should be {.missing = <default_value>}
//
// Take a look at the examples below to get a feel for this.

// clang-format off

// ---------------
// physics section
// ---------------
// physics:
//   flow:
//     mode: <swe|diffusion> # swe by default
//     bed_friction: <true|false> # off by default
//     tiny_h: <value> # 1e-7 by default
//   sediment: <true|false> # off by default
//   salinity: <true|false> # off by default

// mapping of strings to physics flow types
static const cyaml_strval_t physics_flow_modes[] = {
    {"swe",       FLOW_SWE      },
    {"diffusion", FLOW_DIFFUSION},
};

// mapping of physics.flow fields to members of RDyPhysicsFlow
static const cyaml_schema_field_t physics_flow_fields_schema[] = {
    CYAML_FIELD_ENUM("mode", CYAML_FLAG_DEFAULT, RDyPhysicsFlow, mode, physics_flow_modes, CYAML_ARRAY_LEN(physics_flow_modes)),
    CYAML_FIELD(BOOL, "bed_friction", CYAML_FLAG_OPTIONAL, RDyPhysicsFlow, bed_friction, {.missing = false}),
    CYAML_FIELD(FLOAT, "tiny_h", CYAML_FLAG_OPTIONAL, RDyPhysicsFlow, tiny_h, {.missing = 1e-7}),
    CYAML_FIELD_END
};

// mapping of physics fields to members of RDyPhysicsSection
static const cyaml_schema_field_t physics_fields_schema[] = {
    CYAML_FIELD_MAPPING("flow", CYAML_FLAG_DEFAULT, RDyPhysicsSection, flow, physics_flow_fields_schema),
    CYAML_FIELD(BOOL, "sediment", CYAML_FLAG_OPTIONAL, RDyPhysicsSection, sediment, {.missing = false}),
    CYAML_FIELD(BOOL, "salinity", CYAML_FLAG_OPTIONAL, RDyPhysicsSection, salinity, {.missing = false}),
    CYAML_FIELD_END
};

// ----------------
// numerics section
// ----------------
// numerics:
//   spatial: <fv|fe>                 # To begin with, we will only have FV method
//   temporal: <euler,rk4,beuler,...> # These should map on PETSc -ts_type
//   riemann: <roe|hllc>              # To begin with, we will only have Roe

// mapping of strings to numerics spatial types
static const cyaml_strval_t numerics_spatial_types[] = {
    {"fv", SPATIAL_FV},
    {"fe", SPATIAL_FE},
};

// mapping of strings to numerics temporal types
static const cyaml_strval_t numerics_temporal_types[] = {
    {"euler",  TEMPORAL_EULER },
    {"rk4",    TEMPORAL_RK4   },
    {"beuler", TEMPORAL_BEULER},
};

// mapping of strings to numerics riemann solver types
static const cyaml_strval_t numerics_riemann_types[] = {
    {"roe",  RIEMANN_ROE },
    {"hllc", RIEMANN_HLLC},
};

// mapping of numerics fields to members of RDyNumericsSection
static const cyaml_schema_field_t numerics_fields_schema[] = {
    CYAML_FIELD_ENUM("spatial", CYAML_FLAG_DEFAULT, RDyNumericsSection, spatial, numerics_spatial_types, CYAML_ARRAY_LEN(numerics_spatial_types)),
    CYAML_FIELD_ENUM("temporal", CYAML_FLAG_DEFAULT, RDyNumericsSection, temporal, numerics_temporal_types, CYAML_ARRAY_LEN(numerics_temporal_types)),
    CYAML_FIELD_ENUM("riemann", CYAML_FLAG_DEFAULT, RDyNumericsSection, riemann, numerics_riemann_types, CYAML_ARRAY_LEN(numerics_riemann_types)),
    CYAML_FIELD_END
};

// ------------
// time section
// ------------
// time:
//   final_time: <value>
//   unit: <seconds|minutes|hours|days|months|years> # applies to final_time (stored internally in seconds)
//   max_step: <value>
//   time_step: <value> # optional
//   coupling_interval: <value> # optional

// mapping of strings to time units
static const cyaml_strval_t time_units[] = {
    {"seconds", TIME_SECONDS},
    {"minutes", TIME_MINUTES},
    {"hours",   TIME_HOURS  },
    {"days",    TIME_DAYS   },
    {"months",  TIME_MONTHS },
    {"years",   TIME_YEARS  },
};

// mapping of time fields to members of RDyTimeSection
static const cyaml_schema_field_t time_fields_schema[] = {
    CYAML_FIELD(FLOAT, "final_time", CYAML_FLAG_OPTIONAL, RDyTimeSection, final_time, {.missing = INVALID_REAL}),
    CYAML_FIELD_ENUM("unit", CYAML_FLAG_DEFAULT, RDyTimeSection, unit, time_units, CYAML_ARRAY_LEN(time_units)),
    CYAML_FIELD(INT, "max_step", CYAML_FLAG_OPTIONAL, RDyTimeSection, max_step, {.missing = INVALID_INT}),
    CYAML_FIELD(FLOAT, "time_step", CYAML_FLAG_OPTIONAL, RDyTimeSection, time_step, {.missing = INVALID_REAL}),
    CYAML_FIELD(FLOAT, "coupling_interval", CYAML_FLAG_OPTIONAL, RDyTimeSection, coupling_interval, {.missing = INVALID_REAL}),
    CYAML_FIELD_END
};

// ---------------
// logging section
// ---------------
// logging:
//   file: <path> # default: stdout
//   level: <none|warning|info|detail|debug> # <-- increasing levels of logging (default: none)

// mapping of strings to log levels
static const cyaml_strval_t logging_levels[] = {
    {"none",    LOG_NONE   },
    {"warning", LOG_WARNING},
    {"info",    LOG_INFO   },
    {"detail",  LOG_DETAIL },
    {"debug",   LOG_DEBUG  },
};

// mapping of logging fields to members of RDyLoggingSection
static const cyaml_schema_field_t logging_fields_schema[] = {
    CYAML_FIELD_STRING("file", CYAML_FLAG_OPTIONAL, RDyLoggingSection, file, 0),
    CYAML_FIELD_ENUM("level", CYAML_FLAG_OPTIONAL, RDyLoggingSection, level, logging_levels, CYAML_ARRAY_LEN(logging_levels)),
    CYAML_FIELD_END
};

// ------------------
// checkpoint section
// ------------------
// checkpoint:
//   format: <binary|hdf5>
//   interval: <value-in-steps>  # default: 0 (no checkpoints)

// mapping of strings to file formats
static const cyaml_strval_t checkpoint_file_formats[] = {
    {"binary", PETSC_VIEWER_NATIVE    },
    {"hdf5",   PETSC_VIEWER_HDF5_PETSC},
};

// mapping of checkpoint fields to members of RDyCheckpointSection
static const cyaml_schema_field_t checkpoint_fields_schema[] = {
    CYAML_FIELD_ENUM("format", CYAML_FLAG_DEFAULT, RDyCheckpointSection, format, checkpoint_file_formats, CYAML_ARRAY_LEN(checkpoint_file_formats)),
    CYAML_FIELD_INT("interval", CYAML_FLAG_DEFAULT, RDyCheckpointSection, interval),
    CYAML_FIELD_END
};

// ---------------
// restart section
// ---------------
// restart:
//   file: <checkpoint-filename>
//   reinitialize: <true/false>  # default: false

// mapping of restart fields to members of RDyRestartSection
static const cyaml_schema_field_t restart_fields_schema[] = {
    CYAML_FIELD_STRING("file", CYAML_FLAG_OPTIONAL, RDyRestartSection, file, 0),
    CYAML_FIELD(BOOL, "reinitialize", CYAML_FLAG_OPTIONAL, RDyRestartSection, reinitialize, {.missing = PETSC_FALSE}),
    CYAML_FIELD_END
};

// ---------------
// output section
// ---------------
// output:
//   format: <binary|xdmf|cgns>
//   interval: <number-of-steps-between-output-dumps> # default: 0 (no output)
//   batch_size: <number-of-steps-stored-in-each-output-file> # default: 1

// mapping of strings to file formats
static const cyaml_strval_t output_file_formats[] = {
    {"none",   OUTPUT_NONE},
    {"binary", OUTPUT_BINARY},
    {"xdmf",   OUTPUT_XDMF  },
    {"cgns",   OUTPUT_CGNS  },
};

// mapping of time_series fields to members of RDyTimeSeries
static const cyaml_schema_field_t output_time_series_fields_schema[] = {
    CYAML_FIELD_INT("boundary_fluxes", CYAML_FLAG_DEFAULT, RDyTimeSeries, boundary_fluxes),
    CYAML_FIELD_END
};

// mapping of output fields to members of RDyOutputSection
static const cyaml_schema_field_t output_fields_schema[] = {
    CYAML_FIELD_ENUM("format", CYAML_FLAG_OPTIONAL, RDyOutputSection, format, output_file_formats, CYAML_ARRAY_LEN(output_file_formats)),
    CYAML_FIELD_INT("interval", CYAML_FLAG_OPTIONAL, RDyOutputSection, interval),
    CYAML_FIELD_INT("batch_size", CYAML_FLAG_OPTIONAL, RDyOutputSection, batch_size),
    CYAML_FIELD_MAPPING("time_series", CYAML_FLAG_OPTIONAL, RDyOutputSection, time_series, output_time_series_fields_schema),
    CYAML_FIELD_END
};

// ------------
// grid section
// ------------
// grid:
//   file: <path-to-file/grid.{msh,h5,exo}>

// mapping of grid fields to members of RDyGridSection
static const cyaml_schema_field_t grid_fields_schema[] = {
  CYAML_FIELD_STRING("file", CYAML_FLAG_DEFAULT, RDyGridSection, file, 1),
  CYAML_FIELD_END
};

// mapping of strings to input file formats
static const cyaml_strval_t input_file_formats[] = {
    {"",       PETSC_VIEWER_NOFORMAT  },
    {"binary", PETSC_VIEWER_NATIVE    },
    {"hdf5",   PETSC_VIEWER_HDF5_PETSC},
};

// ---------------------------
// surface_composition section
// ---------------------------

// mapping of material specification fields to members of RDyMaterialSpec
static const cyaml_schema_field_t surface_composition_fields_schema[] = {
    CYAML_FIELD_STRING("region", CYAML_FLAG_DEFAULT, RDySurfaceCompositionSpec, region, 1),
    CYAML_FIELD_STRING("material", CYAML_FLAG_DEFAULT, RDySurfaceCompositionSpec, material, 1),
    CYAML_FIELD_END
};

// a single surface composition entry
static const cyaml_schema_value_t surface_composition_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDySurfaceCompositionSpec, surface_composition_fields_schema),
};

// -----------------
// materials section
// -----------------

// mapping of material property fields to RDyMaterialPropertySpec
static const cyaml_schema_field_t material_property_fields_schema[] = {
    CYAML_FIELD_FLOAT("value", CYAML_FLAG_OPTIONAL, RDyMaterialPropertySpec, value),
    CYAML_FIELD_STRING("file", CYAML_FLAG_OPTIONAL, RDyMaterialPropertySpec, file, 1),
    CYAML_FIELD_ENUM("format", CYAML_FLAG_OPTIONAL, RDyMaterialPropertySpec, format, input_file_formats, CYAML_ARRAY_LEN(input_file_formats)),
    CYAML_FIELD_END
};

// mapping of material property fields to RDyMaterialPropertіesSpec
static const cyaml_schema_field_t material_properties_fields_schema[] = {
    CYAML_FIELD_MAPPING("manning", CYAML_FLAG_DEFAULT, RDyMaterialPropertiesSpec, manning, material_property_fields_schema),
    CYAML_FIELD_END
};

// mapping of material fields to RDyMaterialSpec
static const cyaml_schema_field_t material_fields_schema[] = {
    CYAML_FIELD_STRING("name", CYAML_FLAG_DEFAULT, RDyMaterialSpec, name, 1),
    CYAML_FIELD_MAPPING("properties", CYAML_FLAG_DEFAULT, RDyMaterialSpec, properties, material_properties_fields_schema),
    CYAML_FIELD_END
};

// a single material entry
static const cyaml_schema_value_t material_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDyMaterialSpec, material_fields_schema),
};

// ---------------
// regions section
// ---------------
// regions:
//  - name: downstream   # human-readable name for the region
//    grid_region_id: 2  # grid identifier for the region
//  - name: upstream
//    grid_region_id: 1

// schema for region fields
static const cyaml_schema_field_t region_spec_fields_schema[] = {
    CYAML_FIELD_STRING("name", CYAML_FLAG_DEFAULT, RDyRegionSpec, name, 1),
    CYAML_FIELD_INT("grid_region_id", CYAML_FLAG_DEFAULT, RDyRegionSpec, grid_region_id),
    CYAML_FIELD_END
};

// a single region entry
static const cyaml_schema_value_t region_spec_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDyRegionSpec, region_spec_fields_schema),
};

// ---------------------------------------
// initial_conditions and sources sections
// ---------------------------------------
// initial_conditions/sources:
//  - region: <region-name>
//    flow: <name-of-a-flow-condition>
//    sediment: <name-of-a-sediment-condition> # used if physics.sediment == true above
//    salinity: <name-of-a-salinity-condition> # used if physics.salinity == true above
//  - region: <region-name>
//    flow: <name-of-a-flow-condition>
//    sediment: <name-of-a-sediment-condition> # used only if physics.sediment == true above
//    salinity: <name-of-a-salinity-condition> # used only if physics.salinity == true above
// ...

// mapping of conditions fields to members of RDyRegionConditionSpec
static const cyaml_schema_field_t region_condition_spec_fields_schema[] = {
    CYAML_FIELD_STRING("region", CYAML_FLAG_DEFAULT, RDyRegionConditionSpec, region, 1),
    CYAML_FIELD_STRING("flow", CYAML_FLAG_DEFAULT, RDyRegionConditionSpec, flow, 1),
    CYAML_FIELD_STRING("sediment", CYAML_FLAG_OPTIONAL, RDyRegionConditionSpec, sediment, 0),
    CYAML_FIELD_STRING("salinity", CYAML_FLAG_OPTIONAL, RDyRegionConditionSpec, salinity, 0),
    CYAML_FIELD_END
};

// a single regional initial condition / source spec entry
static const cyaml_schema_value_t region_condition_spec_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDyRegionConditionSpec, region_condition_spec_fields_schema),
};

// ------------------
// boundaries section
// ------------------
// boundaries:
//   - name: bottom_wall
//     grid_boundary_id: 3  # grid identifier for the boundary
//   - name: top_wall
//     grid_boundary_id: 2
//   - name: exterior
//     grid_boundary_id: 1

// schema for boundary fields
static const cyaml_schema_field_t boundary_spec_fields_schema[] = {
    CYAML_FIELD_STRING("name", CYAML_FLAG_DEFAULT, RDyBoundarySpec, name, 1),
    CYAML_FIELD_INT("grid_boundary_id", CYAML_FLAG_DEFAULT, RDyBoundarySpec, grid_boundary_id),
    CYAML_FIELD_END
};

// a single boundary entry
static const cyaml_schema_value_t boundary_spec_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDyBoundarySpec, boundary_spec_fields_schema),
};

// ---------------------------
// boundary_conditions section
// ---------------------------
// boundary_conditions:
//   - boundaries: [<boundary-name1>, <boundary-name2>, ...]
//     flow: <name-of-a-flow-condition>
//     sediment: <name-of-a-sediment-condition> # used if physics.sediment = true above
//     salinity: <name-of-a-salinity-condition> # used if physics.salinity = true above
//   - boundaries: [<boundary-name1>, <boundary-name2>, ...]
//     flow: <name-of-a-flow-condition>
//     sediment: <name-of-a-sediment-condition> # used only if physics.sediment = true above
//     salinity: <name-of-a-salinity-condition> # used only if physics.salinity = true above
// ...

// schema for boundary name
static const cyaml_schema_value_t boundary_name_entry = {
    CYAML_VALUE_STRING(CYAML_FLAG_DEFAULT, char, 1, MAX_NAME_LEN),
};

// mapping of conditions fields to members of RDyBoundaryConditionSpec
static const cyaml_schema_field_t boundary_condition_spec_fields_schema[] = {
    CYAML_FIELD_SEQUENCE_COUNT("boundaries", CYAML_FLAG_DEFAULT, RDyBoundaryConditionSpec, boundaries, num_boundaries, &boundary_name_entry, 0, MAX_NUM_BOUNDARIES),
    CYAML_FIELD_STRING("flow", CYAML_FLAG_DEFAULT, RDyBoundaryConditionSpec, flow, 1),
    CYAML_FIELD_STRING("sediment", CYAML_FLAG_OPTIONAL, RDyBoundaryConditionSpec, sediment, 0),
    CYAML_FIELD_STRING("salinity", CYAML_FLAG_OPTIONAL, RDyBoundaryConditionSpec, salinity, 0),
    CYAML_FIELD_END
};

// a single boundary condition spec entry
static const cyaml_schema_value_t boundary_condition_spec_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDyBoundaryConditionSpec, boundary_condition_spec_fields_schema),
};

// -----------------------
// flow_conditions section
// -----------------------
// - name: <name-of-flow-condition-1>
//   type: <dirichlet|neumann|reflecting|critical-outflow>
//   height: <value> # used only by dirichlet
//   momentum: <px, py> # use only by dirichlet
// - name: <name-of-flow-condition-2>
//   type: <dirichlet|neumann|reflecting|critical-outflow>
//   file: <filename>      # used only by dirichlet
//   format: <binary|hdf5> # used only by dirichlet
//   ...

// mapping of strings to types of conditions
static const cyaml_strval_t condition_types[] = {
    {"dirichlet",        CONDITION_DIRICHLET       },
    {"neumann",          CONDITION_NEUMANN         },
    {"reflecting",       CONDITION_REFLECTING      },
    {"critical-outflow", CONDITION_CRITICAL_OUTFLOW},
};

// schema for momentum component (as specified in a 2-item sequence)
static const cyaml_schema_value_t momentum_component_entry = {
    CYAML_VALUE(FLOAT, CYAML_FLAG_DEFAULT, PetscReal, {.missing = INVALID_REAL}),
};

// schema for flow condition fields
static const cyaml_schema_field_t flow_condition_fields_schema[] = {
    CYAML_FIELD_STRING("name", CYAML_FLAG_DEFAULT, RDyFlowCondition, name, 1),
    CYAML_FIELD_ENUM("type", CYAML_FLAG_DEFAULT, RDyFlowCondition, type, condition_types, CYAML_ARRAY_LEN(condition_types)),
    CYAML_FIELD(FLOAT, "height", CYAML_FLAG_OPTIONAL, RDyFlowCondition, height, {.missing = INVALID_REAL}),
    CYAML_FIELD_SEQUENCE_FIXED("momentum", CYAML_FLAG_OPTIONAL, RDyFlowCondition, momentum, &momentum_component_entry, 2),
    CYAML_FIELD_STRING("file", CYAML_FLAG_OPTIONAL, RDyFlowCondition, file, 1),
    CYAML_FIELD_ENUM("format", CYAML_FLAG_OPTIONAL, RDyFlowCondition, format, input_file_formats, CYAML_ARRAY_LEN(input_file_formats)),
    CYAML_FIELD_END
};

// a single flow_conditions entry
static const cyaml_schema_value_t flow_condition_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDyFlowCondition, flow_condition_fields_schema),
};

// ---------------------------
// sediment_conditions section
// ---------------------------
// - name: <name-of-sediment-condition-1>
//   type: <dirichlet|neumann|reflecting|critical>
//   concentration: <value> # used only by dirichlet
// - name: <name-of-sediment-condition-2>
//   type: <dirichlet|neumann|reflecting|critical>
//   file: <filename>      # used only by dirichlet
//   format: <binary|hdf5> # used only by dirichlet
//   ...

// schema for sediment_condition fields
static const cyaml_schema_field_t sediment_condition_fields_schema[] = {
    CYAML_FIELD_STRING("name", CYAML_FLAG_DEFAULT, RDySedimentCondition, name, 1),
    CYAML_FIELD_ENUM("type", CYAML_FLAG_DEFAULT, RDySedimentCondition, type, condition_types, CYAML_ARRAY_LEN(condition_types)),
    CYAML_FIELD_FLOAT("concentration", CYAML_FLAG_OPTIONAL, RDySedimentCondition, concentration),
    CYAML_FIELD_STRING("file", CYAML_FLAG_OPTIONAL, RDySedimentCondition, file, 1),
    CYAML_FIELD_ENUM("format", CYAML_FLAG_OPTIONAL, RDySedimentCondition, format, input_file_formats, CYAML_ARRAY_LEN(input_file_formats)),
    CYAML_FIELD_END
};

// a single sediment_conditions entry
static const cyaml_schema_value_t sediment_condition_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDySedimentCondition, sediment_condition_fields_schema),
};

// ---------------------------
// salinity_conditions section
// ---------------------------
// - name: <name-of-salinity-condition-1>
//   type: <dirichlet|neumann|reflecting|critical>
//   concentration: <value> # used only by dirichlet
// - name: <name-of-salinity-condition-2>
//   type: <dirichlet|neumann|reflecting|critical>
//   file: <filename>      # used only by dirichlet
//   format: <binary|hdf5> # used only by dirichlet
//   ...

// schema for salinity fields
static const cyaml_schema_field_t salinity_condition_fields_schema[] = {
    CYAML_FIELD_STRING("name", CYAML_FLAG_DEFAULT, RDySalinityCondition, name, 1),
    CYAML_FIELD_ENUM("type", CYAML_FLAG_DEFAULT, RDySalinityCondition, type, condition_types, CYAML_ARRAY_LEN(condition_types)),
    CYAML_FIELD_FLOAT("concentration", CYAML_FLAG_OPTIONAL, RDySalinityCondition, concentration),
    CYAML_FIELD_STRING("file", CYAML_FLAG_OPTIONAL, RDySalinityCondition, file, 1),
    CYAML_FIELD_ENUM("format", CYAML_FLAG_OPTIONAL, RDySalinityCondition, format, input_file_formats, CYAML_ARRAY_LEN(input_file_formats)),
    CYAML_FIELD_END
};

// a single salinity_conditions entry
static const cyaml_schema_value_t salinity_condition_entry = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_DEFAULT, RDySalinityCondition, salinity_condition_fields_schema),
};

// ----------------
// top-level schema
// ----------------

// schema for top-level configuration fields
static const cyaml_schema_field_t config_fields_schema[] = {
    CYAML_FIELD_MAPPING("physics", CYAML_FLAG_DEFAULT, RDyConfig, physics, physics_fields_schema),
    CYAML_FIELD_MAPPING("numerics", CYAML_FLAG_DEFAULT, RDyConfig, numerics, numerics_fields_schema),
    CYAML_FIELD_MAPPING("time", CYAML_FLAG_DEFAULT, RDyConfig, time, time_fields_schema),
    CYAML_FIELD_MAPPING("logging", CYAML_FLAG_OPTIONAL, RDyConfig, logging, logging_fields_schema),
    CYAML_FIELD_MAPPING("restart", CYAML_FLAG_OPTIONAL, RDyConfig, restart, restart_fields_schema),
    CYAML_FIELD_MAPPING("output", CYAML_FLAG_OPTIONAL, RDyConfig, output, output_fields_schema),
    CYAML_FIELD_MAPPING("grid", CYAML_FLAG_DEFAULT, RDyConfig, grid, grid_fields_schema),
    CYAML_FIELD_SEQUENCE_COUNT("surface_composition", CYAML_FLAG_DEFAULT, RDyConfig, surface_composition, num_material_assignments, &surface_composition_entry, 0, MAX_NUM_REGIONS),
    CYAML_FIELD_SEQUENCE_COUNT("materials", CYAML_FLAG_DEFAULT, RDyConfig, materials, num_materials, &material_entry, 0, MAX_NUM_MATERIALS),
    CYAML_FIELD_SEQUENCE_COUNT("regions", CYAML_FLAG_DEFAULT, RDyConfig, regions, num_regions,
                               &region_spec_entry, 0, MAX_NUM_REGIONS),
    CYAML_FIELD_SEQUENCE_COUNT("initial_conditions", CYAML_FLAG_DEFAULT, RDyConfig, initial_conditions, num_initial_conditions, &region_condition_spec_entry, 0, MAX_NUM_REGIONS),
    CYAML_FIELD_SEQUENCE_COUNT("boundaries", CYAML_FLAG_OPTIONAL, RDyConfig, boundaries, num_boundaries,
                               &boundary_spec_entry, 0, MAX_NUM_BOUNDARIES),
    CYAML_FIELD_SEQUENCE_COUNT("boundary_conditions", CYAML_FLAG_OPTIONAL, RDyConfig, boundary_conditions, num_boundary_conditions,
                               &boundary_condition_spec_entry, 0, MAX_NUM_BOUNDARIES),
    CYAML_FIELD_SEQUENCE_COUNT("sources", CYAML_FLAG_OPTIONAL, RDyConfig, sources, num_sources, &region_condition_spec_entry, 0, MAX_NUM_REGIONS),
    CYAML_FIELD_SEQUENCE_COUNT("flow_conditions", CYAML_FLAG_DEFAULT, RDyConfig, flow_conditions, num_flow_conditions, &flow_condition_entry, 0,
                               MAX_NUM_CONDITIONS),
    CYAML_FIELD_SEQUENCE_COUNT("sediment_conditions", CYAML_FLAG_OPTIONAL, RDyConfig, sediment_conditions, num_sediment_conditions,
                               &sediment_condition_entry, 0, MAX_NUM_CONDITIONS),
    CYAML_FIELD_SEQUENCE_COUNT("salinity_conditions", CYAML_FLAG_OPTIONAL, RDyConfig, salinity_conditions, num_salinity_conditions,
                               &salinity_condition_entry, 0, MAX_NUM_CONDITIONS),
    CYAML_FIELD_END
};

// schema for top-level configuration datum itself
static const cyaml_schema_value_t config_schema = {
    CYAML_VALUE_MAPPING(CYAML_FLAG_POINTER, RDyConfig, config_fields_schema),
};

// clang-format on

// CYAML log function (of type cyaml_log_fn_t)
static void YamlLog(cyaml_log_t level, void *ctx, const char *fmt, va_list args) {
  // render a log string
  char message[1024];
  vsnprintf(message, 1023, fmt, args);
  MPI_Comm *comm = ctx;
  PetscFPrintf(*comm, stdout, "%s", message);
}

// CYAML memory allocation function (of type cyaml_mem_fn_t)
static void *YamlAlloc(void *ctx, void *ptr, size_t size) {
  if (size) {
    PetscRealloc(size, &ptr);
    return ptr;
  } else {  // free
    PetscFree(ptr);
    return NULL;
  }
}

// parses the given YAML string into the given config representation
static PetscErrorCode ParseYaml(MPI_Comm comm, const char *yaml_str, RDyConfig **config) {
  PetscFunctionBegin;

  // configure our YAML parser
  cyaml_config_t yaml_config = {
      .log_fn    = YamlLog,
      .log_ctx   = &comm,
      .mem_fn    = YamlAlloc,
      .log_level = CYAML_LOG_WARNING,
  };

  const uint8_t *yaml_data     = (const uint8_t *)yaml_str;
  size_t         yaml_data_len = strlen(yaml_str);
  cyaml_err_t    err           = cyaml_load_data(yaml_data, yaml_data_len, &yaml_config, &config_schema, (void **)config, NULL);
  PetscCheck(err == CYAML_OK, comm, PETSC_ERR_USER, "Error parsing config file: %s", cyaml_strerror(err));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// checks config for any invalid or omitted parameters
static PetscErrorCode ValidateConfig(MPI_Comm comm, RDyConfig *config) {
  PetscFunctionBegin;

  // check numerics settings
  if (config->numerics.spatial != SPATIAL_FV) {
    PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only the finite volume spatial method (FV) is currently implemented.");
  }
  if (config->numerics.temporal != TEMPORAL_EULER) {
    PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only the forward euler temporal method (EULER) is currently implemented.");
  }
  if (config->numerics.riemann != RIEMANN_ROE) {
    PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only the roe riemann solver (ROE) is currently implemented.");
  }

  PetscCheck(strlen(config->grid.file), comm, PETSC_ERR_USER, "grid.file not specified!");

  // check time settings
  // 'final_time', 'max_step', 'time_step': exactly two of these three can be specified in the .yaml file.
  PetscInt num_time_settings = 0;
  if (config->time.final_time != INVALID_REAL) ++num_time_settings;
  if (config->time.max_step != INVALID_REAL) ++num_time_settings;
  if (config->time.time_step != INVALID_REAL) ++num_time_settings;
  PetscCheck(num_time_settings, comm, PETSC_ERR_USER,
             "Exactly 2 of time.final_time, time.max_step, time.time_step must be specified (%" PetscInt_FMT " given)", num_time_settings);

  // set the third parameter based on the two that are given
  if (config->time.final_time == INVALID_REAL) {
    config->time.final_time = config->time.max_step * config->time.time_step;
  } else if (config->time.max_step == INVALID_INT) {
    config->time.max_step = (PetscInt)(config->time.final_time / config->time.time_step);
  } else {  // config->time.time_step == INVALID_REAL
    config->time.time_step = config->time.final_time / config->time.max_step;
  }

  // by default, the coupling interval is set to the final time (but in any case,
  // it shouldn't be greater!)
  if (config->time.coupling_interval == INVALID_REAL) {
    config->time.coupling_interval = config->time.final_time;
  } else {
    PetscCheck(config->time.coupling_interval > 0.0, comm, PETSC_ERR_USER, "time.coupling_interval must be positive");
    PetscCheck(config->time.coupling_interval <= config->time.final_time, comm, PETSC_ERR_USER,
               "time.coupling_interval must not exceed time.final_time");
  }

  // we need initial conditions specified for each region
  PetscCheck(config->num_initial_conditions > 0, comm, PETSC_ERR_USER, "No initial conditions were specified!");
  PetscCheck(config->num_initial_conditions == config->num_regions, comm, PETSC_ERR_USER,
             "%" PetscInt_FMT " initial conditions were specified in initial_conditions (exactly %" PetscInt_FMT " needed)",
             config->num_initial_conditions, config->num_regions);

  // we need material properties for each region as well
  PetscCheck(config->num_material_assignments == config->num_regions, comm, PETSC_ERR_USER,
             "Only %" PetscInt_FMT " material <-> region assignments were found in surface_composition (%" PetscInt_FMT " needed)",
             config->num_material_assignments, config->num_regions);

  // validate our materials
  PetscCheck(config->num_materials > 0, comm, PETSC_ERR_USER, "No materials specified!");

  // validate our flow conditions
  for (PetscInt i = 0; i < config->num_flow_conditions; ++i) {
    const RDyFlowCondition *flow_cond = &config->flow_conditions[i];
    PetscCheck(flow_cond->type >= 0, comm, PETSC_ERR_USER, "Flow condition type not set in flow_conditions.%s", flow_cond->name);
    if (flow_cond->type != CONDITION_REFLECTING && flow_cond->type != CONDITION_CRITICAL_OUTFLOW) {
      PetscCheck(flow_cond->height != INVALID_REAL || flow_cond->file[0], comm, PETSC_ERR_USER, "Missing height specification for flow_conditions.%s",
                 flow_cond->name);
      PetscCheck(flow_cond->file[0] || ((flow_cond->momentum[0] != INVALID_REAL) && (flow_cond->momentum[1] != INVALID_REAL)), comm, PETSC_ERR_USER,
                 "Missing or incomplete momentum specification for flow_conditions.%s", flow_cond->name);
    }
  }

  // validate sediment conditions
  for (PetscInt i = 0; i < config->num_sediment_conditions; ++i) {
    const RDySedimentCondition *sed_cond = &config->sediment_conditions[i];
    PetscCheck(sed_cond->type >= 0, comm, PETSC_ERR_USER, "Sediment condition type not set in sediment_conditions.%s", sed_cond->name);
    PetscCheck(sed_cond->concentration != INVALID_REAL, comm, PETSC_ERR_USER, "Missing sediment concentration for sediment_conditions.%s",
               sed_cond->name);
  }

  // validate salinity conditions
  for (PetscInt i = 0; i < config->num_salinity_conditions; ++i) {
    const RDySalinityCondition *sal_cond = &config->salinity_conditions[i];
    PetscCheck(sal_cond->type >= 0, comm, PETSC_ERR_USER, "Salinity condition type not set in salinity_conditions.%s", sal_cond->name);
    PetscCheck(sal_cond->concentration != INVALID_REAL, comm, PETSC_ERR_USER, "Missing salinity concentration for salinity_conditions.%s",
               sal_cond->name);
  }

  // validate output options
  PetscCheck((config->output.format == OUTPUT_NONE) || (config->output.interval > 0), comm, PETSC_ERR_USER,
             "Output interval must be specified as a positive number of steps.");
  PetscCheck((config->output.batch_size == 0) || (config->output.format != OUTPUT_BINARY), comm, PETSC_ERR_USER,
             "Binary output does not support output batching");
  if ((config->output.batch_size == 0) && (config->output.format != OUTPUT_NONE) && config->output.format != OUTPUT_BINARY) {
    config->output.batch_size = 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// adds entries to PETSc's options database based on parsed config info
// (necessary because not all functionality is exposed via PETSc's C API)
static PetscErrorCode SetAdditionalOptions(RDy rdy) {
#define VALUE_LEN 128
  PetscFunctionBegin;
  PetscBool has_param, loc_refine;
  char      value[VALUE_LEN + 1] = {0};

  PetscCall(PetscOptionsHasName(NULL, NULL, "-dm_refine", &loc_refine));
  PetscCall(PetscOptionsHasName(NULL, "ref_", "-dm_refine", &rdy->refine));
  if (rdy->config.grid.file[0] != '\0') PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_filename", rdy->config.grid.file));
  PetscCall(PetscOptionsSetValue(NULL, "-dist_dm_distribute_save_sf", "true"));
  if (loc_refine) PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_transform_label_match_strata", "true"));
  if (rdy->refine) PetscCall(PetscOptionsSetValue(NULL, "-ref_dm_plex_transform_label_match_strata", "true"));

  //--------
  // Output
  //--------

  // set the solution monitoring interval (except for XDMF, which does its own thing)
  if ((rdy->config.output.interval > 0) && (rdy->config.output.format != OUTPUT_XDMF)) {
    PetscCall(PetscOptionsHasName(NULL, NULL, "-ts_monitor_solution_interval", &has_param));
    if (!has_param) {
      snprintf(value, VALUE_LEN, "%" PetscInt_FMT "", rdy->config.output.interval);
      PetscOptionsSetValue(NULL, "-ts_monitor_solution_interval", value);
    }
  }

  // allow the user to override our output format
  PetscBool ts_monitor = PETSC_FALSE;
  char      ts_monitor_opt[128];
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ts_monitor_solution", ts_monitor_opt, 128, &ts_monitor));
  if (ts_monitor) {
    // which is it?
    if (strstr(ts_monitor_opt, "cgns")) {
      rdy->config.output.format = OUTPUT_CGNS;
    } else if (strstr(ts_monitor_opt, "xdmf")) {
      rdy->config.output.format = OUTPUT_XDMF;
    } else if (strstr(ts_monitor_opt, "binary")) {
      rdy->config.output.format = OUTPUT_BINARY;
    }
  } else {  // TS monitoring not set on command line
    // the CGNS viewer's options aren't exposed at all by the C API, so we have
    // to set them here
    if (rdy->config.output.format == OUTPUT_CGNS) {
      // configure timestep monitoring parameters
      char file_pattern[PETSC_MAX_PATH_LEN];
      PetscCall(DetermineOutputFile(rdy, 0, 0.0, "cgns", file_pattern));
      snprintf(value, VALUE_LEN, "cgns:%s", file_pattern);
      PetscOptionsSetValue(NULL, "-ts_monitor_solution", value);
    }
  }

  // set the solution monitoring interval (except for XDMF, which does its own thing)
  if ((rdy->config.output.interval > 0) && (rdy->config.output.format != OUTPUT_XDMF)) {
    PetscCall(PetscOptionsHasName(NULL, NULL, "-ts_monitor_solution_interval", &has_param));
    if (!has_param) {
      snprintf(value, VALUE_LEN, "%" PetscInt_FMT, rdy->config.output.interval);
      PetscOptionsSetValue(NULL, "-ts_monitor_solution_interval", value);
    }
  }

  // adjust the CGNS output batch size if needed
  if (rdy->config.output.format == OUTPUT_CGNS) {
    PetscCall(PetscOptionsHasName(NULL, NULL, "-viewer_cgns_batch_size", &has_param));
    if (!has_param) {
      snprintf(value, VALUE_LEN, "%" PetscInt_FMT "", rdy->config.output.batch_size);
      PetscOptionsSetValue(NULL, "-viewer_cgns_batch_size", value);
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
#undef VALUE_LEN
}

typedef struct {
  const char *pattern;
  const char *substitution;
} Substitution;

// supported string substitutions
static const Substitution substitutions[] = {
    {"${PETSC_ID_TYPE}", PETSC_ID_TYPE},
    {NULL,               NULL         }, // terminator
};

// ON RANK 0 ONLY, reads the given file and performs the given set of string
// substitutions, storing the resulting (newly allocated) string in content
// and its size in content_size
static PetscErrorCode ReadAndSubstitute(MPI_Comm comm, const char *filename, const Substitution substitutions[], char **content,
                                        PetscMPIInt *content_size) {
  PetscFunctionBegin;

  FILE *file = NULL;
  PetscCall(PetscFOpen(comm, filename, "r", &file));

  // determine the file's size and read it into a buffer
  fseek(file, 0, SEEK_END);
  PetscMPIInt raw_size = (PetscMPIInt)ftell(file);
  rewind(file);
  char *raw_content;
  PetscCall(PetscCalloc1(raw_size, &raw_content));
  fread(raw_content, sizeof(char), raw_size, file);
  PetscCall(PetscFClose(comm, file));

  // determine the size of the content with all substitutions applied
  PetscInt num_substitutions = 0;
  *content_size              = raw_size;
  for (PetscInt s = 0; substitutions[s].pattern; ++s) {
    const Substitution sub         = substitutions[s];
    PetscInt           pattern_len = (PetscInt)strlen(sub.pattern);
    PetscInt           subst_len   = (PetscInt)strlen(sub.substitution);
    char              *p           = raw_content;
    while (p != NULL) {
      p = strstr(p, sub.pattern);
      if (p != NULL) {
        *content_size += subst_len - pattern_len;
        p += pattern_len;
        ++num_substitutions;
      }
    }
  }

  // perform any needed string substitutions or just use the raw input
  if (num_substitutions > 0) {
    PetscCall(PetscCalloc1(*content_size, content));
    for (PetscInt s = 0; substitutions[s].pattern; ++s) {
      const Substitution sub         = substitutions[s];
      PetscInt           subst_len   = (PetscInt)strlen(sub.substitution);
      PetscInt           pattern_len = (PetscInt)strlen(sub.pattern);
      char              *p = raw_content, *q = *content;
      while (p != NULL) {
        char *new_p = strstr(p, sub.pattern);
        if (new_p != NULL) {
          memcpy(q, p, new_p - p);
          q += new_p - p;
          p = new_p + pattern_len;
          memcpy(q, sub.substitution, subst_len);
          q += subst_len;
        } else {
          memcpy(q, p, raw_size - (p - raw_content));
          q += raw_size - (p - raw_content);
          p = new_p;
        }
      }
      PetscCheck(q - *content == *content_size, comm, PETSC_ERR_USER, "error performing string substitutions in %s!", filename);
    }
    PetscFree(raw_content);
  } else {
    *content = raw_content;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// reads the config file on process 0, broadcasts it as a string to all other
// processes, and parses the string into rdy->config
PetscErrorCode ReadConfigFile(RDy rdy) {
  PetscFunctionBegin;

  // open the config file on process 0, determine its size, and broadcast its
  // contents to all other processes.
  char       *config_str;
  PetscMPIInt config_size;
  if (rdy->rank == 0) {
    // process 0: read the file and perform substitutions
    PetscCall(ReadAndSubstitute(rdy->comm, rdy->config_file, substitutions, &config_str, &config_size));

    // broadcast the size of the content and then the content itself
    MPI_Bcast(&config_size, 1, MPI_INT, 0, rdy->comm);
    MPI_Bcast(config_str, config_size, MPI_CHAR, 0, rdy->comm);
  } else {
    // other processes: read the size of the content
    MPI_Bcast(&config_size, 1, MPI_LONG, 0, rdy->comm);

    // recreate the configuration string.
    PetscCall(PetscCalloc1(config_size, &config_str));
    MPI_Bcast(config_str, config_size, MPI_CHAR, 0, rdy->comm);
  }

  // parse the YAML config file into a new config struct and validate it
  RDyConfig *config;
  PetscCall(ParseYaml(rdy->comm, config_str, &config));
  PetscCall(ValidateConfig(rdy->comm, config));

  // copy the config into place and dispose of it
  rdy->config = *config;
  PetscFree(config);

  // set any additional options needed in PETSc's options database
  PetscCall(SetAdditionalOptions(rdy));

  // clean up
  PetscFree(config_str);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// =============
//  PrintConfig
// =============

static const char *FlagString(PetscBool flag) { return flag ? "enabled" : "disabled"; }

static PetscErrorCode PrintPhysics(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Physics:");
  RDyLogDetail(rdy, "  Flow:");
  RDyLogDetail(rdy, "    Bed friction: %s", FlagString(rdy->config.physics.flow.bed_friction));
  RDyLogDetail(rdy, "  Sediment model: %s", FlagString(rdy->config.physics.sediment));
  RDyLogDetail(rdy, "  Salinity model: %s", FlagString(rdy->config.physics.salinity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *SpatialString(RDyNumericsSpatial method) {
  static const char *strings[2] = {"finite volume (FV)", "finite element (FE)"};
  return strings[method];
}

static const char *TemporalString(RDyNumericsTemporal method) {
  static const char *strings[3] = {"forward euler", "4th-order Runge-Kutta", "backward euler"};
  return strings[method];
}

static const char *RiemannString(RDyNumericsRiemann solver) {
  static const char *strings[2] = {"roe", "hllc"};
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

static const char *TimeUnitString(RDyTimeUnit unit) {
  static const char *strings[6] = {"seconds", "minutes", "hours", "days", "months", "years"};
  return strings[unit];
}

static PetscErrorCode PrintTime(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Time:");
  RDyLogDetail(rdy, "  Final time: %g %s", rdy->config.time.final_time, TimeUnitString(rdy->config.time.unit));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintCheckpoint(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Checkpoint:");
  if (rdy->config.checkpoint.interval > 0) {
    char format[12];
    if (rdy->config.checkpoint.format == PETSC_VIEWER_NATIVE) {
      strcpy(format, "binary");
    } else {
      strcpy(format, "hdf5");
    }
    RDyLogDetail(rdy, "  File format: %s", format);
    RDyLogDetail(rdy, "  interval: %" PetscInt_FMT, rdy->config.checkpoint.interval);
  } else {
    RDyLogDetail(rdy, "  (disabled)");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintRestart(RDy rdy) {
  PetscFunctionBegin;
  RDyLogDetail(rdy, "Restart:");
  if (rdy->config.restart.file[0]) {
    RDyLogDetail(rdy, "  File: %s", rdy->config.restart.file);
    if (rdy->config.restart.reinitialize) {
      RDyLogDetail(rdy, "  Reinitialize: true");
    } else {
      RDyLogDetail(rdy, "  Reinitialize: false");
    }
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

// Prints config information at the requested log level.
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
