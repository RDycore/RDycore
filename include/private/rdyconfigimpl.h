#ifndef RDYCONFIG_H
#define RDYCONFIG_H

#include <float.h>
#include <limits.h>
#include <petscviewer.h>
#include <private/rdylogimpl.h>

// The types in this file ѕerve as an intermediate representation for our input
// configuration file:
//
// https://rdycore.atlassian.net/wiki/spaces/PD/pages/24576001/RDycore+configuration+file
//

// sentinel values for uninitialized/invalid data
#define INVALID_REAL -DBL_MAX
#define INVALID_INT -INT_MAX

// the maximum number of regions that can be defined on a domain
#define MAX_NUM_REGIONS 32

// the maximum number of boundaries that can be defined on a domain
#define MAX_NUM_BOUNDARIES 32

// the maximum number of materials that can be defined for a simulation
#define MAX_NUM_MATERIALS 32

// the maximum number of flow/sediment/salinity conditions that can be defined for a
// simulation
#define MAX_NUM_CONDITIONS 32

// the maximum length of a string referring to a name in the config file
#define MAX_NAME_LEN 128

// The data structures below are intermediate representations of the sections
// in the YAML configuration file. We parse this file with a YAML parser that
// populates these data structures using a YAML schema. The parser is implemented
// in src/real_config_file.c. Any changes to these data structures must be
// reflected in the corresponding schema in that file.

// ---------------
// physics section
// ---------------

// physics flow modes
typedef enum {
  FLOW_SWE = 0,   // shallow water equations
  FLOW_DIFFUSION  // diffusion equation
} RDyPhysicsFlowMode;

// physics flow parameters
typedef struct {
  RDyPhysicsFlowMode mode;          // flow mode
  PetscBool          bed_friction;  // bed friction enabled?
  PetscReal          tiny_h;        // depth below which no flow occurs
} RDyPhysicsFlow;

// all physics parameters
typedef struct {
  RDyPhysicsFlow flow;      // flow parameters
  PetscBool      sediment;  // sediment effects enabled?
  PetscBool      salinity;  // salinity effects enabled?
} RDyPhysicsSection;

// ----------------
// numerics section
// ----------------

// spatial discretization methods
typedef enum {
  SPATIAL_FV = 0,  // finite volume method
  SPATIAL_FE       // finite element method
} RDyNumericsSpatial;

// temporal discretization methods
typedef enum {
  TEMPORAL_EULER = 0,  // forward euler method
  TEMPORAL_RK4,        // 4th-order Runge-Kutta method
  TEMPORAL_BEULER      // backward euler method
} RDyNumericsTemporal;

// riemann solvers for horizontal flow
typedef enum {
  RIEMANN_ROE = 0,  // Roe solver
  RIEMANN_HLLC      // Harten, Lax, van Leer Contact solver
} RDyNumericsRiemann;

// all numerics parmeters
typedef struct {
  RDyNumericsSpatial  spatial;
  RDyNumericsTemporal temporal;
  RDyNumericsRiemann  riemann;
} RDyNumericsSection;

// ------------
// time section
// ------------

// time units
typedef enum { TIME_SECONDS = 0, TIME_MINUTES, TIME_HOURS, TIME_DAYS, TIME_MONTHS, TIME_YEARS } RDyTimeUnit;

// all time parameters
typedef struct {
  PetscReal   final_time;  // final simulation time [unit]
  RDyTimeUnit unit;        // unit in which time is expressed
  PetscInt    max_step;    // maximum number of simulation time steps
  PetscReal   dtime;       // time step [unit]
} RDyTimeSection;

// ---------------
// logging section
// ---------------

// all logging parameters
typedef struct {
  char        file[PETSC_MAX_PATH_LEN];  // log filename
  RDyLogLevel level;                     // selected log level
} RDyLoggingSection;

// ---------------
// restart section
// ---------------

// all restart parameters
typedef struct {
  PetscViewerFormat format;     // file format
  PetscInt          frequency;  // number of steps between restart dumps
} RDyRestartSection;

// ---------------
// output section
// ---------------

// output file formats
typedef enum { OUTPUT_BINARY = 0, OUTPUT_XDMF, OUTPUT_CGNS } RDyOutputFormat;

// all output parameters
typedef struct {
  RDyOutputFormat format;      // file format
  PetscInt        frequency;   // frequency of output [steps]
  PetscInt        batch_size;  // number of timesteps per output file (if available)
} RDyOutputSection;

// ------------
// grid section
// ------------

// all grid parameters
typedef struct {
  char file[PETSC_MAX_PATH_LEN];  // mesh file
} RDyGridSection;

// file-based domain-wide condition/material specifications
typedef struct {
  char              file[PETSC_MAX_PATH_LEN];  // file specifying domain-wide conditions
  PetscViewerFormat format;                    // format of file
} RDyDomainConditions;

// ---------------------------
// surface_composition section
// ---------------------------

// a named regional material specification
typedef struct {
  PetscInt id;                          // ID of region for related material
  char     material[MAX_NAME_LEN + 1];  // name of related material
} RDyMaterialSpec;

typedef struct {
  char manning[PETSC_MAX_PATH_LEN];
} RDySurfaceCompositionFiles;

typedef struct {
  PetscViewerFormat          format;
  RDySurfaceCompositionFiles files;
} RDySurfaceCompositionDomain;

// all surface composition data
typedef struct {
  RDySurfaceCompositionDomain domain;                        // domain-wide material properties
  PetscInt                    num_regions;                   // number of per-region materials
  RDyMaterialSpec             by_region[MAX_NUM_MATERIALS];  // materials by region
} RDySurfaceCompositionSection;

// -----------------------
// materials section
// -----------------------

// a material with specific properties
// (undefined properties are set to INVALID_INT/INVALID_REAL)
typedef struct {
  char      name[MAX_NAME_LEN + 1];
  PetscReal manning;  // Manning's coefficient [s/m**(1/3)]
} RDyMaterial;

// ---------------------------------------
// initial, boundary and source conditions
// ---------------------------------------
// The following data structures are used in several sections.

// "kinds" of initial/boundary/source conditions applied to regions/boundaries
typedef enum {
  CONDITION_DIRICHLET = 0,    // Dirichlet condition (value is specified)
  CONDITION_NEUMANN,          // Neumann condition (derivative is specified)
  CONDITION_REFLECTING,       // Reflecting condition
  CONDITION_CRITICAL_OUTFLOW  // Critical flow
} RDyConditionType;

// This type defines an initial/boundary condition and/or source/sink with named
// flow, sediment, and salinity conditions.
typedef struct {
  PetscInt id;                          // ID of region or boundary for related condition
  char     flow[MAX_NAME_LEN + 1];      // name of related flow condition
  char     sediment[MAX_NAME_LEN + 1];  // name of related sediment condition
  char     salinity[MAX_NAME_LEN + 1];  // name of related salinity condition
} RDyConditionSpec;

// --------------------------
// initial_conditions section
// --------------------------

// all initial conditions
typedef struct {
  RDyDomainConditions domain;                      // domain-wide conditions
  PetscInt            num_regions;                 // number of per-region conditions defined
  RDyConditionSpec    by_region[MAX_NUM_REGIONS];  // names of types of conditions
} RDyInitialConditionsSection;

// ---------------------------
// boundary_conditions section
// ---------------------------

// (nothing needed here aside from what's provided in initial_conditions)

// ---------------
// sources section
// ---------------

// all source conditions (identical to initial conditions in structure)
typedef RDyInitialConditionsSection RDySourcesSection;

// -----------------------
// flow_conditions section
// -----------------------

// flow-related condition data
typedef struct {
  char             name[MAX_NAME_LEN + 1];
  RDyConditionType type;
  PetscReal        height;
  PetscReal        momentum[2];
} RDyFlowCondition;

// ---------------------------
// sediment_conditions section
// ---------------------------

// sediment-related condition data
typedef struct {
  char             name[MAX_NAME_LEN + 1];
  RDyConditionType type;
  PetscReal        concentration;
} RDySedimentCondition;

// ---------------------------
// salinity_conditions section
// ---------------------------

// salinity-related condition data
typedef struct {
  char             name[MAX_NAME_LEN + 1];
  RDyConditionType type;
  PetscReal        concentration;
} RDySalinityCondition;

// a representation of the config file itself
typedef struct {
  // Each of these fields represents a section of the YAML config file.
  RDyPhysicsSection  physics;
  RDyNumericsSection numerics;
  RDyTimeSection     time;

  RDyLoggingSection logging;
  RDyRestartSection restart;
  RDyOutputSection  output;

  RDyGridSection grid;

  RDySurfaceCompositionSection surface_composition;
  PetscInt                     num_materials;
  RDyMaterial                  materials[MAX_NUM_MATERIALS];

  RDyInitialConditionsSection initial_conditions;
  RDySourcesSection           sources;

  PetscInt         num_boundary_conditions;
  RDyConditionSpec boundary_conditions[MAX_NUM_BOUNDARIES];

  PetscInt             num_flow_conditions;
  RDyFlowCondition     flow_conditions[MAX_NUM_CONDITIONS];
  PetscInt             num_sediment_conditions;
  RDySedimentCondition sediment_conditions[MAX_NUM_CONDITIONS];
  PetscInt             num_salinity_conditions;
  RDySalinityCondition salinity_conditions[MAX_NUM_CONDITIONS];

} RDyConfig;

#endif
