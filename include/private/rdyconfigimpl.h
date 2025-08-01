#ifndef RDYCONFIG_H
#define RDYCONFIG_H

#include <float.h>
#include <limits.h>
#include <petscviewer.h>
#include <private/rdylogimpl.h>
#include <private/rdymmsconfigimpl.h>
#include <rdycore.h>

// The types in this file ѕerve as an intermediate representation for our input
// configuration file:
//
// https://rdycore.github.io/RDycore/user/input.html
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

// the maximum number of flow/sediment/salinity conditions that can be defined for a simulation
#define MAX_NUM_CONDITIONS 32

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

typedef enum {
  SOURCE_SEMI_IMPLICIT = 0,  // semi-implicit treatment
  SOURCE_IMPLICIT_XQ2018     // implicit treatment using Xilin and Qiuhua (2018)
} RDyFlowSourceMethod;

typedef struct {
  RDyFlowSourceMethod method;            // temporal discretization method for source term
  PetscReal           xq2018_threshold;  // threshold for the XQ2018's implicit time integration of source term
} RDyFlowSource;

// physics flow parameters
typedef struct {
  RDyPhysicsFlowMode mode;             // flow mode
  PetscReal          tiny_h;           // depth below which no flow occurs
  PetscReal          h_anuga_regular;  // ANUGA height parameter used for velocity regularization.
  RDyFlowSource      source;
} RDyPhysicsFlow;

typedef struct {
  PetscInt num_classes;  // number of sediment classes
} RDyPhysicsSD;

// all physics parameters
typedef struct {
  RDyPhysicsFlow flow;      // flow parameters
  RDyPhysicsSD   sediment;  // sediment effects enabled?
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

typedef struct {
  PetscBool enable;                 // true = use adaptive time step
  PetscReal target_courant_number;  // target courant number
  PetscReal max_increase_factor;    // max allowable increase in timestep
  PetscReal initial_time_step;      // initial timestep
} RDyTimeAdaptiveSection;

// all time parameters
typedef struct {
  PetscReal              final_time;         // final simulation time [unit]
  RDyTimeUnit            unit;               // unit in which time is expressed
  PetscInt               max_step;           // maximum number of simulation time steps
  PetscReal              time_step;          // minimum internal time step [unit]
  PetscReal              coupling_interval;  // time interval spanned by RDyAdvance [unit]
  RDyTimeAdaptiveSection adaptive;           // adaptive time step for explicity time integration
} RDyTimeSection;

// ---------------
// logging section
// ---------------

// all logging parameters
typedef struct {
  char        file[PETSC_MAX_PATH_LEN];  // log filename
  RDyLogLevel level;                     // selected log level
} RDyLoggingSection;

// -------------------
// checkpoint section
// -------------------

// all checkpoint parameters
typedef struct {
  PetscViewerFormat format;                      // file format
  PetscInt          interval;                    // number of steps between checkpoints [steps]
  char              prefix[PETSC_MAX_PATH_LEN];  // prefix of checkpoint files
} RDyCheckpointSection;

// ---------------
// restart section
// ---------------

// all restart parameters
typedef struct {
  char      file[PETSC_MAX_PATH_LEN];  // checkpoint file from which to restart
  PetscBool reinitialize;              // PETSC_TRUE resets simulation time to 0
} RDyRestartSection;

// --------------
// output section
// --------------

// output file formats
typedef enum { OUTPUT_NONE = 0, OUTPUT_BINARY, OUTPUT_XDMF, OUTPUT_CGNS } RDyOutputFormat;

// observations
typedef struct {
  PetscInt *cells;
  PetscInt  cells_count;
} RDyObservationSites;

typedef struct {
  PetscBool instantaneous;
} RDyObservationTimeSampling;

typedef struct {
  PetscInt                   interval;
  RDyObservationSites        sites;
  RDyObservationTimeSampling time_sampling;
} RDyObservationsSection;

// time series output interval parameters appended to files
typedef struct {
  PetscInt               boundary_fluxes;  // written to "boundary_fluxes.dat" [steps between outputs]
  RDyObservationsSection observations;
} RDyTimeSeries;

// all output parameters
typedef struct {
  PetscBool       enable;                         // true if output is requested
  char            directory[PETSC_MAX_PATH_LEN];  // output directory
  char          **fields;                         // array of selected output field names
  PetscInt        fields_count;                   // number of selected output fields
  RDyOutputFormat format;                         // file format
  PetscInt        output_interval;                // output interval [steps between outputs]
  PetscInt        time_interval;                  // temporal interval at which output is written
  RDyTimeUnit     time_unit;                      // unit of time for temporal interval
  PetscInt        batch_size;                     // number of timesteps per output file (if available)
  RDyTimeSeries   time_series;                    // time series appended to text (.dat) files
  PetscReal       prev_output_time;               // previous time at which output was written
  PetscBool       separate_grid_file;             // whether grid is written to a separate file
} RDyOutputSection;

// ------------
// grid section
// ------------

// all grid parameters
typedef struct {
  char file[PETSC_MAX_PATH_LEN];  // grid file
} RDyGridSection;

// ---------------------------
// surface_composition section
// ---------------------------

// an association between a region and a material
typedef struct {
  char region[MAX_NAME_LEN + 1];    // name of related region
  char material[MAX_NAME_LEN + 1];  // name of related material
} RDySurfaceCompositionSpec;

// -----------------------
// materials section
// -----------------------

// the specification of a single material property, given by a value to be read
// from a file
typedef struct {
  MathExpression    expression;                // expression for property value
  void             *value;                     // muparser-backed functional form
  char              file[PETSC_MAX_PATH_LEN];  // file from which data is to be read
  PetscViewerFormat format;                    // file format
} RDyMaterialPropertySpec;

// the specification of a set of properties defining a material, each of which
// is given by a value or read from a file
typedef struct {
  RDyMaterialPropertySpec manning;  // Manning roughness coefficient
} RDyMaterialPropertiesSpec;

// the specification of a material as a collection of material properties
typedef struct {
  char                      name[MAX_NAME_LEN + 1];  // the name of the material
  RDyMaterialPropertiesSpec properties;              // collection of material properties
} RDyMaterialSpec;

// ----------------------
// regions and boundaries
// ----------------------

// This type associates a named region with an integer ID representing a
// disjoint set of cells in a grid file.
typedef struct {
  char     name[MAX_NAME_LEN + 1];  // human-readable name of region
  PetscInt grid_region_id;          // ID of region cell set within grid file
} RDyRegionSpec;

// This type associates a named boundary with an integer ID representing a
// disjoint set of edges in a grid file.
typedef struct {
  char     name[MAX_NAME_LEN + 1];  // human-readable name of boundary
  PetscInt grid_boundary_id;        // ID of boundary edge set within grid file
} RDyBoundarySpec;

// ---------------------------------------
// initial, boundary and source conditions
// ---------------------------------------
// The following data structures are used in several sections.

// This type defines an initial condition and/or source/sink with named
// flow, sediment, and salinity conditions.
typedef struct {
  char region[MAX_NAME_LEN + 1];                              // name of associated region
  char flow[MAX_NAME_LEN + 1];                                // name of related flow condition
  char sediment[MAX_NUM_SEDIMENT_CLASSES][MAX_NAME_LEN + 1];  // name of related sediment condition
  char salinity[MAX_NAME_LEN + 1];                            // name of related salinity condition
} RDyRegionConditionSpec;

// This type defines a boundary condition with named flow, sediment, and
// salinity conditions.
typedef struct {
  PetscInt num_boundaries;                                        // number of associated boundaries
  char     boundaries[MAX_NUM_BOUNDARIES][MAX_NAME_LEN + 1];      // names of associated boundaries
  char     flow[MAX_NAME_LEN + 1];                                // name of related flow condition
  char     sediment[MAX_NUM_SEDIMENT_CLASSES][MAX_NAME_LEN + 1];  // name of related sediment conditions
  char     salinity[MAX_NAME_LEN + 1];                            // name of related salinity condition
} RDyBoundaryConditionSpec;

// -----------------------
// flow_conditions section
// -----------------------

// flow-related condition data
typedef struct {
  char              name[MAX_NAME_LEN + 1];
  RDyConditionType  type;
  MathExpression    height_expression;      // expression for water height
  MathExpression    x_momentum_expression;  // expression for water y-momentum
  MathExpression    y_momentum_expression;  // expression for water y-momentum
  MathExpression    value_expression;       // expression for value
  void             *height;                 // muparser-backed functional form
  void             *x_momentum;             // muparser-backed functional form
  void             *y_momentum;             // muparser-backed functional form
  void             *value;                  // muparser-backed functional form
  char              file[PETSC_MAX_PATH_LEN];
  PetscViewerFormat format;
} RDyFlowCondition;

// ---------------------------
// sediment_conditions section
// ---------------------------

// sediment-related condition data
typedef struct {
  char              name[MAX_NAME_LEN + 1];
  RDyConditionType  type;
  MathExpression    expression;     // expression for concentration
  void             *concentration;  // muparser-backed functional form
  char              file[PETSC_MAX_PATH_LEN];
  PetscViewerFormat format;
} RDySedimentCondition;

// ---------------------------
// salinity_conditions section
// ---------------------------

// salinity-related condition data
typedef struct {
  char              name[MAX_NAME_LEN + 1];
  RDyConditionType  type;
  MathExpression    expression;     // expression for concentration
  void             *concentration;  // muparser-backed functional form
  char              file[PETSC_MAX_PATH_LEN];
  PetscViewerFormat format;
} RDySalinityCondition;

// ----------------
// ensemble section
// ----------------

// specification of an ensemble member with overridable parameters
typedef struct {
  char                  name[MAX_NAME_LEN + 1];
  RDyGridSection        grid;
  RDyMaterialSpec      *materials;
  PetscInt              materials_count;
  RDyFlowCondition     *flow_conditions;
  PetscInt              flow_conditions_count;
  RDySedimentCondition *sediment_conditions;
  PetscInt              sediment_conditions_count;
  RDySalinityCondition *salinity_conditions;
  PetscInt              salinity_conditions_count;
} RDyEnsembleMember;

// specification of an ensemble
typedef struct {
  PetscInt           size;
  RDyEnsembleMember *members;
  PetscInt           members_count;  // set automatically; must be equal to size!
} RDyEnsembleSection;

// =======================================
// Intermediate Config File Representation
// =======================================

// a representation of the config file itself
typedef struct {
  // Each of these fields represents a section of the YAML config file.
  RDyPhysicsSection  physics;
  RDyNumericsSection numerics;
  RDyTimeSection     time;

  RDyLoggingSection    logging;
  RDyCheckpointSection checkpoint;
  RDyRestartSection    restart;
  RDyOutputSection     output;

  RDyGridSection grid;

  PetscInt        num_materials;
  RDyMaterialSpec materials[MAX_NUM_MATERIALS];

  PetscInt      num_regions;
  RDyRegionSpec regions[MAX_NUM_REGIONS];

  PetscInt                  num_material_assignments;
  RDySurfaceCompositionSpec surface_composition[MAX_NUM_REGIONS];

  PetscInt               num_initial_conditions;
  RDyRegionConditionSpec initial_conditions[MAX_NUM_REGIONS];

  PetscInt               num_sources;
  RDyRegionConditionSpec sources[MAX_NUM_REGIONS];

  PetscInt        num_boundaries;
  RDyBoundarySpec boundaries[MAX_NUM_BOUNDARIES];

  PetscInt                 num_boundary_conditions;
  RDyBoundaryConditionSpec boundary_conditions[MAX_NUM_BOUNDARIES];

  PetscInt             num_flow_conditions;
  RDyFlowCondition     flow_conditions[MAX_NUM_CONDITIONS];
  PetscInt             num_sediment_conditions;
  RDySedimentCondition sediment_conditions[MAX_NUM_CONDITIONS];
  PetscInt             num_salinity_conditions;
  RDySalinityCondition salinity_conditions[MAX_NUM_CONDITIONS];

  RDyEnsembleSection ensemble;

  // MMS-specific section (used only by the MMS driver)
  RDyMMSSection mms;

} RDyConfig;

// ensemble member configuration (see ensemble.c)
PETSC_INTERN PetscErrorCode ConfigureEnsembleMember(RDy);

#endif
