# RDycore YAML Input Specification

You can configure an RDycore simulation by creating a text file that uses
the [YAML](https://yaml.org/) markup language. Typically, these files have a
`.yml` or `.yaml` suffix like [`ex2b.yaml`](https://github.com/RDycore/RDycore/blob/main/driver/tests/swe_roe/ex2b.yaml).
In this section, we describe how to express the specifics for your simulation
using the YAML syntax.

RDycore's YAML input is broken up into several sections, each responsible for a
different aspect of the desired simulation. These sections fall into several
broad categories:

* **Model equations and discretizations**
    * [physics](input.md#physіcs): configures the different physical models
      represented within RDycore
    * [numerics](input.md#numerics): specifies the numerical methods used
      to solve the model equations
    * [grid](input.md#grid): defines RDycore's discrete computational domain
    * [regions](input.md#regions): associates regions (disjoint sets of cells)
      defined in a grid file with human-readable names
    * [boundaries](input.md#boundaries): associates boundaries (disjoint sets of
      edges) defined in a grid file with human-readable names
    * [time](input.md#time): defines the timespan for the simulation, sets
      limits and units for time stepping
* **Simulation diagnostics, output, and restarts**
    * [logging](input.md#logging): controls informational messages logged to
      files or to the terminal
    * [output](input.md#output): configures simulation output, including
      scalable I/O formats and related parameters
    * [restart](input.md#restart): configures simulation restart dumps
* **Material properties**
    * [materials](input.md#materials): defines the materials available to
      the simulation
    * [surface_composition](input.md#surface_composition): associates
      materials defined in the `materials` section with files in which the
      relevant material properties are stored
* **Initial and boundary conditions, source terms**
    * [initial_conditions](input.md#initial_conditions): associates initial
      conditions (as defined in `flow_conditions`, `sediment_conditions`, and/or
      `salinity_conditions`) with specific regions defined in the `regions`
      section
    * [sources](input.md#sources): associates source contributions
      (as defined in `flow_conditions`, `sediment_conditions`, and/or
      `salinity_conditions`) with specific regions defined in the `regions`
      section
    * [boundary_conditions](input.md#boundary_conditions): associates boundary
      conditions (as defined in `flow_conditions`, `sediment_conditions`, and/or
      `salinity_conditions`) with specific boundaries defined in the
      `boundaries` section
    * [flow_conditions](input.md#flow_conditions): defines flow-related
      parameters that can be used to specify initial/boundary conditions and
      sources
    * [sediment_conditions](input.md#sediment_conditions): defines sediment-related
      parameters that can be used to specify initial/boundary conditions and
      sources
    * [salinity_conditions](input.md#salinity_conditions): defines salinity-related
      parameters that can be used to specify initial/boundary conditions and
      sources

Each of these sections is described below, with a motivating example.

## `boundaries`

```yaml
boundaries:
  - name: top_wall
    grid_boundary_id: 2
  - name: bottom_wall
    grid_boundary_id: 3
  - name: exterior
    grid_boundary_id: 1
```

The `boundaries` section is a sequence (list) of boundary definitions, each of
which contains the following parameters:

* `name`: a human-readable name for the boundary
* `grid_boundary_id`: an integer identifier associated with a disjoint set of
  grid edges. If this identifier is not found within the grid file specified
  in the `grid` section, a fatal error occurs.

Boundary definitions can appear in any order within the sequence. The
`boundaries` section is optional, and need not be specified if you don't need to
associate a specific boundary condition with a specific boundary.

## `boundary_conditions`

```yaml
boundary_conditions:
  - boundaries: [top_wall]
    flow: reflecting_bc
  - boundaries: [bottom_wall]
    flow: outflow_bc
```

The `boundary_conditions` section is a sequence (list) associating `flow`,
`sediment`, and `salinity` conditions (as defined in their respective sections)
with boundaries (as defined in the `boundaries` section). A boundary can have
at most one set of boundary conditions associated with it, but a boundary
condition can be associated with multiple boundaries, as indicated by the
`boundaries` field, which accepts a list of boundary names. If no boundary
conditions are given for a specific boundary, that boundary is assigned an
automatically-generated reflecting boundary condition and homogeneous Neumann
sediment and salinity conditions.

The above example shows a valid configuration for a simulation in which sediments
and salinity are not modeled, so only the `flow` parameter is required. The
presence of sediment boundary concentrations requires the `sediment` parameter,
as the presence of salinity concentrations requires the `salinity` parameter.

Like the `boundaries` section, the `boundary_conditions` section is optional.
If no boundary conditions are specified, all boundaries are assumed to have a
reflecting boundary condition.

## `flow_conditions`

```yaml
flow_conditions:
  - name: dam_top_ic
    type: dirichlet
    height: 10
    momentum: [0, 0]
  - name: dam_bottom_ic
    type: dirichlet
    file: dam_ics.dat
    format: binary
  - name: reflecting_bc
    type: reflecting
  - name: outflow_bc
    type: critical-outflow
```

The `flow_conditions` section contains a sequence of sets of parameters defining
flow within a cell or cell boundary (edge). These flow conditions can be used
to define initial conditions, source contributions, and boundary conditions
elsewhere in the file. The parameters that define a flow condition are

* `name`: a human-readable name that can be used to refer to this flow condition
* `type`: the type of constraint applied by this flow condition. Available
  options are
    * `dirichlet`: the condition explicitly specifies the value of relevant
      flow variables. This is useful for Dirichlet boundary conditions,
      initial conditions, and source terms.
    * `neumann`: the condition explicitly specifies the value of the directional
      derivative of relevant flow variables on cell boundaries. This is useful
      only for boundary conditions. Currently, only homogeneous Neumann conditions
      are supported.
    * `reflecting`: the condition specifies that no flow occurs through a given
      boundary, and that the boundary reflects the momentum contained in the flow.
      Useful only for boundary conditions.
    * `critical-outflow`: the condition specifies that flow through a boundary
      is defined by a critical outflow condition. Useful only for boundary
      conditions.

In the case of a Dirichlet condition, flow is prescribed by providing parameters
to set the water height and momentum. This can be done in one of two ways:

1. By specifying the parameters directly using the following fields:

    * `height`: the height of water [m] at the relevant point (within a cell
      or on its boundary)
    * `momentum`: a 2-component sequence/list containing the `x` and `y` components
      of the momentum [kg m/s] at the relevant point (within a cell or on its
      boundary)

2. By specifying a file from which data for these parameters is to be read. The
   data will be read into the components of the solution vector that correspond
   to the cells belonging to the region to which this flow condition is
   assigned:

   * `file`: the path for the file from which data is read, specified relative
     to the directory in which the RDycore executable was run.
   * `format`: the format of the data in the file, which current must be
     `binary` (specifying PETSc's binary format).

## `grid`

```yaml
grid:
  file: breaking-dam.exo
```

The `grid` section defines the computational domain used by RDycore. Currently,
has only a single parameter:

* `file`: the file containing the grid representing the computational domain.
  This grid must be stored in a format supported by PETSc's [DMPlex](https://petsc.org/release/manual/dmplex/)
  data structure.

## `logging`

```yaml
logging:
  file: rdycore.log
  level: info
```

The `logging` section controls how messages emitted by RDycore are logged.
The relevant parameters in this section are

* `file`: the file to which logged messages are written, relative to the
  directory in which RDycore (coupled or uncoupled) is executed. If this
  parameter is omitted, logged messages are written to standard output.
* `level`: the desired level of detail that determines which messages are
  logged. Available options are
    * `none`: no messages are logged
    * `warning`: only warnings/urgent messages are logged
    * `info`: warnings and informational messages are logged
    * `detail`: warnings, informational messages, and messages with some
      degree of technical detail are logged
    * `debug`: all messages including debugging prints are logged

## `materials`

```yaml
materials:
  - name: smooth
    properties:
      manning:
        value: 0.15
  - name: rough
    properties:
      manning:
        file: rough-manning.dat
        format: binary
```

The `materials` section is a sequence (list) of named materials defined by
specific material properties. Each material is essentially a named list of
material properties specified either directly by value or by data in a specific
file with a specific format. A material itself is specified by the following
fields:

* `name`: a human-readable name that can be used to refer to a material
* `properties`: a mapping of material properties, with property names
  mapped to one or more of the following fields:
    * `value`: the value of the material property (omitted when `file` is specified)
    * `file`: the name of a file from which the property is to be read (omitted when `value` is specified)
    * `format`: the format of the specified file (if any)

Valid material properties are:

* `manning` (required): the value of the [Manning roughness coefficient](https://en.wikipedia.org/wiki/Manning_formula)
  for the material

## `numerics`

```yaml
numerics:
  spatial: fv
  temporal: euler
  riemann: roe
```

The `numerics` section defines the spatial and temporal discretizations used by
RDycore. The parameters that define these discretizations are

* `spatial`: determines the spatial discretization. Can be either `fv` for
  a finite volume method, or `fe` for a finite element method. Currently, only
  `fv` is implemented. Default value: `fv`
* `temporal`: determines the method of time integration used. Can be `euler`
  for the forward Euler method, `rk4` for a 4th-order Runge-Kutta method, or
  `beuler` for the L-stable backward Euler method. Currently, only `euler` and
  `rk4` are supported. Default value: `euler`
* `riemann`: determines the form of the Riemann solver used for the shallow
  water equations. Can be `roe` for the Roe solver or `hllc` for the HLLC solver.
  Currently, only `roe` is implemented. Default value: `roe`

## `output`

```yaml
output:
  format: xdmf
  interval: 100
  batch_size: 1
  time_series:
    boundary_fluxes: 10
```

The `output` section control simulation output, including visualization and
time series data (and excluding restart data). Relevant parameters are

* `format`: the format of the output written. Available options are
    * `none`: no output is written. This is the default value.
    * `binary`: output is written using PETSc's binary data format
    * `xdmf`: output is written to the [XDMF](https://xdmf.org/index.php/XDMF_Model_and_Format) format
    * `cgns`: output is written to the [CFD General Notation System (CGNS)](https://cgns.github.io/) format
* `interval`: the number of time steps between output dumps. Default value: 0 (no output)
* `batch_size`: the number of time steps for which output data is stored in a
  single file. For example, a batch size of 10 specifies that each individual
  output file stores data for 10 time steps. Default value: 1
* `time_series`: this subsection controls time series simulation output, which
  is useful for inspection and possibly even coupling. Currently, this subsection
  has only one parameter:
    * `boundary_fluxes`: the interval (number of timesteps) at which boundary
      flux data is appended to a tab-delimited text file

## `physics`

```yaml
physics:
  flow:
    mode: swe
    bed_friction: false
    tiny: 1e-7
  sediment: false
  salinity: false
```

The `physics` section determines which model physics are active in RDycore.
There are three available physical models.

First is the flow model, which is configured in the `flow` subsection. This
model determines how flooding is represented within RDycore. The relevant
parameters in this subsection are:

* `mode`, which determines how the height of floodwater is computed. Valid
   parameters are `swe` ([shallow water equations](https://en.wikipedia.org/wiki/Shallow_water_equations))
   and `diffusive` ([diffusive wave approximation](https://en.wikipedia.org/wiki/Shallow_water_equations#Diffusive_wave),
   not yet supported). This parameter is required and has no default value.
* `bed_friction`, which can be set to `true` or `false` to enable/disable
  riverbed friction. Default value: `false`
* `tiny_h`, which is the water height below which a given point is assumed to
  be dry. Default value: `1e-7`

The second physical model is the sediment model. You can enable or disable this
by setting the `sediment` parameter to `true` or `false`.

The third physical model is the salinity model, which you can also enable or
disable this by setting the `salinity` parameter to `true` or `false`.

## `regions`

```yaml
regions:
  - name: upstream
    grid_region_id: 2
  - name: downstream
    grid_region_id: 1
```

The `regions` section is a sequence (list) of regions definitions, each of
which contains the following parameters:

* `name`: a human-readable name for the boundary
* `grid_region_id`: an integer identifier associated with a disjoint set of
  grid cells. If this identifier is not found within the grid file specified
  in the `grid` section, a fatal error occurs.

Region definitions can appear in any order within the sequence.

## `initial_conditions`

```yaml
initial_conditions:
  - region: upstream
    flow: dam_top_ic
  - region: downstream
    flow: dam_bottom_ic
```

The `initial_conditions` section is a sequence (list) associating `flow`,
`sediment`, and `salinity` conditions (as defined in their respective sections)
with regions (as defined in the `regions` section). A region must have exactly
one set of initial conditions associated with it. The above example shows a valid
configuration for a simulation in which sediments and salinity are not modeled,
so only the `flow` parameter is required. The presence of sediment concentrations
requires the `sediment` parameter, as the presence of salinity concentrations
requires the `salinity` parameter.

## `restart`

This section is not yet implemented.

## `salinity_conditions`

```yaml
salinity_conditions:
  - name: my-sal-condition
    type: dirichlet
    concentration: 1
```

The `salinity_conditions` section contains a sequence of sets of parameters
defining the salinity concentration within a cell or cell boundary (edge).
A salinity condition can be used to define initial conditions, source
contributions, and boundary conditions elsewhere in the file. The parameters
that define a salinity condition are

* `name`: a human-readable name that can be used to refer to the salinity condition
* `type`: the type of constraint applied by this salinity condition. Available
  options are
    * `dirichlet`: the condition explicitly specifies the value of the
      salinity concentration. This is useful for Dirichlet boundary conditions,
      initial conditions, and source terms.
    * `neumann`: the condition explicitly specifies the value of the directional
      derivative of the concentration on cell boundaries. This is useful
      only for boundary conditions. Currently, only homogeneous Neumann conditions
      are supported.

In the case of a Dirichlet condition, a salinity concentration is prescribed by
providing one or more parameters. This can be done in one of two ways:

1. By specifying the concentration directly using the `concentration` parameter

2. By specifying a file from which concentration data is to be read. The
   data will be read into the components of the solution vector that correspond
   to the cells belonging to the region to which this flow condition is
   assigned:

   * `file`: the path for the file from which data is read, specified relative
     to the directory in which the RDycore executable was run.
   * `format`: the format of the data in the file, which current must be
     `binary` (specifying PETSc's binary format).

## `sediment_conditions`

```yaml
sediment_conditions:
  - name: my-sed-condition
    type: dirichlet
    concentration: 1
```

The `sediment_conditions` section contains a sequence of sets of parameters
defining the sediment concentration within a cell or cell boundary (edge).
A sediment condition can be used to define initial conditions, source
contributions, and boundary conditions elsewhere in the file. The parameters
that define a sediment condition are

* `name`: a human-readable name that can be used to refer to the sediment condition
* `type`: the type of constraint applied by this sediment condition. Available
  options are
    * `dirichlet`: the condition explicitly specifies the value of the
      sediment concentration. This is useful for Dirichlet boundary conditions,
      initial conditions, and source terms.
    * `neumann`: the condition explicitly specifies the value of the directional
      derivative of the sediment concentration on cell boundaries. This is
      useful only for boundary conditions. Currently, only homogeneous Neumann
      conditions are supported.

In the case of a Dirichlet condition, a sediment concentration is prescribed by
providing one or more parameters. This can be done in one of two ways:

1. By specifying the concentration directly using the `concentration` parameter

2. By specifying a file from which concentration data is to be read. The
   data will be read into the components of the solution vector that correspond
   to the cells belonging to the region to which this flow condition is
   assigned:

   * `file`: the path for the file from which data is read, specified relative
     to the directory in which the RDycore executable was run.
   * `format`: the format of the data in the file, which current must be
     `binary` (specifying PETSc's binary format).

## `sources`

```yaml
sources:
  - region: upstream
    flow: dam_top_src
  - region: downstream
    flow: dam_bottom_src
```

The `sources` section is a sequence (list) associating `flow`, `sediment`, and
`salinity` sources (as defined in their respective sections) with regions (as
defined in the `regions` section). Sources are optional for each region--if
omitted, a region has no source contributions. A region may have no more than
one set of source conditions associated with it. The above example shows a valid
configuration for a simulation in which sediments and salinity are not modeled,
so only the `flow` parameter is required. The presence of sediment sources
requires the `sediment` parameter, as the presence of salinity sources requires
the `salinity` parameter.

## `surface_composition`

```yaml
surface_composition:
  - region: upstream
    material: smooth
  - region: downstream
    material: rough
```

The `surface_composition` section is a sequence (list) associating materials
(as defined in the `materials` section) with regions (as defined in the
`regions` section). Since regions and materials both have human-readable names,
the association between the two is made clear by the `region` and `material`
parameters in each entry. A region is understood to be completely filled with
the material with which it is associated--the relationship between regions and
materials is necessarily 1:1.

## `time`

```yaml
time:
  final_time: 1
    unit: years
    max_step: 1000
    time_step: 0.001
    coupling_interval: 0.01
```

The `time` section determines the time-stepping strategy used by RDycore using
the following parameters:

* `units`: the units in which time is expressed in the input file. Available
  options are `seconds`, `minutes`, `hours`, `days`, `months`, and `years`.
  This parameter is required and has no default value.
* `final_time`: the time at which the simulation ends (in the desired units)
* `max_step`: the number of steps after which the simulation ends
* `time_step`: a fixed size used for the time step in the desired units.
* `coupling_inverval`: the time interval (in the desired units) at which
  RDycore advances without coupling to E3SM. By default, RDycore runs a single
  time step without coupling to E3SM.

Exactly two of `final_time`, `max_step`, and `time_step` must be specified.
The missing parameter is then computed from those parameters given.
