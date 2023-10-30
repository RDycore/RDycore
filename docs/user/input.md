# RDycore YAML Input Specification

You can configure an RDycore simulation by creating a text file that uses
the [YAML](https://yaml.org/) markup language. Typically, these files have a
`.yml` or `.yaml` suffix like `dam-break.yaml`. In this section, we describe
how to express the specifics for your simulation using the YAML syntax.

RDycore's YAML input is broken up into several sections, each responsible for a
different aspect of the desired simulation. These sections fall into several
broad categories:

* Model equations and discretizations
    * [physics](input.md#physіcs): configures the different physical models
      represented within RDycore
    * [numerics](input.md#numerics): specifies the numerical methods used
      to solve the model equations
* Simulation diagnostics, output, and restarts
    * [logging](input.md#logging): controls informational messages logged to
      files or to the terminal
    * [output](input.md#output): configures simulation output, including
      scalable I/O formats and related parameters
* Material properties
    * [materials](input.md#materials): defines the materials available to
      the simulation
    * [surface_composition](input.md#surface_composition): associates
      materials defined in the `materials` section with files in which the
      relevant material properties are stored
* Spatial discretization
    * [grid](input.md#grid): controls RDycore's spatial discretization
    * [regions](input.md#regions): associates regions (disjoint sets of cells)
      defined in a grid file with human-readable names
    * [boundaries](input.md#boundaries): associates boundaries (disjoint sets of
      edges) defined in a grid file with human-readable names
* Initial and boundary conditions, source terms
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
[boundaries]
  - name: top_wall
    mesh_boundary_id: 2
  - name: bottom_wall
    mesh_boundary_id: 3
  - name: exterior
    mesh_boundary_id: 1
```

The `boundaries` section is a sequence (list) of boundary definitions, each of
which contains the following fields:

* `name`: a human-readable name for the boundary
* `grid_boundary_id`: an integer identifier associated with a disjoint set of
  grid edges. If this identifier is not found within the grid file specified
  in the `grid` section, a fatal error occurs.

Boundary definitions can appear in any order within the sequence.

## `boundary_conditions`

## `flow_conditions`

## `grid`

## `logging`

## `materials`

## `numerics`

## `output`

## `physics`


## `regions`

```yaml
[regions]
  - name: upstream
    grid_region_id: 2
  - name: downstream
    grid_region_id: 1
```

The `regions` section is a sequence (list) of regions definitions, each of
which contains the following fields:

* `name`: a human-readable name for the boundary
* `grid_region_id`: an integer identifier associated with a disjoint set of
  grid cells. If this identifier is not found within the grid file specified
  in the `grid` section, a fatal error occurs.

Region definitions can appear in any order within the sequence.

## `initial_conditions`

## `salinity_conditions`

## `sediment_conditions`

## `sources`

## `surface_composition`

