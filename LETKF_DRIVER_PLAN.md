# RDycore LETKF Driver Plan

## Goal

Create a new `driver/main_da.c` entry point for RDycore that combines:

- RDycore's PETSc + libCEED shallow-water setup and time integration flow from `driver/main.c`
- PETSc LETKF ensemble data assimilation patterns from `petsc_da/src/ml/da/tutorials/ex4.c`

The first implementation is intentionally narrow. It should establish a clean data-assimilation driver structure before adding more sophisticated forcing and observation options.

## First-Cut Scope

Included in the first implementation:

- LETKF-based data assimilation only
- Synthetic observations generated from a separate RDycore truth run
- One MPI job containing:
  - one communicator for the truth run
  - one communicator per ensemble member
- A reduced command-line interface focused on DA controls
- Forecast/analysis loop structure compatible with PETSc `PetscDA`
- Height-only observations for the initial observation operator
- RMSE and basic DA progress reporting

Explicitly excluded from the first implementation:

- Rainfall dataset command-line support from `driver/main.c`
- Boundary-condition dataset command-line support from `driver/main.c`
- File-driven observations
- AMR-specific DA handling
- Sediment/salinity-specific DA handling
- MMS mode
- Rich ensemble output formats beyond lightweight diagnostics

## Runtime Contract For `main_da.c`

The first version of `main_da.c` should:

1. Accept a standard RDycore YAML input file.
2. Parse a reduced set of PETSc options for DA configuration.
3. Create and set up RDycore instances for truth and ensemble members.
4. Build PETSc vectors and matrices needed for LETKF.
5. Advance truth and ensemble states in lockstep.
6. Generate noisy synthetic observations from the truth state.
7. Call `PetscDAAnalysis()` at the configured observation cadence.
8. Report forecast and analysis diagnostics.

## Minimal Option Set

The first implementation should focus on these options:

- `-ensemble_size`
- `-obs_frequency`
- `-obs_stride`
- `-obs_error`
- `-random_seed`
- `-num_observations_vertex`
- `-use_global_localization`
- `-output_prefix`

## Recommended Implementation Sequence

1. Add the new driver target and a minimal `main_da.c` skeleton.
2. Add small RDycore state-transfer helpers so the driver can copy prognostic state into and out of PETSc vectors cleanly.
3. Create truth and ensemble RDycore instances.
4. Build the observation operator and localization data.
5. Configure PETSc `PetscDA` for LETKF.
6. Implement the forecast/analysis loop.
7. Add a small smoke test.

## Integration Notes

- RDycore already provides `RDyCreatePrognosticVec()` and `RDySetInitialConditions()`.
- A clean first implementation should avoid depending on RDycore private headers from `driver/main_da.c`.
- PETSc LETKF setup should follow the pattern in `petsc_da/src/ml/da/tutorials/ex4.c`.
- If distance-based localization is not available in the PETSc build, the driver should fall back to global localization.

## Build

Current local build flow:

```bash
export PETSC_DIR=/Users/gautam.bisht/projects/petsc/petsc_da
export PETSC_ARCH=gcc14-b522cb8c110
cd /Users/gautam.bisht/projects/rdycore/rdycore_da/build-gcc14-b522cb8c110
ninja -j4 install
```