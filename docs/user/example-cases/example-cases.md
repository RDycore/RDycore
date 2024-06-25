# Overview

Here we provide a few examples cases for running [RDycore](https://github.com/RDycore) on DOE's supercomputers.
Each case directory contains the following:

- `index.md` that describes the case
- RDycore input YAML file for the case
- Placeholder batch scripts for the DOE supercomputers on which the case has been previously run
- A bash script that:

  - Compiles RDycore, if needed, and 
  - Generates a batch script that must submitted via `sbatch`

The files for the meshes, boundary conditions, and source-sink terms are not included
in the repository and are instead available in RDycore shared project directory on the
supported DOE's supercomputers.

The following case is supported:

1. [Idealized dam break problem](dam-break/index.md)
