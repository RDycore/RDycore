# mat-ceed

This library provides a CEED-backed Mat ("shell") implementation that allows RDycore to solve sparse
linear systems on CPUs and GPUs using a COO format compatible with libCEED. It is adapted from code
in [HONEE](https://gitlab.com/phypid/honee) and is used for RDycore's implicit solvers.

This library attempts to bridge the gap between the main branch of libCEED and the version used by
RDycore, so additional functions are provided where needed.

At the time of writing (10/16/2025) The code in this library requires PETSc v3.24 (released 9/29/2025)
and the `main` branch of libCEED.
