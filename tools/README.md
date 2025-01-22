# Tools

This directory contains helpful scripts that we use in various stages of
development. Here's a brief rundown of the contents of this directory.

* `build-petsc-docker-image.sh` - A script you can run to generate a Docker
  image with PETSc pre-installed. We can use such images to accelerate our
  automatic build-and-test (a.k.a "continuous integration", or "CI") environment
  on GitHub.
* `Dockerfile.petsc` - a set of instructions for constructing the Docker image
  containing PETSc, used by `build-petsc-docker-image.sh
* `lsan_suppr.txt` - A [LeakSanitizer suppression file](https://github.com/google/sanitizers/wiki/AddressSanitizerLeakSanitizer#suppressions)
  that RDycore can use to detect memory leaks without extra noise
