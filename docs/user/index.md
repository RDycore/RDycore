# RDycore User Guide

RDycore's primary purpose is to provide the DOE E3SM climate model with the
capability to model coastal compound flooding. Accordingly, it has been
constructed as a performance-portable library that makes efficient use of
DOE's leadership-class computing facilities and is invoked by E3SM Fortran code.
This guide explains how we integrate RDycore to E3SM to perform coupled
simulations of compound flooding.

Aside from the RDycore library, we provide standalone drivers that provide
convenient ways of testing and evaluating the model's capabilities:

* standalone C and Fortran driver programs for running uncoupled flood
  simulations given appropriate initial and boundary conditions, source terms,
  etc.

* standalone C and Fortran verification programs that use the method of
  manufactured solutions (MMS) to evaluate the stability and accuracy of the
  underlying numerical methods by computing error norms of simulation results
  measured against analytical solutions. These MMS programs can also compute
  convergence rates, which are useful for identifying algorithmic and
  programming errors.

This guide also describes these standalone programs and their features.

## Coupling RDycore to E3SM

## The Standalone C and Fortran Drivers

* [Standalone YAML input specification](input.md)

## The MMS C and Fortran Drivers

* [MMS driver YAML input specification](mms.md)

