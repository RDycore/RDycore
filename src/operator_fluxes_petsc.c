#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdytracerimpl.h>
#include <private/rdysweimpl.h>

/// Creates a PETSc flux operator appropriate for the given configuration.
/// @param [in]    config                the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                  a mesh containing geometric and topological information for the domain
/// @param [in]    num_boundaries        the number of distinct boundaries bounding the computational domain
/// @param [in]    boundaries            an array of distinct boundaries bounding the computational domain
/// @param [in]    boundary_conditions   an array of boundary conditions corresponding to the domain boundaries
/// @param [inout] boundary_values       an array of sequential Vecs that can store boundary values for each boundary
/// @param [inout] boundary_fluxes       an array of sequential Vecs that can store boundary fluxes for each boundary
/// @param [inout] boundary_fluxes_accum an array of sequential Vecs that can store accumulated boundary fluxes for each boundary
/// @param [out]   flux_op               the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreatePetscFluxOperator(RDyConfig *config, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries,
                                       RDyCondition *boundary_conditions, Vec *boundary_values, Vec *boundary_fluxes, Vec *boundary_fluxes_accum,
                                       OperatorDiagnostics *diagnostics, PetscOperator *flux_op) {
  PetscFunctionBegin;

  PetscCall(PetscOperatorCreateComposite(flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }

  // flux suboperator 0: fluxes between interior cells

  PetscOperator interior_flux_op;
  if (config->physics.sediment.num_classes > 0) {
    PetscCall(CreatePetscTracerInteriorFluxOperator(mesh, *config, diagnostics, &interior_flux_op));
  } else {
    PetscCall(CreatePetscSWEInteriorFluxOperator(mesh, *config, diagnostics, &interior_flux_op));
  }
  PetscCall(PetscOperatorCompositeAddSub(*flux_op, interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    PetscOperator boundary_flux_op;
    RDyBoundary   boundary  = boundaries[b];
    RDyCondition  condition = boundary_conditions[b];
    if (config->physics.sediment.num_classes > 0) {
      PetscCall(CreatePetscTracerBoundaryFluxOperator(mesh, *config, boundary, condition, boundary_values[b], boundary_fluxes[b], diagnostics,
                                                      &boundary_flux_op));
    } else {
      PetscCall(CreatePetscSWEBoundaryFluxOperator(mesh, *config, boundary, condition, boundary_values[b], boundary_fluxes[b],
                                                   boundary_fluxes_accum[b], diagnostics, &boundary_flux_op));
    }
    PetscCall(PetscOperatorCompositeAddSub(*flux_op, boundary_flux_op));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
