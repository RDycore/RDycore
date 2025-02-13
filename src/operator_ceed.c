#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysedimentimpl.h>
#include <private/rdysweimpl.h>

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

/// Creates a CEED flux operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [in]    num_boundaries      the number of distinct boundaries bounding the computational domain
/// @param [in]    boundaries          an array of distinct boundaries bounding the computational domain
/// @param [in]    boundary_conditions an array of boundary conditions corresponding to the domain boundaries
/// @param [out]   flux_op             the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedFluxOperator(RDyConfig *config, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries,
                                      RDyCondition *boundary_conditions, CeedOperator *flux_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedCompositeOperatorCreate(ceed, flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }

  // flux suboperator 0: fluxes between interior cells

  CeedOperator interior_flux_op;
  if (config->physics.sediment.num_classes > 0) {
    PetscCall(CreateSedimentCeedInteriorFluxOperator(mesh, *config, &interior_flux_op));
  } else {
    PetscCall(CreateSWECeedInteriorFluxOperator(mesh, *config, &interior_flux_op));
  }
  PetscCall(CeedCompositeOperatorAddSub(*flux_op, interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    CeedOperator boundary_flux_op;
    RDyBoundary  boundary  = boundaries[b];
    RDyCondition condition = boundary_conditions[b];
    if (config->physics.sediment.num_classes > 0) {
      PetscCall(CreateSedimentCeedBoundaryFluxOperator(mesh, *config, boundary, condition, &boundary_flux_op));
    } else {
      PetscCall(CreateSWECeedBoundaryFluxOperator(mesh, *config, boundary, condition, &boundary_flux_op));
    }
    PetscCall(CeedCompositeOperatorAddSub(*flux_op, boundary_flux_op));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a CEED source operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [out]   source_op           the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedSourceOperator(RDyConfig *config, RDyMesh *mesh, CeedOperator *source_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedCompositeOperatorCreate(ceed, source_op));

  CeedOperator source_0;
  if (config->physics.sediment.num_classes > 0) {
    PetscCall(CreateSedimentCeedSourceOperator(mesh, *config, &source_0));
  } else {
    PetscCall(CreateSWECeedSourceOperator(mesh, *config, &source_0));
  }
  PetscCall(CeedCompositeOperatorAddSub(*source_op, source_0));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
