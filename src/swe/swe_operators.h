#ifndef SWE_OPERATORS_H
#define SWE_OPERATORS_H

#include <ceed/types.h>
#include <private/rdycoreimpl.h>

// Creates a flux operator for the shallow water equations that produces
// solutions to the related riemann problem for cells separated by edges on the
// given computational mesh. The resulting operator can be manipulated by
// libCEED calls.
// @param [in]  ceed The Ceed context used to create the operator
// @param [in]  mesh The computational mesh for which the operator is created
// @param [in]  num_boundaries The number of boundaries (disjoint edge sets) on the mesh
// @param [in]  boundaries An array of disjoint edge sets representing mesh boundaries
// @param [in]  boundary_conditions An array of metadata defining boundary conditions the operator will enforce
// @param [in]  tiny_h the minimum height threshold for water flow
// @param [out] flux_op A pointer to the flux operator to be created
PetscErrorCode CreateSWEFluxOperator(Ceed ceed, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary boundaries[num_boundaries],
                                     RDyCondition boundary_conditions[num_boundaries], PetscReal tiny_h, CeedOperator *flux_op);

// Given a computational mesh, creates a source operator for the shallow water
// equations that computes source terms. The resulting operator can be
// manipulated by libCEED calls.
// @param [in]  ceed The Ceed context used to create the operator
// @param [in]  mesh The computational mesh for which the operator is created
// @param [in]  materials_by_cell An array of RDyMaterials defining cellwise material properties
// @param [in]  tiny_h the minimum height threshold for water flow
// @param [out] flux_op A pointer to the flux operator to be created
PetscErrorCode CreateSWESourceOperator(Ceed ceed, RDyMesh *mesh, RDyMaterial materials_by_cell[mesh->num_cells], PetscReal tiny_h,
                                       CeedOperator *source_op);

// Given a shallow water equations source operator created by
// CreateSWESourceOperator, fetches the field representing the source of water.
// This can be used to implement a time-dependent water source.
PetscErrorCode GetWaterSourceFromSWESourceOperator(CeedOperator source_op, CeedOperatorField *water_source_field);

// Given a shallow water equations source operator created by
// CreateSWESourceOperator, fetches the field representing the Riemann flux.
PetscErrorCode GetRiemannFluxFromSWESourceOperator(CeedOperator source_op, CeedOperatorField *riemann_flux_field);

#endif
