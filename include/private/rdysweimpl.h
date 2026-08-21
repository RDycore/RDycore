#ifndef RDYSWEIMPL_H
#define RDYSWEIMPL_H

#include <ceed/types.h>
#include <petscsys.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>

PETSC_INTERN PetscErrorCode CreateSWEQFunctionContext(Ceed, const RDyConfig, CeedQFunctionContext *);
PETSC_INTERN PetscErrorCode CreatePetscSWEInteriorFluxOperator(RDyMesh *, MPI_Comm, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWEInteriorFluxHROperator(RDyMesh *, const RDyConfig, OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWEBoundaryFluxOperator(RDyMesh *, const RDyConfig, RDyBoundary, RDyCondition, Vec, Vec, Vec,
                                                               OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWESourceOperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSWESourceHROperator(RDyMesh *, const RDyConfig, Vec, Vec, PetscOperator *);

// SWE RHS Jacobian (swe_jacobian_petsc.c): matrix creation and TS registration
PETSC_INTERN PetscErrorCode RegisterSWERHSJacobian(RDy);
PETSC_INTERN PetscErrorCode RegisterSWEIMEXFriction(RDy);
PETSC_INTERN PetscErrorCode DestroySWERHSJacobian(RDy);

#if RDY_HAVE_KOKKOS_JACOBIAN
// Host-side bookkeeping for the Kokkos device RHS (B1): built by
// CreateAnalyticJacobianCOO alongside the device-assembly context (whose
// geometry views the RHS kernels reuse) and attached to
// Operator.petsc.swe_rhs_kokkos; ApplyPetscOperator dispatches to
// ApplySWEPetscOperatorsKokkos when it is present and the state Vec is a
// Kokkos type. Owned by the RDy (freed in DestroySWERHSJacobian); the
// bedge maps and dirichlet staging are borrowed from the rhs_jac_* fields.
typedef struct {
  void            *kokkos;                // the SWEJacobianKokkos context (rdy->rhs_jac_kokkos)
  PetscInt         n_edges, n_bedges;     // compact interior / flattened boundary edge counts
  PetscInt        *edge_id;               // [n_edges] mesh edge id (Courant diagnostics)
  PetscInt        *bedge_bnd, *bedge_idx; // borrowed from rdy->rhs_jac_bedge_*
  PetscScalar     *dirichlet;             // borrowed from rdy->rhs_jac_dirichlet
  PetscScalar     *bflux;                 // [3 * n_bedges] host staging for raw boundary fluxes
  PetscObjectState mp_state, src_state;   // change tracking for the staged host vecs
  PetscBool        primed;
  PetscBool        announced;             // one-time PetscInfo on first device apply
} SWERHSKokkosData;

PETSC_INTERN PetscErrorCode ApplySWEPetscOperatorsKokkos(Operator *, PetscReal, Vec, Vec, PetscBool *);
#endif

#endif
