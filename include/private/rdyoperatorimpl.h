#ifndef RDYOPERATORIMPL_H
#define RDYOPERATORIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyboundaryimpl.h>
#include <private/rdyconfigimpl.h>
#include <private/rdymeshimpl.h>
#include <private/rdyregionimpl.h>

//--------------------------------
// Maximum Wave Speed Diagnostics
//--------------------------------

// This is a diagnostic structure that captures information about the conditions
// under which the maximum courant number is encountered. If you change this
// struct, update the call to MPI_Type_create_struct in InitCourantNumberDiagnostics
// below.
typedef struct {
  PetscReal max_courant_num;  // maximum courant number
  PetscInt  global_edge_id;   // edge at which the max courant number was encountered
  PetscInt  global_cell_id;   // cell in which the max courant number was encountered
  PetscBool is_set;           // true if max_courant_num is set, otherwise false
} CourantNumberDiagnostics;

// MPI datatype and operator for reducing CourantNumberDiagnostics with MPI_AllReduce,
// and related initialization function
extern MPI_Datatype         MPI_COURANT_NUMBER_DIAGNOSTICS;
extern MPI_Op               MPI_MAX_COURANT_NUMBER;
PETSC_INTERN PetscErrorCode InitCourantNumberDiagnostics(void);

// operator material property identifiers
typedef enum {
  OPERATOR_MANNINGS = 0,
  OPERATOR_NUM_MATERIAL_PROPERTIES,
} OperatorMaterialPropertyId;

//------------------------------
// PETSc (CEED-like) "operator"
//------------------------------

typedef struct _p_PetscOperator *PetscOperator;

struct _p_PetscOperator {
  void *context;
  PetscErrorCode (*apply)(void *, PetscReal, Vec, Vec);
  struct {
    PetscInt       num_suboperators;
    PetscOperator *suboperators;
  } composite;
};

PETSC_INTERN PetscErrorCode PetscCompositeOperatorCreate(PetscOperator *);
PETSC_INTERN PetscErrorCode PetscOperatorCreate(void *, PetscErrorCode (*)(void *, Vec, Vec), PetscOperator *);
PETSC_INTERN PetscErrorCode PetscOperatorDestroy(PetscOperator *);
PETSC_INTERN PetscErrorCode PetscCompositeOperatorAddSub(PetscOperator, PetscOperator);
PETSC_INTERN PetscErrorCode PetscOperatorApply(PetscOperator, PetscReal, Vec, Vec);

//----------
// Operator
//----------

// This type and its related functions define an interface for creating a
// nonlinear operator F that computes the time derivative dU/dt of a local
// solution vector U at time t:
//
// F(U, t) -> dU/dt
//
// There are two families of operator implementations:
// 1. CEED implementation: a composite CeedOperator with interior and boundary
//    flux sub-operators and one or more source sub-operators
// 2. PETSc implementation: a set of right-hand-side functions and related
//    vectors
typedef struct Operator {
  // physics configuration defining equations in the system
  RDyPhysicsSection physics_config;

  // number of solution components and component names
  PetscInt num_components;
  char   **field_names;

  // DM and mesh defining the computational domain
  DM       dm;
  RDyMesh *mesh;

  // regions and boundaries in computational domain local to this process
  PetscInt     num_regions, num_boundaries;
  RDyRegion   *regions;
  RDyBoundary *boundaries;

  // CEED/PETSc backends
  union {
    struct {
      // CEED composite operator -- consists of several suboperators. By index,
      // these suboperators are:
      //
      // 0: interior inter-cell fluxes
      // 1-num_boundaries: boundary inter-cell fluxes
      // num_boundaries+1-num_boundaries + num_regions: external sources
      CeedOperator composite;

      // NOTE: all operator data is stored in CeedOperators, so managing
      // NOTE: resources is simpler than in the PETSc case

      // operator timestep last set
      PetscReal dt;

      CeedVector u_local;
      CeedVector rhs, sources;

    } ceed;

    // PETSc operator data
    struct {
      // PETSc composite operator
      PetscOperator composite;

      // array of Dirichlet boundary value vectors, indexed by boundary
      Vec *boundary_values;

      // array of boundary flux vectors, indexed by boundary
      Vec *boundary_fluxes;

      // array of regional external source vectors, indexed by region
      Vec *sources;

      // array of regional material property data Vecs, indexed by
      // [region_index][property_id]
      Vec **material_properties;

    } petsc;

    // global flux divergence data vector for entire domain (used only
    // internally)
    Vec flux_divergence;
  };

  //-------------------------------------------
  // diagnostics (used by both PETSc and CEED)
  //-------------------------------------------

  // boundary fluxes on all domain edges
  Vec boundary_fluxes;

  // courant number diagnostics, local to current MPI process
  CourantNumberDiagnostics courant_number_diags;
} Operator;

PETSC_INTERN PetscErrorCode CreateOperator(RDyPhysicsSection, DM, RDyMesh *, PetscInt, RDyRegion *, PetscInt, RDyBoundary *, Operator **);
PETSC_INTERN PetscErrorCode DestroyOperator(Operator **);

// operator timestepping function
PETSC_INTERN PetscErrorCode ApplyOperator(Operator *, PetscReal, Vec, Vec);

//----------------------
// Operator Data Access
//----------------------

// This type provides access to multi-component operator data.
typedef struct {
  PetscInt    num_components;  // number of data components
  PetscReal **values;          // array of values ([component][index])
  void       *array_pointer;   // pointer to CEED/PETSc array owning data (used internally)
} OperatorData;

PETSC_INTERN PetscErrorCode GetOperatorBoundaryValues(Operator *, RDyBoundary, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorBoundaryValues(Operator *, RDyBoundary, OperatorData *);
PETSC_INTERN PetscErrorCode GetOperatorBoundaryFluxes(Operator *, RDyBoundary, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorBoundaryFluxes(Operator *, RDyBoundary, OperatorData *);
PETSC_INTERN PetscErrorCode GetOperatorExternalSource(Operator *, RDyRegion, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorExternalSource(Operator *, RDyRegion, OperatorData *);
PETSC_INTERN PetscErrorCode GetOperatorMaterialProperty(Operator *, RDyRegion, OperatorMaterialPropertyId, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorMaterialProperty(Operator *, RDyRegion, OperatorMaterialPropertyId, OperatorData *);

// diagnostics
PETSC_INTERN PetscErrorCode GetOperatorCourantNumberDiagnostics(Operator *, CourantNumberDiagnostics *);

#endif
