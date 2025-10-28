#ifndef RDYOPERATORIMPL_H
#define RDYOPERATORIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyboundaryimpl.h>
#include <private/rdyconditionimpl.h>
#include <private/rdyconfigimpl.h>
#include <private/rdymeshimpl.h>
#include <private/rdyregionimpl.h>
#include <rdycore.h>  // for MAX_NAME_LEN

//--------------------------------
// Maximum Wave Speed Diagnostics
//--------------------------------

// This diagnostic structure captures information about the conditions under
// which the maximum courant number is encountered. If you change this struct,
// update the call to MPI_Type_create_struct in InitCourantNumberDiagnostics
// below.
typedef struct {
  PetscReal max_courant_num;  // maximum courant number
  PetscInt  global_edge_id;   // edge at which the max courant number was encountered
  PetscInt  global_cell_id;   // cell in which the max courant number was encountered
} CourantNumberDiagnostics;

// MPI datatype and operator for reducing CourantNumberDiagnostics with MPI_AllReduce,
// and related initialization function
extern MPI_Datatype         MPI_COURANT_NUMBER_DIAGNOSTICS;
extern MPI_Op               MPI_MAX_COURANT_NUMBER;
PETSC_INTERN PetscErrorCode InitCourantNumberDiagnostics(void);

//------------------------------
// General Operator Diagnostics
//------------------------------

/// This type encapsulates all operator diagnostics.
typedef struct {
  /// information about the maximum wave speed
  CourantNumberDiagnostics courant_number;
  /// whether the diagnostics are up to date
  PetscBool updated;
} OperatorDiagnostics;

//------------------------------
// PETSc (CEED-like) "operator"
//------------------------------

// a "field" associating a symbolic name (e. g. "riemannf") with a PETSc Vec
typedef struct {
  char name[MAX_NAME_LEN + 1];
  Vec  vec;
} PetscOperatorField;

// a collection of operator fields
typedef struct {
  PetscInt            num_fields, capacity;
  PetscOperatorField *fields;
} PetscOperatorFields;

PETSC_INTERN PetscErrorCode PetscOperatorFieldsGet(PetscOperatorFields, const char *, Vec *);

// This operator type has a context that stores physics-specific data and two
// operations:
//
// 1. apply(context, dt, u, F), which updates the global right-hand-side vector
//    F by applying the operator to the local vector u over the timestep dt
// 2. destroy(context), which deallocates all resources related to the context
//
// The PetscOperator resembles the CeedOperator type, which allows us to
// structure our PETSc-backed physics similarly to CEED-backed physics. Because
// PETSc runs on CPUs only, the PetscOperator type is much simpler than
// the CeedOperator type, with no need for concepts like active and passive
// inputs, restrictions, etc.
typedef struct _p_PetscOperator *PetscOperator;
struct _p_PetscOperator {
  void               *context;
  PetscOperatorFields fields;
  PetscBool           is_composite;
  PetscErrorCode (*apply)(void *, PetscOperatorFields, PetscReal, Vec, Vec);
  PetscErrorCode (*destroy)(void *);
};

PETSC_INTERN PetscErrorCode PetscOperatorCreate(void *, PetscErrorCode (*)(void *, PetscOperatorFields, PetscReal, Vec, Vec),
                                                PetscErrorCode (*)(void *), PetscOperator *);
PETSC_INTERN PetscErrorCode PetscOperatorDestroy(PetscOperator *);
PETSC_INTERN PetscErrorCode PetscOperatorApply(PetscOperator, PetscReal, Vec, Vec);
PETSC_INTERN PetscErrorCode PetscOperatorSetField(PetscOperator, const char *, Vec);
PETSC_INTERN PetscErrorCode PetscCompositeOperatorCreate(PetscOperator *);
PETSC_INTERN PetscErrorCode PetscCompositeOperatorAddSub(PetscOperator, PetscOperator);

//----------
// Operator
//----------

// This type and its related functions define an interface for creating a
// nonlinear operator F that computes the time derivative du/dt of a local
// solution vector u at time t:
//
// F(u, t) -> du/dt
//
// There are two families of operator implementations:
// 1. CEED implementation: a composite CeedOperator with interior and boundary
//    flux sub-operators and region-based source sub-operators
// 2. PETSc implementation: a composite PetscOperator with a similar set of
//    sub-operators
typedef struct Operator {
  // simulation configuration defining physics, numerics, etc
  RDyConfig *config;

  // number of solution components and component names
  PetscInt num_components;
  char   **field_names;

  // DM and mesh defining the computational domain
  DM       dm;
  RDyMesh *mesh;

  // regions and boundaries in computational domain local to this process
  PetscInt     num_regions;
  RDyRegion   *regions;  // pointer to RDy-owned regions array
  PetscInt     num_boundaries;
  RDyBoundary *boundaries;  // pointer to RDy-owned boundaries array

  // boundary conditions corresponding to boundaries
  RDyCondition *boundary_conditions;  // pointer to RDy-owned boundary_conditions array

  // CEED/PETSc backends
  union {
    struct {
      // CEED flux and source operators (each composed of sub-operators, see
      // CreateOperator in src/operator.c)
      CeedOperator flux, source;

      // timestep last set on operators
      PetscReal dt;

      // bookkeeping vectors
      CeedVector u_local, rhs, sources;

      // domain-wide flux_divergence vector;
      CeedVector flux_divergence;
    } ceed;

    // PETSc operator data
    struct {
      // PETSc composite operators with sub-operators identical in structure to
      // the CEED composite operators above
      PetscOperator flux, source;

      // array of Dirichlet boundary value vectors, indexed by boundary
      Vec *boundary_values;

      // array of boundary flux vectors, indexed by boundary
      Vec *boundary_fluxes;

      // array of boundary flux vectors, indexed by boundary
      Vec *boundary_fluxes_accum;

      // domain-wide external source vector
      Vec external_sources;

      // domain-wide material property vector (# of components == # of scalar properties)
      Vec material_properties;
    } petsc;
  };

  // domain-wide flux divergence data
  Vec flux_divergence;

  //-------------------------------------------
  // diagnostics (used by both PETSc and CEED)
  //-------------------------------------------

  OperatorDiagnostics diagnostics;
} Operator;

PETSC_INTERN PetscErrorCode CreateOperator(RDyConfig *, DM, RDyMesh *, PetscInt, PetscInt, RDyRegion *, PetscInt, RDyBoundary *, RDyCondition *,
                                           Operator **);
PETSC_INTERN PetscErrorCode DestroyOperator(Operator **);

// operator timestepping function
PETSC_INTERN PetscErrorCode ApplyOperator(Operator *, PetscReal, Vec, Vec);

//--------------------------------------------------
// CEED/PETSc Flux and Source Operator Constructors
//--------------------------------------------------

PETSC_INTERN PetscErrorCode CreateCeedFluxOperator(RDyConfig *, RDyMesh *, PetscInt, RDyBoundary *, RDyCondition *, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateCeedSourceOperator(RDyConfig *, RDyMesh *, CeedOperator *);
PETSC_INTERN PetscErrorCode CreatePetscFluxOperator(RDyConfig *, RDyMesh *, PetscInt, RDyBoundary *, RDyCondition *, Vec *, Vec *, Vec *,
                                                    OperatorDiagnostics *, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscSourceOperator(RDyConfig *, RDyMesh *, Vec, Vec, PetscOperator *);

//----------------------
// Operator Data Access
//----------------------

// This type provides access to multi-component operator data.
typedef struct {
  PetscInt    num_components;  // number of data components
  PetscReal **values;          // array of values ([component][index])
  PetscReal  *array_pointer;   // pointer to CEED/PETSc array owning data (used internally)
} OperatorData;

PETSC_INTERN PetscErrorCode GetOperatorBoundaryValues(Operator *, RDyBoundary, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorBoundaryValues(Operator *, RDyBoundary, OperatorData *);

PETSC_INTERN PetscErrorCode GetOperatorBoundaryFluxes(Operator *, RDyBoundary, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorBoundaryFluxes(Operator *, RDyBoundary, OperatorData *);

PETSC_INTERN PetscErrorCode GetOperatorRegionalExternalSource(Operator *, RDyRegion, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorRegionalExternalSource(Operator *, RDyRegion, OperatorData *);
PETSC_INTERN PetscErrorCode GetOperatorDomainExternalSource(Operator *, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorDomainExternalSource(Operator *, OperatorData *);

PETSC_INTERN PetscErrorCode GetOperatorRegionalMaterialProperties(Operator *, RDyRegion, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorRegionalMaterialProperties(Operator *, RDyRegion, OperatorData *);
PETSC_INTERN PetscErrorCode GetOperatorDomainMaterialProperties(Operator *, OperatorData *);
PETSC_INTERN PetscErrorCode RestoreOperatorDomainMaterialProperties(Operator *, OperatorData *);

// diagnostics
PETSC_INTERN PetscErrorCode ResetOperatorDiagnostics(Operator *);
PETSC_INTERN PetscErrorCode UpdateOperatorDiagnostics(Operator *);
PETSC_INTERN PetscErrorCode GetOperatorDiagnostics(Operator *, OperatorDiagnostics *);

#endif
