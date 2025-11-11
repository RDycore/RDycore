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

// This type and its related functions define an interface for solving a nonlinear equation for a
// time-dependent solution u(t) for which the equation
//
// F(t, u, du/dt) = G(t, u)
//
// holds, where F is in general a differential-algebraic expression involving u and its time
// derivative, and G is a "right-hand side function".
//
// In the case of explicit time integration, F(t, u, du/dt) reduces to du/dt itself, and the
// nonlinear system becomes a system of ordinary differential equations:
//
// du/dt = G(t, u),
//
// evaluated at appropriate quadrature points. In this case G contains flux and source terms, with
// the specific forms determined by the underlying physics.
//
// For implicit-explicit time integration (e.g. ARK-IMEX), F(t, u, du/dt) incorporates stiff source
// terms, subtracting them from G, which retains only the explicitly-evaluated flux terms. In this
// case, the jacobian of the operator is dF/du.
//
// For fully-implicit time integration (e.g. BEULER), F again reduces to du/dt and G and its
// jacobian (consisting of flux and source terms) are used in the integration of the system. In this
// case, the jacobian of the operator is dG/du.
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
    // CEED
    struct {
      // Flux and source operators, each composed of sub-operators.
      CeedOperator flux, source;

      // operator for evaluating the Jacobian matrix
      CeedOperator jacobian;

      // timestep last set on operators
      PetscReal dt;

      // bookkeeping vectors
      CeedVector u_local, rhs, sources;

      // domain-wide flux_divergence vector;
      CeedVector flux_divergence;
    } ceed;

    // PETSc operator data
    struct {
      // Flux and source operators, each composed of sub-operators.
      PetscOperator flux, source;

      // operator for evaluating the Jacobian matrix
      PetscOperator jacobian;

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

  // The jacobian matrix for the operator and its preconditioner. In the case of CEED, this matrix
  // is a MatCeed shell that can assemble a matrix with an equivalent non-zero structure; in the
  // case of PETSc, it's just a matrix. In any case, this is used only in implicit and
  // implicit-explicit time discretizations
  Mat mat_jacobian, mat_jacobian_pre;

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

// creates and preallocates a matrix from the operator based on its configuration
PETSC_INTERN PetscErrorCode OperatorCreateJacobianMatrix(Operator *op);

// operator functions for interacting with PETSc's TS solver (RDy passed as context)
PETSC_INTERN PetscErrorCode OperatorIFunction(TS, PetscReal, Vec, Vec, Vec, void *);
PETSC_INTERN PetscErrorCode OperatorIJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PETSC_INTERN PetscErrorCode OperatorRHSFunction(TS, PetscReal, Vec, Vec, void *);
PETSC_INTERN PetscErrorCode OperatorRHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);

//---------------------------------------------------------
// CEED/PETSc Flux, Source, Jacobian Operator Constructors
//---------------------------------------------------------

PETSC_INTERN PetscErrorCode CreateCeedFluxOperator(RDyConfig *, RDyMesh *, PetscInt, RDyBoundary *, RDyCondition *, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateCeedSourceOperator(RDyConfig *, RDyMesh *, CeedOperator *);
PETSC_INTERN PetscErrorCode CreateCeedJacobian(RDyConfig *, RDyMesh *, PetscInt, RDyBoundary *, RDyCondition *, CeedOperator *);

PETSC_INTERN PetscErrorCode CreatePetscFluxOperator(RDyConfig *, RDyMesh *, PetscInt, RDyBoundary *, RDyCondition *, Vec *, Vec *, Vec *,
PETSC_INTERN PetscErrorCode CreatePetscSourceOperator(RDyConfig *, RDyMesh *, Vec, Vec, PetscOperator *);
PETSC_INTERN PetscErrorCode CreatePetscJacobian(RDyConfig *, RDyMesh *, PetscInt, RDyBoundary *, RDyCondition *, PetscOperator *);

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
