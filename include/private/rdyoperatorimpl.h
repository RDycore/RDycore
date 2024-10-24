#ifndef RDYOPERATORSIMPL_H
#define RDYOPERATORSIMPL_H

#include <ceed/ceed.h>
#include <petsc/private/petscimpl.h>
#include <private/rdyboundaryimpl.h>
#include <private/rdymeshimpl.h>

// CEED initialization, availability, context, useful for creating CEED
// sub-operators
PETSC_INTERN PetscErrorCode SetCeedResource(char *);
PETSC_INTERN PetscBool      CeedEnabled(void);
PETSC_INTERN Ceed           CeedContext(void);
PETSC_INTERN PetscErrorCode GetCeedVecType(VecType *);

// creation of sections for configuration-specific physics
PETSC_INTERN PetscErrorCode CreateSection(RDy, PetscSection *);

typedef struct Operator Operator;

// Diagnostic structure that captures information about the conditions under
// which the maximum courant number is encountered. If you change this struct,
// update the call to MPI_Type_create_struct in InitMPITypesAndOps below.
typedef struct {
  PetscReal max_courant_num;  // maximum courant number
  PetscInt  global_edge_id;   // edge at which the max courant number was encountered
  PetscInt  global_cell_id;   // cell in which the max courant number was encountered
  PetscBool is_set;           // true if max_courant_num is set, otherwise false
} CourantNumberDiagnostics;

//-------------
// RDyOperator
//-------------

// This type and its related functions define an interface for creating a
// nonlinear operator F that computes the time derivative dU/dt of a local
// solution vector U at time t:
//
// F(U, t) -> dU/dt
//
// NOTE: the term "operator" here is not the same as a CeedOperator. The CEED
// NOTE: implementation of an operator of this type actually uses two CeedOperators:
// NOTE: 1. A "flux operator", which computes Riemann fluxes using a finite volume method
// NOTE: 2. A "source operator", which add source terms
typedef struct Operator {
  // DM associated with solution vector
  DM dm;

  // mesh representing the computational domain
  RDyMesh *mesh;

  PetscInt num_components;  // number of solution components

  // boundaries in computational domain
  PetscInt     num_boundaries;
  RDyBoundary *boundaries;

  // CEED/PETSc operator implementations
  union {
    struct {
      // NOTE: RDycore uses a global CEED context which is not associated with
      // NOTE: any specific operator

      CeedOperator flux_operator, source_operator;
      CeedVector   u_local;
      CeedVector   rhs, sources;

      Vec flux_divergences;  // storage for accumulated flux divergences

      CeedScalar dt;  // time step associated with operator
    } ceed;
    struct {
      RDy rdy;                                            // here because our PETSc impl uses lots of things (FIXME)
      PetscErrorCode (*apply)(RDy, PetscReal, Vec, Vec);  // apply() function
      void *context;                                      // context pointer -- must be cast to e.g. PetscRiemannDataSWE*
      Vec   sources;                                      // source-sink vector
    } petsc;
  };

  // courant number diagnostics and local update function (no MPI)
  CourantNumberDiagnostics courant_diags;
  PetscErrorCode (*update_local_courant_diags)(Operator *, CourantNumberDiagnostics *);

  // locks on operator data for exclusive access
  struct {
    void **boundary_data;  // per-boundary operator boundary data
    void  *source_data;    // operator source data for the domain
    void  *material_data;  // operator material data for the domain
    void  *flux_div_data;  // operator flux divergence data for the domain
  } lock;
} Operator;

PETSC_INTERN PetscErrorCode CreateCeedOperator(DM, RDyMesh *, PetscInt, PetscInt, RDyBoundary *, Operator *);
PETSC_INTERN PetscErrorCode AddCeedInteriorFluxSubOperator(Operator *, CeedOperator);
PETSC_INTERN PetscErrorCode GetCeedInteriorFluxSubOperator(Operator *, CeedOperator *);
PETSC_INTERN PetscErrorCode AddCeedBoundaryFluxSubOperator(Operator *, RDyBoundary, CeedOperator);
PETSC_INTERN PetscErrorCode GetCeedBoundaryFluxSubOperator(Operator *, RDyBoundary, CeedOperator *);
PETSC_INTERN PetscErrorCode AddCeedSourceSubOperator(Operator *, CeedOperator);
PETSC_INTERN PetscErrorCode GetCeedSourceSubOperator(Operator *, CeedOperator *);

PETSC_INTERN PetscErrorCode CreatePetscOperator(DM, RDyMesh *, PetscInt, PetscInt, RDyBoundary *, Operator *);
PETSC_INTERN PetscErrorCode DestroyOperator(Operator *);

PETSC_INTERN PetscErrorCode SetOperatorTimeStep(Operator *, PetscReal);
PETSC_INTERN PetscErrorCode ApplyOperator(Operator *, PetscReal, Vec, Vec);

PETSC_INTERN PetscErrorCode UpdateOperatorCourantNumberDiagnostics(Operator *);
PETSC_INTERN PetscErrorCode GetOperatorCourantNumberDiagnostics(Operator *, CourantNumberDiagnostics *);

//----------------------
// Operator Data Access
//----------------------

// These types and functions allow access to data within operators, such as
// * boundary values (e.g. for Dirichlet boundary conditions)
// * source terms (water sources, momentum contributions)
// * relevant material properties (e.g. Mannings coefficient)
// * any needed intermediate quantities (e.g. flux divergences computed by the
//   flux operator and passed to the source operator)

// This type provides access to single- or multi-component vector data in either
// CEED or PETSc, depending upon whether CEED is enabled.
typedef struct {
  union {
    struct {
      CeedVector  vec;
      CeedScalar *data;
    } ceed;
    struct {
      Vec        vec;
      PetscReal *data;
    } petsc;
  };
  PetscBool updated;  // true iff updated
} OperatorVectorData;

// This type allows the direct manipulation of per-boundary values for the
// system of equations being solved by RDycore.
typedef struct {
  // associated operator
  Operator *op;
  // associated boundary
  RDyBoundary boundary;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
  OperatorVectorData values;  // boundary values
  OperatorVectorData fluxes;  // boundary fluxes
} OperatorBoundaryData;

PETSC_INTERN PetscErrorCode GetOperatorBoundaryData(Operator *, RDyBoundary, OperatorBoundaryData *);
PETSC_INTERN PetscErrorCode SetOperatorBoundaryValues(OperatorBoundaryData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode GetOperatorBoundaryFluxes(OperatorBoundaryData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorBoundaryData(Operator *, RDyBoundary, OperatorBoundaryData *);

// This type allows the direct manipulation of source values on the entire
// domain for the system of equations being solved by RDycore.
typedef struct {
  // associated operator
  Operator *op;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
  OperatorVectorData sources;
} OperatorSourceData;

PETSC_INTERN PetscErrorCode GetOperatorSourceData(Operator *, OperatorSourceData *);
PETSC_INTERN PetscErrorCode SetOperatorSourceValues(OperatorSourceData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorSourceData(Operator *, OperatorSourceData *);

// This type allows the direct manipulation of operator material properties on
// the entire domain for the system of equations being solved by RDycore.
typedef struct {
  // associated operator
  Operator *op;
  // underlying data storage
  OperatorVectorData mannings;  // mannings coefficient
} OperatorMaterialData;

// operator material properties enum
typedef enum {
  OPERATOR_MANNINGS = 0,
} OperatorMaterialDataIndex;

PETSC_INTERN PetscErrorCode GetOperatorMaterialData(Operator *, OperatorMaterialData *);
PETSC_INTERN PetscErrorCode SetOperatorMaterialValues(OperatorMaterialData *, OperatorMaterialDataIndex, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorMaterialData(Operator *, OperatorMaterialData *);

// This type allows the direct manipulation of flux divergences exchanged
// between the flux and source operators on the entire domain for the system of
// equations being solved by RDycore.
typedef struct {
  // associated operator
  Operator *op;
  // number of components in the underlying system
  PetscInt num_components;
  // underlying data storage
  OperatorVectorData storage;
} OperatorFluxDivergenceData;

PETSC_INTERN PetscErrorCode GetOperatorFluxDivergenceData(Operator *, OperatorFluxDivergenceData *);
PETSC_INTERN PetscErrorCode SetOperatorFluxDivergenceValues(OperatorFluxDivergenceData *, PetscInt, PetscReal *);
PETSC_INTERN PetscErrorCode RestoreOperatorFluxDivergenceData(Operator *, OperatorFluxDivergenceData *);

#endif
