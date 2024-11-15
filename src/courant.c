#include <private/rdycoreimpl.h>

// MPI datatype corresponding to CourantNumberDiagnostics. Created during
// CreateSWEOperator.
MPI_Datatype MPI_COURANT_NUMBER_DIAGNOSTICS = {0};

// MPI operator used to determine the prevailing diagnostics for the maximum
// courant number on all processes. Created during CreateSWEOperator.
MPI_Op MPI_MAX_COURANT_NUMBER = {0};

// function implementing the above MPI operator
static void FindCourantNumberDiagnostics(void *in_vec, void *result_vec, int *len, MPI_Datatype *type) {
  CourantNumberDiagnostics *in_diags     = in_vec;
  CourantNumberDiagnostics *result_diags = result_vec;

  // select the item with the maximum courant number
  for (int i = 0; i < *len; ++i) {
    if (in_diags[i].max_courant_num > result_diags[i].max_courant_num) {
      result_diags[i]        = in_diags[i];
      result_diags[i].is_set = PETSC_TRUE;  // mark as set
    }
  }
}

// this function destroys the above MPI datatype and operator
static void DestroyCourantNumberDiagnostics(void) {
  MPI_Op_free(&MPI_MAX_COURANT_NUMBER);
  MPI_Type_free(&MPI_COURANT_NUMBER_DIAGNOSTICS);
}

// this function initializes some MPI machinery for the above Courant number
// diagnostics, and is called by CreateBasicOperator, which is called when a
// CEED or PETSc Operator is created
PetscErrorCode InitCourantNumberDiagnostics(void) {
  PetscFunctionBegin;

  static PetscBool initialized = PETSC_FALSE;

  if (!initialized) {
    // create an MPI data type for the CourantNumberDiagnostics struct
    const int      num_blocks             = 4;
    const int      block_lengths[4]       = {1, 1, 1, 1};
    const MPI_Aint block_displacements[4] = {
        offsetof(CourantNumberDiagnostics, max_courant_num),
        offsetof(CourantNumberDiagnostics, global_edge_id),
        offsetof(CourantNumberDiagnostics, global_cell_id),
        offsetof(CourantNumberDiagnostics, is_set),
    };
    MPI_Datatype block_types[4] = {MPIU_REAL, MPI_INT, MPI_INT, MPIU_BOOL};
    MPI_Type_create_struct(num_blocks, block_lengths, block_displacements, block_types, &MPI_COURANT_NUMBER_DIAGNOSTICS);
    MPI_Type_commit(&MPI_COURANT_NUMBER_DIAGNOSTICS);

    // create a corresponding reduction operator for the new type
    MPI_Op_create(FindCourantNumberDiagnostics, 1, &MPI_MAX_COURANT_NUMBER);

    // make sure the operator and the type are destroyed upon exit
    PetscCall(RDyOnFinalize(DestroyCourantNumberDiagnostics));

    initialized = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
