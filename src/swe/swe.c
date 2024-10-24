#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>
#include <stddef.h>  // for offsetof
                     //
// maximum length of the name of a prognostic or diagnostic field component
#define MAX_COMP_NAME_LENGTH 20

extern PetscLogEvent RDY_CeedOperatorApply;

// these functions are implemented in swe_petsc.c
PetscErrorCode SWERHSFunctionForInternalEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags);
PetscErrorCode SWERHSFunctionForBoundaryEdges(RDy rdy, Vec F, CourantNumberDiagnostics *courant_num_diags);
PetscErrorCode ComputeSWEDiagnosticVariables(RDy rdy);
PetscErrorCode AddSWESourceTerm(RDy rdy, Vec F);

//-----------------------
// Debugging diagnostics
//-----------------------

// MPI datatype corresponding to CourantNumberDiagnostics. Created during
// CreateSWEOperator.
static MPI_Datatype courant_num_diags_type;

// MPI operator used to determine the prevailing diagnostics for the maximum
// courant number on all processes. Created during CreateSWEOperator.
static MPI_Op courant_num_diags_op;

// function backing the above MPI operator
static void FindCourantNumberDiagnostics(void *in_vec, void *result_vec, int *len, MPI_Datatype *type) {
  CourantNumberDiagnostics *in_diags     = in_vec;
  CourantNumberDiagnostics *result_diags = result_vec;

  // select the item with the maximum courant number
  for (int i = 0; i < *len; ++i) {
    if (in_diags[i].max_courant_num > result_diags[i].max_courant_num) {
      result_diags[i] = in_diags[i];
    }
  }
}

// this function destroys the MPI type and operator associated with CourantNumberDiagnostics
static void DestroyCourantNumberDiagnostics(void) {
  MPI_Op_free(&courant_num_diags_op);
  MPI_Type_free(&courant_num_diags_type);
}

// this function is called by InitSWEOpeⅹator to initialize the above globals
static PetscErrorCode InitMPITypesAndOps(void) {
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
    MPI_Type_create_struct(num_blocks, block_lengths, block_displacements, block_types, &courant_num_diags_type);
    MPI_Type_commit(&courant_num_diags_type);

    // create a corresponding reduction operator for the new type
    MPI_Op_create(FindCourantNumberDiagnostics, 1, &courant_num_diags_op);

    // make sure the operator and the type are destroyed upon exit
    PetscCall(RDyOnFinalize(DestroyCourantNumberDiagnostics));

    initialized = PETSC_TRUE;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// create flux and source operators
static PetscErrorCode CreateOperator(RDy rdy, Operator *operator) {
  PetscFunctionBegin;
  if (CeedEnabled()) {
    RDyLogDebug(rdy, "Creating CEED SWE operator...");
    PetscCall(CreateCeedSWEOperator(rdy, operator));
  } else {
    RDyLogDebug(rdy, "Creating PETSc SWE operator...");
    PetscCall(CreatePetscSWEOperator(rdy, operator));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//---------------------------
// End debugging diagnostics
//---------------------------

// This function creates a PetscSection appropriate for the shallow water equations.
PetscErrorCode CreateSWESection(RDy rdy, PetscSection *section) {
  PetscFunctionBegin;
  PetscInt n_field                             = 1;
  PetscInt n_field_comps[1]                    = {3};
  char     comp_names[3][MAX_COMP_NAME_LENGTH] = {
          "Height",
          "MomentumX",
          "MomentumY",
  };

  PetscCall(PetscSectionCreate(rdy->comm, section));
  PetscCall(PetscSectionSetNumFields(*section, n_field));
  PetscInt n_field_dof_tot = 0;
  for (PetscInt f = 0; f < n_field; ++f) {
    PetscCall(PetscSectionSetFieldComponents(*section, f, n_field_comps[f]));
    for (PetscInt c = 0; c < n_field_comps[f]; ++c, ++n_field_dof_tot) {
      PetscCall(PetscSectionSetComponentName(*section, f, c, comp_names[c]));
    }
  }

  // set the number of degrees of freedom in each cell
  PetscInt c_start, c_end;  // starting and ending cell points
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &c_start, &c_end));
  PetscCall(PetscSectionSetChart(*section, c_start, c_end));
  for (PetscInt c = c_start; c < c_end; ++c) {
    for (PetscInt f = 0; f < n_field; ++f) {
      PetscCall(PetscSectionSetFieldDof(*section, c, f, n_field_comps[f]));
    }
    PetscCall(PetscSectionSetDof(*section, c, n_field_dof_tot));
  }
  PetscCall(PetscSectionSetUp(*section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This function initializes SWE physics for the given dycore.
PetscErrorCode InitSWE(RDy rdy) {
  PetscFunctionBeginUser;

  // set up MPI types and operators used by SWE physics
  PetscCall(InitMPITypesAndOps());

  // sets up solvers, operators, all that stuff
  PetscCall(CreateOperator(rdy, &rdy->operator));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Loops over all internal edges and finds the local maximum courant number.
///        If needed, the data is moved from device to host.
/// @param [in] op_edges A CeedOperator object for edges
/// @param [in] mesh A pointer to a RDyMesh object
/// @param [in] *max_courant_number Local maximum value of courant number
/// @return 0 on sucess, or a non-zero error code on failure
static PetscErrorCode CeedFindMaxCourantNumberInternalEdges(CeedOperator op_edges, RDyMesh *mesh, PetscReal *max_courant_number) {
  PetscFunctionBegin;

  // get the relevant interior sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op_edges, &sub_ops));
  CeedOperator interior_flux_op = sub_ops[0];

  // fetch the field
  CeedOperatorField courant_num;
  PetscCallCEED(CeedOperatorGetFieldByName(interior_flux_op, "courant_number", &courant_num));

  CeedVector courant_num_vec;
  PetscCallCEED(CeedOperatorFieldGetVector(courant_num, &courant_num_vec));

  CeedScalar(*courant_num_data)[2];  // values to the left/right of an edge
  PetscCallCEED(CeedVectorGetArray(courant_num_vec, CEED_MEM_HOST, (CeedScalar **)&courant_num_data));

  // RDyMesh  *mesh  = &rdy->mesh;
  RDyEdges *edges = &mesh->edges;

  PetscInt iedge_owned = 0;
  for (PetscInt ii = 0; ii < mesh->num_internal_edges; ii++) {
    if (edges->is_owned[ii]) {
      CeedScalar local_max = fmax(courant_num_data[iedge_owned][0], courant_num_data[iedge_owned][1]);
      *max_courant_number  = fmax(*max_courant_number, local_max);
      iedge_owned++;
    }
  }
  PetscCallCEED(CeedVectorRestoreArray(courant_num_vec, (CeedScalar **)&courant_num_data));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Loops over all boundary conditions and finds the local maximum Courant number.
///        If needed, the data is moved from device to host.
/// @param [in] op_edges A CeedOperator object for edges
/// @param [in] num_boundaries Total number of boundaries
/// @param [in] boundaries A RDyBoundary object
/// @param [in] *max_courant_number Local maximum value of courant number
/// @return 0 on sucess, or a non-zero error code on failure
static PetscErrorCode CeedFindMaxCourantNumberBoundaryEdges(CeedOperator op_edges, PetscInt num_boundaries, RDyBoundary *boundaries,
                                                            PetscReal *max_courant_number) {
  PetscFunctionBegin;

  // loop over all boundaries
  for (PetscInt b = 0; b < num_boundaries; ++b) {
    RDyBoundary boundary = boundaries[b];

    // get the relevant boundary sub-operator
    CeedOperator *sub_ops;
    PetscCallCEED(CeedCompositeOperatorGetSubList(op_edges, &sub_ops));
    CeedOperator boundary_flux_op = sub_ops[1 + boundary.index];

    // fetch the field
    CeedOperatorField courant_num;
    PetscCallCEED(CeedOperatorGetFieldByName(boundary_flux_op, "courant_number", &courant_num));

    // get access to the data
    CeedVector courant_num_vec;
    PetscCallCEED(CeedOperatorFieldGetVector(courant_num, &courant_num_vec));
    CeedScalar(*courant_num_data)[1];
    PetscCallCEED(CeedVectorGetArray(courant_num_vec, CEED_MEM_HOST, (CeedScalar **)&courant_num_data));

    // find the maximum value
    for (PetscInt e = 0; e < boundary.num_edges; ++e) {
      *max_courant_number = fmax(*max_courant_number, courant_num_data[e][0]);
    }

    // restores the pointer
    PetscCallCEED(CeedVectorRestoreArray(courant_num_vec, (CeedScalar **)&courant_num_data));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Finds the global maximum Courant number across all internal and boundary edges.
/// @param [in] op_edges A CeedOperator object for edges
/// @param [in] num_boundaries Total number of boundaries
/// @param [in] boundaries A RDyBoundary object
/// @param [in] comm A MPI_Comm object
/// @param [out] *max_courant_number Global maximum value of courant number
/// @return 0 on sucess, or a non-zero error code on failure
PetscErrorCode CeedFindMaxCourantNumber(CeedOperator op_edges, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries, MPI_Comm comm,
                                        PetscReal *max_courant_number) {
  PetscFunctionBegin;

  PetscCall(CeedFindMaxCourantNumberInternalEdges(op_edges, mesh, max_courant_number));
  PetscCall(CeedFindMaxCourantNumberBoundaryEdges(op_edges, num_boundaries, boundaries, max_courant_number));

  PetscCall(MPI_Allreduce(MPI_IN_PLACE, max_courant_number, 1, MPI_DOUBLE, MPI_MAX, comm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFindMaxCourantNumber(RDy rdy, PetscReal *max_courant_number) {
  PetscFunctionBegin;
  CourantNumberDiagnostics *courant_num_diags = &rdy->courant_num_diags;
  MPI_Allreduce(MPI_IN_PLACE, courant_num_diags, 1, courant_num_diags_type, courant_num_diags_op, rdy->comm);
  *max_courant_number = courant_num_diags->max_courant_num;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Finds the maximum Courant number for the libCEED and the PETSc version of SWE implementation
/// @param [inout] rdy An RDy object
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode SWEFindMaxCourantNumber(RDy rdy) {
  PetscFunctionBegin;

  CourantNumberDiagnostics *courant_num_diags = &rdy->courant_num_diags;

  if (CeedEnabled()) {
    PetscCall(CeedFindMaxCourantNumber(rdy->operator.ceed.flux_operator, &rdy->mesh, rdy->num_boundaries, rdy->boundaries, rdy->comm,
                                       &courant_num_diags->max_courant_num));
  } else {
    PetscCall(PetscFindMaxCourantNumber(rdy, &courant_num_diags->max_courant_num));
  }
  courant_num_diags->is_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannDataSWECreate(PetscInt N, RiemannDataSWE *data) {
  PetscFunctionBegin;

  data->N = N;
  PetscCall(PetscCalloc1(data->N, &data->h));
  PetscCall(PetscCalloc1(data->N, &data->hu));
  PetscCall(PetscCalloc1(data->N, &data->hv));
  PetscCall(PetscCalloc1(data->N, &data->u));
  PetscCall(PetscCalloc1(data->N, &data->v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannDataSWEDestroy(RiemannDataSWE data) {
  PetscFunctionBegin;

  data.N = 0;
  PetscCall(PetscFree(data.h));
  PetscCall(PetscFree(data.hu));
  PetscCall(PetscFree(data.hv));
  PetscCall(PetscFree(data.u));
  PetscCall(PetscFree(data.v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannEdgeDataSWECreate(PetscInt N, PetscInt ncomp, RiemannEdgeDataSWE *data) {
  PetscFunctionBegin;

  data->N = N;
  PetscCall(PetscCalloc1(data->N, &data->cn));
  PetscCall(PetscCalloc1(data->N, &data->sn));
  PetscCall(PetscCalloc1(data->N * ncomp, &data->flux));
  PetscCall(PetscCalloc1(data->N, &data->amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RiemannEdgeDataSWEDestroy(RiemannEdgeDataSWE data) {
  PetscFunctionBegin;

  data.N = 0;
  PetscCall(PetscFree(data.cn));
  PetscCall(PetscFree(data.sn));
  PetscCall(PetscFree(data.flux));
  PetscCall(PetscFree(data.amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}
