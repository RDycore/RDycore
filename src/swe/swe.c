#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>
#include <stddef.h>  // for offsetof
                     //
// maximum length of the name of a prognostic or diagnostic field component
#define MAX_COMP_NAME_LENGTH 20

extern PetscLogEvent RDY_CeedOperatorApply;

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

  PetscCall(CreateOperator(rdy, &rdy->operator));

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
