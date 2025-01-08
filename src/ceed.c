#include <petscdmceed.h>
#include <private/rdycoreimpl.h>

// global CEED resource and context
static char ceed_resource[PETSC_MAX_PATH_LEN] = {0};
static Ceed ceed_context;

/// returns true iff CEED is enabled
PetscBool CeedEnabled(void) { return (ceed_context != NULL); }

/// returns the global CEED context, which is only valid if CeedEnabled()
/// returns PETSC_TRUE
Ceed CeedContext(void) { return ceed_context; }

/// retrieves the appropriate PETSc Vec type for the selected CEED backend
PetscErrorCode GetCeedVecType(VecType *vec_type) {
  PetscFunctionBegin;

  CeedMemType mem_type_backend;
  PetscCallCEED(CeedGetPreferredMemType(ceed_context, &mem_type_backend));
  switch (mem_type_backend) {
    case CEED_MEM_HOST:
      *vec_type = VECSTANDARD;
      break;
    case CEED_MEM_DEVICE: {
      const char *resolved;
      PetscCallCEED(CeedGetResource(ceed_context, &resolved));
      if (strstr(resolved, "/gpu/cuda")) *vec_type = VECCUDA;
      else if (strstr(resolved, "/gpu/hip")) *vec_type = VECKOKKOS;
      else if (strstr(resolved, "/gpu/sycl")) *vec_type = VECKOKKOS;
      else *vec_type = VECSTANDARD;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// this get called on process exit to clear the CEED resource
static void ClearCeedResource(void) { SetCeedResource(NULL); }

/// Sets the CEED resource string to the given string, initializing the global
/// CEED context if the argument is specified. If CEED has already been enabled
/// with a different resource, a call to this function deletes the global
/// context and recreates it. If the resource string is empty or NULL, CEED is
/// disabled.
/// @param resource a CEED resource string, possibly empty or NULL
PetscErrorCode SetCeedResource(char *resource) {
  PetscFunctionBegin;

  if (ceed_context) {  // we already have a context
    // if it's the same as we're already using, do nothing
    if (resource && !strcmp(resource, ceed_resource)) PetscFunctionReturn(PETSC_SUCCESS);

    // otherwise clear the context before we reset it
    CeedDestroy(&ceed_context);
    ceed_context = NULL;
  }
  if (resource && resource[0]) {
    PetscCallCEED(CeedInit(resource, &ceed_context));
    strncpy(ceed_resource, resource, PETSC_MAX_PATH_LEN);

    static bool firstTime = true;
    if (firstTime) {
      PetscCall(RDyOnFinalize(ClearCeedResource));
      firstTime = false;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
