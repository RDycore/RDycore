#include <private/rdycoreimpl.h>
#include <rdycore.h>

static PetscErrorCode ReadBoundaryConditions(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadFlow(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadGrid(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadInitialConditions(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadNumerics(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadPhysics(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadRestart(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadSediments(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadSourcesAndSinks(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode ReadTime(RDy rdy) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/// Performs any setup needed by RDy after it has been configured.
PetscErrorCode RDySetup(RDy rdy) {
  PetscFunctionBegin;

  // read all relevant YAML "blocks"

  PetscCall(ReadBoundaryConditions(rdy));
  PetscCall(ReadFlow(rdy));
  PetscCall(ReadGrid(rdy));
  PetscCall(ReadInitialConditions(rdy));
  PetscCall(ReadNumerics(rdy));
  PetscCall(ReadPhysics(rdy));
  PetscCall(ReadRestart(rdy));
  PetscCall(ReadSediments(rdy));
  PetscCall(ReadSourcesAndSinks(rdy));
  PetscCall(ReadTime(rdy));

  PetscFunctionReturn(0);
}

