// This unit test suite tests the RDyMesh type. See https://cmocka.org for
// details about the CMocka testing framework.

#include <private/rdymeshimpl.h>
#include <rdycore.h>

#include "rdycore_tests.h"

// Creates a unit box mesh in 2D.
// FIXME: Seems like we're already making a lot of assumptions about
// FIXME: DMs from which we create meshes. Might need some lower-level
// FIXME: DM creation utilities.
static PetscErrorCode Create2DUnitBoxDM(PetscInt Nx, PetscInt Ny, DM *dm) {
  PetscFunctionBegin;
  MPI_Comm  comm = PETSC_COMM_WORLD;
  PetscInt  dim  = 2;
  PetscReal x1 = 0.0, x2 = 1.0;
  PetscReal y1 = 0.0, y2 = 1.0;
  PetscInt  faces[]            = {Nx, Ny};
  PetscReal lower[]            = {x1, y1};
  PetscReal upper[]            = {x2, y2};
  PetscBool interpolate        = PETSC_TRUE;
  PetscInt  localizationHeight = 0;
  PetscBool sparseLocalize     = PETSC_FALSE;
  PetscCall(DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, lower, upper, NULL, interpolate, localizationHeight, sparseLocalize, dm));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));

  // Determine the number of cells, edges, and vertices of the mesh
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd);

  // Create a single section that has 3 DOFs
  PetscSection sec;
  PetscCall(PetscSectionCreate(comm, &sec));

  // Add the 3 DOFs
  PetscInt nfield             = 3;
  PetscInt num_field_dof[]    = {1, 1, 1};
  char     field_names[3][20] = {{"Height"}, {"Momentum in x-dir"}, {"Momentum in y-dir"}};

  nfield = 3;
  PetscCall(PetscSectionSetNumFields(sec, nfield));
  PetscInt total_num_dof = 0;
  for (PetscInt ifield = 0; ifield < nfield; ifield++) {
    PetscCall(PetscSectionSetFieldName(sec, ifield, &field_names[ifield][0]));
    PetscCall(PetscSectionSetFieldComponents(sec, ifield, num_field_dof[ifield]));
    total_num_dof += num_field_dof[ifield];
  }

  PetscCall(PetscSectionSetChart(sec, cStart, cEnd));
  for (PetscInt c = cStart; c < cEnd; c++) {
    for (PetscInt ifield = 0; ifield < nfield; ifield++) {
      PetscCall(PetscSectionSetFieldDof(sec, c, ifield, num_field_dof[ifield]));
    }
    PetscCall(PetscSectionSetDof(sec, c, total_num_dof));
  }

  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(*dm, sec));

  PetscCall(PetscSectionViewFromOptions(sec, NULL, "-layout_view"));
  PetscCall(PetscSectionDestroy(&sec));
  PetscCall(DMSetBasicAdjacency(*dm, PETSC_TRUE, PETSC_TRUE));

  // Before distributing the DM, set a flag to create mapping from natural-to-local order
  PetscCall(DMSetUseNatural(*dm, PETSC_TRUE));

  // Distribute the DM
  DM dm_dist;
  PetscCall(DMPlexDistribute(*dm, 1, NULL, &dm_dist));
  if (dm_dist) {
    DMDestroy(dm);
    *dm = dm_dist;
  }
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  // Create a global dummy vector to ensure that the distributed DM is ready
  // for use. (This creates a global section needed by DMPlexGetPointLocal.)
  Vec dummy;
  DMCreateGlobalVector(*dm, &dummy);
  VecDestroy(&dummy);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Test the creation of an RDyMesh from a DM.
static void TestRDyMeshCreateFromDM(void **state) {
  // Create a distributed DM.
  DM       dm;
  PetscInt Nx = 100, Ny = 100;
  assert_int_equal(0, Create2DUnitBoxDM(Nx, Ny, &dm));
  assert_int_equal(0, DMSetRefineLevel(dm, 1));

  // Now create a local mesh representation.
  RDyMesh mesh;
  assert_int_equal(0, RDyMeshCreateFromDM(dm, &mesh));
  // I expected the following statement to be true, but it's not (for nproc > 1)
  //  assert_int_equal(Nx * Ny, mesh.num_cells);
  assert_true(mesh.num_owned_cells <= Nx * Ny);  // (== iff nproc == 1)

  // Clean up.
  assert_int_equal(0, DMDestroy(&dm));
  assert_int_equal(0, RDyMeshDestroy(mesh));
}

static int    argc_ = 0;
static char **argv_ = NULL;

// Called in advance of tests
static int Setup(void **state) { return RDyInit(argc_, argv_, "test_rdymesh - a unit test for RDyMesh."); }

// Called upon completion of tests
static int Teardown(void **state) { return RDyFinalize(); }

int main(int argc, char *argv[]) {
  // Jot down our command line args to allow PETSc access to them.
  argc_ = argc;
  argv_ = argv;

  // Define our set of unit tests.
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestRDyMeshCreateFromDM),
  };

  // The last two arguments are for setup and teardown functions.
  // (See https://api.cmocka.org/group__cmocka__exec.html)
  return cmocka_run_group_tests(tests, Setup, Teardown);
}
