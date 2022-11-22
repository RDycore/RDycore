// This unit test suite tests the RDyMesh type. See https://cmocka.org for
// details about the CMocka testing framework.

#include <rdycore.h>
#include <private/rdymeshimpl.h>

#include "rdycore_tests.h"

// Creates a unit box mesh in 2D.
// FIXME: Seems like we're already making a lot of assumptions about
// FIXME: DMs from which we create meshes. Might need some lower-level
// FIXME: DM creation utilities.
static PetscErrorCode CreateUnitBoxMesh(PetscInt Nx, PetscInt Ny, DM *dm) {
  PetscFunctionBegin;
  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt dim = 2;
  PetscReal x1 = 0.0, x2 = 1.0;
  PetscReal y1 = 0.0, y2 = 1.0;
  PetscInt  faces[] = {Nx, Ny};
  PetscReal lower[] = {x1, y1};
  PetscReal upper[] = {x2, y2};
  PetscCall(DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, lower, upper,
      PETSC_NULL, PETSC_TRUE, dm));
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

  PetscFunctionReturn(0);
}

// Test the creation of an RDyMesh from a DM.
static void TestRDyMeshCreateFromDM(void **state) {
  DM dm;
  PetscInt Nx = 100, Ny = 100;
  assert_int_equal(0, CreateUnitBoxMesh(Nx, Ny, &dm));

  // Create a mesh that represents the entire DM.
  RDyMesh global_mesh;
  assert_int_equal(0, RDyMeshCreateFromDM(dm, &global_mesh));
  assert_int_equal(Nx*Ny, global_mesh.num_cells);
  assert_int_equal(Nx*Ny, global_mesh.num_cells_local);

  // Distribute the box mesh and create a local representation.
  DM dist_dm;
  assert_int_equal(0, DMPlexDistribute(dm, 0, NULL, &dist_dm));
  if (dist_dm) { // if nproc > 1
    RDyMesh local_mesh;
    assert_int_equal(0, RDyMeshCreateFromDM(dist_dm, &local_mesh));
    assert_int_equal(Nx*Ny, local_mesh.num_cells);
    assert_true(local_mesh.num_cells_local < Nx*Ny);

    assert_int_equal(0, RDyMeshDestroy(local_mesh));
  }

  assert_int_equal(0, RDyMeshDestroy(global_mesh));
}

// Called in advance of tests
static int Setup(void **state) {
  return RDyInitNoArguments();
}

// Called upon completion of tests
static int Teardown(void **state) {
  return RDyFinalize();
}

int main(int argc, char *argv[]) {
  // Define our set of unit tests.
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestRDyMeshCreateFromDM),
  };

  // The last two arguments are for setup and teardown functions.
  // (See https://api.cmocka.org/group__cmocka__exec.html)
  return cmocka_run_group_tests(tests, Setup, Teardown);
}
