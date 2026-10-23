/// Load balancing after an adapt: partition the parents, not the adapted mesh.
///
/// Refinement concentrates on the feature, so one rank ends up owning most of
/// the fine cells. The obvious fix -- DMPlexDistribute() the adapted mesh --
/// breaks everything the adapt rests on: the interpolation matrices and the
/// conservative transfer are sequential per rank and assume a base cell's
/// descendants all live on one rank, and a graph partitioner has no reason to
/// keep them together. p4est gets this free because a space-filling curve
/// keeps a parent's children contiguous; for a plex we have to ask for it.
///
/// So the base mesh is what gets partitioned, weighted by the number of cells
/// each base cell is about to carry, and the adapted mesh is rebuilt from the
/// result. A base cell's descendants are then on one rank by construction.
/// The solution crosses as packed polygons and values (src/rdyamr_transfer.c),
/// which DMPlexDistributeField() moves along the same migration.
///
/// RDycore chooses the partition itself rather than handing the weights to the
/// DM's partitioner. PetscPartitionerDMPlexPartition() does weight each cell
/// by the dofs its closure carries in the local section, but only a real graph
/// partitioner consumes those weights: PETSCPARTITIONERSIMPLE, which is what
/// PETSc falls back to when it is configured without ParMETIS or PTScotch,
/// logs "PETSCPARTITIONERSIMPLE ignores vertex weights" and divides the cells
/// evenly by count. Measured on a build without ParMETIS: setting the section
/// weights changes nothing at all. A chop of the global base-cell order by
/// cumulative weight works on every build and is a deterministic function of
/// the levels, which is what lets the adapt give the same mesh on any number
/// of ranks. It minimizes no edge cut, so where ParMETIS is configured it
/// would give a better partition than this; see plans/amr-petsc-review.md.

#include <petscdmplex.h>
#include <petscsf.h>
#include <private/rdyamrimpl.h>
#include <private/rdycoreimpl.h>
#include <private/rdydmimpl.h>
#include <private/rdymeshimpl.h>

/// @brief Marks which cells of `dm` this rank owns, by local cell index.
static PetscErrorCode OwnedCells(DM dm, PetscInt ncells, PetscBool *owned) {
  PetscFunctionBegin;
  for (PetscInt c = 0; c < ncells; ++c) owned[c] = PETSC_TRUE;

  PetscSF sf;
  PetscCall(DMGetPointSF(dm, &sf));
  if (!sf) PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt        cStart, cEnd, nroots, nleaves;
  const PetscInt *ilocal;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL));
  if (nroots < 0) PetscFunctionReturn(PETSC_SUCCESS);  // no graph set: nothing is a ghost
  for (PetscInt l = 0; l < nleaves; ++l) {
    PetscInt p = ilocal ? ilocal[l] : l;
    if (p >= cStart && p < cEnd) owned[p - cStart] = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Gives each rank a contiguous run of the global base-cell order
///        carrying about the same number of adapted cells.
///
///        The weight of a base cell is how many cells it is about to carry:
///        the number it carries now, which the packed section records exactly,
///        doubled or halved for each level this adapt moves it. Counting it as
///        2^level instead is noticeably worse, because refine_sbr splits a
///        marked triangle into two children or three, so a base cell deep in a
///        refined region holds more than 2^level cells and holds them in a
///        proportion no formula gives.
///
///        The run boundaries fall where the cumulative weight crosses a
///        multiple of total/nranks, so the order of the base cells is
///        preserved across the migration and a balance never reshuffles the
///        mesh wholesale.
static PetscErrorCode WeightedChop(RDy rdy, DM dm_base, const PetscBool *owned, const PetscInt *old_level, PetscInt ndof, PetscSection sec,
                                   PetscPartitioner *part) {
  PetscFunctionBegin;

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(rdy->comm, &size));

  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(dm_base, 0, &cStart, &cEnd));
  const PetscInt nlocal = cEnd - cStart;
  const PetscInt stride = RDyAMRPackStride(ndof);

  // the partitioner numbers only the owned cells, in stratum order
  PetscInt nowned = 0, mine = 0;
  for (PetscInt c = 0; c < nlocal; ++c)
    if (owned[c]) ++nowned;

  PetscInt *w;
  PetscCall(PetscMalloc1(nowned, &w));
  for (PetscInt c = 0; c < nlocal; ++c) {
    if (!owned[c]) continue;
    PetscInt dof;
    PetscCall(PetscSectionGetDof(sec, c + cStart, &dof));
    PetscInt now   = dof / stride;
    PetscInt delta = rdy->amr.base_level[c] - old_level[c];
    PetscInt next  = (delta >= 0) ? (now << delta) : (now >> PetscMin(-delta, 30));
    w[mine++]      = PetscMax(next, 1);
  }

  PetscInt sum = 0;
  for (PetscInt i = 0; i < nowned; ++i) sum += w[i];
  PetscInt before = 0, total = 0;
  PetscCallMPI(MPI_Exscan(&sum, &before, 1, MPIU_INT, MPI_SUM, rdy->comm));
  PetscCallMPI(MPI_Allreduce(&sum, &total, 1, MPIU_INT, MPI_SUM, rdy->comm));

  PetscInt *sizes, *points, *fill;
  PetscCall(PetscCalloc1(size, &sizes));
  PetscCall(PetscCalloc1(size, &fill));
  PetscCall(PetscMalloc1(nowned, &points));

  PetscInt *target;
  PetscCall(PetscMalloc1(nowned, &target));
  PetscInt running = before;
  for (PetscInt i = 0; i < nowned; ++i) {
    PetscInt r = (PetscInt)((PetscReal)running * size / (PetscReal)total);
    target[i]  = PetscMin(PetscMax(r, 0), size - 1);
    sizes[target[i]]++;
    running += w[i];
  }
  // points is a permutation of this rank's cells grouped by destination
  PetscInt off = 0;
  for (PetscInt r = 0; r < size; ++r) {
    fill[r] = off;
    off += sizes[r];
  }
  for (PetscInt i = 0; i < nowned; ++i) points[fill[target[i]]++] = i;

  PetscCall(PetscPartitionerCreate(rdy->comm, part));
  PetscCall(PetscPartitionerSetType(*part, PETSCPARTITIONERSHELL));
  PetscCall(PetscPartitionerShellSetPartition(*part, size, sizes, points));

  PetscCall(PetscFree(w));
  PetscCall(PetscFree(target));
  PetscCall(PetscFree(sizes));
  PetscCall(PetscFree(fill));
  PetscCall(PetscFree(points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Moves one real per cell of `dm_from` onto `dm_to` along `sf`.
///
///        A thin wrapper over DMPlexDistributeField() for the common case of a
///        fixed one value per cell, used for the per-base-cell bookkeeping.
static PetscErrorCode MigrateCellScalars(DM dm_from, DM dm_to, PetscSF sf, const PetscReal *from, PetscReal **to, PetscInt *nto) {
  PetscFunctionBegin;

  PetscInt fStart, fEnd, tStart, tEnd;
  PetscCall(DMPlexGetHeightStratum(dm_from, 0, &fStart, &fEnd));
  PetscCall(DMPlexGetHeightStratum(dm_to, 0, &tStart, &tEnd));

  PetscSection s_from, s_to;
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s_from));
  PetscCall(PetscSectionSetChart(s_from, fStart, fEnd));
  for (PetscInt c = fStart; c < fEnd; ++c) PetscCall(PetscSectionSetDof(s_from, c, 1));
  PetscCall(PetscSectionSetUp(s_from));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s_to));

  Vec          v_from, v_to;
  PetscScalar *a;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, fEnd - fStart, &v_from));
  PetscCall(VecGetArray(v_from, &a));
  for (PetscInt c = 0; c < fEnd - fStart; ++c) a[c] = from[c];
  PetscCall(VecRestoreArray(v_from, &a));
  PetscCall(VecCreate(PETSC_COMM_SELF, &v_to));

  PetscCall(DMPlexDistributeField(dm_to, sf, s_from, v_from, s_to, v_to));

  *nto = tEnd - tStart;
  PetscCall(PetscMalloc1(*nto, to));
  const PetscScalar *b;
  PetscCall(VecGetArrayRead(v_to, &b));
  for (PetscInt c = tStart; c < tEnd; ++c) {
    PetscInt o;
    PetscCall(PetscSectionGetOffset(s_to, c, &o));
    (*to)[c - tStart] = PetscRealPart(b[o]);
  }
  PetscCall(VecRestoreArrayRead(v_to, &b));

  PetscCall(PetscSectionDestroy(&s_from));
  PetscCall(PetscSectionDestroy(&s_to));
  PetscCall(VecDestroy(&v_from));
  PetscCall(VecDestroy(&v_to));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Moves a packed-by-base-cell payload onto another mesh along `sf`.
static PetscErrorCode MigratePacked(DM dm_to, PetscSF sf, PetscSection *sec, PetscScalar **packed) {
  PetscFunctionBegin;

  PetscInt n;
  PetscCall(PetscSectionGetStorageSize(*sec, &n));

  Vec          v_from, v_to;
  PetscScalar *a;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &v_from));
  PetscCall(VecGetArray(v_from, &a));
  for (PetscInt i = 0; i < n; ++i) a[i] = (*packed)[i];
  PetscCall(VecRestoreArray(v_from, &a));
  PetscCall(VecCreate(PETSC_COMM_SELF, &v_to));

  PetscSection s_to;
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &s_to));
  PetscCall(DMPlexDistributeField(dm_to, sf, *sec, v_from, s_to, v_to));

  PetscInt m;
  PetscCall(PetscSectionGetStorageSize(s_to, &m));
  PetscCall(PetscFree(*packed));
  PetscCall(PetscCalloc1(PetscMax(m, 1), packed));
  const PetscScalar *b;
  PetscCall(VecGetArrayRead(v_to, &b));
  for (PetscInt i = 0; i < m; ++i) (*packed)[i] = b[i];
  PetscCall(VecRestoreArrayRead(v_to, &b));

  PetscCall(PetscSectionDestroy(sec));
  *sec = s_to;

  PetscCall(VecDestroy(&v_from));
  PetscCall(VecDestroy(&v_to));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Repartitions the base mesh so that each rank will own about the same
///        number of adapted cells, and brings everything indexed by base cell
///        across with it.
///
///        Call between updating rdy->amr.base_level and rebuilding from the
///        base mesh: the weights are the levels the rebuild is about to use.
///        On one rank, and whenever DMPlexDistribute() declines to move
///        anything, this returns having changed nothing.
///
/// @param [in] rdy          the dycore, whose amr.dm_base is replaced
/// @param [in] dm_base_old  the base mesh in hand, which `sec` is laid out over
/// @param [in] old_level    each base cell's level before this adapt moved it
/// @param [in] ndof         components per cell, for reading `sec`
/// @param [in,out] sec      the packed old cells' layout, re-laid on the new base
/// @param [in,out] packed   the packed old cells, migrated with it
PetscErrorCode RDyAMRBalanceBaseMesh(RDy rdy, DM dm_base_old, const PetscInt *old_level, PetscInt ndof, PetscSection *sec, PetscScalar **packed) {
  PetscFunctionBegin;

  PetscMPIInt size;
  PetscCallMPI(MPI_Comm_size(rdy->comm, &size));
  if (size == 1 || !rdy->amr.balance) PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(dm_base_old, 0, &cStart, &cEnd));
  const PetscInt nlocal = cEnd - cStart;
  PetscCheck(nlocal == rdy->amr.num_base_cells, rdy->comm, PETSC_ERR_PLIB,
             "the base mesh has %" PetscInt_FMT " cells but %" PetscInt_FMT " are recorded", nlocal, rdy->amr.num_base_cells);

  PetscBool *owned;
  PetscCall(PetscMalloc1(nlocal, &owned));
  PetscCall(OwnedCells(dm_base_old, nlocal, owned));

  PetscPartitioner part;
  PetscCall(WeightedChop(rdy, dm_base_old, owned, old_level, ndof, *sec, &part));
  PetscCall(PetscFree(owned));

  // DMPlexDistribute() reads the owned cells and drops the partition overlap,
  // so the base mesh can be handed to it as it stands; the overlap goes back
  // on afterwards, and the rebuild inherits it as before.
  PetscPartitioner was;
  PetscCall(DMPlexGetPartitioner(dm_base_old, &was));
  PetscCall(PetscObjectReference((PetscObject)was));
  PetscCall(DMPlexSetPartitioner(dm_base_old, part));

  // The SF DMPlexDistribute() hands back is composed with whatever migration
  // SF the DM already carries (plexdistribute.c, "Compose with a previous
  // sfMigration if present"). RDycore's base mesh carries the one that defines
  // its natural ordering, so the composed SF is rooted in the original serial
  // mesh on rank 0 -- every other rank gets an SF with no roots, and a field
  // migrated along it arrives empty. Take the SF off for the call, so what
  // comes back runs from the base mesh in hand, and put the composition back
  // together afterwards: the natural ordering is what reads a DEM.
  PetscSF   sf_serial = NULL;
  PetscBool was_natural;
  PetscCall(DMPlexGetMigrationSF(dm_base_old, &sf_serial));
  if (sf_serial) PetscCall(PetscObjectReference((PetscObject)sf_serial));
  PetscCall(DMPlexSetMigrationSF(dm_base_old, NULL));
  PetscCall(DMGetUseNatural(dm_base_old, &was_natural));
  PetscCall(DMSetUseNatural(dm_base_old, PETSC_FALSE));

  DM      dm_flat;
  PetscSF sf_mig;
  PetscCall(DMPlexDistribute(dm_base_old, 0, &sf_mig, &dm_flat));

  PetscCall(DMSetUseNatural(dm_base_old, was_natural));
  PetscCall(DMPlexSetMigrationSF(dm_base_old, sf_serial));
  PetscCall(DMPlexSetPartitioner(dm_base_old, was));
  PetscCall(PetscObjectDereference((PetscObject)was));
  PetscCall(PetscPartitionerDestroy(&part));
  if (!dm_flat) {  // nothing moved
    PetscCall(PetscSFDestroy(&sf_serial));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  DM      dm_base_new;
  PetscSF sf_ovl;
  PetscCall(DMPlexDistributeOverlap(dm_flat, 1, &sf_ovl, &dm_base_new));
  PetscCheck(dm_base_new, rdy->comm, PETSC_ERR_PLIB, "the redistributed base mesh took no partition overlap");
  PetscCall(DMSetBasicAdjacency(dm_base_new, PETSC_TRUE, PETSC_TRUE));

  // serial -> old base -> redistributed -> overlapped, which is the migration
  // SF the natural ordering of the new base mesh is built from
  if (sf_serial) {
    PetscSF to_flat, to_new;
    PetscCall(PetscSFCompose(sf_serial, sf_mig, &to_flat));
    PetscCall(DMPlexRemapMigrationSF(sf_ovl, to_flat, &to_new));
    PetscCall(DMPlexSetMigrationSF(dm_base_new, to_new));
    PetscCall(PetscSFDestroy(&to_flat));
    PetscCall(PetscSFDestroy(&to_new));
    PetscCall(PetscSFDestroy(&sf_serial));
  }

  // the per-base-cell bookkeeping and the packed solution follow the cells:
  // first onto the redistributed mesh, then out onto its overlap
  PetscReal *level_flat, *area_flat, *level_new, *area_new;
  PetscInt   nflat, nnew;
  {
    PetscReal *level_old, *area_old;
    PetscCall(PetscMalloc1(nlocal, &level_old));
    PetscCall(PetscMalloc1(nlocal, &area_old));
    for (PetscInt c = 0; c < nlocal; ++c) {
      level_old[c] = (PetscReal)rdy->amr.base_level[c];
      area_old[c]  = rdy->amr.base_cell_area[c];
    }
    PetscCall(MigrateCellScalars(dm_base_old, dm_flat, sf_mig, level_old, &level_flat, &nflat));
    PetscCall(MigrateCellScalars(dm_base_old, dm_flat, sf_mig, area_old, &area_flat, &nflat));
    PetscCall(PetscFree(level_old));
    PetscCall(PetscFree(area_old));
  }
  PetscCall(MigrateCellScalars(dm_flat, dm_base_new, sf_ovl, level_flat, &level_new, &nnew));
  PetscCall(MigrateCellScalars(dm_flat, dm_base_new, sf_ovl, area_flat, &area_new, &nnew));
  PetscCall(PetscFree(level_flat));
  PetscCall(PetscFree(area_flat));

  PetscCall(MigratePacked(dm_flat, sf_mig, sec, packed));
  PetscCall(MigratePacked(dm_base_new, sf_ovl, sec, packed));

  PetscCall(PetscFree(rdy->amr.base_level));
  PetscCall(PetscFree(rdy->amr.base_cell_area));
  PetscCall(PetscMalloc1(nnew, &rdy->amr.base_level));
  PetscCall(PetscMalloc1(nnew, &rdy->amr.base_cell_area));
  for (PetscInt c = 0; c < nnew; ++c) {
    rdy->amr.base_level[c]     = (PetscInt)PetscFloorReal(level_new[c] + 0.5);
    rdy->amr.base_cell_area[c] = area_new[c];
  }
  rdy->amr.num_base_cells = nnew;
  PetscCall(PetscFree(level_new));
  PetscCall(PetscFree(area_new));

  // hand the new base mesh over, with the one-component DM the bed and the
  // base-to-current interpolation are built on
  PetscCall(DMDestroy(&rdy->amr.dm_base));
  PetscCall(DMDestroy(&rdy->amr.dm_1dof_base));
  rdy->amr.dm_base = dm_base_new;
  // refinement level 0, so the one-component DM takes the natural ordering off
  // the migration SF above: reading a per-base-cell DEM goes through it
  PetscCall(CreateCellCenteredDMFromDM(dm_base_new, 0, rdy->field_1dof, &rdy->amr.dm_1dof_base));

  PetscCall(PetscSFDestroy(&sf_mig));
  PetscCall(PetscSFDestroy(&sf_ovl));
  PetscCall(DMDestroy(&dm_flat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
