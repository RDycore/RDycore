/// RDycore's refinement criterion: the jump in water-surface elevation.
///
/// This follows the loop in PETSc's own finite-volume AMR example,
/// src/ts/tutorials/ex11.c (adaptToleranceFVMSetUp): fill a Vec with a
/// per-cell indicator, threshold it with a refine and a coarsen VecTagger, turn
/// the two index sets into a DMLabel, and adapt from the label. DMAdaptor is
/// the wrapper around that loop for finite-element and metric adaptation; for a
/// cell-centred finite-volume indicator the loop is what a caller writes.
///
/// The difference matters here and is not a matter of taste. ex11 indexes its
/// error vector by local cell, ghosts included, so the indices VecTagger
/// returns are mesh points whether or not the mesh carries a partition overlap.
/// DMAdaptor indexes its error vector by owned cell (it is a global vector on a
/// clone of the mesh) and then uses those indices as mesh points, which is only
/// the same thing while the owned cells lead the chart. They do on a freshly
/// distributed mesh and do not on one refined since, because
/// DMPlexTransformApply numbers children by parent and the children of ghost
/// cells interleave. Driving the loop here, on ex11's convention, is what makes
/// solution-driven AMR give the same answer on any number of ranks.
///
/// The built-in indicators would not fit in any case. DMADAPTORGRADIENT
/// recovers the gradient of the solution field, which here is water depth on a
/// P0 space: on a varying bed that flags water standing still, because depth
/// varies where the free surface does not. The indicator below uses the free
/// surface h + z instead, so a lake at rest indicates nothing and a dam
/// indicates before the first step. DMADAPTORFLUX solves a mixed auxiliary
/// problem that assumes a PetscFE discretization of a diffusion-type operator,
/// which the SWE operator is not.

#include <petscdmplex.h>
#include <petscvec.h>
#include <private/rdyamrimpl.h>
#include <private/rdycoreimpl.h>
#include <private/rdymeshimpl.h>

/// @brief Spreads the indicator outward by `layers` cell layers, so that cells
///        the feature will reach before the next adapt are refined with it.
///        Without this the mesh lags: a front crossing several coarse cells per
///        coupling interval is resolved only after it has already passed.
///
///        The spread runs through the one-component DM's halo exchange, so it
///        crosses partition boundaries and gives the same result on any rank
///        count. `ind` is indexed by owned cell on the way in and out.
static PetscErrorCode DilateIndicator(RDy rdy, PetscScalar *ind, PetscInt layers) {
  PetscFunctionBegin;
  if (layers <= 0) PetscFunctionReturn(PETSC_SUCCESS);

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  Vec work_global, work_local;
  PetscCall(DMGetGlobalVector(rdy->dm_1dof, &work_global));
  PetscCall(DMGetLocalVector(rdy->dm_1dof, &work_local));

  for (PetscInt pass = 0; pass < layers; ++pass) {
    PetscScalar *w;
    PetscCall(VecGetArray(work_global, &w));
    for (PetscInt o = 0; o < mesh->num_owned_cells; ++o) w[o] = ind[o];
    PetscCall(VecRestoreArray(work_global, &w));

    PetscCall(DMGlobalToLocal(rdy->dm_1dof, work_global, INSERT_VALUES, work_local));

    const PetscScalar *wl;
    PetscCall(VecGetArrayRead(work_local, &wl));
    for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
      PetscInt edge_id = edges->internal_edge_ids[e];
      PetscInt l = edges->cell_ids[2 * edge_id], r = edges->cell_ids[2 * edge_id + 1];
      if (l < 0 || r < 0) continue;
      if (cells->is_owned[l]) {
        PetscInt o = cells->local_to_owned[l];
        if (PetscRealPart(wl[r]) > PetscRealPart(ind[o])) ind[o] = wl[r];
      }
      if (cells->is_owned[r]) {
        PetscInt o = cells->local_to_owned[r];
        if (PetscRealPart(wl[l]) > PetscRealPart(ind[o])) ind[o] = wl[l];
      }
    }
    PetscCall(VecRestoreArrayRead(work_local, &wl));
  }

  PetscCall(DMRestoreLocalVector(rdy->dm_1dof, &work_local));
  PetscCall(DMRestoreGlobalVector(rdy->dm_1dof, &work_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief For each local cell, ghosts included, the largest jump in
///        water-surface elevation (h + z) across its internal faces, spread
///        over a buffer and zeroed on cells already at the finest allowed
///        level.
///
///        `errVec` is laid out the way ex11 lays its out: one entry per local
///        cell in mesh point order, so that the indices VecTagger returns are
///        mesh points. The owned values are computed first and then exchanged
///        onto the ghosts, so every rank agrees about a cell it shares -- a
///        ghost's own face neighbourhood is not all local, so computing its
///        value in place would not.
static PetscErrorCode ComputeSurfaceJumpIndicator(RDy rdy, Vec errVec) {
  PetscFunctionBegin;

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscScalar *ind;
  PetscCall(PetscCalloc1(mesh->num_owned_cells, &ind));

  // from the global solution rather than rdy->u_local, which is only filled
  // while the operator is being applied and is still zero before the first step
  Vec locX;
  PetscCall(DMGetLocalVector(rdy->dm, &locX));
  PetscCall(DMGlobalToLocal(rdy->dm, rdy->u_global, INSERT_VALUES, locX));

  const PetscScalar *u;
  PetscCall(VecGetArrayRead(locX, &u));

  PetscInt ndof;
  PetscCall(VecGetBlockSize(rdy->u_global, &ndof));  // h, hu, hv, and any tracers
  for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l = edges->cell_ids[2 * edge_id], r = edges->cell_ids[2 * edge_id + 1];
    if (l < 0 || r < 0) continue;

    PetscReal eta_l = PetscRealPart(u[ndof * l]) + cells->centroids[l].X[2];
    PetscReal eta_r = PetscRealPart(u[ndof * r]) + cells->centroids[r].X[2];
    PetscReal jump  = PetscAbsReal(eta_l - eta_r);

    for (PetscInt i = 0; i < 2; ++i) {
      PetscInt c = (i == 0) ? l : r;
      if (!cells->is_owned[c]) continue;
      PetscInt owned = cells->local_to_owned[c];
      if (jump > PetscRealPart(ind[owned])) ind[owned] = jump;
    }
  }
  PetscCall(VecRestoreArrayRead(locX, &u));
  PetscCall(DMRestoreLocalVector(rdy->dm, &locX));

  // A surface jump below this is not a feature. The floor is what gives the
  // criterion a notion of "nothing is happening", which a relative VecTagger
  // does not have: its box is min + frac * (max - min), so a UNIFORM indicator
  // -- whatever its value, zero included -- gives the degenerate box [v, v] and
  // selects every cell, and roundoff noise around a flat surface selects about
  // half the mesh. A lake at rest is exactly that case, and it is the standard
  // well-balancedness benchmark. The default is the depth the scheme already
  // calls dry, seven orders below the jump across a dam and nine above roundoff.
  PetscReal min_jump = rdy->config.physics.flow.tiny_h;
  PetscCall(PetscOptionsGetReal(NULL, "rdy_amr_", "-min_jump", &min_jump, NULL));
  for (PetscInt o = 0; o < mesh->num_owned_cells; ++o) {
    if (PetscRealPart(ind[o]) < min_jump) ind[o] = 0.0;
  }

  // a front crosses several cells between adapts, so refine a band ahead of it
  PetscInt buffer_layers = 2;
  PetscCall(PetscOptionsGetInt(NULL, "rdy_amr_", "-buffer_layers", &buffer_layers, NULL));
  PetscCall(DilateIndicator(rdy, ind, buffer_layers));

  // cells already at the finest allowed level cannot be refined further, so
  // zero their indicator before the taggers see it: a relative or CDF box
  // would otherwise keep selecting them.
  //
  // The level is read off the area against the cell's OWN base cell, because a
  // bisection halves a cell exactly: a cell max_levels below its base cell has
  // 2^-max_levels of its area, whatever the base cell's size. Comparing against
  // one area for the whole mesh instead only caps the smallest base cell at
  // max_levels; on a graded mesh every larger base cell refines deeper, and
  // the largest without any limit at all on the refine-only path. The test is
  // >=, not >: a cell that has reached max_levels must not be marked again.
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (!cells->is_owned[c]) continue;
    PetscInt base;
    PetscCall(RDyAMRBaseCellOfLocalCell(rdy, c, &base));
    if (base < 0) continue;
    // the ratio is a power of two, so rounding the logarithm absorbs the
    // roundoff that accumulated computing the areas from the coordinates
    PetscInt level = (PetscInt)PetscFloorReal(PetscLog2Real(rdy->amr.base_cell_area[base] / cells->areas[c]) + 0.5);
    if (level >= rdy->amr.max_levels) ind[cells->local_to_owned[c]] = 0.0;
  }

  // spread the owned values onto the ghosts and write them out by local cell
  Vec work_global, work_local;
  PetscCall(DMGetGlobalVector(rdy->dm_1dof, &work_global));
  PetscCall(DMGetLocalVector(rdy->dm_1dof, &work_local));
  PetscScalar *w;
  PetscCall(VecGetArray(work_global, &w));
  for (PetscInt o = 0; o < mesh->num_owned_cells; ++o) w[o] = ind[o];
  PetscCall(VecRestoreArray(work_global, &w));
  PetscCall(DMGlobalToLocal(rdy->dm_1dof, work_global, INSERT_VALUES, work_local));
  PetscCall(VecCopy(work_local, errVec));
  PetscCall(DMRestoreLocalVector(rdy->dm_1dof, &work_local));
  PetscCall(DMRestoreGlobalVector(rdy->dm_1dof, &work_global));

  PetscCall(PetscFree(ind));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates one of the two VecTaggers, with the type and box RDycore
///        wants as defaults and the options database free to override them.
///
///        The refine default takes every cell whose indicator is at least half
///        the largest. Scale-free on purpose: an absolute threshold disables
///        itself, since halving a cell halves the jump across its faces.
static PetscErrorCode CreateTagger(RDy rdy, const char prefix[], PetscReal lo, PetscReal hi, VecTagger *tagger) {
  PetscFunctionBegin;
  VecTaggerBox box = {lo, hi};

  PetscCall(VecTaggerCreate(rdy->comm, tagger));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*tagger, prefix));
  PetscCall(VecTaggerSetType(*tagger, VECTAGGERRELATIVE));
  PetscCall(VecTaggerRelativeSetBox(*tagger, &box));
  PetscCall(VecTaggerSetFromOptions(*tagger));
  PetscCall(VecTaggerSetUp(*tagger));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Marks the cells to refine and to coarsen at this adapt, from the jump
///        in water-surface elevation.
///
///        The thresholds come from the options database, for example
///
///          -rdy_amr_refine_vec_tagger_type cdf -rdy_amr_refine_vec_tagger_box 0.95,1
///          -rdy_amr_coarsen_vec_tagger_box 0,0.05
///
///        Note for a cdf tagger: the error vector counts a shared cell once per
///        rank that sees it, so a quantile is taken over slightly more entries
///        than there are cells. A relative or absolute box is unaffected.
///
/// @param [in] rdy     the dycore
/// @param [out] label  cells marked DM_ADAPT_REFINE and DM_ADAPT_COARSEN,
///                     destroyed by the caller
PetscErrorCode RDyAMRMarkBySurfaceJump(RDy rdy, DMLabel *label) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &cStart, &cEnd));

  // ex11's layout: parallel, so the taggers see the whole domain's range, but
  // with one local entry per local cell so that an index is a mesh point
  Vec errVec;
  PetscCall(VecCreateFromOptions(rdy->comm, NULL, 1, cEnd - cStart, PETSC_DETERMINE, &errVec));
  PetscCall(PetscObjectSetName((PetscObject)errVec, "Error Estimator"));
  PetscCall(VecSetUp(errVec));
  PetscCall(ComputeSurfaceJumpIndicator(rdy, errVec));
  PetscCall(VecViewFromOptions(errVec, NULL, "-rdy_amr_error_vec_view"));

  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", label));

  // Nothing is above the floor anywhere, so there is no feature to resolve and
  // nothing to refine. Asking the taggers would be worse than useless: a
  // relative tagger on a uniform vector selects every cell, so a lake at rest
  // would refine without bound until the cells are small enough to violate the
  // CFL condition at a fixed time step. The coarsen tagger is still asked, and
  // on an all-zero indicator it marks everything, which is the right answer --
  // a mesh with nothing happening on it should relax back towards the base.
  PetscReal err_max;
  PetscCall(VecMax(errVec, NULL, &err_max));

  PetscInt num_refine = 0;
  if (err_max > 0.0) {
    VecTagger refine_tagger;
    IS        refineIS;
    PetscCall(CreateTagger(rdy, "rdy_amr_refine_", 0.5, 1.0, &refine_tagger));
    PetscCall(VecTaggerComputeIS(refine_tagger, errVec, &refineIS, NULL));
    PetscCall(ISGetSize(refineIS, &num_refine));
    if (num_refine) PetscCall(DMLabelSetStratumIS(*label, DM_ADAPT_REFINE, refineIS));
    PetscCall(ISDestroy(&refineIS));
    PetscCall(VecTaggerDestroy(&refine_tagger));
  }

  // only a rebuild from the base mesh can undo a level, so without it a coarsen
  // tagger would cost the work of computing marks nothing can act on
  PetscInt num_coarsen = 0;
  if (rdy->amr.allow_coarsening) {
    VecTagger coarsen_tagger;
    IS        coarsenIS;
    PetscCall(CreateTagger(rdy, "rdy_amr_coarsen_", 0.0, 0.1, &coarsen_tagger));
    PetscCall(VecTaggerComputeIS(coarsen_tagger, errVec, &coarsenIS, NULL));
    PetscCall(ISGetSize(coarsenIS, &num_coarsen));
    if (num_coarsen) PetscCall(DMLabelSetStratumIS(*label, DM_ADAPT_COARSEN, coarsenIS));
    PetscCall(ISDestroy(&coarsenIS));
    PetscCall(VecTaggerDestroy(&coarsen_tagger));
  }
  PetscCall(PetscInfo(rdy->dm, "AMR: marked %" PetscInt_FMT " cells to refine, %" PetscInt_FMT " to coarsen\n", num_refine, num_coarsen));
  PetscCall(VecDestroy(&errVec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Refines the mesh in hand wherever the label marks DM_ADAPT_REFINE,
///        and ignores the coarsen stratum, which a refinement cannot act on.
PetscErrorCode RDyAMRRefineFromLabel(RDy rdy, DMLabel label, DM *dm_new, Mat *c2f_ndof, Mat *f2c_ndof, Mat *c2f_1dof, Mat *f2c_1dof) {
  PetscFunctionBegin;

  DMLabel refine;
  IS      refineIS;
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &refine));
  PetscCall(DMLabelGetStratumIS(label, DM_ADAPT_REFINE, &refineIS));
  if (refineIS) PetscCall(DMLabelSetStratumIS(refine, DM_ADAPT_REFINE, refineIS));
  PetscCall(ISDestroy(&refineIS));

  PetscInt bs;
  PetscCall(VecGetBlockSize(rdy->u_global, &bs));
  PetscCall(RDyAMRRefineByLabel(rdy->dm, refine, bs, dm_new, c2f_ndof, f2c_ndof, c2f_1dof, f2c_1dof));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Moves each base cell's target refinement level by what this adapt's
///        label says about its descendants: a cell marked DM_ADAPT_REFINE
///        takes its base cell one level deeper, one marked DM_ADAPT_COARSEN
///        brings it one level back, and a cell in neither stratum leaves it
///        where it is.
///
///        Separate from the rebuild because load balancing goes between them:
///        the partition is chosen from the levels the rebuild is about to use.
PetscErrorCode RDyAMRUpdateBaseLevelsFromLabel(RDy rdy, DMLabel label) {
  PetscFunctionBegin;

  const PetscInt max_levels = rdy->amr.max_levels;

  // move each base cell's level by what the label says about its descendants
  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(rdy->dm, 0, &cStart, &cEnd));
  PetscInt *delta;
  PetscCall(PetscCalloc1(rdy->amr.num_base_cells, &delta));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt value = 0;
    PetscCall(DMLabelGetValue(label, c, &value));
    if (value != DM_ADAPT_REFINE && value != DM_ADAPT_COARSEN) continue;

    PetscInt base;
    PetscCall(RDyAMRBaseCellOfLocalCell(rdy, c - cStart, &base));
    if (base < 0) continue;
    // refinement wins where a base cell has descendants marked both ways
    if (value == DM_ADAPT_REFINE) delta[base] = 1;
    else if (delta[base] == 0) delta[base] = -1;
  }
  for (PetscInt b = 0; b < rdy->amr.num_base_cells; ++b) {
    rdy->amr.base_level[b] = PetscMax(0, PetscMin(rdy->amr.base_level[b] + delta[b], max_levels));
  }
  PetscCall(PetscFree(delta));
  PetscFunctionReturn(PETSC_SUCCESS);
}
