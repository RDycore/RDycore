/// RDycore's refinement criterion, registered as a PETSc DMAdaptor type.
///
/// Driving the adapt through DMAdaptor rather than beside it means PETSc owns
/// the machinery around the criterion: the refine and coarsen VecTaggers and
/// their options, the adaptation criterion, the error monitors
/// (-rdy_amr_adaptor_monitor_error and friends), and the call into
/// DMAdaptLabel. What RDycore supplies is the part that is specific to the
/// shallow water equations -- the error indicator, and the transfer of the
/// solution between the old mesh and the new one.
///
/// The built-in indicators do not fit this code. DMADAPTORGRADIENT recovers the
/// gradient of the solution field, which here is water depth on a P0 space: on
/// a varying bed that flags water standing still, because depth varies where
/// the free surface does not. The indicator below uses the free surface h + z
/// instead, so a lake at rest indicates nothing and a dam indicates before the
/// first step. DMADAPTORFLUX solves a mixed auxiliary problem that assumes a
/// PetscFE discretization of a diffusion-type operator, which the SWE operator
/// is not.

#include <petscfv.h>  // PetscFVCellGeom and PetscLimiter, used by the DMAdaptor implementation header

#include <petsc/private/dmadaptorimpl.h>
#include <petscdmadaptor.h>
#include <petscdmplex.h>
#include <private/rdyamrimpl.h>
#include <private/rdycoreimpl.h>
#include <private/rdymeshimpl.h>

/// The dycore is carried in a PetscContainer composed on the objects that need
/// it -- the adaptor, which the indicator and the transfer are called with, and
/// the mesh, which is all the registered mesh adaptor is handed. The container
/// is what holds the reference; the pointer it yields is borrowed, so there is
/// nothing to restore.
///
/// An RDy cannot be composed directly: RDyCreate() allocates it with PetscNew,
/// so the PETSCHEADER it carries was never initialised and its reference count
/// starts at zero, which composing and then destroying would drive below zero.
static PetscErrorCode ObjectGetRDy(PetscObject obj, RDy *rdy) {
  PetscFunctionBegin;
  PetscContainer container;
  PetscCall(PetscObjectQuery(obj, "RDy", (PetscObject *)&container));
  PetscCheck(container, PetscObjectComm(obj), PETSC_ERR_ARG_WRONGSTATE, "no RDy attached to this %s", obj->class_name);
  PetscCall(PetscContainerGetPointer(container, (void **)rdy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// attaches the dycore to an object; the composed container holds the reference
static PetscErrorCode ObjectSetRDy(PetscObject obj, RDy rdy) {
  PetscFunctionBegin;
  PetscContainer container;
  PetscCall(PetscContainerCreate(PetscObjectComm(obj), &container));
  PetscCall(PetscContainerSetPointer(container, rdy));
  PetscCall(PetscObjectCompose(obj, "RDy", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));  // composing took the reference this drops
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AdaptorGetRDy(DMAdaptor adaptor, RDy *rdy) {
  PetscFunctionBegin;
  PetscCall(ObjectGetRDy((PetscObject)adaptor, rdy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGetRDy(DM dm, RDy *rdy) {
  PetscFunctionBegin;
  PetscCall(ObjectGetRDy((PetscObject)dm, rdy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Spreads the indicator outward by `layers` cell layers, so that cells
///        the feature will reach before the next adapt are refined with it.
///        Without this the mesh lags: a front crossing several coarse cells per
///        coupling interval is resolved only after it has already passed.
///
///        The spread runs through the solution DM's halo exchange (component 0
///        of a 3-dof work vector carries the indicator), so it crosses
///        partition boundaries and gives the same result on any rank count.
static PetscErrorCode DilateIndicator(RDy rdy, PetscScalar *ind, PetscInt layers) {
  PetscFunctionBegin;
  if (layers <= 0) PetscFunctionReturn(PETSC_SUCCESS);

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  Vec work_global, work_local;
  PetscCall(DMGetGlobalVector(rdy->dm, &work_global));
  PetscCall(DMGetLocalVector(rdy->dm, &work_local));

  for (PetscInt pass = 0; pass < layers; ++pass) {
    PetscScalar *w;
    PetscCall(VecZeroEntries(work_global));
    PetscCall(VecGetArray(work_global, &w));
    for (PetscInt o = 0; o < mesh->num_owned_cells; ++o) w[3 * o] = ind[o];
    PetscCall(VecRestoreArray(work_global, &w));

    PetscCall(DMGlobalToLocal(rdy->dm, work_global, INSERT_VALUES, work_local));

    const PetscScalar *wl;
    PetscCall(VecGetArrayRead(work_local, &wl));
    for (PetscInt e = 0; e < mesh->num_internal_edges; ++e) {
      PetscInt edge_id = edges->internal_edge_ids[e];
      PetscInt l = edges->cell_ids[2 * edge_id], r = edges->cell_ids[2 * edge_id + 1];
      if (l < 0 || r < 0) continue;
      if (cells->is_owned[l]) {
        PetscInt o = cells->local_to_owned[l];
        if (PetscRealPart(wl[3 * r]) > PetscRealPart(ind[o])) ind[o] = wl[3 * r];
      }
      if (cells->is_owned[r]) {
        PetscInt o = cells->local_to_owned[r];
        if (PetscRealPart(wl[3 * l]) > PetscRealPart(ind[o])) ind[o] = wl[3 * l];
      }
    }
    PetscCall(VecRestoreArrayRead(work_local, &wl));
  }

  PetscCall(DMRestoreLocalVector(rdy->dm, &work_local));
  PetscCall(DMRestoreGlobalVector(rdy->dm, &work_global));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief The error indicator of the "surfacejump" DMAdaptor type: for each
///        owned cell, the largest jump in water-surface elevation (h + z)
///        across its internal faces, spread over a buffer and zeroed on cells
///        already at the finest allowed level.
///
///        DMAdaptor hands in the local solution and an error Vec living on a
///        P0 field over the same cells, which the two VecTaggers then threshold.
static PetscErrorCode ComputeErrorIndicator_SurfaceJump(DMAdaptor adaptor, Vec locX, Vec errVec) {
  PetscFunctionBegin;

  RDy rdy;
  PetscCall(AdaptorGetRDy(adaptor, &rdy));

  RDyMesh  *mesh  = &rdy->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscScalar *ind;
  PetscCall(VecZeroEntries(errVec));
  PetscCall(VecGetArray(errVec, &ind));

  const PetscScalar *u;
  PetscCall(VecGetArrayRead(locX, &u));

  const PetscInt ndof = 3;  // h, hu, hv
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

  // a front crosses several cells between adapts, so refine a band ahead of it
  PetscInt buffer_layers = 2;
  PetscCall(PetscOptionsGetInt(NULL, "rdy_amr_", "-buffer_layers", &buffer_layers, NULL));
  PetscCall(DilateIndicator(rdy, ind, buffer_layers));

  // cells already at the finest allowed level cannot be refined further, so
  // zero their indicator before the taggers see it: a relative or CDF box
  // would otherwise keep selecting them
  PetscInt max_levels = 3;
  PetscCall(PetscOptionsGetInt(NULL, "rdy_amr_", "-max_levels", &max_levels, NULL));
  PetscReal min_area = rdy->amr.base_min_cell_area / PetscPowReal(2.0, (PetscReal)max_levels);
  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (!cells->is_owned[c]) continue;
    if (cells->areas[c] < min_area) ind[cells->local_to_owned[c]] = 0.0;
  }

  PetscCall(VecRestoreArray(errVec, &ind));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief The transfer of the "surfacejump" type: carries the solution from the
///        mesh being replaced to the one replacing it.
///
///        PETSc has no default for this case. With DM_ADAPTATION_LABEL and no
///        transfer function set, DMAdaptorTransferSolution falls through to
///        DMForestTransferVec, which a DMPlex is not.
///
///        The new mesh is a refinement of the old one, so each of its cells
///        takes its parent's value: that is the injection in the interpolation
///        matrices the refinement's transform defines. The matrices are kept on
///        the RDy for RDyPerformAMR, which needs them to carry regions across
///        the adapt as well.
static PetscErrorCode TransferSolution_SurfaceJump(DMAdaptor adaptor, DM dm_old, Vec x_old, DM dm_new, Vec x_new, void *ctx) {
  PetscFunctionBegin;

  RDy rdy;
  PetscCall(AdaptorGetRDy(adaptor, &rdy));

  // a rebuilt mesh is not a refinement of the old one, so injection would be
  // wrong: RDyPerformAMR carries the solution across conservatively once the
  // new mesh has been built, which is also when its geometry exists
  if (rdy->amr.mesh_was_rebuilt) PetscFunctionReturn(PETSC_SUCCESS);

  PetscInt bs;
  PetscCall(VecGetBlockSize(x_old, &bs));
  PetscCall(RDyAMRBuildInterpolationMatrices(dm_old, dm_new, bs, &rdy->amr.adapt_coarse_to_fine_ndof, &rdy->amr.adapt_fine_to_coarse_ndof,
                                             &rdy->amr.adapt_coarse_to_fine_1dof, &rdy->amr.adapt_fine_to_coarse_1dof));

  // the matrices are indexed by local cells, ghosts included
  Vec local_old, local_new;
  PetscCall(DMGetLocalVector(dm_old, &local_old));
  PetscCall(DMGetLocalVector(dm_new, &local_new));
  PetscCall(DMGlobalToLocal(dm_old, x_old, INSERT_VALUES, local_old));
  PetscCall(MatMult(rdy->amr.adapt_coarse_to_fine_ndof, local_old, local_new));
  PetscCall(DMLocalToGlobal(dm_new, local_new, INSERT_VALUES, x_new));
  PetscCall(DMRestoreLocalVector(dm_old, &local_old));
  PetscCall(DMRestoreLocalVector(dm_new, &local_new));
  PetscFunctionReturn(PETSC_SUCCESS);
}


/// @brief The "rdycore_rebuild" mesh adaptor: builds the adapted mesh from the
///        base mesh rather than by refining the mesh in hand, which is what
///        lets a region coarsen again.
///
///        Registered with DMGenerateRegister, so DMAdaptLabel dispatches here
///        when -adapt_dm_adaptor is rdycore_rebuild, and both strata of the
///        label mean what they say: a cell marked DM_ADAPT_REFINE takes its
///        base cell one level deeper, one marked DM_ADAPT_COARSEN brings it one
///        level back, and a cell in neither stratum leaves it where it is. The
///        built-in cellrefiner cannot do this -- it refines every cell the
///        label carries, whatever value it carries.
///
///        The interpolation matrices the rebuild produces are left on the RDy
///        for RDyPerformAMR, which needs them to carry the regions across.
static PetscErrorCode DMAdaptLabel_RDyRebuild(DM dm, PETSC_UNUSED Vec metric, DMLabel label, PETSC_UNUSED DMLabel rgLabel, DM *rdm) {
  PetscFunctionBegin;

  RDy rdy;
  PetscCall(DMGetRDy(dm, &rdy));

  PetscInt max_levels = 3;
  PetscCall(PetscOptionsGetInt(NULL, "rdy_amr_", "-max_levels", &max_levels, NULL));

  // move each base cell's level by what the label says about its descendants
  PetscInt cStart, cEnd;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscInt *delta;
  PetscCall(PetscCalloc1(rdy->amr.num_base_cells, &delta));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt value = 0;
    PetscCall(DMLabelGetValue(label, c, &value));
    if (value != DM_ADAPT_REFINE && value != DM_ADAPT_COARSEN) continue;

    PetscInt base = c - cStart;
    if (rdy->amr.BaseToCurrentMat1Dof) {
      PetscInt           ncols;
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscCall(MatGetRow(rdy->amr.BaseToCurrentMat1Dof, c - cStart, &ncols, &cols, &vals));
      base = ncols ? cols[0] : -1;
      PetscCall(MatRestoreRow(rdy->amr.BaseToCurrentMat1Dof, c - cStart, &ncols, &cols, &vals));
    }
    if (base < 0) continue;
    // refinement wins where a base cell has descendants marked both ways
    if (value == DM_ADAPT_REFINE) delta[base] = 1;
    else if (delta[base] == 0) delta[base] = -1;
  }
  for (PetscInt b = 0; b < rdy->amr.num_base_cells; ++b) {
    rdy->amr.base_level[b] = PetscMax(0, PetscMin(rdy->amr.base_level[b] + delta[b], max_levels));
  }
  PetscCall(PetscFree(delta));

  PetscInt bs;
  PetscCall(VecGetBlockSize(rdy->u_global, &bs));
  PetscCall(RDyAMRRebuildMeshFromBase(rdy, bs, max_levels, rdm, &rdy->amr.adapt_coarse_to_fine_ndof, &rdy->amr.adapt_fine_to_coarse_ndof,
                                      &rdy->amr.adapt_coarse_to_fine_1dof, &rdy->amr.adapt_fine_to_coarse_1dof));
  rdy->amr.mesh_was_rebuilt = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMAdaptorCreate_SurfaceJump(DMAdaptor adaptor) {
  PetscFunctionBegin;
  // the transfer has a public setter; the indicator does not, so it is set
  // through the implementation header
  adaptor->ops->computeerrorindicator = ComputeErrorIndicator_SurfaceJump;
  PetscCall(DMAdaptorSetTransferFunction(adaptor, TransferSolution_SurfaceJump));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Sets an option unless the user has already given it, so that everything
/// RDycore's AMR needs can still be overridden from the command line.
static PetscErrorCode SetOptionDefault(const char name[], const char value[]) {
  PetscFunctionBegin;
  PetscBool set;
  PetscCall(PetscOptionsHasName(NULL, NULL, name, &set));
  if (!set) PetscCall(PetscOptionsSetValue(NULL, name, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates the DMAdaptor that drives RDycore's AMR: the "surfacejump"
///        criterion, a relative refine box and a coarsen box that only an
///        adaptor able to coarsen will act on.
///
///        Thresholds, monitors and the criterion all come from PETSc under the
///        "rdy_amr_" prefix, so for example
///
///          -rdy_amr_refine_vec_tagger_type cdf -rdy_amr_refine_vec_tagger_box 0.95,1
///          -rdy_amr_coarsen_vec_tagger_box 0,0.05
///          -rdy_amr_adaptor_monitor_error
///
/// @param [in] rdy      the dycore, attached to the adaptor for the indicator
/// @param [out] adaptor the adaptor, destroyed by RDyAMRDestroyAdaptor()
PetscErrorCode RDyAMRCreateAdaptor(RDy rdy, DMAdaptor *adaptor) {
  PetscFunctionBegin;

  static PetscBool registered = PETSC_FALSE;
  if (!registered) {
    PetscCall(DMAdaptorRegister(RDYADAPTORSURFACEJUMP, DMAdaptorCreate_SurfaceJump));
    PetscCall(DMGenerateRegister(RDYMESHADAPTORREBUILD, NULL, NULL, DMAdaptLabel_RDyRebuild, -1));
    registered = PETSC_TRUE;
  }

  PetscCall(DMAdaptorCreate(rdy->comm, adaptor));
  // DMAdaptorSetOptionsPrefix, not PetscObjectSetOptionsPrefix: it also
  // prefixes the two taggers, so their options are -rdy_amr_refine_vec_tagger_*
  // and -rdy_amr_coarsen_vec_tagger_* rather than the bare -refine_vec_tagger_*
  PetscCall(DMAdaptorSetOptionsPrefix(*adaptor, "rdy_amr_"));
  PetscCall(DMAdaptorSetType(*adaptor, RDYADAPTORSURFACEJUMP));
  // the indicator and the transfer are handed the adaptor; the registered mesh
  // adaptor is handed only the mesh
  PetscCall(ObjectSetRDy((PetscObject)*adaptor, rdy));
  PetscCall(ObjectSetRDy((PetscObject)rdy->dm, rdy));

  // DMAdaptor takes its mesh from the solver's DM
  SNES snes;
  PetscCall(TSGetSNES(rdy->ts, &snes));
  PetscCall(SNESSetDM(snes, rdy->dm));
  PetscCall(DMAdaptorSetSolver(*adaptor, snes));

  PetscCall(DMAdaptorSetCriterion(*adaptor, DM_ADAPTATION_LABEL));
  PetscCall(DMAdaptorSetSequenceLength(*adaptor, 1));

  // DMAdaptor appends "adapt_" to the mesh's options prefix for the duration of
  // the DMAdaptLabel call, so the refinement's own options have to be given
  // under that prefix. AMR in RDycore needs simplex bisection, and needs the
  // transform to match strata or the boundary edges of the refined mesh come
  // back as the wrong mesh points.
  PetscCall(SetOptionDefault("-adapt_dm_adaptor", rdy->amr.allow_coarsening ? RDYMESHADAPTORREBUILD : "cellrefiner"));
  PetscCall(SetOptionDefault("-adapt_dm_plex_transform_type", "refine_sbr"));
  PetscCall(SetOptionDefault("-adapt_dm_plex_transform_label_match_strata", NULL));

  // Refine the cells whose indicator is at least half the largest. Scale-free
  // on purpose: an absolute threshold disables itself, since halving a cell
  // halves the jump across its faces.
  //
  // The taggers are configured through options rather than through their
  // objects -- the same way a DMPlex is -- so that everything here can be
  // overridden on the command line and nothing is hidden in the code.
  PetscCall(SetOptionDefault("-rdy_amr_refine_vec_tagger_type", "relative"));
  PetscCall(SetOptionDefault("-rdy_amr_refine_vec_tagger_box", "0.5,1"));

  // The coarsening tagger keeps PETSc's default, which tags nothing, unless
  // the mesh adaptor can act on it: cellrefiner refines every cell the label
  // carries, whatever value it carries, so a cell tagged for coarsening there
  // would be refined instead.
  if (rdy->amr.allow_coarsening) {
    PetscCall(SetOptionDefault("-rdy_amr_coarsen_vec_tagger_type", "relative"));
    PetscCall(SetOptionDefault("-rdy_amr_coarsen_vec_tagger_box", "0,0.1"));
  }

  PetscCall(DMAdaptorSetFromOptions(*adaptor));
  PetscCall(DMAdaptorSetUp(*adaptor));
  PetscFunctionReturn(PETSC_SUCCESS);
}
