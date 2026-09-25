#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysweimpl.h>
#include <private/rdytracerimpl.h>

#include "operator_identity_ceed.h"
#include "swe/swe_fluxes_hydro_recon_ceed.h"
#include "tracer/tracer_fluxes_hydro_recon_ceed.h"

// The CEED flux operator for hydrostatic reconstruction (HR) consists of:
//
// * for the entire domain: an interior flux sub-operator that computes fluxes
//   on pairs of cells on the interior of the domain after performing hydrostatic
//   reconstruction at each interface.
// * for each domain boundary: a boundary flux sub-operator that computes fluxes
//   into/out of cells adjacent to boundary edges using HR.
//
// Unlike the standard well-balancing paths, the HR operator does NOT require
// eta_vertices. Instead, the geom field carries cell-centered bed elevations
// (zc) for the left and right cells.

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

static inline CeedInt NumTracers(const RDyConfig config) {
  return config.physics.sediment.num_classes + ((config.physics.salinity) ? 1 : 0) + ((config.physics.heat) ? 1 : 0);
}

// frees a data context allocated using PETSc, returning a libCEED error code
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

static PetscErrorCode CreateHRInteriorFluxQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;

  CeedInt num_tracers = NumTracers(config);

  CeedQFunctionContext qf_context;
  if (num_tracers == 0) {  // flow only
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEFlux_HydroRecon_Roe, SWEFlux_HydroRecon_Roe_loc, qf));
    PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
  } else {  // flow + tracers
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, TracerFlux_HydroRecon_Roe, TracerFlux_HydroRecon_Roe_loc, qf));
    PetscCall(CreateTracerQFunctionContext(ceed, config, &qf_context));
  }
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for HR fluxes on interior edges.
///
/// Active input fields:
///    * `q_left[num_interior_edges][3]`
///
/// Passive input fields:
///    * `geom[num_interior_edges][HR_NUM_COMP_INTERIOR_GEOM]`:
///        see HRInteriorGeomIndex for component layout
///    * `q_right[num_interior_edges][3]` - bound to the passive vector returned in
///      `q_right_restricted`, which the caller must refresh (by applying
///      `q_right_refresh_op`, from the same active input vector) before every
///      application of `subop`. It cannot be made active here alongside `q_left`
///      because CEED's CUDA backends only support a single active input field per operator.
///
/// Active output fields:
///    * `cell_left[num_interior_edges][3]`
///    * `cell_right[num_interior_edges][3]`
///
/// Passive output fields:
///    * `flux[num_owned_cells][3]`
///    * `courant_number[num_interior_edges][2]`
///
/// @param [out] q_right_refresh_op  the CeedOperator used to refresh q_right_restricted from the active
///                                  input vector ahead of every application of `subop`
/// @param [out] q_right_restricted_out the passive CeedVector backing `subop`'s `q_right` field
static PetscErrorCode CreateCeedInteriorFluxHydroReconSuboperator(const RDyConfig config, RDyMesh *mesh, CeedOperator *q_right_refresh_op,
                                                                  CeedVector *q_right_restricted_out, CeedOperator *subop) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_flow_comp = 3;  // NOTE: SWE assumed!
  CeedInt num_tracers   = NumTracers(config);
  CeedInt num_comp      = num_flow_comp + num_tracers;

  RDyCells    *cells    = &mesh->cells;
  RDyEdges    *edges    = &mesh->edges;
  RDyVertices *vertices = &mesh->vertices;

  CeedQFunction qf;
  PetscCall(CreateHRInteriorFluxQFunction(ceed, config, &qf));

  // add inputs and outputs
  CeedInt num_comp_geom = HR_NUM_COMP_INTERIOR_GEOM;
  CeedInt num_comp_cnum = 2;
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction q_restrict_l, q_restrict_r, c_restrict_l, c_restrict_r, restrict_geom, restrict_flux, restrict_cnum, restrict_q_right_local;
  CeedVector          geom, flux, cnum, q_right_restricted;
  {
    CeedInt num_edges = mesh->num_owned_internal_edges;

    // create a vector of geometric factors
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    CeedScalar(*g)[HR_NUM_COMP_INTERIOR_GEOM];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l                                = edges->cell_ids[2 * iedge];
      CeedInt r                                = edges->cell_ids[2 * iedge + 1];
      g[owned_edge][HR_INTERIOR_SN]            = edges->sn[iedge];
      g[owned_edge][HR_INTERIOR_CN]            = edges->cn[iedge];
      g[owned_edge][HR_INTERIOR_NEG_L_OVER_AL] = -edges->lengths[iedge] / cells->areas[l];
      g[owned_edge][HR_INTERIOR_L_OVER_AR]     = edges->lengths[iedge] / cells->areas[r];

      if (config.grid.cell_elevation.file[0]) {
        // use cell-centered z from file if provided
        g[owned_edge][HR_INTERIOR_ZC_LEFT]  = cells->centroids[l].X[2];
        g[owned_edge][HR_INTERIOR_ZC_RIGHT] = cells->centroids[r].X[2];

      } else {
        // Use vertex-averaged z for cell bed elevation.  DMPlex's centroid z
        // (from DMPlexComputeCellGeometryFVM) differs from the vertex average
        // for non-planar quads, which breaks lake-at-rest consistency with ICs
        // computed as h = eta - mean(vertex_z).
        CeedScalar zc_l = 0.0;
        for (CeedInt v = cells->vertex_offsets[l]; v < cells->vertex_offsets[l + 1]; v++) {
          zc_l += vertices->points[cells->vertex_ids[v]].X[2];
        }
        zc_l /= (CeedScalar)cells->num_vertices[l];

        CeedScalar zc_r = 0.0;
        for (CeedInt v = cells->vertex_offsets[r]; v < cells->vertex_offsets[r + 1]; v++) {
          zc_r += vertices->points[cells->vertex_ids[v]].X[2];
        }
        zc_r /= (CeedScalar)cells->num_vertices[r];

        g[owned_edge][HR_INTERIOR_ZC_LEFT]  = zc_l;
        g[owned_edge][HR_INTERIOR_ZC_RIGHT] = zc_r;
      }

      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector to store inter-cell fluxes
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));

    // create a vector to store the courant number for each edge
    CeedInt cnum_strides[] = {num_comp_cnum, 1, num_comp_cnum};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_cnum, num_edges * num_comp_cnum, cnum_strides, &restrict_cnum));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_cnum, &cnum, NULL));
    PetscCallCEED(CeedVectorSetValue(cnum, 0.0));

    // create element restrictions for (active) left and right input/output states
    CeedInt *q_offset_l, *q_offset_r, *c_offset_l, *c_offset_r;
    PetscCall(PetscMalloc2(num_edges, &q_offset_l, num_edges, &q_offset_r));
    PetscCall(PetscMalloc2(num_edges, &c_offset_l, num_edges, &c_offset_r));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l              = edges->cell_ids[2 * iedge];
      CeedInt r              = edges->cell_ids[2 * iedge + 1];
      q_offset_l[owned_edge] = l * num_comp;
      q_offset_r[owned_edge] = r * num_comp;
      c_offset_l[owned_edge] = cells->local_to_owned[l] * num_comp;
      c_offset_r[owned_edge] = cells->local_to_owned[r] * num_comp;
      owned_edge++;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, q_offset_l,
                                            &q_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, q_offset_r,
                                            &q_restrict_r));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, c_offset_l,
                                            &c_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, c_offset_r,
                                            &c_restrict_r));
    PetscCall(PetscFree2(q_offset_l, q_offset_r));
    PetscCall(PetscFree2(c_offset_l, c_offset_r));

    // q_right is bound to a passive q_right_restricted vector rather than
    // active alongside q_left, since CEED's CUDA backends support only a
    // single active input field per operator. q_right_refresh_op (built
    // below) refreshes it via a real CeedOperator apply rather than a
    // standalone CeedElemRestrictionApply() call; see
    // libceed-cuda-restriction-apply-broadcast-bug.md.
    CeedInt qr_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp, num_edges * num_comp, qr_strides, &restrict_q_right_local));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_q_right_local, &q_right_restricted, NULL));

    {
      IdentityCopyContext id_ctx;
      PetscCall(PetscCalloc1(1, &id_ctx));
      id_ctx->num_comp = num_comp;

      CeedQFunctionContext qf_id_context;
      PetscCallCEED(CeedQFunctionContextCreate(ceed, &qf_id_context));
      PetscCallCEED(CeedQFunctionContextSetData(qf_id_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*id_ctx), id_ctx));
      PetscCallCEED(CeedQFunctionContextSetDataDestroy(qf_id_context, CEED_MEM_HOST, FreeContextPetsc));

      CeedQFunction qf_identity;
      PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, IdentityCopy, IdentityCopy_loc, &qf_identity));
      PetscCallCEED(CeedQFunctionAddInput(qf_identity, "u", num_comp, CEED_EVAL_NONE));
      PetscCallCEED(CeedQFunctionAddOutput(qf_identity, "v", num_comp, CEED_EVAL_NONE));
      PetscCallCEED(CeedQFunctionSetContext(qf_identity, qf_id_context));
      PetscCallCEED(CeedQFunctionContextDestroy(&qf_id_context));

      PetscCallCEED(CeedOperatorCreate(ceed, qf_identity, NULL, NULL, q_right_refresh_op));
      PetscCallCEED(CeedOperatorSetField(*q_right_refresh_op, "u", q_restrict_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
      PetscCallCEED(CeedOperatorSetField(*q_right_refresh_op, "v", restrict_q_right_local, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
      PetscCallCEED(CeedQFunctionDestroy(&qf_identity));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, subop));
  PetscCallCEED(CeedOperatorSetField(*subop, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*subop, "q_left", q_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "q_right", restrict_q_right_local, CEED_BASIS_NONE, q_right_restricted));
  PetscCallCEED(CeedOperatorSetField(*subop, "cell_left", c_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "cell_right", c_restrict_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*subop, "flux", restrict_flux, CEED_BASIS_NONE, flux));
  PetscCallCEED(CeedOperatorSetField(*subop, "courant_number", restrict_cnum, CEED_BASIS_NONE, cnum));

  // hand ownership of q_right_restricted to the caller (in addition to
  // *q_right_refresh_op, already set above): it must apply
  // *q_right_refresh_op to refresh q_right_restricted ahead of every
  // application of *subop, and destroy both when the operator is torn down
  *q_right_restricted_out = q_right_restricted;

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q_right_local));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// Creates a CEED flux operator for hydrostatic reconstruction.
/// @param [in]    config              configuration defining physics and numerics
/// @param [in]    mesh                mesh with geometric/topological data
/// @param [in]    num_boundaries      the number of distinct boundaries
/// @param [in]    boundaries          array of boundaries
/// @param [in]    boundary_conditions array of boundary conditions
/// @param [out]   q_right_refresh_op  the CeedOperator used to refresh q_right_restricted
/// @param [out]   q_right_restricted  the passive CeedVector backing the interior suboperator's q_right field
/// @param [out]   flux_op             the newly created composite operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedFluxHROperator(RDyConfig *config, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries,
                                        RDyCondition *boundary_conditions, CeedOperator *q_right_refresh_op, CeedVector *q_right_restricted,
                                        CeedOperator *flux_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedOperatorCreateComposite(ceed, flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }

  // flux suboperator 0: fluxes between interior cells
  CeedOperator interior_flux_op;
  PetscCall(CreateCeedInteriorFluxHydroReconSuboperator(*config, mesh, q_right_refresh_op, q_right_restricted, &interior_flux_op));
  PetscCall(CeedOperatorCompositeAddSub(*flux_op, interior_flux_op));
  PetscCall(CeedOperatorDestroy(&interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges
  // Since zc_L == zc_R for boundary cells, the HR reconstruction is a no-op
  // and the standard boundary suboperator with zero eta_vertices gives
  // identical results (same as WELL_BALANCING_NONE).
  CeedVector eta_vertices;
  PetscCall(CreateCeedEtaVerticesVector(mesh, &eta_vertices));
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    CeedOperator boundary_flux_op;
    RDyCondition condition = boundary_conditions[b];
    PetscCall(CreateCeedBoundaryFluxSuboperator(*config, mesh, &eta_vertices, &boundaries[b], condition, &boundary_flux_op));
    PetscCall(CeedOperatorCompositeAddSub(*flux_op, boundary_flux_op));
    PetscCall(CeedOperatorDestroy(&boundary_flux_op));
  }
  PetscCallCEED(CeedVectorDestroy(&eta_vertices));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
