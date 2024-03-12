#include <petscdmceed.h>
#include <private/rdysweimpl.h>

#include "swe_ceed_impl.h"

// frees a data context allocated using PETSc, returning a libCEED error code
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

// creates a QFunction context for a flux or source operator with the given
// minimum water height threshold
static PetscErrorCode CreateQFunctionContext(Ceed ceed, PetscReal tiny_h, CeedQFunctionContext *qf_context) {
  PetscFunctionBeginUser;

  SWEContext swe_ctx;
  PetscCall(PetscCalloc1(1, &swe_ctx));

  swe_ctx->dtime   = 0.0;
  swe_ctx->tiny_h  = tiny_h;
  swe_ctx->gravity = 9.806;

  PetscCallCEED(CeedQFunctionContextCreate(ceed, qf_context));
  PetscCallCEED(CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*swe_ctx), swe_ctx));
  PetscCallCEED(CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "time step", offsetof(struct SWEContext_, dtime), 1, "Time step of TS"));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "small h value", offsetof(struct SWEContext_, tiny_h), 1,
                                                   "Height threshold below which dry condition is assumed"));
  PetscCallCEED(CeedQFunctionContextRegisterDouble(*qf_context, "gravity", offsetof(struct SWEContext_, gravity), 1, "Accelaration due to gravity"));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

static PetscErrorCode CreateInteriorFluxOperator(Ceed ceed, RDyMesh *mesh, PetscReal tiny_h, CeedOperator *flux_op) {
  PetscFunctionBeginUser;

  CeedInt   num_comp = 3;
  RDyCells *cells    = &mesh->cells;
  RDyEdges *edges    = &mesh->edges;

  CeedQFunction qf;
  CeedInt       num_comp_geom = 4;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEFlux_Roe, SWEFlux_Roe_loc, &qf));
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));

  CeedQFunctionContext qf_context;
  PetscCallCEED(CreateQFunctionContext(ceed, tiny_h, &qf_context));
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  CeedElemRestriction q_restrict_l, q_restrict_r, c_restrict_l, c_restrict_r, restrict_geom, restrict_flux;
  CeedVector          geom, flux;
  {
    CeedInt num_edges = mesh->num_owned_internal_edges;

    // create an element restriction for geometric factors that convert
    // fluxes to cell states
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));

    // create an element restriction for accumulated fluxes
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));

    // create element restrictions for left and right input/output states,
    // populate offsets for these states, and set the (invariant)
    // geometric parameters
    CeedInt *q_offset_l, *q_offset_r, *c_offset_l, *c_offset_r;
    PetscCall(PetscMalloc2(num_edges, &q_offset_l, num_edges, &q_offset_r));
    PetscCall(PetscMalloc2(num_edges, &c_offset_l, num_edges, &c_offset_r));
    CeedScalar(*g)[4];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, oe = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l      = edges->cell_ids[2 * iedge];
      CeedInt r      = edges->cell_ids[2 * iedge + 1];
      q_offset_l[oe] = l * num_comp;
      q_offset_r[oe] = r * num_comp;
      c_offset_l[oe] = cells->local_to_owned[l] * num_comp;
      c_offset_r[oe] = cells->local_to_owned[r] * num_comp;

      g[oe][0] = edges->sn[iedge];
      g[oe][1] = edges->cn[iedge];
      g[oe][2] = -edges->lengths[iedge] / cells->areas[l];
      g[oe][3] = edges->lengths[iedge] / cells->areas[r];
      oe++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create element restrictions for left and right cell states
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
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(q_restrict_r, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_r, stdout));
    }
  }

  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, flux_op));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "q_left", q_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "q_right", q_restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "cell_left", c_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "cell_right", c_restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "flux", restrict_flux, CEED_BASIS_COLLOCATED, flux));

  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_r));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

static PetscErrorCode CreateBoundaryFluxOperator(Ceed ceed, RDyMesh *mesh, RDyBoundary boundary, RDyCondition boundary_condition, PetscReal tiny_h,
                                                 CeedOperator *flux_op) {
  PetscFunctionBeginUser;

  CeedInt   num_comp = 3;
  RDyCells *cells    = &mesh->cells;
  RDyEdges *edges    = &mesh->edges;

  CeedQFunctionUser func;
  const char       *func_loc;
  switch (boundary_condition.flow->type) {
    case CONDITION_DIRICHLET:
      func     = SWEBoundaryFlux_Dirichlet_Roe;
      func_loc = SWEBoundaryFlux_Dirichlet_Roe_loc;
      break;
    case CONDITION_REFLECTING:
      func     = SWEBoundaryFlux_Reflecting_Roe;
      func_loc = SWEBoundaryFlux_Reflecting_Roe_loc;
      break;
    case CONDITION_CRITICAL_OUTFLOW:
      func     = SWEBoundaryFlux_Outflow_Roe;
      func_loc = SWEBoundaryFlux_Outflow_Roe_loc;
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n",
                 boundary.id);
  }
  CeedQFunction qf;
  CeedInt       num_comp_geom = 3;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, func, func_loc, &qf));
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    PetscCallCEED(CeedQFunctionAddInput(qf, "q_dirichlet", num_comp, CEED_EVAL_NONE));
  }
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));

  CeedQFunctionContext qf_context;
  PetscCallCEED(CreateQFunctionContext(ceed, tiny_h, &qf_context));
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  CeedElemRestriction q_restrict_l, c_restrict_l, restrict_dirichlet, restrict_geom, restrict_flux;
  CeedVector          geom, flux, dirichlet;
  {
    CeedInt num_edges = boundary.num_edges;

    // create element restrictions for left and right input/output states
    CeedInt *q_offset_l, *c_offset_l, *offset_dirichlet = NULL;
    PetscCall(PetscMalloc1(num_edges, &q_offset_l));
    PetscCall(PetscMalloc1(num_edges, &c_offset_l));
    if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
      PetscCall(PetscMalloc1(num_edges, &offset_dirichlet));
    }

    // create an element restriction for geometric factors that convert
    // fluxes to cell states
    CeedInt num_owned_edges = 0;
    for (CeedInt e = 0; e < boundary.num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (edges->is_owned[iedge]) num_owned_edges++;
    }
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));

    // create an element restriction for accumulated fluxes
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));

    // create an element restriction for the "left" (interior) input/output
    // states, populate offsets for these states, and set the (invariant)
    // geometric parameters
    CeedScalar(*g)[3];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, oe = 0; e < num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l      = edges->cell_ids[2 * iedge];
      q_offset_l[oe] = l * num_comp;
      c_offset_l[oe] = cells->local_to_owned[l] * num_comp;
      if (offset_dirichlet) {  // Dirichlet boundary values
        offset_dirichlet[oe] = e * num_comp;
      }

      g[oe][0] = edges->sn[iedge];
      g[oe][1] = edges->cn[iedge];
      g[oe][2] = -edges->lengths[iedge] / cells->areas[l];
      oe++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create the element restriction for the left cell states
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            q_offset_l, &q_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            c_offset_l, &c_restrict_l));
    PetscCall(PetscFree(q_offset_l));
    PetscCall(PetscFree(c_offset_l));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
    }

    // if we have Dirichlet boundary values, create a restriction and passive
    // input vector for them
    if (offset_dirichlet) {
      PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, num_edges * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                              offset_dirichlet, &restrict_dirichlet));
      PetscCall(PetscFree(offset_dirichlet));
      if (0) PetscCallCEED(CeedElemRestrictionView(restrict_dirichlet, stdout));
      PetscCallCEED(CeedElemRestrictionCreateVector(restrict_dirichlet, &dirichlet, NULL));
      PetscCallCEED(CeedVectorSetValue(dirichlet, 0.0));
    }
  }

  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, flux_op));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "q_left", q_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    PetscCallCEED(CeedOperatorSetField(*flux_op, "q_dirichlet", restrict_dirichlet, CEED_BASIS_COLLOCATED, dirichlet));
  }
  PetscCallCEED(CeedOperatorSetField(*flux_op, "cell_left", c_restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*flux_op, "flux", restrict_flux, CEED_BASIS_COLLOCATED, flux));

  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// Creates a flux operator for the shallow water equations that produces
// solutions to the related riemann problem for cells separated by edges on the
// given computational mesh. The resulting operator can be manipulated by
// libCEED calls.
// @param [in]  ceed The Ceed context used to create the operator
// @param [in]  mesh The computational mesh for which the operator is created
// @param [in]  num_boundaries The number of boundaries (disjoint edge sets) on the mesh
// @param [in]  boundaries An array of disjoint edge sets representing mesh boundaries
// @param [in]  boundary_conditions An array of metadata defining boundary conditions the operator will enforce
// @param [in]  tiny_h the minimum height threshold for water flow
// @param [out] flux_op A pointer to the flux operator to be created
PetscErrorCode CreateSWEFluxOperator(Ceed ceed, RDyMesh *mesh, CeedInt num_boundaries, RDyBoundary boundaries[num_boundaries],
                                     RDyCondition boundary_conditions[num_boundaries], PetscReal tiny_h, CeedOperator *flux_op) {
  PetscFunctionBeginUser;

  // create a composite operator consisting of interior and boundary flux
  // operators
  PetscCallCEED(CeedCompositeOperatorCreate(ceed, flux_op));

  CeedOperator interior_op;
  PetscCall(CreateInteriorFluxOperator(ceed, mesh, tiny_h, &interior_op));
  PetscCallCEED(CeedCompositeOperatorAddSub(*flux_op, interior_op));
  PetscCallCEED(CeedOperatorDestroy(&interior_op));

  for (CeedInt b = 0; b < num_boundaries; b++) {
    CeedOperator boundary_op;
    RDyBoundary  boundary           = boundaries[b];
    RDyCondition boundary_condition = boundary_conditions[b];
    PetscCall(CreateBoundaryFluxOperator(ceed, mesh, boundary, boundary_condition, tiny_h, &boundary_op));
    PetscCallCEED(CeedCompositeOperatorAddSub(*flux_op, boundary_op));
    PetscCallCEED(CeedOperatorDestroy(&boundary_op));
  }

  if (0) PetscCallCEED(CeedOperatorView(*flux_op, stdout));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// updates the time step used by the SWE flux operator
PetscErrorCode SWEFluxOperatorSetTimeStep(CeedOperator flux_op, PetscReal dt) {
  PetscFunctionBeginUser;

  CeedContextFieldLabel label;
  PetscCallCEED(CeedOperatorGetContextFieldLabel(flux_op, "time step", &label));
  PetscCallCEED(CeedOperatorSetContextDouble(flux_op, label, &dt));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// Gets the field representing the boundary flux for the given boundary.
PetscErrorCode SWEFluxOperatorGetBoundaryFlux(CeedOperator flux_op, RDyBoundary boundary, CeedOperatorField *boundary_flux) {
  PetscFunctionBeginUser;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(flux_op, &sub_ops));
  CeedOperator boundary_flux_op = sub_ops[1 + boundary.index];

  // fetch the field
  PetscCallCEED(CeedOperatorGetFieldByName(boundary_flux_op, "flux", boundary_flux));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// Gets the field representing Dirіchlet boundary values for the given boundary.
PetscErrorCode SWEFluxOperatorGetDirichletBoundaryValues(CeedOperator flux_op, RDyBoundary boundary, CeedOperatorField *boundary_values) {
  PetscFunctionBeginUser;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(flux_op, &sub_ops));
  CeedOperator boundary_flux_op = sub_ops[1 + boundary.index];

  // fetch the field
  PetscCallCEED(CeedOperatorGetFieldByName(boundary_flux_op, "q_dirichlet", boundary_values));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

PetscErrorCode SWEFluxOperatorSetDirichletBoundaryValues(CeedOperator flux_op, RDyMesh *mesh, RDyBoundary boundary, PetscInt size,
                                                         PetscReal boundary_values[size]) {
  PetscFunctionBeginUser;

  // fetch the array storing the boundary values
  CeedOperatorField dirichlet_field;
  PetscCall(SWEFluxOperatorGetDirichletBoundaryValues(flux_op, boundary, &dirichlet_field));
  CeedVector dirichlet_vector;
  PetscCallCEED(CeedOperatorFieldGetVector(dirichlet_field, &dirichlet_vector));
  CeedInt num_comp = 3;
  CeedScalar(*dirichlet_ceed)[num_comp];
  PetscCallCEED(CeedVectorGetArray(dirichlet_vector, CEED_MEM_HOST, (CeedScalar **)&dirichlet_ceed));

  // set the boundary values
  for (CeedInt i = 0; i < boundary.num_edges; ++i) {
    dirichlet_ceed[i][0] = boundary_values[num_comp * i];
    dirichlet_ceed[i][1] = boundary_values[num_comp * i + 1];
    dirichlet_ceed[i][2] = boundary_values[num_comp * i + 2];
  }

  // copy the values into the CEED operator
  PetscCallCEED(CeedVectorRestoreArray(dirichlet_vector, (CeedScalar **)&dirichlet_ceed));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// Given a computational mesh, creates a source operator for the shallow water
// equations that computes source terms. The resulting operator can be
// manipulated by libCEED calls.
// @param [in]  ceed The Ceed context used to create the operator
// @param [in]  mesh The computational mesh for which the operator is created
// @param [in]  num_cells Number of cells
// @param [in]  materials_by_cell An array of RDyMaterials defining cellwise material properties
// @param [in]  tiny_h the minimum height threshold for water flow
// @param [out] flux_op A pointer to the flux operator to be created
PetscErrorCode CreateSWESourceOperator(Ceed ceed, RDyMesh *mesh, PetscInt num_cells, RDyMaterial materials_by_cell[num_cells], PetscReal tiny_h,
                                       CeedOperator *source_op) {
  PetscFunctionBeginUser;

  PetscCallCEED(CeedCompositeOperatorCreate(ceed, source_op));
  CeedInt   num_comp = 3;
  RDyCells *cells    = &mesh->cells;

  {
    // source term
    CeedQFunction qf;
    CeedInt       num_comp_geom = 2, num_comp_swe_src = 3, num_comp_mannings_n = 1;
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourceTerm, SWESourceTerm_loc, &qf));
    PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "swe_src", num_comp_swe_src, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "mannings_n", num_comp_mannings_n, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
    PetscCallCEED(CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE));

    CeedQFunctionContext qf_context;
    PetscCallCEED(CreateQFunctionContext(ceed, tiny_h, &qf_context));
    if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
    PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
    PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

    CeedElemRestriction restrict_c, restrict_q, restrict_geom, restrict_swe, restrict_mannings_n, restrict_riemannf;
    CeedVector          geom;
    CeedVector          swe_src;
    CeedVector          mannings_n;
    CeedVector          riemannf;
    {  // Create element restrictions for state
      CeedInt *offset_c, *offset_q;
      CeedScalar(*g)[num_comp_geom];
      CeedScalar(*n)[num_comp_mannings_n];
      CeedInt num_owned_cells = mesh->num_cells_local;

      CeedInt strides_geom[] = {num_comp_geom, 1, num_comp_geom};
      PetscCallCEED(
          CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom));
      PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
      PetscCallCEED(CeedVectorSetValue(geom, 0.0));

      CeedInt strides_swe_src[] = {num_comp_swe_src, 1, num_comp_swe_src};
      PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_swe_src, num_owned_cells * num_comp_swe_src, strides_swe_src,
                                                     &restrict_swe));
      PetscCallCEED(CeedElemRestrictionCreateVector(restrict_swe, &swe_src, NULL));
      PetscCallCEED(CeedVectorSetValue(swe_src, 0.0));

      CeedInt strides_mannings_n[] = {num_comp_mannings_n, 1, num_comp_mannings_n};
      PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_mannings_n, num_owned_cells * num_comp_mannings_n,
                                                     strides_mannings_n, &restrict_mannings_n));
      PetscCallCEED(CeedElemRestrictionCreateVector(restrict_mannings_n, &mannings_n, NULL));
      PetscCallCEED(CeedVectorSetValue(mannings_n, 0.0));

      CeedInt strides_riemannf[] = {num_comp, 1, num_comp};
      PetscCallCEED(
          CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp, num_owned_cells * num_comp, strides_riemannf, &restrict_riemannf));
      PetscCallCEED(CeedElemRestrictionCreateVector(restrict_riemannf, &riemannf, NULL));
      PetscCallCEED(CeedVectorSetValue(riemannf, 0.0));

      PetscCall(PetscMalloc1(num_owned_cells, &offset_q));
      PetscCall(PetscMalloc1(num_owned_cells, &offset_c));
      PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
      PetscCallCEED(CeedVectorGetArray(mannings_n, CEED_MEM_HOST, (CeedScalar **)&n));
      for (CeedInt c = 0, oc = 0; c < mesh->num_cells; c++) {
        if (!cells->is_local[c]) continue;

        offset_q[oc] = c * num_comp;
        offset_c[oc] = cells->local_to_owned[c] * num_comp;

        g[oc][0] = cells->dz_dx[c];
        g[oc][1] = cells->dz_dy[c];

        n[oc][0] = materials_by_cell[c].manning;

        oc++;
      }
      PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));
      PetscCallCEED(CeedVectorRestoreArray(mannings_n, (CeedScalar **)&n));
      PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_cells, 1, num_comp, 1, num_owned_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                              offset_q, &restrict_q));
      PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_cells, 1, num_comp, 1, num_owned_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                              offset_c, &restrict_c));
      PetscCall(PetscFree(offset_c));
      PetscCall(PetscFree(offset_q));
      if (0) {
        PetscCallCEED(CeedElemRestrictionView(restrict_q, stdout));
        PetscCallCEED(CeedElemRestrictionView(restrict_c, stdout));
      }
    }

    {
      CeedOperator op;
      PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, &op));
      PetscCallCEED(CeedOperatorSetField(op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom));
      PetscCallCEED(CeedOperatorSetField(op, "swe_src", restrict_swe, CEED_BASIS_COLLOCATED, swe_src));
      PetscCallCEED(CeedOperatorSetField(op, "mannings_n", restrict_mannings_n, CEED_BASIS_COLLOCATED, mannings_n));
      PetscCallCEED(CeedOperatorSetField(op, "riemannf", restrict_riemannf, CEED_BASIS_COLLOCATED, riemannf));
      PetscCallCEED(CeedOperatorSetField(op, "q", restrict_q, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
      PetscCallCEED(CeedOperatorSetField(op, "cell", restrict_c, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));
      PetscCallCEED(CeedCompositeOperatorAddSub(*source_op, op));
      PetscCallCEED(CeedOperatorDestroy(&op));
    }
    PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
    PetscCallCEED(CeedElemRestrictionDestroy(&restrict_mannings_n));
    PetscCallCEED(CeedElemRestrictionDestroy(&restrict_c));
    PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q));
    PetscCallCEED(CeedVectorDestroy(&geom));
  }

  if (0) PetscCallCEED(CeedOperatorView(*source_op, stdout));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// updates the time step used by the SWE source operator
PetscErrorCode SWESourceOperatorSetTimeStep(CeedOperator source_op, PetscReal dt) {
  PetscFunctionBeginUser;

  CeedContextFieldLabel label;
  PetscCallCEED(CeedOperatorGetContextFieldLabel(source_op, "time step", &label));
  PetscCallCEED(CeedOperatorSetContextDouble(source_op, label, &dt));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// Given a shallow water equations source operator created by
// CreateSWESourceOperator, fetches the field associated with the field_name.
// This can be used to implement a time-dependent water source.
static PetscErrorCode SWESubOperatorGetOperatorField(CeedOperator op, CeedInt sub_op_idx, const char *field_name, CeedOperatorField *op_field) {
  PetscFunctionBeginUser;

  // check if the requested suboperator index is valid
  CeedInt num_sub_op;
  PetscCallCEED(CeedCompositeOperatorGetNumSub(op, &num_sub_op));
  PetscCheck(sub_op_idx < num_sub_op, PETSC_COMM_WORLD, PETSC_ERR_USER,
             "Trying to extract info about Ceed subporator = %d, but total number of Ceed operators = %d", sub_op_idx, num_sub_op);

  // get the source sub-operator responsible for the water source (the first one)
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(op, &sub_ops));
  CeedOperator sub_op = sub_ops[sub_op_idx];

  // fetch the field
  PetscCallCEED(CeedOperatorGetFieldByName(sub_op, field_name, op_field));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// For the given CeedOperator, first extract the 'sub_op_idx'-th sub-operator.
/// Then, extract the CeedField associated with 'field_name' for the sub-operator.
/// Finally, set values in the icomp-th component of the CeedField
/// @param [inout]  op The Ceed source operator
/// @param [in] sub_op_idx Index of the Ceed sub-operator
/// @param [in] field_name Name of the field in the Ceed sub-operator
/// @param [in]  icomp The component of the field whose values will be set
/// @param [in]  *swe_src The array containing values for the source term
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode SetOperatorFieldComponent(CeedOperator op, CeedInt sub_op_idx, const char *field_name, CeedInt icomp, PetscReal *value) {
  PetscFunctionBeginUser;

  CeedOperatorField swe_field;
  SWESubOperatorGetOperatorField(op, sub_op_idx, field_name, &swe_field);

  CeedElemRestriction restrict_swe;
  PetscCallCEED(CeedOperatorFieldGetElemRestriction(swe_field, &restrict_swe));

  CeedVector swe_vec;
  CeedOperatorFieldGetVector(swe_field, &swe_vec);

  CeedInt num_comp;
  PetscCallCEED(CeedElemRestrictionGetNumComponents(restrict_swe, &num_comp));

  CeedScalar(*data_ceed)[num_comp];
  PetscCallCEED(CeedVectorGetArray(swe_vec, CEED_MEM_HOST, (CeedScalar **)&data_ceed));

  CeedSize len;
  PetscCallCEED(CeedVectorGetLength(swe_vec, &len));
  for (CeedInt i = 0; i < len / num_comp; ++i) {
    data_ceed[i][icomp] = value[i];
  }

  PetscCallCEED(CeedVectorRestoreArray(swe_vec, (CeedScalar **)&data_ceed));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// sets the per-cell water source for the given CEED SWE source operator
PetscErrorCode SWESourceOperatorSetWaterSource(CeedOperator source_op, PetscReal *swe_src) {
  PetscFunctionBeginUser;
  CeedInt sub_op_idx = 0, icomp = 0;
  PetscCall(SetOperatorFieldComponent(source_op, sub_op_idx, "swe_src", icomp, swe_src));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// sets the per-cell x-momentum source for the given CEED SWE source operator
PetscErrorCode SWESourceOperatorSetXMomentumSource(CeedOperator source_op, PetscReal *swe_src) {
  PetscFunctionBeginUser;
  CeedInt sub_op_idx = 0, icomp = 1;
  PetscCall(SetOperatorFieldComponent(source_op, sub_op_idx, "swe_src", icomp, swe_src));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// sets the per-cell y-momentum source for the given CEED SWE source operator
PetscErrorCode SWESourceOperatorSetYMomentumSource(CeedOperator source_op, PetscReal *swe_src) {
  PetscFunctionBeginUser;
  CeedInt sub_op_idx = 0, icomp = 2;
  PetscCall(SetOperatorFieldComponent(source_op, sub_op_idx, "swe_src", icomp, swe_src));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

// Given a shallow water equations source operator created by
// CreateSWESourceOperator, fetches the field representing the Riemann flux.
PetscErrorCode SWESourceOperatorGetRiemannFlux(CeedOperator source_op, CeedOperatorField *riemann_flux_field) {
  PetscFunctionBeginUser;

  // get the source sub-operator responsible for the water source (the first one)
  CeedOperator *sub_ops;
  PetscCallCEED(CeedCompositeOperatorGetSubList(source_op, &sub_ops));
  CeedOperator riemannf_source_op = sub_ops[0];

  // fetch the field
  PetscCallCEED(CeedOperatorGetFieldByName(riemannf_source_op, "riemannf", riemann_flux_field));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

PetscErrorCode SWESourceOperatorSetManningsN(CeedOperator source_op, PetscReal *n_values) {
  PetscFunctionBeginUser;
  CeedInt sub_op_idx = 0, icomp = 0;
  PetscCall(SetOperatorFieldComponent(source_op, sub_op_idx, "mannings_n", icomp, n_values));
  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}
