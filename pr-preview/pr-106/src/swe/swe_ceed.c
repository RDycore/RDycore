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

  CeedQFunctionContextCreate(ceed, qf_context);
  CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*swe_ctx), swe_ctx);
  CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(*qf_context, "time step", offsetof(struct SWEContext_, dtime), 1, "Time step of TS");
  CeedQFunctionContextRegisterDouble(*qf_context, "small h value", offsetof(struct SWEContext_, tiny_h), 1,
                                     "Height threshold below which dry condition is assumed");
  CeedQFunctionContextRegisterDouble(*qf_context, "gravity", offsetof(struct SWEContext_, gravity), 1, "Accelaration due to gravity");

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateInteriorFluxOperator(Ceed ceed, RDyMesh *mesh, PetscReal tiny_h, CeedOperator *flux_op) {
  PetscFunctionBeginUser;

  CeedInt   num_comp = 3;
  RDyCells *cells    = &mesh->cells;
  RDyEdges *edges    = &mesh->edges;

  CeedQFunction qf;
  CeedInt       num_comp_geom = 4;
  CeedQFunctionCreateInterior(ceed, 1, SWEFlux_Roe, SWEFlux_Roe_loc, &qf);
  CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf, "cell_right", num_comp, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE);

  CeedQFunctionContext qf_context;
  CreateQFunctionContext(ceed, tiny_h, &qf_context);
  if (0) CeedQFunctionContextView(qf_context, stdout);
  CeedQFunctionSetContext(qf, qf_context);
  CeedQFunctionContextDestroy(&qf_context);

  CeedElemRestriction restrict_l, restrict_r, restrict_geom, restrict_flux;
  CeedVector          geom, flux;
  {
    CeedInt num_edges = mesh->num_owned_internal_edges;

    // create an element restriction for geometric factors that convert
    // fluxes to cell states
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom);
    CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL);
    CeedVectorSetValue(geom, 0.0);

    // create an element restriction for accumulated fluxes
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux);
    CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL);
    CeedVectorSetValue(flux, 0.0);

    // create element restrictions for left and right input/output states,
    // populate offsets for these states, and set the (invariant)
    // geometric parameters
    CeedInt *offset_l, *offset_r;
    PetscCall(PetscMalloc2(num_edges, &offset_l, num_edges, &offset_r));
    CeedScalar(*g)[4];
    CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g);
    for (CeedInt e = 0, oe = 0; e < mesh->num_internal_edges; e++) {
      PetscInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      PetscInt l   = edges->cell_ids[2 * iedge];
      PetscInt r   = edges->cell_ids[2 * iedge + 1];
      offset_l[oe] = l * num_comp;
      offset_r[oe] = r * num_comp;

      g[oe][0] = edges->sn[iedge];
      g[oe][1] = edges->cn[iedge];
      g[oe][2] = -edges->lengths[iedge] / cells->areas[l];
      g[oe][3] = edges->lengths[iedge] / cells->areas[r];
      oe++;
    }
    CeedVectorRestoreArray(geom, (CeedScalar **)&g);

    // create element restrictions for left and right cell states
    CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_l, &restrict_l);
    CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_r, &restrict_r);
    PetscCall(PetscFree2(offset_l, offset_r));
    if (0) {
      CeedElemRestrictionView(restrict_l, stdout);
      CeedElemRestrictionView(restrict_r, stdout);
    }
  }

  CeedOperatorCreate(ceed, qf, NULL, NULL, flux_op);
  CeedOperatorSetField(*flux_op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom);
  CeedOperatorSetField(*flux_op, "q_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*flux_op, "q_right", restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*flux_op, "cell_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*flux_op, "cell_right", restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*flux_op, "flux", restrict_flux, CEED_BASIS_COLLOCATED, flux);

  CeedElemRestrictionDestroy(&restrict_geom);
  CeedElemRestrictionDestroy(&restrict_flux);
  CeedElemRestrictionDestroy(&restrict_l);
  CeedElemRestrictionDestroy(&restrict_r);
  CeedVectorDestroy(&geom);
  CeedVectorDestroy(&flux);

  PetscFunctionReturn(PETSC_SUCCESS);
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
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %d\n", boundary.id);
  }
  CeedQFunction qf;
  CeedInt       num_comp_geom = 3;
  CeedQFunctionCreateInterior(ceed, 1, func, func_loc, &qf);
  CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE);
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    CeedQFunctionAddInput(qf, "q_dirichlet", num_comp, CEED_EVAL_NONE);
  }
  CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE);

  CeedQFunctionContext qf_context;
  CreateQFunctionContext(ceed, tiny_h, &qf_context);
  if (0) CeedQFunctionContextView(qf_context, stdout);
  CeedQFunctionSetContext(qf, qf_context);
  CeedQFunctionContextDestroy(&qf_context);

  CeedElemRestriction restrict_l, restrict_dirichlet, restrict_geom, restrict_flux;
  CeedVector          geom, flux, dirichlet;
  {
    CeedInt num_edges = boundary.num_edges;

    // create element restrictions for left and right input/output states
    CeedInt *offset_l, *offset_dirichlet = NULL;
    PetscCall(PetscMalloc1(num_edges, &offset_l));
    if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
      PetscCall(PetscMalloc1(num_edges, &offset_dirichlet));
    }

    // create an element restriction for geometric factors that convert
    // fluxes to cell states
    CeedInt num_owned_edges = 0;
    for (CeedInt e = 0; e < boundary.num_edges; e++) {
      PetscInt iedge = boundary.edge_ids[e];
      if (edges->is_owned[iedge]) num_owned_edges++;
    }
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom);
    CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL);
    CeedVectorSetValue(geom, 0.0);

    // create an element restriction for accumulated fluxes
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux);
    CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL);
    CeedVectorSetValue(flux, 0.0);

    // create an element restriction for the "left" (interior) input/output
    // states, populate offsets for these states, and set the (invariant)
    // geometric parameters
    CeedScalar(*g)[3];
    CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g);
    for (CeedInt e = 0, oe = 0; e < num_edges; e++) {
      PetscInt iedge = boundary.edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      PetscInt l   = edges->cell_ids[2 * iedge];
      PetscInt r   = edges->cell_ids[2 * iedge + 1];
      offset_l[oe] = l * num_comp;
      if (offset_dirichlet) {  // Dirichlet boundary values
        offset_dirichlet[oe] = r * num_comp;
      }

      g[oe][0] = edges->sn[iedge];
      g[oe][1] = edges->cn[iedge];
      g[oe][2] = -edges->lengths[iedge] / cells->areas[l];
      oe++;
    }
    CeedVectorRestoreArray(geom, (CeedScalar **)&g);

    // create the element restriction for the left cell states
    CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_l,
                              &restrict_l);
    PetscCall(PetscFree(offset_l));
    if (0) CeedElemRestrictionView(restrict_l, stdout);

    // if we have Dirichlet boundary values, create a restriction and passive
    // input vector for them
    if (offset_dirichlet) {
      CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_dirichlet,
                                &restrict_dirichlet);
      PetscCall(PetscFree(offset_dirichlet));
      if (0) CeedElemRestrictionView(restrict_dirichlet, stdout);
      CeedElemRestrictionCreateVector(restrict_dirichlet, &dirichlet, NULL);
      CeedVectorSetValue(dirichlet, 0.0);
    }
  }

  CeedOperatorCreate(ceed, qf, NULL, NULL, flux_op);
  CeedOperatorSetField(*flux_op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom);
  CeedOperatorSetField(*flux_op, "q_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    CeedOperatorSetField(*flux_op, "q_dirichlet", restrict_dirichlet, CEED_BASIS_COLLOCATED, dirichlet);
  }
  CeedOperatorSetField(*flux_op, "cell_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(*flux_op, "flux", restrict_flux, CEED_BASIS_COLLOCATED, flux);

  CeedElemRestrictionDestroy(&restrict_geom);
  CeedElemRestrictionDestroy(&restrict_flux);
  CeedElemRestrictionDestroy(&restrict_l);
  CeedVectorDestroy(&geom);
  CeedVectorDestroy(&flux);

  PetscFunctionReturn(PETSC_SUCCESS);
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
PetscErrorCode CreateSWEFluxOperator(Ceed ceed, RDyMesh *mesh, int num_boundaries, RDyBoundary boundaries[num_boundaries],
                                     RDyCondition boundary_conditions[num_boundaries], PetscReal tiny_h, CeedOperator *flux_op) {
  PetscFunctionBeginUser;

  // create a composite operator consisting of interior and boundary flux
  // operators
  CeedCompositeOperatorCreate(ceed, flux_op);

  CeedOperator interior_op;
  PetscCall(CreateInteriorFluxOperator(ceed, mesh, tiny_h, &interior_op));
  CeedCompositeOperatorAddSub(*flux_op, interior_op);
  CeedOperatorDestroy(&interior_op);

  for (PetscInt b = 0; b < num_boundaries; b++) {
    CeedOperator boundary_op;
    RDyBoundary  boundary           = boundaries[b];
    RDyCondition boundary_condition = boundary_conditions[b];
    PetscCall(CreateBoundaryFluxOperator(ceed, mesh, boundary, boundary_condition, tiny_h, &boundary_op));
    CeedCompositeOperatorAddSub(*flux_op, boundary_op);
    CeedOperatorDestroy(&boundary_op);
  }

  if (0) CeedOperatorView(*flux_op, stdout);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// updates the time step used by the SWE flux operator
PetscErrorCode SWEFluxOperatorSetTimeStep(CeedOperator flux_op, PetscReal dt) {
  PetscFunctionBeginUser;

  CeedContextFieldLabel label;
  CeedOperatorGetContextFieldLabel(flux_op, "time step", &label);
  CeedOperatorSetContextDouble(flux_op, label, &dt);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Gets the field representing the boundary flux for the given boundary.
PetscErrorCode SWEFluxOperatorGetBoundaryFlux(CeedOperator flux_op, RDyBoundary boundary, CeedOperatorField *boundary_flux) {
  PetscFunctionBeginUser;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  CeedCompositeOperatorGetSubList(flux_op, &sub_ops);
  CeedOperator boundary_flux_op = sub_ops[1 + boundary.index];

  // fetch the field
  CeedOperatorGetFieldByName(boundary_flux_op, "flux", boundary_flux);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Gets the field representing Dirіchlet boundary values for the given boundary.
PetscErrorCode SWEFluxOperatorGetDirichletBoundaryValues(CeedOperator flux_op, RDyBoundary boundary, CeedOperatorField *boundary_values) {
  PetscFunctionBeginUser;

  // get the relevant boundary sub-operator
  CeedOperator *sub_ops;
  CeedCompositeOperatorGetSubList(flux_op, &sub_ops);
  CeedOperator boundary_flux_op = sub_ops[1 + boundary.index];

  // fetch the field
  CeedOperatorGetFieldByName(boundary_flux_op, "q_dirichlet", boundary_values);

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SWEFluxOperatorSetDirichletBoundaryValues(CeedOperator flux_op, RDyBoundary boundary,
                                                         PetscReal boundary_values[3 * boundary.num_edges]) {
  PetscFunctionBeginUser;

  // fetch the array storing the boundary values
  CeedOperatorField dirichlet_field;
  PetscCall(SWEFluxOperatorGetDirichletBoundaryValues(flux_op, boundary, &dirichlet_field));
  CeedVector dirichlet_vector;
  CeedOperatorFieldGetVector(dirichlet_field, &dirichlet_vector);
  PetscInt num_comp = 3;
  CeedScalar(*dirichlet_ceed)[num_comp];
  CeedVectorGetArray(dirichlet_vector, CEED_MEM_HOST, (CeedScalar **)&dirichlet_ceed);

  // set the boundary values
  for (PetscInt i = 0; i < boundary.num_edges; ++i) {
    dirichlet_ceed[i][0] = boundary_values[num_comp * i];
    dirichlet_ceed[i][1] = boundary_values[num_comp * i + 1];
    dirichlet_ceed[i][2] = boundary_values[num_comp * i + 2];
  }

  // copy the values into the CEED operator
  CeedVectorRestoreArray(dirichlet_vector, (CeedScalar **)&dirichlet_ceed);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Given a computational mesh, creates a source operator for the shallow water
// equations that computes source terms. The resulting operator can be
// manipulated by libCEED calls.
// @param [in]  ceed The Ceed context used to create the operator
// @param [in]  mesh The computational mesh for which the operator is created
// @param [in]  materials_by_cell An array of RDyMaterials defining cellwise material properties
// @param [in]  tiny_h the minimum height threshold for water flow
// @param [out] flux_op A pointer to the flux operator to be created
PetscErrorCode CreateSWESourceOperator(Ceed ceed, RDyMesh *mesh, RDyMaterial materials_by_cell[mesh->num_cells], PetscReal tiny_h,
                                       CeedOperator *source_op) {
  PetscFunctionBeginUser;

  CeedCompositeOperatorCreate(ceed, source_op);
  CeedInt   num_comp = 3;
  RDyCells *cells    = &mesh->cells;

  {
    // source term
    CeedQFunction qf;
    CeedInt       num_comp_geom = 2, num_comp_water_src = 1, num_comp_mannings_n = 1;
    CeedQFunctionCreateInterior(ceed, 1, SWESourceTerm, SWESourceTerm_loc, &qf);
    CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf, "water_src", num_comp_water_src, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf, "mannings_n", num_comp_mannings_n, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE);
    CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE);
    CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE);

    CeedQFunctionContext qf_context;
    CreateQFunctionContext(ceed, tiny_h, &qf_context);
    if (0) CeedQFunctionContextView(qf_context, stdout);
    CeedQFunctionSetContext(qf, qf_context);
    CeedQFunctionContextDestroy(&qf_context);

    CeedElemRestriction restrict_c, restrict_geom, restrict_water_src, restrict_mannings_n, restrict_riemannf;
    CeedVector          geom;
    CeedVector          water_src;
    CeedVector          mannings_n;
    CeedVector          riemannf;
    {  // Create element restrictions for state
      CeedInt *offset_c;
      CeedScalar(*g)[num_comp_geom];
      CeedScalar(*n)[num_comp_mannings_n];
      CeedInt num_owned_cells = mesh->num_cells_local;

      CeedInt strides_geom[] = {num_comp_geom, 1, num_comp_geom};
      CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom);
      CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL);
      CeedVectorSetValue(geom, 0.0);

      CeedInt strides_water_src[] = {num_comp_water_src, 1, num_comp_water_src};
      CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_water_src, num_owned_cells * num_comp_water_src, strides_water_src,
                                       &restrict_water_src);
      CeedElemRestrictionCreateVector(restrict_water_src, &water_src, NULL);
      CeedVectorSetValue(water_src, 0.0);

      CeedInt strides_mannings_n[] = {num_comp_mannings_n, 1, num_comp_mannings_n};
      CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_mannings_n, num_owned_cells * num_comp_mannings_n, strides_mannings_n,
                                       &restrict_mannings_n);
      CeedElemRestrictionCreateVector(restrict_mannings_n, &mannings_n, NULL);
      CeedVectorSetValue(mannings_n, 0.0);

      CeedInt strides_riemannf[] = {num_comp, 1, num_comp};
      CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp, num_owned_cells * num_comp, strides_riemannf, &restrict_riemannf);
      CeedElemRestrictionCreateVector(restrict_riemannf, &riemannf, NULL);
      CeedVectorSetValue(riemannf, 0.0);

      PetscCall(PetscMalloc1(num_owned_cells, &offset_c));
      CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g);
      CeedVectorGetArray(mannings_n, CEED_MEM_HOST, (CeedScalar **)&n);
      for (CeedInt c = 0, oc = 0; c < mesh->num_cells; c++) {
        if (!cells->is_local[c]) continue;

        offset_c[oc] = c * num_comp;

        g[oc][0] = cells->dz_dx[c];
        g[oc][1] = cells->dz_dy[c];

        n[oc][0] = materials_by_cell[c].manning;

        oc++;
      }
      CeedVectorRestoreArray(geom, (CeedScalar **)&g);
      CeedVectorRestoreArray(mannings_n, (CeedScalar **)&n);
      CeedElemRestrictionCreate(ceed, num_owned_cells, 1, num_comp, 1, num_owned_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_c,
                                &restrict_c);
      PetscCall(PetscFree(offset_c));
      if (0) CeedElemRestrictionView(restrict_c, stdout);
    }

    {
      CeedOperator op;
      CeedOperatorCreate(ceed, qf, NULL, NULL, &op);
      CeedOperatorSetField(op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom);
      CeedOperatorSetField(op, "water_src", restrict_water_src, CEED_BASIS_COLLOCATED, water_src);
      CeedOperatorSetField(op, "mannings_n", restrict_mannings_n, CEED_BASIS_COLLOCATED, mannings_n);
      CeedOperatorSetField(op, "riemannf", restrict_riemannf, CEED_BASIS_COLLOCATED, riemannf);
      CeedOperatorSetField(op, "q", restrict_c, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
      CeedOperatorSetField(op, "cell", restrict_c, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
      CeedCompositeOperatorAddSub(*source_op, op);
      CeedOperatorDestroy(&op);
    }
    CeedElemRestrictionDestroy(&restrict_geom);
    CeedElemRestrictionDestroy(&restrict_mannings_n);
    CeedElemRestrictionDestroy(&restrict_c);
    CeedVectorDestroy(&geom);
  }

  if (0) CeedOperatorView(*source_op, stdout);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// updates the time step used by the SWE source operator
PetscErrorCode SWESourceOperatorSetTimeStep(CeedOperator source_op, PetscReal dt) {
  PetscFunctionBeginUser;

  CeedContextFieldLabel label;
  CeedOperatorGetContextFieldLabel(source_op, "time step", &label);
  CeedOperatorSetContextDouble(source_op, label, &dt);

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Given a shallow water equations source operator created by
// CreateSWESourceOperator, fetches the field representing the source of water.
// This can be used to implement a time-dependent water source.
PetscErrorCode SWESourceOperatorGetWaterSource(CeedOperator source_op, CeedOperatorField *water_source_field) {
  PetscFunctionBeginUser;

  // get the source sub-operator responsible for the water source (the first one)
  CeedOperator *sub_ops;
  CeedCompositeOperatorGetSubList(source_op, &sub_ops);
  CeedOperator water_source_op = sub_ops[0];

  // fetch the field
  CeedOperatorGetFieldByName(water_source_op, "water_src", water_source_field);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// sets the per-cell water source for the given CEED SWE source operator
PetscErrorCode SWESourceOperatorSetWaterSource(CeedOperator source_op, PetscReal *water_src) {
  PetscFunctionBeginUser;

  CeedOperatorField water_src_field;
  SWESourceOperatorGetWaterSource(source_op, &water_src_field);
  CeedVector water_src_vec;
  CeedOperatorFieldGetVector(water_src_field, &water_src_vec);

  PetscInt num_comp_water_src = 1;
  CeedScalar(*wat_src_ceed)[num_comp_water_src];
  CeedVectorGetArray(water_src_vec, CEED_MEM_HOST, (CeedScalar **)&wat_src_ceed);

  CeedSize water_src_len;
  CeedVectorGetLength(water_src_vec, &water_src_len);
  for (PetscInt i = 0; i < water_src_len; ++i) {
    wat_src_ceed[i][0] = water_src[i];
  }

  CeedVectorRestoreArray(water_src_vec, (CeedScalar **)&wat_src_ceed);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Given a shallow water equations source operator created by
// CreateSWESourceOperator, fetches the field representing the Riemann flux.
PetscErrorCode SWESourceOperatorGetRiemannFlux(CeedOperator source_op, CeedOperatorField *riemann_flux_field) {
  PetscFunctionBeginUser;

  // get the source sub-operator responsible for the water source (the first one)
  CeedOperator *sub_ops;
  CeedCompositeOperatorGetSubList(source_op, &sub_ops);
  CeedOperator riemannf_source_op = sub_ops[0];

  // fetch the field
  CeedOperatorGetFieldByName(riemannf_source_op, "riemannf", riemann_flux_field);
  PetscFunctionReturn(PETSC_SUCCESS);
}
