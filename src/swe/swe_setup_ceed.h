#ifndef SWE_SETUP_CEED_H
#define SWE_SETUP_CEED_H

#include "swe_flux_ceed.h"

// Free a plain data context that was allocated using PETSc; returning libCEED error codes
static int FreeContextPetsc(void *data) {
  if (PetscFree(data)) return CeedError(NULL, CEED_ERROR_ACCESS, "PetscFree failed");
  return CEED_ERROR_SUCCESS;
}

static PetscErrorCode CreateQFunctionContextForSWE(RDy rdy, Ceed ceed, CeedQFunctionContext *qf_context) {
  PetscFunctionBeginUser;

  SWEContext swe_ctx;
  PetscCall(PetscCalloc1(1, &swe_ctx));

  swe_ctx->dtime   = 0.0;
  swe_ctx->tiny_h  = rdy->config.physics.flow.tiny_h;
  swe_ctx->gravity = 9.806;

  CeedQFunctionContextCreate(ceed, qf_context);
  CeedQFunctionContextSetData(*qf_context, CEED_MEM_HOST, CEED_USE_POINTER, sizeof(*swe_ctx), swe_ctx);
  CeedQFunctionContextSetDataDestroy(*qf_context, CEED_MEM_HOST, FreeContextPetsc);
  CeedQFunctionContextRegisterDouble(*qf_context, "time step", offsetof(struct SWEContext_, dtime), 1, "Time step of TS");
  CeedQFunctionContextRegisterDouble(*qf_context, "samll h value", offsetof(struct SWEContext_, tiny_h), 1,
                                     "Height threshold below which dry condition is assumed");
  CeedQFunctionContextRegisterDouble(*qf_context, "gravity", offsetof(struct SWEContext_, gravity), 1, "Accelaration due to gravity");

  PetscFunctionReturn(0);
}

static PetscErrorCode RDyCeedOperatorSetUp(RDy rdy) {
  PetscFunctionBeginUser;

  PetscInt op_id = -1;

  if (rdy->ceed_resource[0] && !rdy->ceed_rhs.op_edges) {
    rdy->ceed_rhs.dt = 0.0;

    CeedCompositeOperatorCreate(rdy->ceed, &rdy->ceed_rhs.op_edges);
    CeedInt   num_comp = 3;
    RDyMesh  *mesh     = &rdy->mesh;
    RDyCells *cells    = &mesh->cells;
    RDyEdges *edges    = &mesh->edges;
    {  // interior operator
      CeedQFunction qf;
      CeedInt       num_comp_geom = 4;
      CeedQFunctionCreateInterior(rdy->ceed, 1, SWEFlux_Roe, SWEFlux_Roe_loc, &qf);
      CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE);
      CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddOutput(qf, "cell_right", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE);

      CeedQFunctionContext qf_context;
      CreateQFunctionContextForSWE(rdy, rdy->ceed, &qf_context);
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
        CeedElemRestrictionCreateStrided(rdy->ceed, num_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom);
        CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL);
        CeedVectorSetValue(geom, 0.0);

        // create an element restriction for accumulated fluxes
        CeedInt f_strides[] = {num_comp, 1, num_comp};
        CeedElemRestrictionCreateStrided(rdy->ceed, num_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux);
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
        CeedElemRestrictionCreate(rdy->ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_l,
                                  &restrict_l);
        CeedElemRestrictionCreate(rdy->ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_r,
                                  &restrict_r);
        PetscCall(PetscFree2(offset_l, offset_r));
        if (0) {
          CeedElemRestrictionView(restrict_l, stdout);
          CeedElemRestrictionView(restrict_r, stdout);
        }

        CeedVectorCreate(rdy->ceed, mesh->num_cells * num_comp, &rdy->ceed_rhs.u_local_ceed);
        CeedVectorCreate(rdy->ceed, mesh->num_cells * num_comp, &rdy->ceed_rhs.f_ceed);
      }

      {
        CeedOperator op;
        CeedOperatorCreate(rdy->ceed, qf, NULL, NULL, &op);
        CeedOperatorSetField(op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom);
        CeedOperatorSetField(op, "q_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op, "q_right", restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op, "cell_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op, "cell_right", restrict_r, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op, "flux", restrict_flux, CEED_BASIS_COLLOCATED, flux);
        CeedCompositeOperatorAddSub(rdy->ceed_rhs.op_edges, op);
        op_id++;
        CeedOperatorDestroy(&op);
      }
      CeedElemRestrictionDestroy(&restrict_geom);
      CeedElemRestrictionDestroy(&restrict_flux);
      CeedElemRestrictionDestroy(&restrict_l);
      CeedElemRestrictionDestroy(&restrict_r);
      CeedVectorDestroy(&geom);
      CeedVectorDestroy(&flux);
    }
    for (PetscInt b = 0; b < rdy->num_boundaries; b++) {
      RDyBoundary      *boundary = &rdy->boundaries[b];
      CeedQFunctionUser func;
      const char       *func_loc;
      switch (rdy->boundary_conditions[b].flow->type) {
        case CONDITION_REFLECTING:
          func     = SWEBoundaryFlux_Reflecting_Roe;
          func_loc = SWEBoundaryFlux_Reflecting_Roe_loc;
          break;
        case CONDITION_CRITICAL_OUTFLOW:
          func     = SWEBoundaryFlux_Outflow_Roe;
          func_loc = SWEBoundaryFlux_Outflow_Roe_loc;
          break;
        default:
          PetscCheck(PETSC_FALSE, rdy->comm, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %d\n", rdy->boundary_ids[b]);
      }
      CeedQFunction qf;
      CeedInt       num_comp_geom = 3;
      CeedQFunctionCreateInterior(rdy->ceed, 1, func, func_loc, &qf);
      CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE);
      CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE);

      CeedQFunctionContext qf_context;
      CreateQFunctionContextForSWE(rdy, rdy->ceed, &qf_context);
      if (0) CeedQFunctionContextView(qf_context, stdout);
      CeedQFunctionSetContext(qf, qf_context);
      CeedQFunctionContextDestroy(&qf_context);

      CeedElemRestriction restrict_l, restrict_geom, restrict_flux;
      CeedVector          geom, flux;
      {
        CeedInt num_edges = boundary->num_edges;

        // create element restrictions for left and right input/output states
        CeedInt *offset_l;
        PetscCall(PetscMalloc1(num_edges, &offset_l));

        // create an element restriction for geometric factors that convert
        // fluxes to cell states
        CeedInt num_owned_edges = 0;
        for (CeedInt e = 0; e < boundary->num_edges; e++) {
          PetscInt iedge = boundary->edge_ids[e];
          if (edges->is_owned[iedge]) num_owned_edges++;
        }
        CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
        CeedElemRestrictionCreateStrided(rdy->ceed, num_owned_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom);
        CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL);
        CeedVectorSetValue(geom, 0.0);  // initialize to ensure the arrays is allocated
                                        //
        // create an element restriction for accumulated fluxes
        CeedInt f_strides[] = {num_comp, 1, num_comp};
        CeedElemRestrictionCreateStrided(rdy->ceed, num_owned_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux);
        CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL);
        CeedVectorSetValue(flux, 0.0);

        // create an element restrictions for the "left" (interior) input/output
        // states, populate offsets for these states, and set the (invariant)
        // geometric parameters
        CeedScalar(*g)[3];
        CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g);
        for (CeedInt e = 0, oe = 0; e < num_edges; e++) {
          PetscInt iedge = boundary->edge_ids[e];
          if (!edges->is_owned[iedge]) continue;
          PetscInt l   = edges->cell_ids[2 * iedge];
          offset_l[oe] = l * num_comp;

          g[oe][0] = edges->sn[iedge];
          g[oe][1] = edges->cn[iedge];
          g[oe][2] = -edges->lengths[iedge] / cells->areas[l];
          oe++;
        }
        CeedVectorRestoreArray(geom, (CeedScalar **)&g);

        // create the element restriction for the left cell states
        CeedElemRestrictionCreate(rdy->ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_l,
                                  &restrict_l);
        PetscCall(PetscFree(offset_l));
        if (0) CeedElemRestrictionView(restrict_l, stdout);
      }

      {
        CeedOperator op;
        CeedOperatorCreate(rdy->ceed, qf, NULL, NULL, &op);
        CeedOperatorSetField(op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom);
        CeedOperatorSetField(op, "q_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op, "cell_left", restrict_l, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op, "flux", restrict_flux, CEED_BASIS_COLLOCATED, flux);
        CeedCompositeOperatorAddSub(rdy->ceed_rhs.op_edges, op);
        op_id++;
        CeedOperatorDestroy(&op);
      }
      CeedElemRestrictionDestroy(&restrict_geom);
      CeedElemRestrictionDestroy(&restrict_flux);
      CeedElemRestrictionDestroy(&restrict_l);
      CeedVectorDestroy(&geom);
      CeedVectorDestroy(&flux);
    }

    if (0) CeedOperatorView(rdy->ceed_rhs.op_edges, stdout);
  }

  op_id = -1;
  if (rdy->ceed_resource[0] && !rdy->ceed_rhs.op_src) {
    CeedCompositeOperatorCreate(rdy->ceed, &rdy->ceed_rhs.op_src);
    CeedInt   num_comp = 3;
    RDyMesh  *mesh     = &rdy->mesh;
    RDyCells *cells    = &mesh->cells;

    {
      // source term
      CeedQFunction qf;
      CeedInt       num_comp_geom = 2, num_comp_water_src = 1, num_comp_mannings_n = 1;
      CeedQFunctionCreateInterior(rdy->ceed, 1, SWESourceTerm, SWESourceTerm_loc, &qf);
      CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE);
      CeedQFunctionAddInput(qf, "water_src", num_comp_water_src, CEED_EVAL_NONE);
      CeedQFunctionAddInput(qf, "mannings_n", num_comp_mannings_n, CEED_EVAL_NONE);
      CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE);
      CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE);

      CeedQFunctionContext qf_context;
      CreateQFunctionContextForSWE(rdy, rdy->ceed, &qf_context);
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
        CeedElemRestrictionCreateStrided(rdy->ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom);
        CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL);
        CeedVectorSetValue(geom, 0.0);  // initialize to ensure the arrays is allocated

        CeedInt strides_water_src[] = {num_comp_water_src, 1, num_comp_water_src};
        CeedElemRestrictionCreateStrided(rdy->ceed, num_owned_cells, 1, num_comp_water_src, num_owned_cells * num_comp_water_src, strides_water_src,
                                         &restrict_water_src);
        CeedElemRestrictionCreateVector(restrict_water_src, &water_src, NULL);
        CeedVectorSetValue(water_src, 0.0);  // initialize to ensure the arrays is allocated

        CeedInt strides_mannings_n[] = {num_comp_mannings_n, 1, num_comp_mannings_n};
        CeedElemRestrictionCreateStrided(rdy->ceed, num_owned_cells, 1, num_comp_mannings_n, num_owned_cells * num_comp_mannings_n,
                                         strides_mannings_n, &restrict_mannings_n);
        CeedElemRestrictionCreateVector(restrict_mannings_n, &mannings_n, NULL);
        CeedVectorSetValue(mannings_n, 0.0);  // initialize to ensure the arrays is allocated

        CeedInt strides_riemannf[] = {num_comp, 1, num_comp};
        CeedElemRestrictionCreateStrided(rdy->ceed, num_owned_cells, 1, num_comp, num_owned_cells * num_comp, strides_riemannf, &restrict_riemannf);
        CeedElemRestrictionCreateVector(restrict_riemannf, &riemannf, NULL);
        CeedVectorSetValue(riemannf, 0.0);  // initialize to ensure the arrays is allocated

        PetscCall(PetscMalloc1(num_owned_cells, &offset_c));
        CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g);
        CeedVectorGetArray(mannings_n, CEED_MEM_HOST, (CeedScalar **)&n);
        for (CeedInt c = 0, oc = 0; c < mesh->num_cells; c++) {
          if (!cells->is_local[c]) continue;

          offset_c[oc] = c * num_comp;

          g[oc][0] = cells->dz_dx[c];
          g[oc][1] = cells->dz_dy[c];

          n[oc][0] = rdy->materials_by_cell[c].manning;

          oc++;
        }
        CeedVectorRestoreArray(geom, (CeedScalar **)&g);
        CeedVectorRestoreArray(mannings_n, (CeedScalar **)&n);
        CeedElemRestrictionCreate(rdy->ceed, num_owned_cells, 1, num_comp, 1, num_owned_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, offset_c,
                                  &restrict_c);
        PetscCall(PetscFree(offset_c));
        if (0) CeedElemRestrictionView(restrict_c, stdout);
      }

      {
        CeedOperator op;
        CeedOperatorCreate(rdy->ceed, qf, NULL, NULL, &op);
        CeedOperatorSetField(op, "geom", restrict_geom, CEED_BASIS_COLLOCATED, geom);
        CeedOperatorSetField(op, "water_src", restrict_water_src, CEED_BASIS_COLLOCATED, water_src);
        CeedOperatorSetField(op, "mannings_n", restrict_water_src, CEED_BASIS_COLLOCATED, mannings_n);
        CeedOperatorSetField(op, "riemannf", restrict_riemannf, CEED_BASIS_COLLOCATED, riemannf);
        CeedOperatorSetField(op, "q", restrict_c, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedOperatorSetField(op, "cell", restrict_c, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);
        CeedCompositeOperatorAddSub(rdy->ceed_rhs.op_src, op);
        op_id++;
        rdy->ceed_rhs.water_src_op_id = op_id;
        CeedOperatorDestroy(&op);
      }
      CeedElemRestrictionDestroy(&restrict_geom);
      CeedElemRestrictionDestroy(&restrict_mannings_n);
      CeedElemRestrictionDestroy(&restrict_c);
      CeedVectorDestroy(&geom);
    }

    CeedVectorCreate(rdy->ceed, mesh->num_cells_local * num_comp, &rdy->ceed_rhs.s_ceed);
    CeedVectorCreate(rdy->ceed, mesh->num_cells_local * num_comp, &rdy->ceed_rhs.u_ceed);

    if (0) CeedOperatorView(rdy->ceed_rhs.op_src, stdout);
  }

  PetscFunctionReturn(0);
}

#endif  // SWE_SETUP_CEED_H
