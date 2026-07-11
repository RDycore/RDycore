#ifndef SWE_RIEMANN_PETSC_H
#define SWE_RIEMANN_PETSC_H

// Shared PETSc-backend helpers for the SWE Riemann operators. These small
// routines are used by both the standard/HR flux operators in swe_petsc.c and
// the HR + second-order MUSCL operator in swe_hr_muscl_petsc.c.
//
// Include this AFTER the private RDycore headers (rdyoperatorimpl.h /
// rdysweimpl.h) so that RDyMesh / RDyConfig are defined, mirroring how
// swe_roe_flux_petsc.h is included.

#include "swe_types_petsc.h"

// silence unused-function warnings in translation units that use only a subset
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"

static PetscErrorCode CreateRiemannStateData(PetscInt num_states, RiemannStateData *data) {
  PetscFunctionBegin;

  data->num_states = num_states;
  PetscCall(PetscCalloc1(num_states, &data->h));
  PetscCall(PetscCalloc1(num_states, &data->hu));
  PetscCall(PetscCalloc1(num_states, &data->hv));
  PetscCall(PetscCalloc1(num_states, &data->u));
  PetscCall(PetscCalloc1(num_states, &data->v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyRiemannStateData(RiemannStateData data) {
  PetscFunctionBegin;

  data.num_states = 0;
  PetscCall(PetscFree(data.h));
  PetscCall(PetscFree(data.hu));
  PetscCall(PetscFree(data.hv));
  PetscCall(PetscFree(data.u));
  PetscCall(PetscFree(data.v));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateRiemannEdgeData(PetscInt num_edges, PetscInt num_comp, RiemannEdgeData *data) {
  PetscFunctionBegin;

  data->num_edges = num_edges;
  PetscCall(PetscCalloc1(num_edges, &data->cn));
  PetscCall(PetscCalloc1(num_edges, &data->sn));
  PetscCall(PetscCalloc1(num_edges * num_comp, &data->fluxes));
  PetscCall(PetscCalloc1(num_edges, &data->amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyRiemannEdgeData(RiemannEdgeData data) {
  PetscFunctionBegin;

  data.num_edges = 0;
  PetscCall(PetscFree(data.cn));
  PetscCall(PetscFree(data.sn));
  PetscCall(PetscFree(data.fluxes));
  PetscCall(PetscFree(data.amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRiemannVelocities(PetscReal tiny_h, PetscReal h_anuga, RiemannStateData *data) {
  PetscFunctionBeginUser;
  PetscReal denom;

  for (PetscInt n = 0; n < data->num_states; n++) {
    if (data->h[n] < tiny_h) {
      data->u[n] = 0.0;
      data->v[n] = 0.0;
    } else {
      denom      = Square(data->h[n]) + Square(h_anuga);
      data->u[n] = data->hu[n] * data->h[n] / denom;
      data->v[n] = data->hv[n] * data->h[n] / denom;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Computes the per-cell bed elevation zc[] used by the hydrostatic
// reconstruction operators. Allocates *zc (length mesh->num_cells) and fills it
// from the cell_elevation file (cell centroid z) when provided, otherwise from
// the vertex-averaged bed elevation. Caller frees *zc.
static PetscErrorCode ComputeCellBedElevation(RDyMesh *mesh, const RDyConfig config, PetscReal **zc) {
  PetscFunctionBeginUser;

  RDyCells    *cells    = &mesh->cells;
  RDyVertices *vertices = &mesh->vertices;

  PetscCall(PetscCalloc1(mesh->num_cells, zc));
  for (PetscInt c = 0; c < mesh->num_cells; c++) {
    if (config.grid.cell_elevation.file[0]) {
      // if cell elevation is provided via the file
      (*zc)[c] = cells->centroids[c].X[2];
    } else {
      // otherwise, compute vertex-averaged bed elevation
      PetscReal z_sum = 0.0;
      for (PetscInt v = cells->vertex_offsets[c]; v < cells->vertex_offsets[c + 1]; v++) {
        z_sum += vertices->points[cells->vertex_ids[v]].X[2];
      }
      (*zc)[c] = z_sum / (PetscReal)cells->num_vertices[c];
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif  // SWE_RIEMANN_PETSC_H
