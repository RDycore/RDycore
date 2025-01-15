#ifndef SEDIMENT_PETSC_H
#define SEDIMENT_PETSC_H

#include <petscsys.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

// riemann left and right states
typedef struct {
  PetscInt   num_states;         // number of states
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_sediment_comp;  // number of sediment components
  PetscReal *h, *hu, *hv, *hc;   // prognostic variables
  PetscReal *u, *v, *c;          // diagnostic variables
} SedimentRiemannStateData;

typedef struct {
  PetscInt   num_edges;          // number of edges
  PetscInt   num_flow_comp;      // number of flow components
  PetscInt   num_sediment_comp;  // number of sediment components
  PetscReal *cn, *sn;            // cosine and sine of the angle between edges and y-axis
  PetscReal *fluxes;             // fluxes through the edge
  PetscReal *amax;               // courant number on edges
} SedimentRiemannEdgeData;

static PetscErrorCode CreateSedimentRiemannStateData(PetscInt num_states, PetscInt num_flow_comp, PetscInt num_sediment_comp,
                                                     SedimentRiemannStateData *data) {
  PetscFunctionBegin;

  data->num_states        = num_states;
  data->num_flow_comp     = num_flow_comp;
  data->num_sediment_comp = num_sediment_comp;

  PetscCall(PetscCalloc1(num_states, &data->h));
  PetscCall(PetscCalloc1(num_states, &data->hu));
  PetscCall(PetscCalloc1(num_states, &data->hv));
  PetscCall(PetscCalloc1(num_states, &data->u));
  PetscCall(PetscCalloc1(num_states, &data->v));

  PetscCall(PetscCalloc1(num_states * num_sediment_comp, &data->hc));
  PetscCall(PetscCalloc1(num_states * num_sediment_comp, &data->c));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentRiemannStateData(SedimentRiemannStateData data) {
  PetscFunctionBegin;

  data.num_states        = 0;
  data.num_flow_comp     = 0;
  data.num_sediment_comp = 0;
  PetscCall(PetscFree(data.h));
  PetscCall(PetscFree(data.hu));
  PetscCall(PetscFree(data.hv));
  PetscCall(PetscFree(data.hc));
  PetscCall(PetscFree(data.u));
  PetscCall(PetscFree(data.v));
  PetscCall(PetscFree(data.c));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentRiemannEdgeData(SedimentRiemannEdgeData data) {
  PetscFunctionBegin;

  data.num_edges         = 0;
  data.num_flow_comp     = 0;
  data.num_sediment_comp = 0;

  PetscCall(PetscFree(data.cn));
  PetscCall(PetscFree(data.sn));
  PetscCall(PetscFree(data.fluxes));
  PetscCall(PetscFree(data.amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSedimentRiemannEdgeData(PetscInt num_edges, PetscInt num_flow_comp, PetscInt num_sediment_comp,
                                                    SedimentRiemannEdgeData *data) {
  PetscFunctionBegin;

  data->num_edges         = num_edges;
  data->num_flow_comp     = num_flow_comp;
  data->num_sediment_comp = num_sediment_comp;

  PetscCall(PetscCalloc1(num_edges, &data->cn));
  PetscCall(PetscCalloc1(num_edges, &data->sn));

  PetscCall(PetscCalloc1(num_edges * (num_flow_comp + num_sediment_comp), &data->fluxes));
  PetscCall(PetscCalloc1(num_edges, &data->amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeRiemannVelocitiesAndConcentration(PetscReal tiny_h, SedimentRiemannStateData *data) {
  PetscFunctionBeginUser;

  PetscInt index;
  for (PetscInt n = 0; n < data->num_states; n++) {
    if (data->h[n] < tiny_h) {
      data->u[n] = 0.0;
      data->v[n] = 0.0;
      for (PetscInt s = 0; s < data->num_sediment_comp; s++) {
        index          = n * data->num_sediment_comp + s;
        data->c[index] = 0.0;
      }
    } else {
      data->u[n] = data->hu[n] / data->h[n];
      data->v[n] = data->hv[n] / data->h[n];
      for (PetscInt s = 0; s < data->num_sediment_comp; s++) {
        index          = n * data->num_sediment_comp + s;
        data->c[index] = data->hc[index] / data->h[n];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Interior Flux Operator
//------------------------

typedef struct {
  RDyMesh                 *mesh;          // domain mesh
  PetscReal                tiny_h;        // minimum water height for wet conditions
  SedimentRiemannStateData left_states;   // "left" riemann states on interior edges
  SedimentRiemannStateData right_states;  // "right" riemann states on interior edges
  SedimentRiemannEdgeData  edges;         // riemann fluxes on interior edges
  OperatorDiagnostics     *diagnostics;   // courant number, etc
} SedimentInteriorFluxOperator;

static PetscErrorCode ApplySedimentInteriorFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Add code in ApplySedimentInteriorFlux");

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentInteriorFlux(void *context) {
  PetscFunctionBegin;

  SedimentInteriorFluxOperator *interior_flux_op = context;

  DestroySedimentRiemannStateData(interior_flux_op->left_states);
  DestroySedimentRiemannStateData(interior_flux_op->right_states);
  DestroySedimentRiemannEdgeData(interior_flux_op->edges);

  PetscCall(PetscFree(interior_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSedimentPetscInteriorFluxOperator(RDyMesh *mesh, PetscInt num_flow_comp, PetscInt num_sediment_comp,
                                                       OperatorDiagnostics *diagnostics, PetscReal tiny_h, PetscOperator *petsc_op) {
  PetscFunctionBegin;

  SedimentInteriorFluxOperator *interior_flux_op;
  PetscCall(PetscCalloc1(1, &interior_flux_op));
  *interior_flux_op = (SedimentInteriorFluxOperator){
      .mesh        = mesh,
      .diagnostics = diagnostics,
      .tiny_h      = tiny_h,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateSedimentRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_sediment_comp, &interior_flux_op->left_states));
  PetscCall(CreateSedimentRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_sediment_comp, &interior_flux_op->right_states));
  PetscCall(CreateSedimentRiemannEdgeData(mesh->num_internal_edges, num_flow_comp, num_sediment_comp, &interior_flux_op->edges));

  // copy mesh geometry data into place
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id       = edges->internal_edge_ids[e];
    PetscInt right_cell_id = edges->cell_ids[2 * edge_id + 1];

    if (right_cell_id != -1) {
      interior_flux_op->edges.cn[e] = edges->cn[edge_id];
      interior_flux_op->edges.sn[e] = edges->sn[edge_id];
    }
  }

  PetscCall(PetscOperatorCreate(interior_flux_op, ApplySedimentInteriorFlux, DestroySedimentInteriorFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Boundary Flux Operator
//------------------------

typedef struct {
  RDyMesh                 *mesh;                // domain mesh
  RDyBoundary              boundary;            // boundary associated with this sub-operator
  RDyCondition             boundary_condition;  // boundary condition associated with this sub-operator
  Vec                      boundary_values;     // Dirichlet boundary values vector
  Vec                      boundary_fluxes;     // boundary flux values vector
  OperatorDiagnostics     *diagnostics;         // courant number, boundary fluxes
  PetscReal                tiny_h;              // minimum water height for wet conditions
  SedimentRiemannStateData left_states;
  SedimentRiemannStateData right_states;
  SedimentRiemannEdgeData  edges;
  PetscReal               *cosines, *sines;  // cosine and sine of the angle between the edge and y-axis
  PetscReal               *a_max;            // maximum courant number
} SedimentBoundaryFluxOperator;

static PetscErrorCode ApplySedimentBoundaryFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Add code in ApplySedimentBoundaryFlux");

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentBoundaryFlux(void *context) {
  PetscFunctionBegin;

  SedimentBoundaryFluxOperator *boundary_flux_op = context;

  DestroySedimentRiemannStateData(boundary_flux_op->left_states);
  DestroySedimentRiemannStateData(boundary_flux_op->right_states);
  DestroySedimentRiemannEdgeData(boundary_flux_op->edges);

  PetscCall(PetscFree(boundary_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSedimentPetscBoundaryFluxOperator(RDyMesh *mesh, PetscInt num_flow_comp, PetscInt num_sediment_comp, RDyBoundary boundary,
                                                       RDyCondition boundary_condition, Vec boundary_values, Vec boundary_fluxes,
                                                       OperatorDiagnostics *diagnostics, PetscReal tiny_h, PetscOperator *petsc_op) {
  PetscFunctionBegin;
  SedimentBoundaryFluxOperator *boundary_flux_op;
  PetscCall(PetscCalloc1(1, &boundary_flux_op));
  *boundary_flux_op = (SedimentBoundaryFluxOperator){
      .mesh               = mesh,
      .boundary           = boundary,
      .boundary_condition = boundary_condition,
      .boundary_values    = boundary_values,
      .boundary_fluxes    = boundary_fluxes,
      .diagnostics        = diagnostics,
      .tiny_h             = tiny_h,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateSedimentRiemannStateData(boundary.num_edges, num_flow_comp, num_sediment_comp, &boundary_flux_op->left_states));
  PetscCall(CreateSedimentRiemannStateData(boundary.num_edges, num_flow_comp, num_sediment_comp, &boundary_flux_op->right_states));
  PetscCall(CreateSedimentRiemannEdgeData(boundary.num_edges, num_flow_comp, num_sediment_comp, &boundary_flux_op->edges));

  // copy mesh geometry data into place
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id              = boundary.edge_ids[e];
    boundary_flux_op->edges.cn[e] = edges->cn[edge_id];
    boundary_flux_op->edges.sn[e] = edges->sn[edge_id];
  }
  PetscCall(PetscOperatorCreate(boundary_flux_op, ApplySedimentBoundaryFlux, DestroySedimentBoundaryFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//-----------------
// Source Operator
//-----------------

typedef struct {
  RDyMesh  *mesh;               // domain mesh
  PetscInt  num_flow_comp;      // number of flow components
  PetscInt  num_sediment_comp;  // number of sediment components
  Vec       external_sources;   // external source vector
  Vec       mannings;           // mannings coefficient vector
  PetscReal tiny_h;             // minimum water height for wet conditions
  PetscReal xq2018_threshold;   // threshold for the XQ2018's implicit time integration of source term
} SedimentSourceOperator;

static PetscErrorCode ApplySedimentSourceSemiImplicit(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Add code in ApplySedimentSourceSemiImplicit");

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroySedimentSource(void *context) {
  PetscFunctionBegin;
  SedimentSourceOperator *source_op = context;
  PetscFree(source_op);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateSedimentPetscSourceOperator(RDyMesh *mesh, PetscInt num_flow_comp, PetscInt num_sediment_comp, Vec external_sources,
                                                 Vec mannings, RDyFlowSourceMethod method, PetscReal tiny_h, PetscReal xq2018_threshold,
                                                 PetscOperator *petsc_op) {
  PetscFunctionBegin;
  SedimentSourceOperator *source_op;
  PetscCall(PetscCalloc1(1, &source_op));
  *source_op = (SedimentSourceOperator){
      .mesh              = mesh,
      .num_flow_comp     = num_flow_comp,
      .num_sediment_comp = num_sediment_comp,
      .external_sources  = external_sources,
      .mannings          = mannings,
      .tiny_h            = tiny_h,
      .xq2018_threshold  = xq2018_threshold,
  };

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)external_sources, &comm));

  switch (method) {
    case SOURCE_SEMI_IMPLICIT:
      PetscCall(PetscOperatorCreate(source_op, ApplySedimentSourceSemiImplicit, DestroySedimentSource, petsc_op));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only semi_implicit and implicit_xq2018 are supported in the PETSc version");
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif