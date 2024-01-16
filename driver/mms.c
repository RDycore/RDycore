#include <petscsys.h>
#include <rdycore.h>

PetscReal GRAVITY = 9.806;

static const char *help_str =
    "rdycore_mms - a standalone RDycore driver for various MMS problems\n"
    "usage: rdycore_mms [options] <filename.yaml>\n";

static void usage(const char *exe_name) {
  fprintf(stderr, "%s: usage:\n", exe_name);
  fprintf(stderr, "%s <input.yaml>\n\n", exe_name);
}

typedef enum { H, DH_DX, DH_DY, DH_DT, U, DU_DX, DU_DY, DU_DT, V, DV_DX, DV_DY, DV_DT, HU, HV, Z, DZ_DX, DZ_DY, N } DataType;

typedef struct {
  PetscReal Lx, Ly;
  PetscReal h0, u0, v0;
  PetscReal n0;
  PetscReal z0;
  PetscReal t0;
} ProblemData;

static PetscErrorCode Problem1_Init(ProblemData *pdata) {
  PetscFunctionBegin;
  pdata->Lx = 5.0;
  pdata->Ly = 5.0;
  pdata->h0 = 0.005;
  pdata->u0 = 0.025;
  pdata->v0 = 0.025;
  pdata->t0 = 20.0;
  pdata->n0 = 0.01;
  pdata->z0 = pdata->h0 / 2.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Problem1_GetData(ProblemData *pdata, PetscReal t, PetscReal xc, PetscReal yc, DataType dtype, PetscReal *data) {
  PetscFunctionBegin;

  PetscReal Lx = pdata->Lx;
  PetscReal Ly = pdata->Ly;
  PetscReal h0 = pdata->h0;
  PetscReal u0 = pdata->u0;
  PetscReal v0 = pdata->v0;
  PetscReal t0 = pdata->t0;
  PetscReal n0 = pdata->n0;
  PetscReal z0 = pdata->z0;

  PetscReal sin_xc = PetscSinScalar(PETSC_PI * xc / Lx);
  PetscReal cos_xc = PetscCosScalar(PETSC_PI * xc / Lx);
  PetscReal sin_yc = PetscSinScalar(PETSC_PI * yc / Ly);
  PetscReal cos_yc = PetscCosScalar(PETSC_PI * yc / Ly);
  PetscReal exp_t  = PetscExpScalar(t / t0);

  switch (dtype) {
    case H:
      *data = h0 * (1 + sin_xc * sin_yc) * exp_t;
      break;
    case DH_DX:
      *data = PETSC_PI * h0 / Lx * exp_t * sin_yc * cos_xc;
      break;
    case DH_DY:
      *data = PETSC_PI * h0 / Ly * exp_t * sin_xc * cos_yc;
      break;
    case DH_DT:
      *data = h0 / t0 * (1 + sin_xc * sin_yc) * exp_t;
      break;
    case U:
      *data = u0 * cos_xc * sin_yc * exp_t;
      break;
    case DU_DX:
      *data = (-1) * PETSC_PI * u0 / Lx * sin_xc * sin_yc * exp_t;
      break;
    case DU_DY:
      *data = PETSC_PI * u0 / Ly * cos_xc * cos_yc * exp_t;
      break;
    case DU_DT:
      *data = u0 / t0 * cos_xc * sin_yc * exp_t;
      break;
    case V:
      *data = v0 * sin_xc * cos_yc * exp_t;
      break;
    case DV_DX:
      *data = PETSC_PI * v0 / Lx * cos_xc * cos_yc * exp_t;
      break;
    case DV_DY:
      *data = (-1) * PETSC_PI * v0 / Ly * sin_xc * sin_yc * exp_t;
      break;
    case DV_DT:
      *data = v0 / t0 * sin_xc * cos_yc * exp_t;
      break;
    case HU:
      *data = (h0 * (1 + sin_xc * sin_yc) * exp_t) * (u0 * cos_xc * sin_yc * exp_t);
      break;
    case HV:
      *data = (h0 * (1 + sin_xc * sin_yc) * exp_t) * (v0 * sin_xc * cos_yc * exp_t);
      break;
    case Z:
      *data = z0 * sin_xc * sin_yc;
      break;
    case DZ_DX:
      *data = z0 * PETSC_PI / Lx * cos_xc * sin_yc;
      break;
    case DZ_DY:
      *data = z0 * PETSC_PI / Ly * sin_xc * cos_yc;
      break;
    case N:
      *data = n0 * (1.0 + sin_xc * sin_yc);
      break;
    default:
      break;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Problem1_SourceTerm(ProblemData *pdata, PetscReal t, PetscInt ncells, PetscReal *xc, PetscReal *yc, PetscReal *h_source,
                                          PetscReal *hu_source, PetscReal *hv_source) {
  PetscFunctionBegin;

  for (PetscInt icell = 0; icell < ncells; icell++) {
    PetscReal h, dhdx, dhdy, dhdt;
    PetscReal u, dudx, dudy, dudt;
    PetscReal v, dvdx, dvdy, dvdt;
    PetscReal dzdx, dzdy;
    PetscReal n;

    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], H, &h));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DH_DX, &dhdx));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DH_DY, &dhdy));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DH_DT, &dhdt));

    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], U, &u));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DU_DX, &dudx));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DU_DY, &dudy));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DU_DT, &dudt));

    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], V, &v));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DV_DX, &dvdx));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DV_DY, &dvdy));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DV_DT, &dvdt));

    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], N, &n));

    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DZ_DX, &dzdx));
    PetscCall(Problem1_GetData(pdata, t, xc[icell], yc[icell], DZ_DY, &dzdy));

    PetscReal Cd = GRAVITY * (n * n) * PetscPowReal(h, -1.0 / 3.0);

    h_source[icell] = dhdt + u * dhdx + h * dudx + v * dhdy + h * dvdy;

    hu_source[icell] = u * dhdt + h * dudt;
    hu_source[icell] += 2.0 * u * h * dudx + u * u * dhdx + GRAVITY * h * dhdx;
    hu_source[icell] += u * h * dvdy + v * h * dudy + u * v * dhdy;
    hu_source[icell] += dzdx * GRAVITY * h;
    hu_source[icell] += Cd * u * PetscSqrtReal(u * u + v * v);

    hv_source[icell] = v * dhdt + h * dvdt;
    hv_source[icell] += u * h * dvdx + v * h * dudx + u * v * dhdx;
    hv_source[icell] += v * v * dhdy + 2.0 * v * h * dvdy + GRAVITY * h * dhdy;
    hv_source[icell] += dzdy * GRAVITY * h;
    hv_source[icell] += Cd * v * PetscSqrtReal(u * u + v * v);

  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Problem1_DirichletValue(ProblemData *pdata, PetscReal t, PetscInt nedges, PetscReal *xc_bnd_cell, PetscReal *yc_bnd_cell,
                                              PetscReal *xc_edge, PetscReal *yc_edge, PetscReal *h_bnd, PetscReal *hu_bnd, PetscReal *hv_bnd,
                                              PetscReal *bc_values) {
  PetscFunctionBegin;

  for (PetscInt iedge = 0; iedge < nedges; iedge++) {
    PetscReal dx = xc_bnd_cell[iedge] - xc_edge[iedge];
    PetscReal dy = yc_bnd_cell[iedge] - yc_edge[iedge];
    PetscReal xc = xc_edge[iedge] - dx;
    PetscReal yc = yc_edge[iedge] - dy;

    PetscCall(Problem1_GetData(pdata, t, xc, yc, H, &h_bnd[iedge]));

    PetscReal u_bnd, v_bnd;
    PetscCall(Problem1_GetData(pdata, t, xc, yc, U, &u_bnd));
    PetscCall(Problem1_GetData(pdata, t, xc, yc, V, &v_bnd));

    hu_bnd[iedge] = h_bnd[iedge] * u_bnd;
    hv_bnd[iedge] = h_bnd[iedge] * v_bnd;

    bc_values[iedge * 3 + 0] = h_bnd[iedge];
    bc_values[iedge * 3 + 1] = hu_bnd[iedge];
    bc_values[iedge * 3 + 2] = hv_bnd[iedge];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[]) {
  // print usage info if no arguments given
  if (argc < 2) {
    usage(argv[0]);
    exit(-1);
  }

  // initialize subsystems
  PetscCall(RDyInit(argc, argv, help_str));

  PetscMPIInt myrank, commsize;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &myrank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &commsize));

  if (strcmp(argv[1], "-help")) {  // if given a config file

    // create rdycore and set it up with the given file
    MPI_Comm comm = PETSC_COMM_WORLD;
    RDy      rdy;

    PetscCall(RDyCreate(comm, argv[1], &rdy));

    PetscCall(RDySetup(rdy));

    PetscInt   ncells = 0;
    PetscReal *xc_cell, *yc_cell, *area_cell;
    PetscInt  *nat_id_cell;
    PetscReal *h_source, *hu_source, *hv_source;
    PetscReal *h_soln, *hu_soln, *hv_soln;
    PetscReal *h_anal, *hu_anal, *hv_anal;

    PetscInt   nedges = 0;
    PetscReal *xc_edge, *yc_edge;
    PetscReal *xc_bnd_cell, *yc_bnd_cell;
    PetscReal *h_bnd, *hu_bnd, *hv_bnd, *bc_values;
    PetscInt  *nat_id_bnd_cell;

    PetscInt bc_idx = 0;

    // get information about cells
    {
      PetscCall(RDyGetNumLocalCells(rdy, &ncells));

      PetscCalloc1(ncells, &xc_cell);
      PetscCalloc1(ncells, &yc_cell);
      PetscCalloc1(ncells, &area_cell);
      PetscCalloc1(ncells, &nat_id_cell);
      PetscCalloc1(ncells, &h_source);
      PetscCalloc1(ncells, &hu_source);
      PetscCalloc1(ncells, &hv_source);
      PetscCalloc1(ncells, &h_soln);
      PetscCalloc1(ncells, &hu_soln);
      PetscCalloc1(ncells, &hv_soln);
      PetscCalloc1(ncells, &h_anal);
      PetscCalloc1(ncells, &hu_anal);
      PetscCalloc1(ncells, &hv_anal);

      PetscCall(RDyGetLocalCellXCentroids(rdy, ncells, xc_cell));
      PetscCall(RDyGetLocalCellYCentroids(rdy, ncells, yc_cell));
      PetscCall(RDyGetLocalCellNaturalIDs(rdy, ncells, nat_id_cell));
      PetscCall(RDyGetLocalCellAreas(rdy, ncells, area_cell));
    }

    // get information about boundary conditions
    {
      PetscInt nbcs, bc_type;
      PetscCall(RDyGetNumBoundaryConditions(rdy, &nbcs));
      if (nbcs > 0) {
        PetscCall(RDyGetNumBoundaryEdges(rdy, bc_idx, &nedges));
        PetscCall(RDyGetBoundaryConditionFlowType(rdy, bc_idx, &bc_type));
        PetscCheck(bc_type == CONDITION_DIRICHLET, comm, PETSC_ERR_USER, "Only expecting CONDITION_DIRICHLET to be present in the yaml");

        if (nedges > 0) {
          PetscCalloc1(nedges, &xc_edge);
          PetscCalloc1(nedges, &yc_edge);
          PetscCalloc1(nedges, &xc_bnd_cell);
          PetscCalloc1(nedges, &yc_bnd_cell);
          PetscCalloc1(nedges, &nat_id_bnd_cell);
          PetscCalloc1(nedges, &h_bnd);
          PetscCalloc1(nedges, &hu_bnd);
          PetscCalloc1(nedges, &hv_bnd);
          PetscCalloc1(nedges * 3, &bc_values);

          PetscCall(RDyGetBoundaryEdgeXCentroids(rdy, bc_idx, nedges, xc_edge));
          PetscCall(RDyGetBoundaryEdgeYCentroids(rdy, bc_idx, nedges, yc_edge));
          PetscCall(RDyGetBoundaryCellXCentroids(rdy, bc_idx, nedges, xc_bnd_cell));
          PetscCall(RDyGetBoundaryCellYCentroids(rdy, bc_idx, nedges, yc_bnd_cell));
          PetscCall(RDyGetBoundaryCellNaturalIDs(rdy, bc_idx, nedges, nat_id_bnd_cell));
        }
      }
    }

    PetscReal cur_time = 0.0;

    ProblemData pdata;
    PetscCall(Problem1_Init(&pdata));

    PetscReal *mannings_n;
    PetscCalloc1(ncells, &mannings_n);
    for (PetscInt icell = 0; icell < ncells; icell++) {
      PetscCall(Problem1_GetData(&pdata, 0.0, xc_cell[icell], yc_cell[icell], N, &mannings_n[icell]));
    }
    PetscCall(RDySetManningsNForLocalCell(rdy, ncells, mannings_n));

    Vec ic_vec;
    PetscScalar *ic_ptr;
    PetscCall(RDyCreatePrognosticVec(rdy, &ic_vec));

    PetscCall(VecGetArray(ic_vec, &ic_ptr));
    for (PetscInt icell = 0; icell < ncells; icell++) {
      PetscCall(Problem1_GetData(&pdata, 0.0, xc_cell[icell], yc_cell[icell], H, &h_anal[icell]));
      PetscCall(Problem1_GetData(&pdata, 0.0, xc_cell[icell], yc_cell[icell], HU, &hu_anal[icell]));
      PetscCall(Problem1_GetData(&pdata, 0.0, xc_cell[icell], yc_cell[icell], HV, &hv_anal[icell]));
      ic_ptr[icell * 3 + 0] = h_anal[icell];
      ic_ptr[icell * 3 + 1] = hu_anal[icell];
      ic_ptr[icell * 3 + 2] = hv_anal[icell];
    }
    PetscCall(VecRestoreArray(ic_vec, &ic_ptr));
    PetscCall(RDySetInitialConditions(rdy, ic_vec));
    PetscCall(VecDestroy(&ic_vec));

    while (!RDyFinished(rdy)) {
      PetscCall(RDyGetTime(rdy, &cur_time));

      PetscCall(Problem1_SourceTerm(&pdata, cur_time, ncells, xc_cell, yc_cell, h_source, hu_source, hv_source));
      if (nedges > 0) {
        PetscCall(Problem1_DirichletValue(&pdata, cur_time, nedges, xc_bnd_cell, yc_bnd_cell, xc_edge, yc_edge, h_bnd, hu_bnd, hv_bnd, bc_values));
      }

      // set the MMS source terms
      PetscCall(RDySetWaterSourceForLocalCell(rdy, ncells, h_source));
      PetscCall(RDySetXMomentumSourceForLocalCell(rdy, ncells, hu_source));
      PetscCall(RDySetYMomentumSourceForLocalCell(rdy, ncells, hv_source));

      // set dirchlet BC
      if (nedges > 0) {
        PetscCall(RDySetDirichletBoundaryValues(rdy, bc_idx, nedges, 3, bc_values));
      }

      // advance the solution by the coupling interval specified in the config file
      PetscCall(RDyAdvance(rdy));
    }

    PetscCall(RDyGetLocalCellHeights(rdy, ncells, h_soln));
    PetscCall(RDyGetLocalCellXMomentums(rdy, ncells, hu_soln));
    PetscCall(RDyGetLocalCellYMomentums(rdy, ncells, hv_soln));

    PetscReal err[3], err1[3], err1_glb[3], err2[3], err2_glb[3], errm[3], errm_glb[3];
    PetscReal area_cell_sum = 0.0, area_cell_sum_glb = 0.0;

    for (PetscInt idof = 0; idof < 3; idof++) {
      err1[idof] = 0.0;
      err2[idof] = 0.0;
      errm[idof] = 0.0;
    }

    PetscCall(RDyGetTime(rdy, &cur_time));
    for (PetscInt icell = 0; icell < ncells; icell++) {
      PetscCall(Problem1_GetData(&pdata, cur_time, xc_cell[icell], yc_cell[icell], H, &h_anal[icell]));
      PetscCall(Problem1_GetData(&pdata, cur_time, xc_cell[icell], yc_cell[icell], HU, &hu_anal[icell]));
      PetscCall(Problem1_GetData(&pdata, cur_time, xc_cell[icell], yc_cell[icell], HV, &hv_anal[icell]));

      err[0] = PetscAbs(h_soln[icell] - h_anal[icell]);
      err[1] = PetscAbs(hu_soln[icell] - hu_anal[icell]);
      err[2] = PetscAbs(hv_soln[icell] - hv_anal[icell]);

      area_cell_sum += area_cell[icell];

      for (PetscInt idof = 0; idof < 3; idof++) {
        err1[idof] += err[idof] * area_cell[icell];
        err2[idof] += PetscPowReal(err[idof], 2.0) * area_cell[icell];
        errm[idof] = PetscMax(err[idof], errm[idof]);
      }
    }

    PetscMPIInt ncells_glb;
    PetscCall(MPI_Reduce(&ncells, &ncells_glb, 1, MPI_INTEGER, MPI_SUM, 0, PETSC_COMM_WORLD));

    PetscCall(MPI_Reduce(&err1, &err1_glb, 3, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
    PetscCall(MPI_Reduce(&err2, &err2_glb, 3, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
    PetscCall(MPI_Reduce(&errm, &errm_glb, 3, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));

    PetscCall(MPI_Reduce(&area_cell_sum, &area_cell_sum_glb, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));

    if (!myrank) {
      for (PetscInt idof = 0; idof < 3; idof++) {
        err2_glb[idof] = PetscPowReal(err2_glb[idof], 0.5);
      }

      printf("Avg-cell-area    : %18.16f\n",area_cell_sum_glb/ncells_glb);
      printf("Avg-length-scale : %18.16f\n",PetscPowReal(area_cell_sum_glb/ncells_glb, 0.5));

      printf("Error-Norm-1     : ");
      for (PetscInt idof = 0; idof < 3; idof++) printf ("%18.16f ", err1_glb[idof]);
      printf("\n");

      printf("Error-Norm-2     : ");
      for (PetscInt idof = 0; idof < 3; idof++) printf ("%18.16f ", err2_glb[idof]);
      printf("\n");

      printf("Error-Norm-Max   : ");
      for (PetscInt idof = 0; idof < 3; idof++) printf ("%18.16f ", errm_glb[idof]);
      printf("\n");

    }

    // free up memory
    PetscFree(xc_cell);
    PetscFree(yc_cell);
    PetscFree(area_cell);
    PetscFree(h_source);
    PetscFree(hu_source);
    PetscFree(hv_source);
    PetscFree(h_soln);
    PetscFree(hu_soln);
    PetscFree(hv_soln);
    PetscFree(h_anal);
    PetscFree(hu_anal);
    PetscFree(hv_anal);
    if (nedges > 0) {
      PetscFree(xc_edge);
      PetscFree(yc_edge);
      PetscFree(xc_bnd_cell);
      PetscFree(yc_bnd_cell);
      PetscFree(nat_id_bnd_cell);
      PetscFree(h_bnd);
      PetscFree(hu_bnd);
      PetscFree(hv_bnd);
    }
  }

  PetscCall(RDyFinalize());
  return 0;
}