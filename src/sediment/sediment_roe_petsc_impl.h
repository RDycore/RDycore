#ifndef SEDIMENT_ROE_PETSC_IMPL_H
#define SEDIMENT_ROE_PETSC_IMPL_H

#include <rdycore.h>

#include "../swe/swe_roe_petsc_impl.h"
#include "sediment_petsc_impl.h"

/// @brief Computes the flux for SWE and sediments across the edge using Roe's approximate Riemann solve
/// @param [in] *datal A SedimentRiemannStateData for values left of the edges
/// @param [in] *datar A SedimentRiemannStateData for values right of the edges
/// @param [in] sn array containing sines of the angles between edges and y-axis
/// @param [in] cn array containing cosines of the angles between edges and y-axis
/// @param [out] fij array containing fluxes through edges
/// @param [out] amax array storing maximum courant number on edges
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ComputeSedimentRoeFlux(SedimentRiemannStateData *datal, SedimentRiemannStateData *datar, const PetscReal *sn,
                                             const PetscReal *cn, PetscReal *fij, PetscReal *amax) {
  PetscFunctionBeginUser;

  PetscReal *hl  = datal->h;
  PetscReal *ul  = datal->u;
  PetscReal *vl  = datal->v;
  PetscReal *cil = datal->ci;

  PetscReal *hr  = datar->h;
  PetscReal *ur  = datar->u;
  PetscReal *vr  = datar->v;
  PetscReal *cir = datar->ci;

  PetscAssert(datal->num_states == datar->num_states, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Size of data left and right of edges is not the same!");

  PetscInt num_states = datal->num_states;
  PetscInt flow_ncomp = datal->num_flow_comp;
  PetscInt sed_ncomp  = datal->num_sediment_comp;
  PetscInt soln_ncomp = flow_ncomp + sed_ncomp;

  PetscInt ci_index_offset;

  PetscReal cihat[MAX_NUM_FIELD_COMPONENTS]                       = {0};
  PetscReal dch[MAX_NUM_FIELD_COMPONENTS]                         = {0};
  PetscReal R[MAX_NUM_FIELD_COMPONENTS][MAX_NUM_FIELD_COMPONENTS] = {0};
  PetscReal A[MAX_NUM_FIELD_COMPONENTS]                           = {0};
  PetscReal dW[MAX_NUM_FIELD_COMPONENTS]                          = {0};
  PetscReal FL[MAX_NUM_FIELD_COMPONENTS]                          = {0};
  PetscReal FR[MAX_NUM_FIELD_COMPONENTS]                          = {0};

  for (PetscInt i = 0; i < num_states; ++i) {
    // compute the eigenspectrum for the shallow water equations
    PetscReal A_swe[3], R_swe[3][3], dW_swe[3], amax_swe;
    ComputeSWERoeEigenspectrum(hl[i], ul[i], vl[i], hr[i], ur[i], vr[i], sn[i], cn[i], A_swe, R_swe, dW_swe, &amax_swe);

    PetscReal duml   = sqrt(hl[i]);
    PetscReal dumr   = sqrt(hr[i]);
    PetscReal dh     = hr[i] - hl[i];
    PetscReal uperpl = ul[i] * cn[i] + vl[i] * sn[i];
    PetscReal uperpr = ur[i] * cn[i] + vr[i] * sn[i];

    ci_index_offset = i * sed_ncomp;
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      cihat[j] = (duml * cil[ci_index_offset + j] + dumr * cir[ci_index_offset + j]) / (duml + dumr);
      dch[j]   = cir[ci_index_offset + j] * hr[i] - cil[ci_index_offset + j] * hl[i];
    }
    if ((i == 3796 || i == 3987 || i == 3991)){
      printf("\n");
      printf(" hl  = %+e, hr  = %+e \n",hl[i],hr[i]);
      printf(" hul = %+e, hur = %+e \n",hl[i]*ul[i],hr[i]*ur[i]);
      printf(" hvl = %+e, hvr = %+e \n",hl[i]*vl[i],hr[i]*vr[i]);
      printf(" hcl = %+e, hcr = %+e \n",hl[i]*cil[ci_index_offset],hr[i]*cir[ci_index_offset]);
      printf(" ul  = %+e, ur  = %+e \n",ul[i],ur[i]);
      printf(" vl  = %+e, vr  = %+e \n",vl[i],vr[i]);
      printf(" cil = %+e, cir = %+e, cihat = %+e \n",cil[ci_index_offset],cir[ci_index_offset],cihat[0]);
      printf(" uperpl = %+e, uperpr = %+e \n",uperpl,uperpr);
      printf(" amax_swe = %+e \n",amax_swe);
    }

    for (PetscInt i = 0; i < 3; ++i) {
      for (PetscInt j = 0; j < 3; ++j) {
        R[i][j] = R_swe[i][j];
      }
    }
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      R[j + 3][0]     = cihat[j];
      R[j + 3][2]     = cihat[j];
      R[j + 3][j + 3] = 1.0;
    }

    A[0] = A_swe[0];
    A[1] = A_swe[1];
    A[2] = A_swe[2];
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      A[j + 3] = A[1];
    }

    dW[0] = dW_swe[0];
    dW[1] = dW_swe[1];
    dW[2] = dW_swe[2];
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      dW[j + 3] = dch[j] - cihat[j] * dh;
    }

    // compute interface fluxes
    FL[0] = uperpl * hl[i];
    FL[1] = ul[i] * uperpl * hl[i] + 0.5 * GRAVITY * hl[i] * hl[i] * cn[i];
    FL[2] = vl[i] * uperpl * hl[i] + 0.5 * GRAVITY * hl[i] * hl[i] * sn[i];

    FR[0] = uperpr * hr[i];
    FR[1] = ur[i] * uperpr * hr[i] + 0.5 * GRAVITY * hr[i] * hr[i] * cn[i];
    FR[2] = vr[i] * uperpr * hr[i] + 0.5 * GRAVITY * hr[i] * hr[i] * sn[i];

    ci_index_offset = i * sed_ncomp;
    for (PetscInt j = 0; j < sed_ncomp; j++) {
      FL[j + 3] = hl[i] * uperpl * cil[ci_index_offset + j];
      FR[j + 3] = hr[i] * uperpr * cir[ci_index_offset + j];
    }

    // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
    for (PetscInt dof1 = 0; dof1 < soln_ncomp; dof1++) {
      for (PetscInt dof2 = 0; dof2 < soln_ncomp; dof2++) {
        if (dof2 == 0) {
          fij[soln_ncomp * i + dof1] = 0.5 * (FL[dof1] + FR[dof1]);
          //printf("dof1 = %d \n",dof1);
        }
        fij[soln_ncomp * i + dof1] = fij[soln_ncomp * i + dof1] - 0.5 * R[dof1][dof2] * A[dof2] * dW[dof2];
        if ((i == 3796 || i == 3987 || i == 3991) && dof1 == soln_ncomp -1) {
        //if ((i == 3796 || i == 3987 || i == 3991)) {
          printf(" dof1 = %d dof2 = %d, R = %+e, A = %e, dW = %+e, fij = %+e \n",dof1,dof2,R[dof1][dof2],A[dof2],dW[dof2],fij[soln_ncomp * i + dof1]);
          if (dof2 == soln_ncomp -1) printf("\n");
        }
      }
    }
    if ((i == 3796 || i == 3987 || i == 3991)){
      printf("fij[0] = %+e\n",fij[soln_ncomp * i + 0]);
      printf("fij[1] = %+e\n",fij[soln_ncomp * i + 1]);
      printf("fij[2] = %+e\n",fij[soln_ncomp * i + 2]);
      printf("fij[3] = %+e\n",fij[soln_ncomp * i + 3]);
      PetscReal hc_l = hl[i] * cil[ci_index_offset];
      PetscReal hc_r = hr[i] * cir[ci_index_offset];
      PetscReal huc_l = hc_l * uperpl;
      PetscReal huc_r = hc_r * uperpr;
      //printf(" FluxC_edge = %e\n", 0.5*( huc_l + huc_r - cihat[0] * (hc_r - hc_l)));
    }

    amax[i] = amax_swe;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
