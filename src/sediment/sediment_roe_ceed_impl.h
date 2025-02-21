#ifndef SEDIMENT_ROE_CEED_IMPL_H
#define SEDIMENT_ROE_CEED_IMPL_H

#include "../swe/swe_ceed_impl.h"

// we disable compiler warnings for implicitly-declared math functions known to
// the JIT compiler
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-function-declaration"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"

/// computes the flux across an edge using Roe's approximate Riemann solver
/// for flow and sediment transport
CEED_QFUNCTION_HELPER void SedimentRiemannFlux_Roe(const CeedScalar gravity, const CeedScalar tiny_h, SedimentState qL, SedimentState qR,
                                                   CeedScalar sn, CeedScalar cn, CeedInt flow_ndof, CeedInt sed_ndof, CeedScalar flux[],
                                                   CeedScalar *amax) {
  const CeedScalar hl = qL.h, hr = qR.h;
  const CeedScalar ul = SafeDiv(qL.hu, hl, hl, tiny_h);
  const CeedScalar vl = SafeDiv(qL.hv, hl, hl, tiny_h);
  CeedScalar       cil[MAX_NUM_SEDIMENT_CLASSES], cir[MAX_NUM_SEDIMENT_CLASSES];
  for (CeedInt j = 0; j < sed_ndof; ++j) {
    cil[j] = SafeDiv(qL.hci[j], hl, hl, tiny_h);
    cir[j] = SafeDiv(qR.hci[j], hr, hl, tiny_h);
  }
  const CeedScalar ur = SafeDiv(qR.hu, hr, hr, tiny_h);
  const CeedScalar vr = SafeDiv(qR.hv, hr, hr, tiny_h);

  // compute the eigenspectrum for the shallow water equations
  CeedScalar A_swe[3], R_swe[3][3], dW_swe[3], amax_swe;
  ComputeSWEEigenspectrum_Roe(hl, ul, vl, hr, ur, vr, sn, cn, gravity, A_swe, R_swe, dW_swe, &amax_swe);

  // compute sediment contributions
  CeedScalar cihat[MAX_NUM_FIELD_COMPONENTS]                       = {0};
  CeedScalar dch[MAX_NUM_FIELD_COMPONENTS]                         = {0};
  CeedScalar dW[MAX_NUM_FIELD_COMPONENTS]                          = {0};
  CeedScalar R[MAX_NUM_FIELD_COMPONENTS][MAX_NUM_FIELD_COMPONENTS] = {0};
  CeedScalar A[MAX_NUM_FIELD_COMPONENTS]                           = {0};
  CeedScalar FL[MAX_NUM_FIELD_COMPONENTS]                          = {0};
  CeedScalar FR[MAX_NUM_FIELD_COMPONENTS]                          = {0};

  CeedScalar duml   = sqrt(hl);
  CeedScalar dumr   = sqrt(hr);
  CeedScalar dh     = hr - hl;
  CeedScalar uperpl = ul * cn + vl * sn;
  CeedScalar uperpr = ur * cn + vr * sn;

  for (CeedInt j = 0; j < sed_ndof; j++) {
    cihat[j] = (duml * cil[j] + dumr * cir[j]) / (duml + dumr);
    dch[j]   = cir[j] * hr - cil[j] * hl;
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R[i][j] = R_swe[i][j];
    }
  }
  for (CeedInt j = 0; j < sed_ndof; j++) {
    R[j + 3][0]     = cihat[j];
    R[j + 3][2]     = cihat[j];
    R[j + 3][j + 3] = 1.0;
  }

  A[0] = A_swe[0];
  A[1] = A_swe[1];
  A[2] = A_swe[2];
  for (CeedInt j = 0; j < sed_ndof; j++) {
    A[j + 3] = A[1];
  }

  dW[0] = dW_swe[0];
  dW[1] = dW_swe[1];
  dW[2] = dW_swe[2];
  for (CeedInt j = 0; j < sed_ndof; j++) {
    dW[j + 3] = dch[j] - cihat[j] * dh;
  }

  // compute interface fluxes
  FL[0] = uperpl * hl;
  FL[1] = ul * uperpl * hl + 0.5 * gravity * hl * hl * cn;
  FL[2] = vl * uperpl * hl + 0.5 * gravity * hl * hl * sn;

  FR[0] = uperpr * hr;
  FR[1] = ur * uperpr * hr + 0.5 * gravity * hr * hr * cn;
  FR[2] = vr * uperpr * hr + 0.5 * gravity * hr * hr * sn;

  for (CeedInt j = 0; j < sed_ndof; j++) {
    FL[j + 3] = hl * uperpl * cil[j];
    FR[j + 3] = hr * uperpr * cir[j];
  }

  // flux = 0.5*(FL + FR - matmul(R,matmul(A,dW))
  CeedInt soln_ncomp = flow_ndof + sed_ndof;
  for (CeedInt dof1 = 0; dof1 < soln_ncomp; dof1++) {
    flux[dof1] = 0.5 * (FL[dof1] + FR[dof1]);

    for (CeedInt dof2 = 0; dof2 < soln_ncomp; dof2++) {
      flux[dof1] = flux[dof1] - 0.5 * R[dof1][dof2] * A[dof2] * dW[dof2];
    }
  }

  // max wave speed
  *amax = amax_swe;
}

#pragma GCC diagnostic   pop
#pragma clang diagnostic pop

#endif
