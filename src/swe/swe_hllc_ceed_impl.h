#pragma once
#include "rdycore.h"

CEED_QFUNCTION_HELPER void SWERiemannFlux_HLLC(CeedScalar gravity, CeedScalar tiny_h, CeedScalar h_anuga,
                                               SWEState qL, SWEState qR, CeedScalar sn, CeedScalar cn,
                                               CeedScalar flux[3], CeedScalar *amax) {
  const CeedScalar unL = (qL.hu * cn + qL.hv * sn) / qL.h;
  const CeedScalar unR = (qR.hu * cn + qR.hv * sn) / qR.h;

  const CeedScalar cL = sqrt(gravity * qL.h);
  const CeedScalar cR = sqrt(gravity * qR.h);

  const CeedScalar sL = fmin(unL - cL, unR - cR);
  const CeedScalar sR = fmax(unL + cL, unR + cR);
  const CeedScalar s_star = (qR.hu - qL.hu + sL * qL.h - sR * qR.h) / (qL.h - qR.h);

  *amax = fmax(fabs(sL), fabs(sR));

  // Compute fluxes left/right
  CeedScalar FL[3], FR[3];
  FL[0] = qL.hu * cn + qL.hv * sn;
  FL[1] = (qL.hu * unL + 0.5 * gravity * qL.h * qL.h) * cn;
  FL[2] = (qL.hv * unL + 0.5 * gravity * qL.h * qL.h) * sn;

  FR[0] = qR.hu * cn + qR.hv * sn;
  FR[1] = (qR.hu * unR + 0.5 * gravity * qR.h * qR.h) * cn;
  FR[2] = (qR.hv * unR + 0.5 * gravity * qR.h * qR.h) * sn;

  if (0 <= sL) {
    flux[0] = FL[0];
    flux[1] = FL[1];
    flux[2] = FL[2];
  } else if (sL <= 0 && 0 <= s_star) {
    const CeedScalar h_star = qL.h * (sL - unL) / (sL - s_star);
    const CeedScalar hu_star = h_star * s_star * cn;
    const CeedScalar hv_star = h_star * s_star * sn;

    flux[0] = FL[0] + sL * (h_star - qL.h);
    flux[1] = FL[1] + sL * (hu_star - qL.hu);
    flux[2] = FL[2] + sL * (hv_star - qL.hv);
  } else if (s_star <= 0 && 0 <= sR) {
    const CeedScalar h_star = qR.h * (sR - unR) / (sR - s_star);
    const CeedScalar hu_star = h_star * s_star * cn;
    const CeedScalar hv_star = h_star * s_star * sn;

    flux[0] = FR[0] + sR * (h_star - qR.h);
    flux[1] = FR[1] + sR * (hu_star - qR.hu);
    flux[2] = FR[2] + sR * (hv_star - qR.hv);
  } else {
    flux[0] = FR[0];
    flux[1] = FR[1];
    flux[2] = FR[2];
  }
}
