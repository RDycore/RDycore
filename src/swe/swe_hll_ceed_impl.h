#pragma once

#include <math.h>
#include "rdycore.h"
// HLL Riemann flux implementation for RDycore SWE
CEED_QFUNCTION_HELPER void SWERiemannFlux_HLL(CeedScalar gravity, CeedScalar tiny_h, CeedScalar h_anuga,
                                              SWEState qL, SWEState qR, CeedScalar sn, CeedScalar cn,
                                              CeedScalar flux[3], CeedScalar *amax) {
  const CeedScalar unL = (qL.hu * cn + qL.hv * sn) / qL.h;
  const CeedScalar unR = (qR.hu * cn + qR.hv * sn) / qR.h;

  const CeedScalar cL = sqrt(gravity * qL.h);
  const CeedScalar cR = sqrt(gravity * qR.h);

  const CeedScalar sL = fmin(unL - cL, unR - cR);
  const CeedScalar sR = fmax(unL + cL, unR + cR);
  *amax = fmax(fabs(sL), fabs(sR));

  // Inviscid fluxes
  CeedScalar FL[3], FR[3];
  FL[0] = qL.hu * cn + qL.hv * sn;
  FL[1] = (qL.hu * unL + 0.5 * gravity * qL.h * qL.h) * cn;
  FL[2] = (qL.hv * unL + 0.5 * gravity * qL.h * qL.h) * sn;

  FR[0] = qR.hu * cn + qR.hv * sn;
  FR[1] = (qR.hu * unR + 0.5 * gravity * qR.h * qR.h) * cn;
  FR[2] = (qR.hv * unR + 0.5 * gravity * qR.h * qR.h) * sn;

  // Final flux
  if (0 <= sL) {
    flux[0] = FL[0];
    flux[1] = FL[1];
    flux[2] = FL[2];
  } else if (sR <= 0) {
    flux[0] = FR[0];
    flux[1] = FR[1];
    flux[2] = FR[2];
  } else {
    for (int i = 0; i < 3; i++) {
      const CeedScalar delta_q = (i == 0 ? qR.h - qL.h :
                                  i == 1 ? qR.hu - qL.hu :
                                           qR.hv - qL.hv);
      flux[i] = (sR * FL[i] - sL * FR[i] + sL * sR * delta_q) / (sR - sL);
    }
  }
}
