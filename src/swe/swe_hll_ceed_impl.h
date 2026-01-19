#pragma once
#include <math.h>

CEED_QFUNCTION_HELPER void SWERiemannFlux_HLL(CeedScalar gravity, CeedScalar tiny_h, CeedScalar h_anuga,
                                             SWEState qL, SWEState qR, CeedScalar sn, CeedScalar cn,
                                             CeedScalar flux[3], CeedScalar *amax) {
  (void)h_anuga; 

  if (qL.h <= tiny_h && qR.h <= tiny_h) {
    flux[0] = flux[1] = flux[2] = 0.0;
    *amax = 0.0;
    return;
  }

  const CeedScalar unL = SafeDiv(qL.hu * cn + qL.hv * sn, qL.h, fabs(qL.h), RDY_TINY);
  const CeedScalar unR = SafeDiv(qR.hu * cn + qR.hv * sn, qR.h, fabs(qR.h), RDY_TINY);

  const CeedScalar cL = sqrt(gravity * fmax(qL.h, 0.0));
  const CeedScalar cR = sqrt(gravity * fmax(qR.h, 0.0));

  const CeedScalar sL = fmin(unL - cL, unR - cR);
  const CeedScalar sR = fmax(unL + cL, unR + cR);
  *amax = fmax(fabs(sL), fabs(sR));

  // Physical normal fluxes: Fn = E*nx + G*ny
  CeedScalar FL[3], FR[3];
  SWEPhysicalNormalFlux(gravity, qL, sn, cn, FL);
  SWEPhysicalNormalFlux(gravity, qR, sn, cn, FR);

  if (0.0 <= sL) {
    flux[0] = FL[0];
    flux[1] = FL[1];
    flux[2] = FL[2];
  } else if (sR <= 0.0) {
    flux[0] = FR[0];
    flux[1] = FR[1];
    flux[2] = FR[2];
  } else {
    const CeedScalar denom = SafeDiv(1.0, (sR - sL), fabs(sR - sL), RDY_TINY);
    flux[0] = (sR * FL[0] - sL * FR[0] + sL * sR * (qR.h  - qL.h )) * denom;
    flux[1] = (sR * FL[1] - sL * FR[1] + sL * sR * (qR.hu - qL.hu)) * denom;
    flux[2] = (sR * FL[2] - sL * FR[2] + sL * sR * (qR.hv - qL.hv)) * denom;
  }
}
