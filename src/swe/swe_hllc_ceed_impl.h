#pragma once
#include <math.h>

CEED_QFUNCTION_HELPER void SWERiemannFlux_HLLC(CeedScalar gravity, CeedScalar tiny_h, CeedScalar h_anuga,
                                              SWEState qL, SWEState qR, CeedScalar sn, CeedScalar cn,
                                              CeedScalar flux[3], CeedScalar *amax) {
  (void)h_anuga; 

  if (qL.h <= tiny_h && qR.h <= tiny_h) {
    flux[0] = flux[1] = flux[2] = 0.0;
    *amax = 0.0;
    return;
  }

  // Geometry: n = (nx, ny)
  const CeedScalar nx = cn, ny = sn;

  // Velocities 
  const CeedScalar uL = SafeDiv(qL.hu, qL.h, fabs(qL.h), RDY_TINY);
  const CeedScalar vL = SafeDiv(qL.hv, qL.h, fabs(qL.h), RDY_TINY);
  const CeedScalar uR = SafeDiv(qR.hu, qR.h, fabs(qR.h), RDY_TINY);
  const CeedScalar vR = SafeDiv(qR.hv, qR.h, fabs(qR.h), RDY_TINY);

  // Normal and tangential velocities
  const CeedScalar unL = uL * nx + vL * ny;
  const CeedScalar utL = -uL * ny + vL * nx;

  const CeedScalar unR = uR * nx + vR * ny;
  const CeedScalar utR = -uR * ny + vR * nx;

  // Wave speeds 
  const CeedScalar cL = sqrt(gravity * fmax(qL.h, 0.0));
  const CeedScalar cR = sqrt(gravity * fmax(qR.h, 0.0));

  const CeedScalar sL = fmin(unL - cL, unR - cR);
  const CeedScalar sR = fmax(unL + cL, unR + cR);

  *amax = fmax(fabs(sL), fabs(sR));

  // Physical normal fluxes
  CeedScalar FL[3], FR[3];
  SWEPhysicalNormalFlux(gravity, qL, sn, cn, FL);
  SWEPhysicalNormalFlux(gravity, qR, sn, cn, FR);

  // Normal momenta m_n = h * u_n
  const CeedScalar mL = qL.hu * nx + qL.hv * ny;
  const CeedScalar mR = qR.hu * nx + qR.hv * ny;

  // Contact wave speed s*
  const CeedScalar denom = (qL.h - qR.h);
  const CeedScalar s_star = SafeDiv((mR - mL) + sL * qL.h - sR * qR.h, denom, fabs(denom), RDY_TINY);

  // Select region
  if (0.0 <= sL) {
    flux[0] = FL[0];
    flux[1] = FL[1];
    flux[2] = FL[2];
    return;
  }

  if (sR <= 0.0) {
    flux[0] = FR[0];
    flux[1] = FR[1];
    flux[2] = FR[2];
    return;
  }

  // Left star region
  if (sL <= 0.0 && 0.0 <= s_star) {
    const CeedScalar h_star  = qL.h * (sL - unL) / (sL - s_star);
    const CeedScalar mn_star = h_star * s_star;
    const CeedScalar mt_star = h_star * utL; 

    // Rotate back to (hu, hv)
    const CeedScalar hu_star = mn_star * nx - mt_star * ny;
    const CeedScalar hv_star = mn_star * ny + mt_star * nx;

    flux[0] = FL[0] + sL * (h_star - qL.h);
    flux[1] = FL[1] + sL * (hu_star - qL.hu);
    flux[2] = FL[2] + sL * (hv_star - qL.hv);
    return;
  }

  // Right star region
  if (s_star <= 0.0 && 0.0 <= sR) {
    const CeedScalar h_star  = qR.h * (sR - unR) / (sR - s_star);
    const CeedScalar mn_star = h_star * s_star;
    const CeedScalar mt_star = h_star * utR; // preserve tangential velocity from right

    const CeedScalar hu_star = mn_star * nx - mt_star * ny;
    const CeedScalar hv_star = mn_star * ny + mt_star * nx;

    flux[0] = FR[0] + sR * (h_star - qR.h);
    flux[1] = FR[1] + sR * (hu_star - qR.hu);
    flux[2] = FR[2] + sR * (hv_star - qR.hv);
    return;
  }

  // Fallback
  flux[0] = 0.0;
  flux[1] = 0.0;
  flux[2] = 0.0;
}
