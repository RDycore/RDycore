#ifndef SOURCE_QFUNCTION_HELPERS_H
#define SOURCE_QFUNCTION_HELPERS_H

#ifndef Square
#define Square(x) ((x) * (x))
#endif

#ifndef SafeDiv
#define SafeDiv(a, b, c, tiny) ((c) > (tiny) ? (a) / (b) : 0.0)
#endif

CEED_QFUNCTION_HELPER int BedElevationSlopeTerm(CeedScalar gravity, CeedScalar h, CeedScalar dz_dx,
                                                CeedScalar dz_dy, CeedScalar *bedx, CeedScalar *bedy) {
  *bedx = dz_dx * gravity * h;
  *bedy = dz_dy * gravity * h;
  return 0;
}

CEED_QFUNCTION_HELPER int SemiImplicitBedFrictionRoughnessTerm(CeedScalar gravity, CeedScalar mannings_n,
                                                               CeedScalar h, CeedScalar u, CeedScalar v,
                                                               CeedScalar dt, CeedScalar F_riemann[3],
                                                               CeedScalar *tbx, CeedScalar *tby) {
  const CeedScalar Cd = gravity * Square(mannings_n) * pow(h, -1.0 / 3.0);
  const CeedScalar velocity_mag = sqrt(Square(velocity[0]) + Square(velocity[1]));
  const CeedScalar tb = Cd * velocity / h;
  const CeedScalar factor = tb / (1.0 + dt * tb);

  const CeedScalar Fsum_x = F_riemann[1];
  const CeedScalar Fsum_y = F_riemann[2];

  source[0] = (hu + dt * Fsum_x - dt * bedx) * factor;
  source[1] = (hv + dt * Fsum_y - dt * bedy) * factor;
  source[0] = elevation_slope[0] * gravity * h;
  source[1] = elevation_slope[1] * gravity * h;
  return 0;
}

#endif
