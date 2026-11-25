// Unit tests for the analytic Jacobians used by the fully implicit SWE solver:
//   - SWERiemannFluxJacobian_Roe (src/swe/swe_roe_flux_jacobian_ceed.h)
//   - SWEEdgeFluxJacobian_HydroRecon / _AugmentedRoe (src/swe/swe_edge_flux_jacobian_ceed.h),
//     checked against the residual Q-functions SWEFlux_HydroRecon_Roe / SWEFlux_AugmentedRoe
//   - SWESourceJacobian, SWEReflectingStateJacobian, SWECriticalOutflowStateJacobian
//     (src/swe/swe_sources_jacobian_ceed.h)
// The references are (a) the residual kernels themselves (flux values must match
// SWERiemannFlux_Roe exactly), (b) central finite differences of those kernels on
// states away from the non-smooth points of the Roe flux, and (c) the closed-form
// physical flux Jacobian of paper Section 6.1 in the consistent case U_L = U_R.
// See https://cmocka.org for details about the CMocka testing framework.

#include <math.h>
#include <rdycore.h>

#ifndef Square
#define Square(x) ((x) * (x))
#endif
#ifndef SafeDiv
#define SafeDiv(a, b, c, tiny) ((c) > (tiny) ? (a) / (b) : 0.0)
#endif

#include "../swe/swe_types_ceed.h"
// clang-format off
#include "../swe/swe_roe_flux_ceed.h"
// clang-format on
#include "../swe/swe_edge_flux_jacobian_ceed.h"
#include "../swe/swe_fluxes_augmented_roe_ceed.h"
#include "../swe/swe_fluxes_hydro_recon_ceed.h"
#include "../swe/swe_sources_jacobian_ceed.h"
#include "rdycore_tests.h"

static const PetscReal G      = 9.806;
static const PetscReal TINY_H = 1e-7;

// deterministic pseudo-random numbers in [0, 1)
static unsigned long long rng_state = 12345ULL;
static PetscReal          Rand01(void) {
           rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
           return (PetscReal)((rng_state >> 11) * (1.0 / 9007199254740992.0));
}
static PetscReal RandRange(PetscReal a, PetscReal b) { return a + (b - a) * Rand01(); }

static void SetComp(SWEState *q, PetscInt k, PetscReal val) {
  if (k == 0) q->h = val;
  else if (k == 1) q->hu = val;
  else q->hv = val;
}
static PetscReal GetComp(SWEState q, PetscInt k) { return (k == 0) ? q.h : ((k == 1) ? q.hu : q.hv); }

static void RoeFlux(PetscReal h_anuga, SWEState qL, SWEState qR, PetscReal sn, PetscReal cn, PetscReal flux[3]) {
  PetscReal amax;
  SWERiemannFlux_Roe(G, TINY_H, h_anuga, qL, qR, sn, cn, 0.0, flux, &amax);
}

// Checks one column of the flux Jacobian (derivative w.r.t. component k of side
// 0=left/1=right) against central differences. Columns whose one-sided differences
// disagree (a kink of |.|, max(0,.) or the entropy fix lies inside the stencil) are
// skipped; returns 1 if the column was checked, 0 if skipped.
static int CheckFluxColumn(PetscReal h_anuga, SWEState qL, SWEState qR, PetscReal sn, PetscReal cn, PetscInt side, PetscInt k, PetscReal dF[3][3]) {
  SWEState  q   = side ? qR : qL;
  PetscReal x   = GetComp(q, k);
  PetscReal eps = 1e-6 * fmax(1.0, fabs(x)) * (k == 0 ? fmax(q.h, 1e-3) : 1.0);
  PetscReal fp[3], fm[3], f0[3];

  SWEState qp = q, qm = q;
  SetComp(&qp, k, x + eps);
  SetComp(&qm, k, x - eps);
  RoeFlux(h_anuga, side ? qL : qp, side ? qp : qR, sn, cn, fp);
  RoeFlux(h_anuga, side ? qL : qm, side ? qm : qR, sn, cn, fm);
  RoeFlux(h_anuga, qL, qR, sn, cn, f0);

  for (PetscInt i = 0; i < 3; i++) {
    PetscReal fwd = (fp[i] - f0[i]) / eps, bwd = (f0[i] - fm[i]) / eps;
    if (fabs(fwd - bwd) > 1e-4 * fmax(1.0, fabs(fwd))) return 0;  // kink inside the stencil
  }
  for (PetscInt i = 0; i < 3; i++) {
    PetscReal fd    = (fp[i] - fm[i]) / (2.0 * eps);
    PetscReal scale = fmax(1.0, fabs(fd));
    if (fabs(fd - dF[i][k]) > 1e-6 * scale) {
      print_message("flux Jacobian mismatch: side %d, row %d, col %d: analytic %.12e, FD %.12e\n", (int)side, (int)i, (int)k, dF[i][k], fd);
      fail();
    }
  }
  return 1;
}

static void RandomWetState(SWEState *q) {
  q->h  = RandRange(0.01, 3.0);
  q->hu = q->h * RandRange(-3.0, 3.0);
  q->hv = q->h * RandRange(-3.0, 3.0);
}

// the Jacobian routine returns the same flux as the residual kernel, and its
// derivatives agree with central differences of that kernel
static void TestRoeFluxJacobianVsFD(void **state) {
  PetscReal h_anugas[2] = {0.0, 0.05};
  int       checked = 0, total = 0;
  for (int a = 0; a < 2; a++) {
    for (int trial = 0; trial < 400; trial++) {
      SWEState qL, qR;
      RandomWetState(&qL);
      RandomWetState(&qR);
      if (trial % 5 == 0) {  // nearby states (smooth, small jumps)
        qR.h  = qL.h * RandRange(0.9, 1.1);
        qR.hu = qL.hu + RandRange(-0.1, 0.1);
        qR.hv = qL.hv + RandRange(-0.1, 0.1);
      }
      PetscReal theta = RandRange(0.0, 2.0 * PETSC_PI);
      PetscReal sn = sin(theta), cn = cos(theta);

      PetscReal flux[3], flux_ref[3], dFL[3][3], dFR[3][3], amax;
      SWERiemannFluxJacobian_Roe(G, TINY_H, h_anugas[a], qL, qR, sn, cn, flux, dFL, dFR, &amax);
      RoeFlux(h_anugas[a], qL, qR, sn, cn, flux_ref);
      for (int i = 0; i < 3; i++) assert_true(fabs(flux[i] - flux_ref[i]) <= 1e-13 * fmax(1.0, fabs(flux_ref[i])));

      for (PetscInt k = 0; k < 3; k++) {
        checked += CheckFluxColumn(h_anugas[a], qL, qR, sn, cn, 0, k, dFL);
        checked += CheckFluxColumn(h_anugas[a], qL, qR, sn, cn, 1, k, dFR);
        total += 2;
      }
    }
  }
  print_message("Roe flux Jacobian: %d of %d columns checked against FD (rest straddle a kink)\n", checked, total);
  assert_true(checked > total / 2);
}

// in the consistent case U_L = U_R the dissipation vanishes to first order... but not
// its derivative; the sum dF/dU_L + dF/dU_R must equal the physical flux Jacobian of
// paper Section 6.1, because d/dU F(U, U) = dF_perp/dU for a consistent flux
static void TestRoeFluxJacobianConsistency(void **state) {
  for (int trial = 0; trial < 100; trial++) {
    SWEState q;
    RandomWetState(&q);
    PetscReal theta = RandRange(0.0, 2.0 * PETSC_PI);
    PetscReal sn = sin(theta), cn = cos(theta);

    PetscReal flux[3], dFL[3][3], dFR[3][3], amax;
    SWERiemannFluxJacobian_Roe(G, TINY_H, 0.0, q, q, sn, cn, flux, dFL, dFR, &amax);

    PetscReal q1 = q.h, q2 = q.hu, q3 = q.hv;
    PetscReal J[3][3] = {
        {0.0,                                                                cn,                                sn                               },
        {-q2 * q2 * cn / (q1 * q1) - q2 * q3 * sn / (q1 * q1) + G * q1 * cn, 2.0 * q2 * cn / q1 + q3 * sn / q1, q2 * sn / q1                     },
        {-q2 * q3 * cn / (q1 * q1) - q3 * q3 * sn / (q1 * q1) + G * q1 * sn, q3 * cn / q1,                      q2 * cn / q1 + 2.0 * q3 * sn / q1},
    };
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        PetscReal sum = dFL[i][j] + dFR[i][j];
        if (fabs(sum - J[i][j]) > 1e-10 * fmax(1.0, fabs(J[i][j]))) {
          print_message("consistency mismatch (%d,%d): dF/dUL + dF/dUR = %.12e, paper 6.1 = %.12e\n", i, j, sum, J[i][j]);
          fail();
        }
      }
    }
  }
}

// cell source S(U) = [0, -g h zx - tbx, -g h zy - tby], friction as in SWEFullyImplicitBedFrictionRoughness
static void Source(PetscReal h_anuga, SWEState q, PetscReal zx, PetscReal zy, PetscReal n, PetscReal S[3]) {
  PetscReal tbx = 0.0, tby = 0.0;
  if (q.h > TINY_H) {
    PetscReal denom = q.h * q.h + h_anuga * h_anuga;
    PetscReal u = q.hu * q.h / denom, v = q.hv * q.h / denom;
    PetscReal tb = G * n * n * pow(q.h, -1.0 / 3.0) * sqrt(u * u + v * v) / q.h;
    tbx          = q.hu * tb;
    tby          = q.hv * tb;
  }
  S[0] = 0.0;
  S[1] = -G * q.h * zx - tbx;
  S[2] = -G * q.h * zy - tby;
}

static void TestSourceJacobianVsFD(void **state) {
  PetscReal h_anugas[2] = {0.0, 0.05};
  for (int a = 0; a < 2; a++) {
    for (int trial = 0; trial < 200; trial++) {
      SWEState q;
      RandomWetState(&q);
      PetscReal zx = RandRange(-0.1, 0.1), zy = RandRange(-0.1, 0.1), n = RandRange(0.0, 0.1);
      PetscReal dS[3][3];
      SWESourceJacobian(G, TINY_H, h_anugas[a], q, zx, zy, n, dS);
      for (PetscInt k = 0; k < 3; k++) {
        PetscReal x   = GetComp(q, k);
        PetscReal eps = 1e-7 * fmax(1.0, fabs(x));
        SWEState  qp = q, qm = q;
        SetComp(&qp, k, x + eps);
        SetComp(&qm, k, x - eps);
        PetscReal Sp[3], Sm[3];
        Source(h_anugas[a], qp, zx, zy, n, Sp);
        Source(h_anugas[a], qm, zx, zy, n, Sm);
        for (int i = 0; i < 3; i++) {
          PetscReal fd = (Sp[i] - Sm[i]) / (2.0 * eps);
          if (fabs(fd - dS[i][k]) > 1e-6 * fmax(1.0, fabs(fd))) {
            print_message("source Jacobian mismatch (%d,%d): analytic %.12e, FD %.12e\n", i, (int)k, dS[i][k], fd);
            fail();
          }
        }
      }
    }
  }

  // at rest (|q| = 0) the momentum columns of the friction Jacobian vanish
  SWEState  q0 = {1.0, 0.0, 0.0};
  PetscReal dS[3][3];
  SWESourceJacobian(G, TINY_H, 0.0, q0, 0.0, 0.0, 0.05, dS);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) assert_true(dS[i][j] == 0.0);
}

// boundary state maps U_R(U_L) and the chain rule dF/dU_L + dF/dU_R dU_R/dU_L
static void BoundaryFlux(int type, SWEState qL, PetscReal sn, PetscReal cn, PetscReal F[3]) {
  SWEState qR;
  if (type == 0) {  // reflecting
    PetscReal dum1 = sn * sn - cn * cn, dum2 = 2.0 * sn * cn;
    qR = (SWEState){qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1};
  } else {  // critical outflow
    PetscReal q = qL.hu * cn + qL.hv * sn, hR = pow(q * q / G, 1.0 / 3.0), s = sqrt(G * hR);
    qR = (SWEState){hR, hR * s * cn, hR * s * sn};
  }
  RoeFlux(0.0, qL, qR, sn, cn, F);
}

static void TestBoundaryJacobianVsFD(void **state) {
  int checked = 0;
  for (int type = 0; type < 2; type++) {
    for (int trial = 0; trial < 200; trial++) {
      SWEState  qL;
      PetscReal theta = RandRange(0.0, 2.0 * PETSC_PI);
      PetscReal sn = sin(theta), cn = cos(theta);
      RandomWetState(&qL);
      if (type == 1) {  // outflow: flow toward the boundary (q > 0) and away from q = 0
        PetscReal q = qL.hu * cn + qL.hv * sn;
        if (q < 0.05) {
          qL.hu += (0.05 - q + RandRange(0.0, 1.0)) * cn;
          qL.hv += (0.05 - q + RandRange(0.0, 1.0)) * sn;
        }
      }
      SWEState  qR;
      PetscReal M[3][3];
      if (type == 0) {
        PetscReal dum1 = sn * sn - cn * cn, dum2 = 2.0 * sn * cn;
        qR = (SWEState){qL.h, qL.hu * dum1 - qL.hv * dum2, -qL.hu * dum2 - qL.hv * dum1};
        SWEReflectingStateJacobian(sn, cn, M);
      } else {
        PetscReal q = qL.hu * cn + qL.hv * sn, hR = pow(q * q / G, 1.0 / 3.0), s = sqrt(G * hR);
        qR = (SWEState){hR, hR * s * cn, hR * s * sn};
        SWECriticalOutflowStateJacobian(G, qL, sn, cn, M);
      }
      PetscReal flux[3], dFL[3][3], dFR[3][3], dF[3][3], amax;
      SWERiemannFluxJacobian_Roe(G, TINY_H, 0.0, qL, qR, sn, cn, flux, dFL, dFR, &amax);
      SWEBoundaryFluxTotalJacobian(dFL, dFR, M, dF);

      for (PetscInt k = 0; k < 3; k++) {
        PetscReal x   = GetComp(qL, k);
        PetscReal eps = 1e-6 * fmax(1.0, fabs(x));
        SWEState  qp = qL, qm = qL;
        SetComp(&qp, k, x + eps);
        SetComp(&qm, k, x - eps);
        PetscReal Fp[3], Fm[3], F0[3];
        BoundaryFlux(type, qp, sn, cn, Fp);
        BoundaryFlux(type, qm, sn, cn, Fm);
        BoundaryFlux(type, qL, sn, cn, F0);
        PetscBool kink = PETSC_FALSE;
        for (int i = 0; i < 3; i++) {
          PetscReal fwd = (Fp[i] - F0[i]) / eps, bwd = (F0[i] - Fm[i]) / eps;
          if (fabs(fwd - bwd) > 1e-4 * fmax(1.0, fabs(fwd))) kink = PETSC_TRUE;
        }
        if (kink) continue;
        checked++;
        for (int i = 0; i < 3; i++) {
          PetscReal fd = (Fp[i] - Fm[i]) / (2.0 * eps);
          if (fabs(fd - dF[i][k]) > 1e-6 * fmax(1.0, fabs(fd))) {
            print_message("boundary (%s) Jacobian mismatch (%d,%d): analytic %.12e, FD %.12e\n", type ? "outflow" : "reflecting", i, (int)k, dF[i][k],
                          fd);
            fail();
          }
        }
      }
    }
  }
  print_message("boundary Jacobians: %d columns checked against FD\n", checked);
  assert_true(checked > 600);
}

// Evaluates the left/right edge fluxes G_L, G_R of the HR (scheme 1) or augmented Roe (scheme 2)
// residual Q-function for one edge, using weights w_L = -1, w_R = +1 so that
// cell_L = -G_L and cell_R = G_R.
static void EdgeFluxesFromQFunction(int scheme, SWEState qL, SWEState qR, PetscReal zL, PetscReal zR, PetscReal sn, PetscReal cn, PetscReal h_anuga,
                                    PetscReal GL[3], PetscReal GR[3]) {
  struct SWEContext_ ctx     = {.dtime = 1.0, .tiny_h = TINY_H, .h_anuga_regular = h_anuga, .gravity = G, .xq2018_threshold = 0.0};
  CeedScalar         geom[6] = {sn, cn, -1.0, 1.0, zL, zR};
  CeedScalar         ql[3] = {qL.h, qL.hu, qL.hv}, qr[3] = {qR.h, qR.hu, qR.hv};
  CeedScalar         cl[3], cr[3], fl[3], cnum[2];
  const CeedScalar  *in[3]  = {geom, ql, qr};
  CeedScalar        *out[4] = {cl, cr, fl, cnum};
  if (scheme == 1) SWEFlux_HydroRecon_Roe(&ctx, 1, in, out);
  else SWEFlux_AugmentedRoe(&ctx, 1, in, out);
  for (int i = 0; i < 3; i++) {
    GL[i] = -cl[i];
    GR[i] = cr[i];
  }
}

static void EdgeJacobian(int scheme, SWEState qL, SWEState qR, PetscReal zL, PetscReal zR, PetscReal sn, PetscReal cn, PetscReal h_anuga,
                         PetscReal GL[3], PetscReal GR[3], PetscReal J[4][3][3]) {
  if (scheme == 1) SWEEdgeFluxJacobian_HydroRecon(G, TINY_H, h_anuga, qL, qR, zL, zR, sn, cn, GL, GR, J);
  else SWEEdgeFluxJacobian_AugmentedRoe(G, TINY_H, h_anuga, qL, qR, zL, zR, sn, cn, GL, GR, J);
}

// HR and augmented Roe: the Jacobian routine reproduces the residual Q-function's fluxes, and its
// four blocks agree with central differences of that Q-function away from kinks
static void TestEdgeSchemeJacobianVsFD(int scheme) {
  const char *name    = scheme == 1 ? "hydrostatic reconstruction" : "augmented Roe";
  int         checked = 0, total = 0;
  for (int a = 0; a < 2; a++) {
    PetscReal h_anuga = a ? 0.05 : 0.0;
    for (int trial = 0; trial < 400; trial++) {
      SWEState qL, qR;
      RandomWetState(&qL);
      RandomWetState(&qR);
      PetscReal zL = RandRange(-0.5, 0.5), zR = RandRange(-0.5, 0.5);
      if (trial % 4 == 0) zR = zL;                               // flat edge
      if (trial % 7 == 0) zR = zL + qL.h + RandRange(0.1, 1.0);  // step above the left surface
      if (trial % 9 == 0) qR = (SWEState){0.0, 0.0, 0.0};        // dry right neighbor (wet/dry front)
      if (trial % 11 == 0) qL = (SWEState){0.0, 0.0, 0.0};       // dry left neighbor
      PetscReal theta = RandRange(0.0, 2.0 * PETSC_PI), sn = sin(theta), cn = cos(theta);

      PetscReal GL[3], GR[3], GLref[3], GRref[3], J[4][3][3];
      EdgeJacobian(scheme, qL, qR, zL, zR, sn, cn, h_anuga, GL, GR, J);
      EdgeFluxesFromQFunction(scheme, qL, qR, zL, zR, sn, cn, h_anuga, GLref, GRref);
      for (int i = 0; i < 3; i++) {
        if (fabs(GL[i] - GLref[i]) > 1e-12 * fmax(1.0, fabs(GLref[i])) || fabs(GR[i] - GRref[i]) > 1e-12 * fmax(1.0, fabs(GRref[i]))) {
          print_message("%s flux value mismatch, component %d: G_L %.15e vs %.15e, G_R %.15e vs %.15e\n", name, i, GL[i], GLref[i], GR[i], GRref[i]);
          fail();
        }
      }

      for (PetscInt side = 0; side < 2; side++) {
        for (PetscInt k = 0; k < 3; k++) {
          total++;
          SWEState  q   = side ? qR : qL;
          PetscReal x   = GetComp(q, k);
          PetscReal eps = 1e-6 * fmax(1.0, fabs(x)) * (k == 0 ? fmax(q.h, 1e-3) : 1.0);
          SWEState  qp = q, qm = q;
          SetComp(&qp, k, x + eps);
          SetComp(&qm, k, x - eps);
          PetscReal GLp[3], GRp[3], GLm[3], GRm[3];
          EdgeFluxesFromQFunction(scheme, side ? qL : qp, side ? qp : qR, zL, zR, sn, cn, h_anuga, GLp, GRp);
          EdgeFluxesFromQFunction(scheme, side ? qL : qm, side ? qm : qR, zL, zR, sn, cn, h_anuga, GLm, GRm);
          PetscBool kink = PETSC_FALSE;
          for (int i = 0; i < 3; i++) {
            PetscReal f[2] = {(GLp[i] - GLref[i]) / eps, (GRp[i] - GRref[i]) / eps}, b[2] = {(GLref[i] - GLm[i]) / eps, (GRref[i] - GRm[i]) / eps};
            for (int t = 0; t < 2; t++)
              if (fabs(f[t] - b[t]) > 1e-4 * fmax(1.0, fabs(f[t]))) kink = PETSC_TRUE;
          }
          if (kink) continue;
          checked++;
          for (int i = 0; i < 3; i++) {
            PetscReal fdL = (GLp[i] - GLm[i]) / (2.0 * eps), fdR = (GRp[i] - GRm[i]) / (2.0 * eps);
            PetscReal aL = J[side][i][k], aR = J[2 + side][i][k];
            if (fabs(fdL - aL) > 1e-6 * fmax(1.0, fabs(fdL)) || fabs(fdR - aR) > 1e-6 * fmax(1.0, fabs(fdR))) {
              print_message("%s Jacobian mismatch: side %d, row %d, col %d: dG_L analytic %.12e FD %.12e; dG_R analytic %.12e FD %.12e\n", name,
                            (int)side, i, (int)k, aL, fdL, aR, fdR);
              fail();
            }
          }
        }
      }
    }
  }
  print_message("%s Jacobian: %d of %d columns checked against FD (rest straddle a kink)\n", name, checked, total);
  assert_true(checked > total / 2);
}

static void TestHydroReconJacobianVsFD(void **state) { TestEdgeSchemeJacobianVsFD(1); }
static void TestAugmentedRoeJacobianVsFD(void **state) { TestEdgeSchemeJacobianVsFD(2); }

// on a flat edge, HR reduces to plain Roe (identical blocks)
static void TestHydroReconFlatBedIsRoe(void **state) {
  for (int trial = 0; trial < 100; trial++) {
    SWEState qL, qR;
    RandomWetState(&qL);
    RandomWetState(&qR);
    PetscReal theta = RandRange(0.0, 2.0 * PETSC_PI), sn = sin(theta), cn = cos(theta), z = RandRange(-1.0, 1.0);
    PetscReal GL[3], GR[3], J[4][3][3], GL0[3], GR0[3], J0[4][3][3];
    SWEEdgeFluxJacobian_HydroRecon(G, TINY_H, 0.0, qL, qR, z, z, sn, cn, GL, GR, J);
    SWEEdgeFluxJacobian_Roe(G, TINY_H, 0.0, qL, qR, sn, cn, GL0, GR0, J0);
    for (int k = 0; k < 4; k++)
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) assert_true(fabs(J[k][i][j] - J0[k][i][j]) <= 1e-12 * fmax(1.0, fabs(J0[k][i][j])));
  }
}

static int    argc_ = 0;
static char **argv_ = NULL;

static int Setup(void **state) { return RDyInit(argc_, argv_, "test_swe_roe_jacobian - unit tests for the implicit SWE Jacobians."); }
static int Teardown(void **state) { return RDyFinalize(); }

int main(int argc, char *argv[]) {
  argc_                           = argc;
  argv_                           = argv;
  const struct CMUnitTest tests[] = {
      cmocka_unit_test(TestRoeFluxJacobianVsFD),      cmocka_unit_test(TestRoeFluxJacobianConsistency), cmocka_unit_test(TestSourceJacobianVsFD),
      cmocka_unit_test(TestBoundaryJacobianVsFD),     cmocka_unit_test(TestHydroReconJacobianVsFD),     cmocka_unit_test(TestHydroReconFlatBedIsRoe),
      cmocka_unit_test(TestAugmentedRoeJacobianVsFD),
  };
  return cmocka_run_group_tests(tests, Setup, Teardown);
}
