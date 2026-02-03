#ifndef RDYMMSCONFIG_H
#define RDYMMSCONFIG_H

#include <float.h>
#include <petsc.h>
#include <rdycore.h>

// The types in this file ѕerve as an intermediate representation for our MMS
// driver's input configuration file:
//
// https://rdycore.github.io/RDycore/user/mms.html
//

// the maximum length of a string containing a mathematical expression
#define MAX_EXPRESSION_LEN 128

typedef char MathExpression[MAX_EXPRESSION_LEN + 1];

// specification of a set of named constants for the MMS driver, each
// represented by a single capital letter
typedef struct {
  PetscReal A, B, C, D, E, F, G, H, J, I_, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z;
} RDyMMSConstants;

// manufactured solutions for the shallow water equations (SWE)
typedef struct {
  struct {
    // water height h(x, y, t) and partial derivatives
    MathExpression h, dhdx, dhdy, dhdt;
    // flow x-velocity u(x, y, t) and partial derivatives
    MathExpression u, dudx, dudy, dudt;
    // flow y-velocity v(x, y, t) and partial derivatives
    MathExpression v, dvdx, dvdy, dvdt;
    // elevation z(x, y) and partial derivatives
    MathExpression z, dzdx, dzdy;
    // Manning's roughness coefficient n(x, y)
    MathExpression n;
  } expressions;

  struct {
    // water height h(x, y, t) and partial derivatives
    void *h, *dhdx, *dhdy, *dhdt;
    // flow x-velocity u(x, y, t) and partial derivatives
    void *u, *dudx, *dudy, *dudt;
    // flow y-velocity v(x, y, t) and partial derivatives
    void *v, *dvdx, *dvdy, *dvdt;
    // elevation z(x, y) and partial derivatives
    void *z, *dzdx, *dzdy;
    // Manning's roughness coefficient n(x, y)
    void *n;
  } solutions;
} RDyMMSSWESolutions;

// manufactured solutions for tracers
typedef struct {
  struct {
    // tracer value ci(x, y, t) and partial derivatives
    MathExpression c[MAX_NUM_TRACERS], dcdx[MAX_NUM_TRACERS], dcdy[MAX_NUM_TRACERS], dcdt[MAX_NUM_TRACERS];
  } expressions;

  struct {
    // tracer value ci(x, y, t) and partial derivatives
    // NOTE: uintptr_t is an integer big enough to store a pointer, so we can
    // NOTE: use it as an element of an array (unlike void)
    uintptr_t c[MAX_NUM_TRACERS], dcdx[MAX_NUM_TRACERS], dcdy[MAX_NUM_TRACERS], dcdt[MAX_NUM_TRACERS];
  } solutions;
} RDyMMSTracerSolutions;

typedef struct {
  PetscReal L1, L2, Linf;
} RDyMMSErrorNorms;

typedef struct {
  RDyMMSErrorNorms h, hu, hv, c[MAX_NUM_TRACERS];
} RDyMMSConvergenceRates;

typedef struct {
  PetscInt               num_refinements;
  PetscInt               base_refinement;
  RDyMMSConvergenceRates expected_rates;
} RDyMMSConvergence;

// constants and expressions for manufactured solutions
typedef struct {
  RDyMMSConstants         constants;
  RDyMMSSWESolutions      swe;
  RDyMMSTracerSolutions   tracers;
  RDyMMSConvergence       convergence;
} RDyMMSSection;

#endif
