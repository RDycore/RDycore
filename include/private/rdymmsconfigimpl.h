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

// the maximum length of a string referring to a name in the config file
#define MAX_MMS_EXPRESSION_LEN 128

// a string containing an expression for a manufactured solution
typedef char ManufacturedSolution[MAX_MMS_EXPRESSION_LEN + 1];

// specification of a set of named constants for the MMS driver, each
// represented by a single capital letter
typedef struct {
  PetscReal A, B, C, D, E, F, G, H, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z;
} RDyMMSConstants;

// specification of a set of manufactured solutions for the
// shallow water equations (SWE)
typedef struct {
  // water height h(x, y, t) and partial derivatives
  ManufacturedSolution h, dhdx, dhdy, dhdt;

  // flow x-velocity u(x, y, t) and partial derivatives
  ManufacturedSolution u, dudx, dudy, dudt;

  // flow y-velocity v(x, y, t) and partial derivatives
  ManufacturedSolution v, dvdx, dvdy, dvdt;

  // elevation z(x, y) and partial derivatives
  ManufacturedSolution z, dzdx, dzdy;
} RDyMMSSWESolutions;

// specification of an ensemble
typedef struct {
  RDyMMSConstants    constants;
  RDyMMSSWESolutions swe;
} RDyMMSSection;

#endif
