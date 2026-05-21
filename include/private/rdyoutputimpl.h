#ifndef RDYOUTPUTIMPL_H
#define RDYOUTPUTIMPL_H

#include <ceed/ceed.h>
#include <petsc.h>
#include <private/rdydmimpl.h>

/// Groups all state for one output variable (instantaneous + time-averaged).
/// Does NOT own the vectors — it holds references to vectors on Operator
/// (for prim_vars/sources) or on RDy (for solution).
///
/// Branching convention: functions branch on ceed_accum (or ceed_inst) being
/// non-NULL rather than calling CeedEnabled(). NULL encodes "use PETSc path".
typedef struct {
  SectionFieldSpec inst_fields;           // field spec for instantaneous output
  SectionFieldSpec avg_fields;            // field spec for time-averaged output
  PetscReal        accumulated_time;      // total simulated time since last reset
  PetscBool        skip_first_component;  // if PETSC_TRUE, skip component 0 on write/guard (prim vars only)

  // References to vectors holding the actual data (non-owning except soln_output.petsc_accum).
  // For prim_vars/sources: point to Operator-owned vectors.
  // For solution: petsc_inst = u_global (non-owning), petsc_accum = RDy-owned Vec.
  Vec        petsc_inst;   // PETSc Vec with instantaneous data (CEED staging area when ceed_inst != NULL)
  Vec        petsc_accum;  // PETSc Vec with accumulated data (CEED staging area when ceed_accum != NULL)
  CeedVector ceed_inst;    // CEED instantaneous; NULL if not using CEED (e.g. solution, or PETSc-only)
  CeedVector ceed_accum;   // CEED accumulator; NULL if not using CEED (e.g. solution, or PETSc-only)
} OutputVar;

#endif
