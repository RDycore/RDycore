#ifndef RDYPRELOADIMPL_H
#define RDYPRELOADIMPL_H

#include <petsclog.h>

// These macros reimplement PETSc's preloading macros in petsclog.h (e.g.
// https://petsc.org/release/manualpages/Profiling/PetscPreLoadBegin/),
// removing the requirement that they be called inside a function that is
// invoked only once.
//
// To use the macros, #include this file and replace "Petsc" with "RDy" in each
// of the macros and variables associated with preloading.

#define RDyPreLoadBegin(flag, name)                                                              \
  do {                                                                                           \
    PetscBool     RDyPreLoading = flag;                                                          \
    int           RDyPreLoadMax, RDyPreLoadIt;                                                   \
    PetscLogStage _stageNum;                                                                     \
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-preload", &RDyPreLoading, NULL));                \
    RDyPreLoadMax     = (int)(RDyPreLoading);                                                    \
    RDyPreLoadingUsed = RDyPreLoading ? PETSC_TRUE : RDyPreLoadingUsed;                          \
    for (RDyPreLoadIt = 0; RDyPreLoadIt <= RDyPreLoadMax; RDyPreLoadIt++) {                      \
      RDyPreLoadingOn = RDyPreLoading;                                                           \
      PetscCall(PetscBarrier(NULL));                                                             \
      PetscCall(PetscLogStageGetId(name, &_stageNum));                                           \
      if (_stageNum == -1) PetscCall(PetscLogStageRegister(name, &_stageNum));                   \
      PetscCall(PetscLogStageSetActive(_stageNum, (PetscBool)(!RDyPreLoadMax || RDyPreLoadIt))); \
      PetscCall(PetscLogStagePush(_stageNum));

#define RDyPreLoadEnd()          \
  PetscCall(PetscLogStagePop()); \
  RDyPreLoading = PETSC_FALSE;   \
  }                              \
  }                              \
  while (0)

#define RDyPreLoadStage(name)                                                                  \
  do {                                                                                         \
    PetscCall(PetscLogStagePop());                                                             \
    PetscCall(PetscLogStageGetId(name, &_stageNum));                                           \
    if (_stageNum == -1) PetscCall(PetscLogStageRegister(name, &_stageNum));                   \
    PetscCall(PetscLogStageSetActive(_stageNum, (PetscBool)(!RDyPreLoadMax || RDyPreLoadIt))); \
    PetscCall(PetscLogStagePush(_stageNum));                                                   \
  } while (0)

PETSC_EXTERN PetscBool RDyPreLoadingUsed;
PETSC_EXTERN PetscBool RDyPreLoadingOn;

#endif
