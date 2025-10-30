// SPDX-FileCopyrightText: Copyright (c) 2017-2024, HONEE contributors.
// SPDX-License-Identifier: Apache-2.0 OR BSD-2-Clause
#pragma once

#include <ceed.h>
#include <petsc-ceed.h>
#include <petscdm.h>
#include <petscmat.h>

#define MATCEED "ceed"

// Core functionality
PETSC_CEED_EXTERN PetscErrorCode MatCreateCeed(DM dm_x, DM dm_y, CeedOperator op_mult, CeedOperator op_mult_transpose, Mat *mat);
PETSC_CEED_EXTERN PetscErrorCode MatCeedCopy(Mat mat_ceed, Mat mat_other);
PETSC_CEED_EXTERN PetscErrorCode MatCeedSetAssemblyDataUpdateNeeded(Mat mat_ceed, PetscBool update_needed);
PETSC_CEED_EXTERN PetscErrorCode MatCeedCreateMatCOO(Mat mat_ceed, Mat *mat_coo);
PETSC_CEED_EXTERN PetscErrorCode MatCeedSetPreallocationCOO(Mat mat_ceed, Mat mat_coo);
PETSC_CEED_EXTERN PetscErrorCode MatCeedAssembleCOO(Mat mat_ceed, Mat mat_coo);

PETSC_CEED_INTERN PetscErrorCode MatCeedSetContextDouble(Mat mat, const char *name, double value);
PETSC_CEED_INTERN PetscErrorCode MatCeedGetContextDouble(Mat mat, const char *name, double *value);
PETSC_CEED_EXTERN PetscErrorCode MatCeedSetContextReal(Mat mat, const char *name, PetscReal value);
PETSC_CEED_EXTERN PetscErrorCode MatCeedGetContextReal(Mat mat, const char *name, PetscReal *value);
PETSC_CEED_INTERN PetscErrorCode MatCeedSetTime(Mat mat, PetscReal time);
PETSC_CEED_INTERN PetscErrorCode MatCeedGetTime(Mat mat, PetscReal *time);
PETSC_CEED_INTERN PetscErrorCode MatCeedSetDt(Mat mat, PetscReal dt);
PETSC_CEED_INTERN PetscErrorCode MatCeedSetShifts(Mat mat, PetscReal shift_v, PetscReal shift_a);

// Advanced functionality
PETSC_CEED_EXTERN PetscErrorCode MatCeedSetContext(Mat mat, PetscCtxDestroyFn f, void *ctx);
PETSC_CEED_EXTERN PetscErrorCode MatCeedGetContext(Mat mat, void *ctx);

PETSC_CEED_EXTERN PetscErrorCode MatCeedSetOperation(Mat mat, MatOperation op, void (*g)(void));
PETSC_CEED_EXTERN PetscErrorCode MatCeedSetCOOMatType(Mat mat, MatType type);
PETSC_CEED_EXTERN PetscErrorCode MatCeedGetCOOMatType(Mat mat, MatType *type);
PETSC_CEED_EXTERN PetscErrorCode MatCeedSetReusePreallocationCOO(Mat mat, PetscBool coo_reuse_preallocation);
PETSC_CEED_EXTERN PetscErrorCode MatCeedGetReusePreallocationCOO(Mat mat, PetscBool *coo_reuse_preallocation);

PETSC_CEED_EXTERN PetscErrorCode MatCeedSetLocalVectors(Mat mat, Vec X_loc, Vec Y_loc_transpose);
PETSC_CEED_EXTERN PetscErrorCode MatCeedGetLocalVectors(Mat mat, Vec *X_loc, Vec *Y_loc_transpose);
PETSC_CEED_EXTERN PetscErrorCode MatCeedRestoreLocalVectors(Mat mat, Vec *X_loc, Vec *Y_loc_transpose);

PETSC_CEED_EXTERN PetscErrorCode MatCeedGetCeedOperators(Mat mat, CeedOperator *op_mult, CeedOperator *op_mult_transpose);
PETSC_CEED_EXTERN PetscErrorCode MatCeedRestoreCeedOperators(Mat mat, CeedOperator *op_mult, CeedOperator *op_mult_transpose);

PETSC_CEED_EXTERN PetscErrorCode MatCeedSetLogEvents(Mat mat, PetscLogEvent log_event_mult, PetscLogEvent log_event_mult_transpose);
PETSC_CEED_EXTERN PetscErrorCode MatCeedGetLogEvents(Mat mat, PetscLogEvent *log_event_mult, PetscLogEvent *log_event_mult_transpose);
PETSC_CEED_EXTERN PetscErrorCode MatCeedSetCeedOperatorLogEvents(Mat mat, PetscLogEvent log_event_mult, PetscLogEvent log_event_mult_transpose);
PETSC_CEED_EXTERN PetscErrorCode MatCeedGetCeedOperatorLogEvents(Mat mat, PetscLogEvent *log_event_mult, PetscLogEvent *log_event_mult_transpose);
