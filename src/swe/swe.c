#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

extern PetscLogEvent RDY_CeedOperatorApply;

static inline CeedMemType MemTypeP2C(PetscMemType mem_type) { return PetscMemTypeDevice(mem_type) ? CEED_MEM_DEVICE : CEED_MEM_HOST; }

static PetscErrorCode RDyCeedOperatorApply(RDy rdy, PetscReal dt, Vec U_local, Vec F) {
  PetscFunctionBeginUser;

  // update the timestep for the ceed operators if necessary
  if (rdy->ceed.dt != dt) {
    PetscCall(SWEFluxOperatorSetTimeStep(rdy->ceed.flux_operator, dt));
    PetscCall(SWESourceOperatorSetTimeStep(rdy->ceed.source_operator, dt));
    rdy->ceed.dt = dt;
  }

  {
    // The computation of fluxes across internal and boundary edges via CeedOperator is done
    // in the following three stages:
    //
    // a) Pre-CeedOperatorApply stage:
    //    - Set memory pointer of a CeedVector (u_local_ceed) is set to PETSc Vec (U_local)
    //    - Ask DM to "get" a PETSc Vec (F_local), then set memory pointer of a CeedVector (f_ceed)
    //      to the F_local PETSc Vec.
    //
    // b) CeedOperatorApply stage
    //    - Apply the CeedOparator in which u_local_ceed is an input, while f_ceed is an output.
    //
    // c) Post-CeedOperatorApply stage:
    //    - Add values in F_local to F via Local-to-Global scatter.
    //    - Clean up memory

    PetscScalar *u_local, *f;
    PetscMemType mem_type;
    Vec          F_local;

    CeedVector u_local_ceed = rdy->ceed.u_local;
    CeedVector f_ceed       = rdy->ceed.rhs;

    // 1. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> U_local
    PetscCall(VecGetArrayAndMemType(U_local, &u_local, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u_local));

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: f_ceed --> F_local
    PetscCall(DMGetLocalVector(rdy->dm, &F_local));
    PetscCall(VecGetArrayAndMemType(F_local, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(f_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    // 3. Apply the CeedOpeator associated with the internal and boundary edges
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(rdy->ceed.flux_operator, u_local_ceed, f_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    // 4. Resets memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(f_ceed, MemTypeP2C(mem_type), &f));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u_local));

    // 5. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(F_local, &f));
    PetscCall(VecRestoreArrayAndMemType(U_local, &u_local));

    // 6. Zero out values in F and then add F_local to F via Local-to-Global scatter
    PetscCall(VecZeroEntries(F));
    PetscCall(DMLocalToGlobal(rdy->dm, F_local, ADD_VALUES, F));

    // 7. Restor the F_local
    PetscCall(DMRestoreLocalVector(rdy->dm, &F_local));
  }

  {
    // The computation of contribution of the source-sink term via CeedOperator is done
    // in the following three stages:
    //
    // a) Pre-CeedOperatorApply stage:
    //    - Set memory pointer of a CeedVector (u_local_ceed) to PETSc Vec (U_local)
    //    - A copy of the PETSc Vec F is made as host_fluxes. Then, memory pointer of a CeedVector (riemannf_ceed)
    //      to host_fluxes.
    //    - Set memory pointer of a CeedVector (s_ceed) to PETSc Vec (F)
    //
    // b) CeedOperatorApply stage
    //    - Apply the CeedOparator in which u_local_ceed is an input, while s_ceed is an output.
    //
    // c) Post-CeedOperatorApply stage:
    //    - Clean up memory

    // 1. Get the CeedVector associated with the "riemannf" CeedOperatorField
    CeedOperatorField riemannf_field;
    CeedVector        riemannf_ceed;
    SWESourceOperatorGetRiemannFlux(rdy->ceed.source_operator, &riemannf_field);
    PetscCallCEED(CeedOperatorFieldGetVector(riemannf_field, &riemannf_ceed));

    PetscScalar *u, *f, *host_fluxes;
    PetscMemType mem_type;
    CeedVector   u_local_ceed = rdy->ceed.u_local;
    CeedVector   s_ceed       = rdy->ceed.sources;

    // 2. Sets the pointer of a CeedVector to a PETSc Vec: u_local_ceed --> U_local
    PetscCall(VecGetArrayAndMemType(U_local, &u, &mem_type));
    PetscCallCEED(CeedVectorSetArray(u_local_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, u));

    // 3. Make a duplicate copy of the F as the values will be used as input for the CeedOperator
    //    corresponding to the source-sink term
    PetscCall(VecCopy(F, rdy->ceed.host_fluxes));

    // 4. Sets the pointer of a CeedVector to a PETSc Vec: host_fluxes --> riemannf_ceed
    PetscCall(VecGetArrayAndMemType(rdy->ceed.host_fluxes, &host_fluxes, &mem_type));
    PetscCallCEED(CeedVectorSetArray(riemannf_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, host_fluxes));

    // 5. Sets the pointer of a CeedVector to a PETSc Vec: F --> s_ceed
    PetscCall(VecGetArrayAndMemType(F, &f, &mem_type));
    PetscCallCEED(CeedVectorSetArray(s_ceed, MemTypeP2C(mem_type), CEED_USE_POINTER, f));

    // 6. Apply the source CeedOperator
    PetscCall(PetscLogEventBegin(RDY_CeedOperatorApply, U_local, F, 0, 0));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCEED(CeedOperatorApply(rdy->ceed.source_operator, u_local_ceed, s_ceed, CEED_REQUEST_IMMEDIATE));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogEventEnd(RDY_CeedOperatorApply, U_local, F, 0, 0));

    // 7. Reset memory pointer of CeedVectors
    PetscCallCEED(CeedVectorTakeArray(s_ceed, MemTypeP2C(mem_type), &f));
    PetscCallCEED(CeedVectorTakeArray(riemannf_ceed, MemTypeP2C(mem_type), &host_fluxes));
    PetscCallCEED(CeedVectorTakeArray(u_local_ceed, MemTypeP2C(mem_type), &u));

    // 8. Restore pointers to the PETSc Vecs
    PetscCall(VecRestoreArrayAndMemType(U_local, &u));
    PetscCall(VecRestoreArrayAndMemType(F, &f));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
