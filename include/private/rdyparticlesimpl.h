#ifndef RDYPARTICLESIMPL_H
#define RDYPARTICLESIMPL_H

#include <petscdmswarm.h>
#include <petscts.h>

// Forward declaration — full definition is in rdycoreimpl.h
typedef struct _p_RDy *RDy;

// Advection context passed to the FreeStreaming RHS function.
// Mirrors the AdvCtx pattern from PETSc ex77.c, adapted for SWE cell-center
// velocity (FV P0 data rather than FEM).
typedef struct {
  Vec        u_pde;        // alias for rdy->u_global (the live PDE solution)
  PetscReal  tiny_h;       // minimum water height threshold (from config)
  RDy        rdy;          // back-pointer to RDy for mesh/config access
  const char *cellid_field; // cached cell-ID field name (constant for swarm lifetime)
} ParticleAdvCtx;

// Container for all particle-tracer state attached to an RDy instance.
typedef struct {
  DM             dm_swarm;           // the DMSwarm (DMSWARM_PIC type)
  TS             ts_particles;       // separate TS for particle advection (ex77.c pattern)
  ParticleAdvCtx adv_ctx;            // advection context passed to FreeStreaming RHS
  PetscInt       particles_per_cell; // number of particles seeded per cell (0 = disabled)
  PetscBool      enabled;            // whether particle tracing is active
} RDyParticles;

// Internal API — implemented in src/rdyparticles.c
PETSC_INTERN PetscErrorCode CreateParticleSwarm(RDy rdy);
PETSC_INTERN PetscErrorCode DestroyParticleSwarm(RDy rdy);
PETSC_INTERN PetscErrorCode WriteParticleOutput(RDy rdy, PetscInt step, PetscReal time);

#endif  // RDYPARTICLESIMPL_H
