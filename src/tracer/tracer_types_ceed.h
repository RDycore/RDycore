#ifndef TRACERS_TYPES_CEED_H
#define TRACERS_TYPES_CEED_H

#include <ceed.h>
#include <private/config.h>

// Q-function context with data attached
typedef struct TracerContext_ *TracerContext;
struct TracerContext_ {
  CeedScalar dtime;
  CeedScalar tiny_h;
  CeedScalar gravity;
  CeedScalar xq2018_threshold;
  CeedScalar kp_constant;
  CeedScalar settling_velocity;
  CeedScalar tau_critical_erosion;
  CeedScalar tau_critical_deposition;
  CeedScalar rhow;
  CeedInt    tracer_ndof;
  CeedInt    flow_ndof;
};

struct TracerState_ {
  CeedScalar h, hu, hv, hci[MAX_NUM_SEDIMENT_CLASSES], s, T;
};
typedef struct TracerState_ TracerState;

#endif
