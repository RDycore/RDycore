#ifndef SEDIMENT_TYPES_CEED_H
#define SEDIMENT_TYPES_CEED_H

#include "sediment_ceed.h"

// Q-function context with data attached
typedef struct SedimentContext_ *SedimentContext;
struct SedimentContext_ {
  CeedScalar dtime;
  CeedScalar tiny_h;
  CeedScalar gravity;
  CeedScalar xq2018_threshold;
  CeedScalar kp_constant;
  CeedScalar settling_velocity;
  CeedScalar tau_critical_erosion;
  CeedScalar tau_critical_deposition;
  CeedScalar rhow;
  CeedInt    sed_ndof;
  CeedInt    flow_ndof;
};

struct SedimentState_ {
  CeedScalar h, hu, hv, hci[MAX_NUM_SEDIMENT_CLASSES];
};
typedef struct SedimentState_ SedimentState;

#endif
