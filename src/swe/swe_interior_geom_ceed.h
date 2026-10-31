#ifndef SWE_INTERIOR_GEOM_CEED_H
#define SWE_INTERIOR_GEOM_CEED_H

// geom component indices for interior edges of the edge-based bed-source
// methods (hydrostatic reconstruction, augmented Roe), 6 components
#ifndef HR_INTERIOR_GEOM_INDEX_DEFINED
#define HR_INTERIOR_GEOM_INDEX_DEFINED
typedef enum {
  HR_INTERIOR_SN = 0,
  HR_INTERIOR_CN,
  HR_INTERIOR_NEG_L_OVER_AL,
  HR_INTERIOR_L_OVER_AR,
  HR_INTERIOR_ZC_LEFT,
  HR_INTERIOR_ZC_RIGHT,
  HR_NUM_COMP_INTERIOR_GEOM,
} HRInteriorGeomIndex;
#endif

#endif
