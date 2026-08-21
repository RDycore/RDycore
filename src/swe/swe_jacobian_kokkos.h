#ifndef SWE_JACOBIAN_KOKKOS_H
#define SWE_JACOBIAN_KOKKOS_H

// C interface to the Kokkos device assembly of the analytic SWE RHS Jacobian
// (swe_jacobian_kokkos.kokkos.cxx). The context captures the static geometry
// and the COO value-buffer layout at setup; each Assemble fills the device
// values buffer with kernels that share the flux-Jacobian math header with
// the host loop (RDY_MATH_FN), and returns a DEVICE pointer suitable for
// MatSetValuesCOO on a Kokkos matrix type.

#include <petscsystypes.h>

// BC types the device boundary kernel understands (the C side maps
// RDyConditionType to these; RegisterSWERHSJacobian already rejects others)
typedef enum {
  SWE_JK_BC_DIRICHLET        = 0,
  SWE_JK_BC_REFLECTING       = 1,
  SWE_JK_BC_CRITICAL_OUTFLOW = 2,
} SWEJacobianKokkosBCType;

// setup-time description; all arrays are HOST arrays copied to device views
// by Create (the caller may free them afterwards)
typedef struct {
  // interior edges, in COO pattern order
  PetscInt         n_edges;
  const PetscInt  *edge_l, *edge_r;    // ghosted local cell ids
  const PetscReal *edge_sn, *edge_cn;  // edge normal angle
  const PetscReal *edge_wl, *edge_wr;  // -len/A_l, +len/A_r
  const PetscInt  *edge_owned;         // bit 0: l owned, bit 1: r owned
  const PetscInt  *edge_offset;        // scalar offset of the edge's block run in coo_v
  // boundary edges (all boundaries flattened), in COO pattern order
  PetscInt         n_bedges;
  const PetscInt  *bedge_cell;                  // owned interior cell l
  const PetscReal *bedge_sn, *bedge_cn, *bedge_wl;
  const PetscInt  *bedge_type;                  // SWEJacobianKokkosBCType
  const PetscInt  *bedge_offset;                // scalar offset of the (l,l) block
  // owned cells (diagonal source blocks), in COO pattern order
  PetscInt         n_cells;
  const PetscInt  *cell_id;         // ghosted local cell id
  const PetscInt  *cell_owned_idx;  // index into the material-properties array
  const PetscReal *cell_dzdx, *cell_dzdy;
  const PetscInt  *cell_offset;  // scalar offset of the (c,c) block
  // sizes and physics constants
  PetscInt  ncoo;         // total scalars in the values buffer
  PetscInt  n_u_local;    // length of the ghosted state array
  PetscInt  matprop_len;  // length of the material-properties array
  PetscInt  matprop_stride, matprop_manning;  // layout: manning of owned cell o at [stride*o + manning]
  PetscReal tiny_h, h_anuga;
  PetscBool friction_in_rhs;  // full source Jacobian vs bed-slope-only (ARK-IMEX)
} SWEJacobianKokkosSetup;

typedef struct SWEJacobianKokkos SWEJacobianKokkos;  // opaque C++ context

#ifdef __cplusplus
extern "C" {
#endif

PetscErrorCode SWEJacobianKokkosCreate(const SWEJacobianKokkosSetup *setup, SWEJacobianKokkos **jk);

// u_local/mat_props may be host or device pointers (pass the memtype from
// VecGetArrayReadAndMemType); dirichlet is a HOST array of 3 scalars per
// flattened boundary edge (unused slots for non-Dirichlet edges), or NULL
// when there are no boundary edges. On return *coo_v is a DEVICE pointer to
// the filled values buffer, valid until the next Assemble/Destroy.
PetscErrorCode SWEJacobianKokkosAssemble(SWEJacobianKokkos *jk, const PetscScalar *u_local, PetscMemType u_memtype, const PetscScalar *mat_props,
                                         PetscMemType matprop_memtype, const PetscScalar *dirichlet, const PetscScalar **coo_v);

PetscErrorCode SWEJacobianKokkosDestroy(SWEJacobianKokkos **jk);

#ifdef __cplusplus
}
#endif

#endif  // SWE_JACOBIAN_KOKKOS_H
