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
  SWE_JK_BC_FREE_OUTFLOW     = 3,
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

// setup-time description of the device RHS machinery, which reuses the
// geometry views of an existing Jacobian context (B1: device RHS). All arrays
// are HOST arrays copied to device views by SetupRHS (caller may free them
// afterwards). The gather CSR runs over the SAME owned-cell list as the
// Jacobian setup (cell_id order) and lists, for each cell, its flux-buffer
// contributions in exactly the host accumulation order: interior edges by
// ascending compact-edge index, then boundary edges by ascending flattened
// index -- so the device sums are bitwise identical to the host loop's.
typedef struct {
  PetscInt         n_gather;                  // total gather entries
  const PetscInt  *gather_start;              // [n_cells + 1] CSR offsets
  const PetscInt  *gather_idx;                // flux-buffer block index: interior k in [0, n_edges), boundary n_edges + m
  const PetscReal *gather_w;                  // -len/A or +len/A, matching the host expression
  PetscInt         source_method;             // 0 = explicit (friction in RHS), 1 = ark-imex (bed slope only)
  PetscInt         src_len;                   // length of the external-sources array (3 * owned cells)
} SWERHSKokkosSetup;

typedef struct SWEJacobianKokkos SWEJacobianKokkos;  // opaque C++ context

#ifdef __cplusplus
extern "C" {
#endif

PetscErrorCode SWEJacobianKokkosCreate(const SWEJacobianKokkosSetup *setup, SWEJacobianKokkos **jk);

// Adds the RHS machinery (flux buffers, gather CSR, staging views) to an
// existing Jacobian context.
PetscErrorCode SWEKokkosSetupRHS(SWEJacobianKokkos *jk, const SWERHSKokkosSetup *setup);

// Flux stage of the device RHS: computes interior-edge and boundary-edge Roe
// fluxes and gathers them into f (a Kokkos-memory-space pointer of length
// 3 * owned cells, fully overwritten). u is the ghosted state in Kokkos
// memory space; dirichlet is the HOST staging array of 3 scalars per
// flattened boundary edge (as in Assemble). On return bflux_host (HOST,
// 3 * n_bedges) holds the raw boundary fluxes (for the boundary-flux vecs),
// and *cfac_max/*cfac_loc the maximum Courant factor amax*len/min(A) over wet
// edges and its location (compact interior index, or n_edges + m for
// boundary; *cfac_loc = -1 when no wet edge contributed).
PetscErrorCode SWEKokkosApplyFlux(SWEJacobianKokkos *jk, const PetscScalar *u, const PetscScalar *dirichlet, PetscScalar *f,
                                  PetscScalar *bflux_host, PetscReal *cfac_max, PetscInt *cfac_loc);

// Source stage of the device RHS: f += sources (Kokkos memory space,
// read-modify-write) and pv (Kokkos memory space, 3 * owned cells) is
// overwritten with regularized primitive variables. mat_props and ext_src are
// HOST arrays staged into cached device views keyed on their vecs' object
// states (pass PetscObjectStateGet of the source vec; re-uploaded only when
// the state changed).
PetscErrorCode SWEKokkosApplySource(SWEJacobianKokkos *jk, const PetscScalar *u, const PetscScalar *mat_props, PetscObjectState matprop_state,
                                    const PetscScalar *ext_src, PetscObjectState src_state, PetscScalar *f, PetscScalar *pv);

// u_local/mat_props may be host or device pointers (pass the memtype from
// VecGetArrayReadAndMemType); a host mat_props is staged into the same
// state-keyed cached device view the RHS source stage uses (pass the
// material-properties vec's object state). dirichlet is a HOST array of 3
// scalars per flattened boundary edge (unused slots for non-Dirichlet edges),
// or NULL when there are no boundary edges. On return *coo_v is a DEVICE
// pointer to the filled values buffer, valid until the next Assemble/Destroy.
PetscErrorCode SWEJacobianKokkosAssemble(SWEJacobianKokkos *jk, const PetscScalar *u_local, PetscMemType u_memtype, const PetscScalar *mat_props,
                                         PetscMemType matprop_memtype, PetscObjectState matprop_state, const PetscScalar *dirichlet,
                                         const PetscScalar **coo_v);

// Device values for the explicit-drag parameter Jacobian dF/dn (COO order:
// for owned cell o, entries (3o+1, o) then (3o+2, o)). u_owned is the GLOBAL
// (owned-dof) state as a device pointer; mat_props is the HOST array staged
// through the shared state-keyed cache. On return *vals is a DEVICE pointer
// (2 scalars per owned cell) for MatSetValuesCOO, valid until the next
// call/Destroy.
PetscErrorCode SWEKokkosJacobianP(SWEJacobianKokkos *jk, const PetscScalar *u_owned, const PetscScalar *mat_props, PetscObjectState matprop_state,
                                  const PetscScalar **vals);

PetscErrorCode SWEJacobianKokkosDestroy(SWEJacobianKokkos **jk);

#ifdef __cplusplus
}
#endif

#endif  // SWE_JACOBIAN_KOKKOS_H
