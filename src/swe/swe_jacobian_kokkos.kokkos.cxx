// Kokkos device assembly of the analytic SWE RHS Jacobian: fills the COO
// values buffer laid out by CreateAnalyticJacobianCOO with one kernel over
// interior edges, one over boundary edges, and one over owned cells. The
// per-edge/per-cell math is the SAME code the host loop runs -- the shared
// header below is included with RDY_MATH_FN = KOKKOS_INLINE_FUNCTION -- and
// each edge/cell writes at a precomputed offset, so the buffer is filled in
// exactly the host loop's order (dry/dry and non-contributing branches write
// zero blocks, keeping offsets and values in lockstep).

#include <Kokkos_Core.hpp>
#include <string>
#include <petscsys.h>

#define RDY_MATH_FN static KOKKOS_INLINE_FUNCTION
#include "swe_jacobian_kokkos.h"
#include "swe_roe_flux_jacobian_petsc.h"  // also pulls in swe_roe_flux_petsc.h (ComputeSWERoeFluxEdge)

namespace {
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;
template <typename T>
using View = Kokkos::View<T *, MemSpace>;
template <typename T>
static View<T> ToDevice(const char *name, const T *host, PetscInt n) {
  View<T> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(name)), n);
  Kokkos::deep_copy(d, Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(host, n));
  return d;
}
}  // namespace

struct SWEJacobianKokkos {
  // static geometry (device)
  View<PetscInt>  edge_l, edge_r, edge_owned, edge_offset;
  View<PetscReal> edge_sn, edge_cn, edge_wl, edge_wr;
  View<PetscInt>  bedge_cell, bedge_type, bedge_offset;
  View<PetscReal> bedge_sn, bedge_cn, bedge_wl;
  View<PetscInt>  cell_id, cell_owned_idx, cell_offset;
  View<PetscReal> cell_dzdx, cell_dzdy;
  // per-assembly buffers (device)
  View<PetscScalar> v;         // the COO values buffer
  View<PetscScalar> u_stage;   // staging when the state arrives as a host pointer
  View<PetscScalar> mp_stage;  // staging when material properties arrive as a host pointer
  View<PetscScalar> dirichlet; // per-bedge ghost states, uploaded each assembly
  PetscInt          n_edges, n_bedges, n_cells, ncoo, n_u_local, matprop_len, matprop_stride, matprop_manning;
  PetscReal         tiny_h, h_anuga;
  PetscBool         friction_in_rhs;
  // RHS machinery (B1, SWEKokkosSetupRHS; extents stay 0 until then)
  View<PetscInt>    gather_start, gather_idx;  // CSR over the owned-cell list
  View<PetscReal>   gather_w;
  View<PetscScalar> fluxg;      // gather flux buffer: 3 * (n_edges + n_bedges)
  View<PetscScalar> braw;       // raw boundary fluxes: 3 * n_bedges (D2H each eval)
  View<PetscReal>   cfac;       // Courant factor candidates: n_edges + n_bedges
  View<PetscScalar> src_stage;  // cached external sources
  View<PetscScalar> jacp_v;     // dF/dn COO values buffer (2 per owned cell)
  PetscInt          rhs_source_method, src_len;
  bool              has_dirichlet = false;  // any SWE_JK_BC_DIRICHLET boundary edge (else the dirichlet staging is never read)
  bool              rhs_ready = false, mp_primed = false, src_primed = false;
  PetscObjectState  mp_state = 0, src_state = 0;  // source-vec states of the staged copies (valid when *_primed)
};

namespace {
// Stage a host array into a cached device view, re-uploading only when the
// source vec's object state changed since the copy was made (or on first use).
static PetscErrorCode StageTracked(View<PetscScalar> &stage, const char *name, const PetscScalar *host, PetscInt len, PetscObjectState state,
                                   bool *primed, PetscObjectState *primed_state) {
  PetscFunctionBegin;
  if (stage.extent(0) == 0) PetscCallCXX(stage = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(name)), len));
  if (!*primed || state != *primed_state) {
    PetscCallCXX(Kokkos::deep_copy(stage, Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(host, len)));
    PetscCall(PetscLogCpuToGpu(1.0 * len * sizeof(PetscScalar)));
    *primed       = true;
    *primed_state = state;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
}  // namespace

PetscErrorCode SWEJacobianKokkosCreate(const SWEJacobianKokkosSetup *s, SWEJacobianKokkos **jk_out)
{
  SWEJacobianKokkos *jk;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCallCXX(jk = new SWEJacobianKokkos());
  jk->n_edges         = s->n_edges;
  jk->n_bedges        = s->n_bedges;
  jk->n_cells         = s->n_cells;
  jk->ncoo            = s->ncoo;
  jk->n_u_local       = s->n_u_local;
  jk->matprop_len     = s->matprop_len;
  jk->matprop_stride  = s->matprop_stride;
  jk->matprop_manning = s->matprop_manning;
  jk->tiny_h          = s->tiny_h;
  jk->h_anuga         = s->h_anuga;
  jk->friction_in_rhs = s->friction_in_rhs;
  PetscCallCXX(jk->edge_l = ToDevice("swejk_edge_l", s->edge_l, s->n_edges));
  PetscCallCXX(jk->edge_r = ToDevice("swejk_edge_r", s->edge_r, s->n_edges));
  PetscCallCXX(jk->edge_owned = ToDevice("swejk_edge_owned", s->edge_owned, s->n_edges));
  PetscCallCXX(jk->edge_offset = ToDevice("swejk_edge_offset", s->edge_offset, s->n_edges));
  PetscCallCXX(jk->edge_sn = ToDevice("swejk_edge_sn", s->edge_sn, s->n_edges));
  PetscCallCXX(jk->edge_cn = ToDevice("swejk_edge_cn", s->edge_cn, s->n_edges));
  PetscCallCXX(jk->edge_wl = ToDevice("swejk_edge_wl", s->edge_wl, s->n_edges));
  PetscCallCXX(jk->edge_wr = ToDevice("swejk_edge_wr", s->edge_wr, s->n_edges));
  if (s->n_bedges) {
    for (PetscInt e = 0; e < s->n_bedges && !jk->has_dirichlet; ++e) jk->has_dirichlet = (s->bedge_type[e] == SWE_JK_BC_DIRICHLET);
    PetscCallCXX(jk->bedge_cell = ToDevice("swejk_bedge_cell", s->bedge_cell, s->n_bedges));
    PetscCallCXX(jk->bedge_type = ToDevice("swejk_bedge_type", s->bedge_type, s->n_bedges));
    PetscCallCXX(jk->bedge_offset = ToDevice("swejk_bedge_offset", s->bedge_offset, s->n_bedges));
    PetscCallCXX(jk->bedge_sn = ToDevice("swejk_bedge_sn", s->bedge_sn, s->n_bedges));
    PetscCallCXX(jk->bedge_cn = ToDevice("swejk_bedge_cn", s->bedge_cn, s->n_bedges));
    PetscCallCXX(jk->bedge_wl = ToDevice("swejk_bedge_wl", s->bedge_wl, s->n_bedges));
    PetscCallCXX(jk->dirichlet = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_dirichlet")), 3 * s->n_bedges));
  }
  PetscCallCXX(jk->cell_id = ToDevice("swejk_cell_id", s->cell_id, s->n_cells));
  PetscCallCXX(jk->cell_owned_idx = ToDevice("swejk_cell_owned_idx", s->cell_owned_idx, s->n_cells));
  PetscCallCXX(jk->cell_offset = ToDevice("swejk_cell_offset", s->cell_offset, s->n_cells));
  PetscCallCXX(jk->cell_dzdx = ToDevice("swejk_cell_dzdx", s->cell_dzdx, s->n_cells));
  PetscCallCXX(jk->cell_dzdy = ToDevice("swejk_cell_dzdy", s->cell_dzdy, s->n_cells));
  PetscCallCXX(jk->v = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_coo_v")), s->ncoo));
  *jk_out = jk;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SWEJacobianKokkosAssemble(SWEJacobianKokkos *jk, const PetscScalar *u_local, PetscMemType u_memtype, const PetscScalar *mat_props,
                                         PetscMemType matprop_memtype, PetscObjectState matprop_state, const PetscScalar *dirichlet,
                                         const PetscScalar **coo_v)
{
  Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u, mp;

  PetscFunctionBegin;
  // stage host-resident inputs on device (kokkos vec types hand us device
  // pointers directly; plain vec types come through host)
  if (PetscMemTypeHost(u_memtype)) {
    if (jk->u_stage.extent(0) == 0) PetscCallCXX(jk->u_stage = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_u_stage")), jk->n_u_local));
    PetscCallCXX(Kokkos::deep_copy(jk->u_stage, Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(u_local, jk->n_u_local)));
    u = Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(jk->u_stage.data(), jk->n_u_local);
  } else {
    u = Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(u_local, jk->n_u_local);
  }
  if (PetscMemTypeHost(matprop_memtype)) {
    // the same cached copy the RHS source stage uses; re-uploaded only when the vec changed
    PetscCall(StageTracked(jk->mp_stage, "swejk_mp_stage", mat_props, jk->matprop_len, matprop_state, &jk->mp_primed, &jk->mp_state));
    mp = Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(jk->mp_stage.data(), jk->matprop_len);
  } else {
    mp = Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(mat_props, jk->matprop_len);
  }
  if (jk->has_dirichlet) {  // only Dirichlet edges read the staging; skip the per-assembly upload otherwise
    PetscCheck(dirichlet, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Dirichlet boundary edges present but no dirichlet staging array");
    PetscCallCXX(Kokkos::deep_copy(jk->dirichlet, Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(dirichlet, 3 * jk->n_bedges)));
    PetscCall(PetscLogCpuToGpu(3.0 * jk->n_bedges * sizeof(PetscScalar)));
  }

  const PetscReal tiny_h = jk->tiny_h, h_anuga = jk->h_anuga;
  auto            v      = jk->v;

  PetscCall(PetscLogGpuTimeBegin());
  {  // interior-edge flux blocks (mirrors the host loop in SWERHSJacobianAnalytic)
    auto edge_l = jk->edge_l, edge_r = jk->edge_r, edge_owned = jk->edge_owned, edge_offset = jk->edge_offset;
    auto edge_sn = jk->edge_sn, edge_cn = jk->edge_cn, edge_wl = jk->edge_wl, edge_wr = jk->edge_wr;
    PetscCallCXX(Kokkos::parallel_for(
      "swejk_interior", Kokkos::RangePolicy<ExecSpace>(0, jk->n_edges), KOKKOS_LAMBDA(const PetscInt e) {
        const PetscInt l = edge_l(e), r = edge_r(e), owned = edge_owned(e);
        PetscInt       cursor  = edge_offset(e);
        const PetscInt nblocks = 2 * ((owned & 1) != 0) + 2 * ((owned & 2) != 0);

        const PetscReal uL[3] = {PetscRealPart(u(3 * l)), PetscRealPart(u(3 * l + 1)), PetscRealPart(u(3 * l + 2))};
        const PetscReal uR[3] = {PetscRealPart(u(3 * r)), PetscRealPart(u(3 * r + 1)), PetscRealPart(u(3 * r + 2))};
        if (uL[0] < tiny_h && uR[0] < tiny_h) {  // RHS skips dry/dry edges
          for (PetscInt k = 0; k < 9 * nblocks; ++k) v(cursor + k) = 0.0;
          return;
        }
        PetscReal dFdUL[3][3], dFdUR[3][3];
        SWERoeFluxJacobian(uL, uR, edge_sn(e), edge_cn(e), tiny_h, h_anuga, dFdUL, dFdUR);
        const PetscReal wl = edge_wl(e), wr = edge_wr(e);
        if (owned & 1) {
          for (PetscInt i = 0; i < 3; ++i)
            for (PetscInt j = 0; j < 3; ++j) v(cursor + 3 * i + j) = wl * dFdUL[i][j];
          cursor += 9;
          for (PetscInt i = 0; i < 3; ++i)
            for (PetscInt j = 0; j < 3; ++j) v(cursor + 3 * i + j) = wl * dFdUR[i][j];
          cursor += 9;
        }
        if (owned & 2) {
          for (PetscInt i = 0; i < 3; ++i)
            for (PetscInt j = 0; j < 3; ++j) v(cursor + 3 * i + j) = wr * dFdUR[i][j];
          cursor += 9;
          for (PetscInt i = 0; i < 3; ++i)
            for (PetscInt j = 0; j < 3; ++j) v(cursor + 3 * i + j) = wr * dFdUL[i][j];
        }
      }));
  }
  if (jk->n_bedges) {  // boundary-edge (l,l) blocks (mirrors ApplyBoundaryFlux's Jacobian)
    auto bedge_cell = jk->bedge_cell, bedge_type = jk->bedge_type, bedge_offset = jk->bedge_offset;
    auto bedge_sn = jk->bedge_sn, bedge_cn = jk->bedge_cn, bedge_wl = jk->bedge_wl;
    auto dir = jk->dirichlet;
    PetscCallCXX(Kokkos::parallel_for(
      "swejk_boundary", Kokkos::RangePolicy<ExecSpace>(0, jk->n_bedges), KOKKOS_LAMBDA(const PetscInt e) {
        const PetscInt  l = bedge_cell(e), off = bedge_offset(e);
        const PetscReal sn = bedge_sn(e), cn = bedge_cn(e), wl = bedge_wl(e);

        const PetscReal consL[3] = {PetscRealPart(u(3 * l)), PetscRealPart(u(3 * l + 1)), PetscRealPart(u(3 * l + 2))};
        PetscReal       qL[3], PL[3][3];
        SWEReconstructPrimitiveWithJacobian(consL, tiny_h, h_anuga, qL, PL);

        PetscReal qR[3] = {0, 0, 0}, G[3][3] = {{0}};
        bool      contribute = true;
        switch (bedge_type(e)) {
          case SWE_JK_BC_DIRICHLET: {
            const PetscReal consR[3] = {PetscRealPart(dir(3 * e)), PetscRealPart(dir(3 * e + 1)), PetscRealPart(dir(3 * e + 2))};
            PetscReal       PR_unused[3][3];
            SWEReconstructPrimitiveWithJacobian(consR, tiny_h, h_anuga, qR, PR_unused);
            // ghost independent of the interior state: dG stays zero
          } break;
          case SWE_JK_BC_REFLECTING: {
            const PetscReal d1 = Square(sn) - Square(cn), d2 = 2.0 * sn * cn;
            qR[0]   = qL[0];
            qR[1]   = qL[1] * d1 - qL[2] * d2;
            qR[2]   = -qL[1] * d2 - qL[2] * d1;
            G[0][0] = 1.0;
            G[1][1] = d1;
            G[1][2] = -d2;
            G[2][1] = -d2;
            G[2][2] = -d1;
          } break;
          case SWE_JK_BC_CRITICAL_OUTFLOW: {
            const PetscReal uperp = qL[1] * cn + qL[2] * sn;
            if (uperp < 0.0) {
              contribute = false;  // code zeroes BOTH states: flux is identically zero
            } else {
              const PetscReal q   = qL[0] * uperp;
              qR[0]               = PetscPowReal(Square(q) / GRAVITY, 1.0 / 3.0);
              const PetscReal vel = PetscSqrtReal(GRAVITY * qR[0]);
              qR[1]               = vel * cn;
              qR[2]               = vel * sn;
              if (q > 1e-14) {
                const PetscReal dq[3]   = {uperp, qL[0] * cn, qL[0] * sn};
                const PetscReal hR_coef = (2.0 / 3.0) * qR[0] / q;
                const PetscReal v_coef  = (vel > 0.0) ? 0.5 * GRAVITY / vel : 0.0;
                for (PetscInt j = 0; j < 3; ++j) {
                  G[0][j] = hR_coef * dq[j];
                  G[1][j] = cn * v_coef * G[0][j];
                  G[2][j] = sn * v_coef * G[0][j];
                }
              }
            }
          } break;
        }
        if (!contribute || (qL[0] < tiny_h && qR[0] < tiny_h)) {  // zero flux / dry/dry boundary edge
          for (PetscInt k = 0; k < 9; ++k) v(off + k) = 0.0;
          return;
        }
        for (PetscInt j = 0; j < 3; ++j) {
          const PetscReal dirL[3] = {PL[0][j], PL[1][j], PL[2][j]};
          PetscReal       dirR[3], dF[3];
          for (PetscInt i = 0; i < 3; ++i) dirR[i] = G[i][0] * dirL[0] + G[i][1] * dirL[1] + G[i][2] * dirL[2];
          SWERoeFluxDifferentialPrim(qL, qR, sn, cn, dirL, dirR, dF);
          for (PetscInt i = 0; i < 3; ++i) v(off + 3 * i + j) = wl * dF[i];
        }
      }));
  }
  {  // per-cell source blocks
    auto cell_id = jk->cell_id, cell_owned_idx = jk->cell_owned_idx, cell_offset = jk->cell_offset;
    auto cell_dzdx = jk->cell_dzdx, cell_dzdy = jk->cell_dzdy;
    const PetscInt  mp_stride = jk->matprop_stride, mp_manning = jk->matprop_manning;
    const PetscBool friction  = jk->friction_in_rhs;
    PetscCallCXX(Kokkos::parallel_for(
      "swejk_cells", Kokkos::RangePolicy<ExecSpace>(0, jk->n_cells), KOKKOS_LAMBDA(const PetscInt ci) {
        const PetscInt  c = cell_id(ci), off = cell_offset(ci);
        const PetscReal n_manning = PetscRealPart(mp(mp_stride * cell_owned_idx(ci) + mp_manning));
        const PetscReal cons[3]   = {PetscRealPart(u(3 * c)), PetscRealPart(u(3 * c + 1)), PetscRealPart(u(3 * c + 2))};
        PetscReal       D[3][3];
        if (friction) {
          SWESourceJacobian(cons, n_manning, cell_dzdx(ci), cell_dzdy(ci), tiny_h, D);
        } else {
          for (PetscInt i = 0; i < 3; ++i)
            for (PetscInt j = 0; j < 3; ++j) D[i][j] = 0.0;
          D[1][0] = -GRAVITY * cell_dzdx(ci);
          D[2][0] = -GRAVITY * cell_dzdy(ci);
        }
        for (PetscInt i = 0; i < 3; ++i)
          for (PetscInt j = 0; j < 3; ++j) v(off + 3 * i + j) = D[i][j];
      }));
  }
  PetscCallCXX(Kokkos::fence());
  PetscCall(PetscLogGpuTimeEnd());
  *coo_v = jk->v.data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// Device RHS (B1): the flux stage computes per-edge Roe fluxes (interior and
// boundary kernels sharing ComputeSWERoeFluxEdge with the host loops) and a
// deterministic per-owned-cell gather whose CSR lists contributions in the
// host loops' accumulation order, so the sums are bitwise identical to the
// host RHS. The source stage mirrors ApplySourceExplicit / ApplySourceARKImex.
// ---------------------------------------------------------------------------

PetscErrorCode SWEKokkosSetupRHS(SWEJacobianKokkos *jk, const SWERHSKokkosSetup *s)
{
  PetscFunctionBegin;
  PetscCallCXX(jk->gather_start = ToDevice("swejk_gather_start", s->gather_start, jk->n_cells + 1));
  if (s->n_gather) {
    PetscCallCXX(jk->gather_idx = ToDevice("swejk_gather_idx", s->gather_idx, s->n_gather));
    PetscCallCXX(jk->gather_w = ToDevice("swejk_gather_w", s->gather_w, s->n_gather));
  }
  PetscCallCXX(jk->fluxg = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_fluxg")), 3 * (jk->n_edges + jk->n_bedges)));
  if (jk->n_bedges) PetscCallCXX(jk->braw = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_braw")), 3 * jk->n_bedges));
  PetscCallCXX(jk->cfac = View<PetscReal>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_cfac")), jk->n_edges + jk->n_bedges));
  jk->rhs_source_method = s->source_method;
  jk->src_len           = s->src_len;
  jk->rhs_ready         = true;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SWEKokkosApplyFlux(SWEJacobianKokkos *jk, const PetscScalar *u_ptr, const PetscScalar *dirichlet, PetscScalar *f_ptr,
                                  PetscScalar *bflux_host, PetscReal *cfac_max, PetscInt *cfac_loc)
{
  PetscFunctionBegin;
  PetscCheck(jk->rhs_ready, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "SWEKokkosSetupRHS has not been called on this context");
  Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u(u_ptr, jk->n_u_local);
  Kokkos::View<PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>       f(f_ptr, 3 * jk->n_cells);

  if (jk->has_dirichlet) {  // only Dirichlet edges read the staging; skip the per-eval upload otherwise
    PetscCheck(dirichlet, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Dirichlet boundary edges present but no dirichlet staging array");
    PetscCallCXX(Kokkos::deep_copy(jk->dirichlet, Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(dirichlet, 3 * jk->n_bedges)));
    PetscCall(PetscLogCpuToGpu(3.0 * jk->n_bedges * sizeof(PetscScalar)));
  }

  const PetscReal tiny_h = jk->tiny_h, h_anuga = jk->h_anuga;
  const PetscInt  n_e = jk->n_edges, n_b = jk->n_bedges;
  auto            fluxg = jk->fluxg;
  auto            cfac  = jk->cfac;

  PetscCall(PetscLogGpuTimeBegin());
  {  // interior-edge fluxes (mirrors ApplyInteriorFlux)
    auto edge_l = jk->edge_l, edge_r = jk->edge_r;
    auto edge_sn = jk->edge_sn, edge_cn = jk->edge_cn, edge_wl = jk->edge_wl, edge_wr = jk->edge_wr;
    PetscCallCXX(Kokkos::parallel_for(
      "swejk_rhs_interior", Kokkos::RangePolicy<ExecSpace>(0, n_e), KOKKOS_LAMBDA(const PetscInt e) {
        const PetscInt  l = edge_l(e), r = edge_r(e);
        const PetscReal hl = PetscRealPart(u(3 * l)), hul = PetscRealPart(u(3 * l + 1)), hvl = PetscRealPart(u(3 * l + 2));
        const PetscReal hr = PetscRealPart(u(3 * r)), hur = PetscRealPart(u(3 * r + 1)), hvr = PetscRealPart(u(3 * r + 2));
        if (hr < tiny_h && hl < tiny_h) {  // host skips dry/dry edges: zero contribution, no Courant candidate
          fluxg(3 * e) = fluxg(3 * e + 1) = fluxg(3 * e + 2) = 0.0;
          cfac(e)                                            = -1.0;
          return;
        }
        PetscReal ul, vl, ur, vr, fij[3], amax;
        ComputeSWERiemannVelocity(hl, hul, hvl, tiny_h, h_anuga, &ul, &vl);
        ComputeSWERiemannVelocity(hr, hur, hvr, tiny_h, h_anuga, &ur, &vr);
        ComputeSWERoeFluxEdge(hl, ul, vl, hr, ur, vr, edge_sn(e), edge_cn(e), fij, &amax);
        fluxg(3 * e)     = fij[0];
        fluxg(3 * e + 1) = fij[1];
        fluxg(3 * e + 2) = fij[2];
        // amax * len/min(A_l, A_r): wl = -len/A_l, wr = +len/A_r
        cfac(e) = amax * fmax(-edge_wl(e), edge_wr(e));
      }));
  }
  if (n_b) {  // boundary-edge fluxes (mirrors ApplyBoundaryFlux and the BC appliers)
    auto bedge_cell = jk->bedge_cell, bedge_type = jk->bedge_type;
    auto bedge_sn = jk->bedge_sn, bedge_cn = jk->bedge_cn, bedge_wl = jk->bedge_wl;
    auto dir  = jk->dirichlet;
    auto braw = jk->braw;
    PetscCallCXX(Kokkos::parallel_for(
      "swejk_rhs_boundary", Kokkos::RangePolicy<ExecSpace>(0, n_b), KOKKOS_LAMBDA(const PetscInt m) {
        const PetscInt  l = bedge_cell(m);
        const PetscReal sn = bedge_sn(m), cn = bedge_cn(m);
        PetscReal       hl = PetscRealPart(u(3 * l)), hul = PetscRealPart(u(3 * l + 1)), hvl = PetscRealPart(u(3 * l + 2));
        PetscReal       ul, vl;
        ComputeSWERiemannVelocity(hl, hul, hvl, tiny_h, h_anuga, &ul, &vl);

        PetscReal hr = 0.0, ur = 0.0, vr = 0.0;
        switch (bedge_type(m)) {
          case SWE_JK_BC_DIRICHLET: {
            hr = PetscRealPart(dir(3 * m));
            ComputeSWERiemannVelocity(hr, PetscRealPart(dir(3 * m + 1)), PetscRealPart(dir(3 * m + 2)), tiny_h, h_anuga, &ur, &vr);
          } break;
          case SWE_JK_BC_REFLECTING: {
            hr                   = hl;
            const PetscReal dum1 = Square(sn) - Square(cn);
            const PetscReal dum2 = 2.0 * sn * cn;
            ur                   = ul * dum1 - vl * dum2;
            vr                   = -ul * dum2 - vl * dum1;
          } break;
          case SWE_JK_BC_CRITICAL_OUTFLOW: {
            const PetscReal uperp = ul * cn + vl * sn;
            if (uperp < 0.0) {  // inflow: host zeroes BOTH states
              hl = ul = vl = 0.0;
            } else {
              const PetscReal q = hl * fabs(uperp);
              hr                = PetscPowReal(Square(q) / GRAVITY, 1.0 / 3.0);

              const PetscReal velocity = PetscPowReal(GRAVITY * hr, 0.5);
              ur                       = velocity * cn;
              vr                       = velocity * sn;
            }
          } break;
        }

        PetscReal fij[3], amax;
        ComputeSWERoeFluxEdge(hl, ul, vl, hr, ur, vr, sn, cn, fij, &amax);
        braw(3 * m)     = fij[0];  // raw flux for the boundary-flux vec (matches host, which stores it unconditionally)
        braw(3 * m + 1) = fij[1];
        braw(3 * m + 2) = fij[2];
        const bool wet  = !(hl < tiny_h && hr < tiny_h);  // host accumulation/Courant guard (post-BC states)
        const PetscInt g = n_e + m;
        fluxg(3 * g)     = wet ? fij[0] : 0.0;
        fluxg(3 * g + 1) = wet ? fij[1] : 0.0;
        fluxg(3 * g + 2) = wet ? fij[2] : 0.0;
        cfac(g)          = wet ? amax * (-bedge_wl(m)) : -1.0;  // amax * len/A_l
      }));
  }
  {  // deterministic per-owned-cell gather (host accumulation order; overwrites f, which arrives zeroed)
    auto gather_start = jk->gather_start, gather_idx = jk->gather_idx;
    auto gather_w       = jk->gather_w;
    auto cell_owned_idx = jk->cell_owned_idx;
    PetscCallCXX(Kokkos::parallel_for(
      "swejk_rhs_gather", Kokkos::RangePolicy<ExecSpace>(0, jk->n_cells), KOKKOS_LAMBDA(const PetscInt ci) {
        const PetscInt o = cell_owned_idx(ci);
        PetscReal      val0 = 0.0, val1 = 0.0, val2 = 0.0;
        for (PetscInt j = gather_start(ci); j < gather_start(ci + 1); ++j) {
          const PetscInt  k = gather_idx(j);
          const PetscReal w = gather_w(j);
          val0 += PetscRealPart(fluxg(3 * k)) * w;
          val1 += PetscRealPart(fluxg(3 * k + 1)) * w;
          val2 += PetscRealPart(fluxg(3 * k + 2)) * w;
        }
        f(3 * o)     = val0;
        f(3 * o + 1) = val1;
        f(3 * o + 2) = val2;
      }));
  }
  // max Courant factor and its location (for the diagnostics the host loops
  // update inline); ties resolve to an arbitrary wet edge, which only affects
  // the reported edge/cell ids, not the max
  {
    PetscReal maxval = -1.0;
    PetscInt  maxloc = -1;
    if (n_e + n_b > 0) {
      Kokkos::MaxLoc<PetscReal, PetscInt>::value_type result;
      PetscCallCXX(Kokkos::parallel_reduce(
        "swejk_rhs_cfacmax", Kokkos::RangePolicy<ExecSpace>(0, n_e + n_b),
        KOKKOS_LAMBDA(const PetscInt i, Kokkos::MaxLoc<PetscReal, PetscInt>::value_type &lmax) {
          if (cfac(i) > lmax.val) {
            lmax.val = cfac(i);
            lmax.loc = i;
          }
        },
        Kokkos::MaxLoc<PetscReal, PetscInt>(result)));
      if (result.val > 0.0) {
        maxval = result.val;
        maxloc = result.loc;
      }
    }
    *cfac_max = maxval;
    *cfac_loc = maxloc;
  }
  PetscCallCXX(Kokkos::fence());
  PetscCall(PetscLogGpuTimeEnd());

  if (n_b) {  // raw boundary fluxes back to the host staging array
    PetscCallCXX(Kokkos::deep_copy(Kokkos::View<PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(bflux_host, 3 * n_b), jk->braw));
    PetscCall(PetscLogGpuToCpu(3.0 * n_b * sizeof(PetscScalar)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SWEKokkosApplySource(SWEJacobianKokkos *jk, const PetscScalar *u_ptr, const PetscScalar *mat_props, PetscObjectState matprop_state,
                                    const PetscScalar *ext_src, PetscObjectState src_state, PetscScalar *f_ptr, PetscScalar *pv_ptr)
{
  PetscFunctionBegin;
  PetscCheck(jk->rhs_ready, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "SWEKokkosSetupRHS has not been called on this context");
  Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u(u_ptr, jk->n_u_local);
  Kokkos::View<PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>       f(f_ptr, 3 * jk->n_cells);
  Kokkos::View<PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>       pv(pv_ptr, 3 * jk->n_cells);

  // stage host-resident inputs, re-uploading only when their source vecs changed
  PetscCall(StageTracked(jk->mp_stage, "swejk_mp_stage", mat_props, jk->matprop_len, matprop_state, &jk->mp_primed, &jk->mp_state));
  PetscCall(StageTracked(jk->src_stage, "swejk_src_stage", ext_src, jk->src_len, src_state, &jk->src_primed, &jk->src_state));

  const PetscReal tiny_h = jk->tiny_h, h_anuga = jk->h_anuga;
  const PetscInt  mp_stride = jk->matprop_stride, mp_manning = jk->matprop_manning;
  const bool      friction = (jk->rhs_source_method == 0);  // explicit drag in the RHS; ark-imex keeps it implicit
  auto            mp = jk->mp_stage;
  auto            src = jk->src_stage;
  auto            cell_id = jk->cell_id, cell_owned_idx = jk->cell_owned_idx;
  auto            cell_dzdx = jk->cell_dzdx, cell_dzdy = jk->cell_dzdy;

  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCXX(Kokkos::parallel_for(
    "swejk_rhs_source", Kokkos::RangePolicy<ExecSpace>(0, jk->n_cells), KOKKOS_LAMBDA(const PetscInt ci) {
      const PetscInt  c = cell_id(ci), o = cell_owned_idx(ci);
      const PetscReal h = PetscRealPart(u(3 * c)), hu = PetscRealPart(u(3 * c + 1)), hv = PetscRealPart(u(3 * c + 2));

      // bed slope (include_bed_slope is always true on the non-HR path)
      const PetscReal bedx = cell_dzdx(ci) * GRAVITY * h;
      const PetscReal bedy = cell_dzdy(ci) * GRAVITY * h;

      PetscReal tbx = 0.0, tby = 0.0;
      if (friction && h >= tiny_h) {  // mirrors ApplySourceExplicit's wet branch
        const PetscReal uu = hu / h;
        const PetscReal vv = hv / h;

        const PetscReal N_mannings = PetscRealPart(mp(mp_stride * o + mp_manning));
        const PetscReal Cd         = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);

        const PetscReal velocity = PetscSqrtReal(Square(uu) + Square(vv));
        const PetscReal tb       = Cd * velocity / h;

        tbx = tb * hu;
        tby = tb * hv;
      }

      f(3 * o)     += src(3 * o);
      f(3 * o + 1) += -bedx - tbx + PetscRealPart(src(3 * o + 1));
      f(3 * o + 2) += -bedy - tby + PetscRealPart(src(3 * o + 2));

      // primitive variables (h, u, v) with ANUGA regularization
      const PetscReal denom = Square(h) + Square(h_anuga);
      pv(3 * o)             = h;
      pv(3 * o + 1)         = (h >= tiny_h) ? (hu * h / denom) : 0.0;
      pv(3 * o + 2)         = (h >= tiny_h) ? (hv * h / denom) : 0.0;
    }));
  PetscCallCXX(Kokkos::fence());
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SWEKokkosJacobianP(SWEJacobianKokkos *jk, const PetscScalar *u_ptr, const PetscScalar *mat_props, PetscObjectState matprop_state,
                                  const PetscScalar **vals)
{
  PetscFunctionBegin;
  PetscCall(StageTracked(jk->mp_stage, "swejk_mp_stage", mat_props, jk->matprop_len, matprop_state, &jk->mp_primed, &jk->mp_state));
  if (jk->jacp_v.extent(0) == 0) PetscCallCXX(jk->jacp_v = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_jacp_v")), 2 * jk->n_cells));
  Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u(u_ptr, 3 * jk->n_cells);

  const PetscReal tiny_h    = jk->tiny_h;
  const PetscInt  mp_stride = jk->matprop_stride, mp_manning = jk->matprop_manning;
  auto            mp = jk->mp_stage;
  auto            jv = jk->jacp_v;

  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCXX(Kokkos::parallel_for(
    "swejk_jacp", Kokkos::RangePolicy<ExecSpace>(0, jk->n_cells), KOKKOS_LAMBDA(const PetscInt o) {
      const PetscReal h = PetscRealPart(u(3 * o)), hu = PetscRealPart(u(3 * o + 1)), hv = PetscRealPart(u(3 * o + 2));
      PetscReal       v1 = 0.0, v2 = 0.0;
      if (h >= tiny_h) {  // mirrors the host loop's dry / motionless skips (which leave zeros)
        const PetscReal m = PetscSqrtReal(Square(hu) + Square(hv));
        if (m != 0.0) {
          const PetscReal n_manning = PetscRealPart(mp(mp_stride * o + mp_manning));
          const PetscReal coeff     = -2.0 * GRAVITY * n_manning * PetscPowReal(h, -7.0 / 3.0) * m;
          v1                        = coeff * hu;
          v2                        = coeff * hv;
        }
      }
      jv(2 * o)     = v1;
      jv(2 * o + 1) = v2;
    }));
  PetscCallCXX(Kokkos::fence());
  PetscCall(PetscLogGpuTimeEnd());
  *vals = jk->jacp_v.data();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SWEJacobianKokkosDestroy(SWEJacobianKokkos **jk)
{
  PetscFunctionBegin;
  PetscCallCXX(delete *jk);
  *jk = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
