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
#include "swe_roe_flux_jacobian_petsc.h"

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
};

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
                                         PetscMemType matprop_memtype, const PetscScalar *dirichlet, const PetscScalar **coo_v)
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
    if (jk->mp_stage.extent(0) == 0) PetscCallCXX(jk->mp_stage = View<PetscScalar>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("swejk_mp_stage")), jk->matprop_len));
    PetscCallCXX(Kokkos::deep_copy(jk->mp_stage, Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(mat_props, jk->matprop_len)));
    mp = Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(jk->mp_stage.data(), jk->matprop_len);
  } else {
    mp = Kokkos::View<const PetscScalar *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(mat_props, jk->matprop_len);
  }
  if (jk->n_bedges) {
    PetscCheck(dirichlet, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "boundary edges present but no dirichlet staging array");
    PetscCallCXX(Kokkos::deep_copy(jk->dirichlet, Kokkos::View<const PetscScalar *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(dirichlet, 3 * jk->n_bedges)));
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

PetscErrorCode SWEJacobianKokkosDestroy(SWEJacobianKokkos **jk)
{
  PetscFunctionBegin;
  PetscCallCXX(delete *jk);
  *jk = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
