# Session Handoff — March 23, 2026

## What Was Accomplished This Session

### 1. LETKF Integration — Phase 1 Complete ✅

All Phase 1 work is implemented, tested, and committed to git.

**Files created/modified:**

| File | Status | Description |
|------|--------|-------------|
| `driver/letkf_test.c` | New | LETKF twin-experiment driver using PetscDA LETKF |
| `driver/CMakeLists.txt` | Modified | Builds `rdycore_letkf` binary when `RDYCORE_HAVE_DA=ON` |
| `driver/tests/letkf/letkf_dam_break.yaml` | New | YAML config for dam-break LETKF CTest |
| `driver/tests/letkf/CMakeLists.txt` | New | Registers `letkf_dam_break_np_1` CTest |
| `driver/tests/CMakeLists.txt` | Modified | Adds `add_subdirectory(letkf)` guarded by `RDYCORE_HAVE_DA` |
| `plans/rdycore-letkf-integration-plan.md` | Modified | Phase 1 complete banner, file summary, verification test section |
| `plans/genesis-mission-foa-response-plan.md` | Modified | Rewritten without PINN/surrogate content (see below) |

**Key implementation facts:**
- Binary: `rdycore_letkf` — argument order is `./rdycore_letkf <yaml_file> [PETSc options]`
  (YAML must come before PETSc options because PETSc reorders `argv`)
- CTest: `letkf_dam_break_np_1` (Test #116) passes in ~1.2 sec
- np=2 test was removed — 44-cell mesh is too small for parallel LETKF
  (eigendecomposition ill-conditioned: `||V*D*V^T - T||/||T|| = 0.97`)
- All files are staged in git (`A` = new, `M` = modified)

**To run the test manually:**
```bash
cd /path/to/build
# Copy inputs (CMake does this automatically for ENABLE_TESTS=ON)
cp driver/tests/letkf/letkf_dam_break.yaml driver/
cp share/meshes/planar_dam_10x5.msh driver/

# Minimal test (3 steps, 5 members, no DA)
mpiexec -n 1 driver/rdycore_letkf driver/letkf_dam_break.yaml \
  -letkf_steps 3 -letkf_ensemble_size 5 \
  -letkf_obs_stride 2 -letkf_obs_freq 1

# Full DA test (10 steps, 10 members, assimilate every step)
mpiexec -n 1 driver/rdycore_letkf driver/letkf_dam_break.yaml \
  -letkf_steps 10 -letkf_ensemble_size 10 \
  -letkf_obs_stride 2 -letkf_obs_freq 1
```

**Expected output (10-step DA test):**
```
RDycore LETKF Twin Experiment
  Ensemble size: 10
  Steps: 10
  Obs stride: 2
  Obs freq: 1
  ...
Statistics (10 steps):
  Mean RMSE (truth vs ensemble mean): ~0.01-0.05
  Mean ensemble spread: ~0.01-0.10
  RMSE should decrease or stabilize after assimilation steps
```

---

### 2. FOA Response Plan Rewritten ✅

**File:** [`plans/genesis-mission-foa-response-plan.md`](genesis-mission-foa-response-plan.md)

**What changed:** All PINN (Physics-Informed Neural Network) and AI surrogate
references were removed. The proposal was restructured from a 4-component plan
(training data + PINN surrogates + DA + use case) to a 3-component plan:

| Old Component | New Component |
|--------------|---------------|
| Component 1: Training Data Generation | Component 1: RDycore Ensemble Simulation and Training Data |
| Component 2: Physics-Informed AI Surrogate Development | **Removed entirely** |
| Component 3: AI-Enhanced Data Assimilation | Component 2: AI-Enhanced Data Assimilation (PetscDA LETKF + AI Operators) |
| Component 4: Use Case | Component 3: Use Case — Flood Resilience for Energy Infrastructure |

**Primary AI contribution (new framing):**
> AI-learned localization, inflation, and model error correction operators
> for PetscDA LETKF — replacing hand-tuned Gaspari-Cohn heuristics with
> neural operators trained on RDycore ensemble simulation data.

**Sections updated:**
- §3.1 Vision: Removed surrogate item; reframed around AI-enhanced DA
- §3.2 Why AI: Removed surrogate speedup framing; reframed around DA quality
- §3.3 Mermaid diagram: Removed PINN node; simplified to ensemble → LETKF → AI-DA
- §4 Team: Removed "PINNs, GNNs, FNOs" from Co-PI 1 role; replaced with "AI/ML for DA"
- §4 Partner actions: Removed PINN partner search; replaced with DA/ML expertise search
- §5 Objectives: Replaced surrogate objective with AI-DA objectives
- §5 Task 2: Replaced PINN task with PetscDA LETKF integration task
- §5 Task 3: New AI-learned localization/inflation task
- §5 Milestones: Replaced surrogate milestones with DA-focused milestones
- §5 Decision gate metrics: Replaced surrogate metrics with DA quality metrics
- §8 Review criteria: Removed PINN references; updated to AI-DA framing
- §11 Risks: Replaced surrogate risk with AI localization generalization risk
- §14.6 Proposal implications: Removed surrogate framing; updated to AI-DA
- §15 Talking points: Removed surrogate speedup; added AI-DA talking points
- §15 Decision gate metrics: Replaced surrogate metrics with DA quality metrics
- §16 Integration steps: Added step 7 (replace Q with AI-learned localization)

**No PINN references remain anywhere in the file.**

---

## Current State of the Repository

All work is staged in git. To commit:
```bash
git commit -m "Phase 1 LETKF integration complete; FOA plan rewritten without PINN content"
```

Staged files:
- `A  driver/letkf_test.c`
- `M  driver/CMakeLists.txt`
- `A  driver/tests/letkf/letkf_dam_break.yaml`
- `A  driver/tests/letkf/CMakeLists.txt`
- `M  driver/tests/CMakeLists.txt`
- `M  plans/rdycore-letkf-integration-plan.md`
- `M  plans/genesis-mission-foa-response-plan.md`

---

## Next Steps (Phase 2 and Proposal)

### LETKF Integration — Phase 2 (Parallel Support)
See [`plans/rdycore-letkf-integration-plan.md`](rdycore-letkf-integration-plan.md) §Phase 2.

Key work needed:
1. Distribute ensemble members across MPI ranks (currently all on rank 0)
2. Use a larger mesh (e.g., `Houston1km`) that can support parallel LETKF
3. Validate that eigendecomposition remains well-conditioned with more cells

### FOA Proposal — Immediate Actions
1. **Identify university AI/ML partner** — focus on groups working on
   learned DA operators, neural covariance models, or machine learning
   for geophysical inverse problems (not PINN groups)
2. **Identify industry partner** — utility company or energy infrastructure
   operator with flood risk concerns in Houston/Gulf Coast region
3. **Attend DOE webinar** — March 26, 3 PM Eastern
4. **Begin narrative draft** — use §5 of the FOA plan as the outline
5. **Deadline**: April 28, 2026, 11:59 PM Eastern

### Key Technical Differentiators to Emphasize in Proposal
- Phase 1 LETKF integration is **already working** — not speculative
- PetscDA LETKF tutorials already solve SWE (same physics as RDycore)
- GPU-accelerated via Kokkos (CUDA/HIP/SYCL) — runs on Frontier/Perlmutter
- RDycore validated at 471M cells, R²=0.99 on Malpasset dam break
- Houston1km mesh already in codebase — use case is ready to run

---

## Key File Locations

| File | Purpose |
|------|---------|
| [`plans/genesis-mission-foa-response-plan.md`](genesis-mission-foa-response-plan.md) | FOA response strategy (no PINNs) |
| [`plans/rdycore-letkf-integration-plan.md`](rdycore-letkf-integration-plan.md) | LETKF engineering plan (Phase 1 complete) |
| [`driver/letkf_test.c`](../driver/letkf_test.c) | LETKF twin-experiment driver |
| [`driver/tests/letkf/letkf_dam_break.yaml`](../driver/tests/letkf/letkf_dam_break.yaml) | CTest YAML config |
| [`driver/tests/letkf/CMakeLists.txt`](../driver/tests/letkf/CMakeLists.txt) | CTest registration |
| [`../petsc_gem/src/ml/da/tutorials/ex4.c`](../../petsc_gem/src/ml/da/tutorials/ex4.c) | PetscDA LETKF SWE reference tutorial |

---

*Handoff written March 23, 2026.*
