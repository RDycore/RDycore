#----------------------------------------------------------
# Tests for the LETKF data assimilation driver
# Requires RDYCORE_HAVE_DA (PetscDA support in petsc_gem)
#----------------------------------------------------------

set(letkf_driver ${CMAKE_CURRENT_BINARY_DIR}/../../rdycore_letkf)

# Copy the YAML config and mesh to the binary directory so the driver
# can find them by relative path (grid: file: planar_dam_10x5.msh)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/letkf_dam_break.yaml
          ${MESH_DIR}/planar_dam_10x5.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Basic LETKF twin experiment: 5 ensemble members, 3 steps, obs every step
# Tests that the driver runs without error and produces output
# Note: np=1 only -- the 44-cell planar_dam_10x5 mesh is too small for
# meaningful parallel LETKF testing (eigendecomposition becomes ill-conditioned
# with small ensemble + distributed observations). Use a larger mesh for np>1.
add_test(letkf_dam_break_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1
         ${letkf_driver} letkf_dam_break.yaml
         -letkf_steps 3
         -letkf_ensemble_size 5
         -letkf_obs_stride 2
         -letkf_obs_freq 1)
