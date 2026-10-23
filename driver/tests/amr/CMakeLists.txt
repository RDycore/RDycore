# h5diff is only needed by the bit-identical gate below, and is not part of
# every HDF5 installation's bin directory (on Perlmutter it comes from the
# cray-hdf5-parallel module rather than from PETSc), so it is not required
find_program(H5DIFF h5diff PATHS ${PETSC_DIR}/${PETSC_ARCH}/bin)
set(amr_c_driver ${amr_driver})

# arguments for specific configurations ("basic" config has no options)
set(ceed_args -ceed /cpu/self)
set(preload_args -preload)

# AMR driver
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/amr_dx1.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mms_triangles_dx1-${PETSC_ID_TYPE})
file(COPY
     ${REFINEMENT_DIR}/mms_triangles_dx1/2011-01-01:00-00.${PETSC_ID_TYPE}.bin
     ${REFINEMENT_DIR}/mms_triangles_dx1/2011-01-01:00-01.${PETSC_ID_TYPE}.bin
     ${REFINEMENT_DIR}/mms_triangles_dx1/2011-01-01:00-02.${PETSC_ID_TYPE}.bin
     ${REFINEMENT_DIR}/mms_triangles_dx1/2011-01-01:00-03.${PETSC_ID_TYPE}.bin
     ${REFINEMENT_DIR}/mms_triangles_dx1/2011-01-01:00-04.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/mms_triangles_dx1-${PETSC_ID_TYPE}
    )
foreach(np 3)
  foreach(config basic)
    foreach(lang c)
      # -refine_check_volume so that this case asserts something rather than
      # only exiting 0; it conserves to 5.7e-16 against the 1e-11 held to
      add_test(amr_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_${lang}_driver} amr_dx1.yaml -options_left -dm_adaptor cellrefiner -dm_plex_transform_type refine_sbr -dm_plex_transform_label_match_strata -dm_view -dm_fine_view -refine_data_start_date 2011,01,01,00,01 -refine_data_dir mms_triangles_dx1-${PETSC_ID_TYPE} -refine_check_volume ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# Demonstration cases refined by the solution itself (water-surface jump), each
# reporting the water volume across every adapt.
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/channel_wave_amr.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/dam_break_amr.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/still_water_ramp.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/still_water_dem.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/graded_wave_amr.yaml
     ${MESH_DIR}/channel_100x4_tri.msh
     ${MESH_DIR}/planar_dam_20x10_tri.msh
     ${MESH_DIR}/ramp_100x4_tri.msh
     ${MESH_DIR}/graded_40x2_tri.msh
     ${MESH_DIR}/channel_ramp_elevation.${PETSC_ID_TYPE}.bin
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# the transform RDycore's AMR needs is set in the code, so the cases ask only
# for the criterion.
#
# -refine_check_volume is a gate (exactness class: machine precision, with the
# mechanism -- the area-weighted transfer reproduces sum(h * area) exactly and
# only the summation order changes). It used to print the number and check
# nothing, so these cases passed on exit status and a transfer that lost mass
# would have gone through them; the worst departure measured over every case in
# this directory is 1.4e-14 relative, against the 1e-11 the driver holds it to.
set(amr_jump_args -refine_criterion jump -refine_at_start -refine_check_volume)

# 1D dam break on a 2D mesh: the case checked against the Stoker solution
foreach(np 1 3)
  add_test(amr_channel_wave_c_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_c_driver} channel_wave_amr.yaml -options_left ${amr_jump_args})
endforeach()

# 2D dam break: refined around the dam before the first step, then tracking the
# jet through the breach and the fronts leaving it
foreach(np 1 3)
  add_test(amr_dam_break_c_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_c_driver} dam_break_amr.yaml -options_left ${amr_jump_args})
endforeach()

# The same two cases with coarsening: each adapt rebuilds the mesh from the base
# mesh, so the cell count falls again behind a feature that has moved on, and
# the solution moves between meshes by the conservative transfer
foreach(np 1 3)
  add_test(amr_channel_wave_coarsen_c_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_c_driver} channel_wave_amr.yaml -options_left
           ${amr_jump_args} -refine_allow_coarsening)
endforeach()
add_test(amr_dam_break_coarsen_c_np_1 ${MPIEXEC} ${MPIEXEC_FLAGS} -n 1 ${amr_c_driver} dam_break_amr.yaml -options_left
         ${amr_jump_args} -refine_allow_coarsening)

# Gate: solution-driven AMR chooses the same mesh on one rank and on three,
# with and without coarsening. See the script for what used to go wrong.
foreach(mode refine coarsen)
  set(refine_extra "")
  set(coarsen_extra "-refine_allow_coarsening")
  # and the same under CEED, which is what caught the operator scattering its
  # fluxes by the wrong index on an adapted mesh: 1148 cells on one rank
  # against 1150 on three
  set(basic_ceed "")
  set(ceed_ceed "-ceed /cpu/self")
  foreach(backend basic ceed)
    set(suffix "")
    if (backend STREQUAL ceed)
      set(suffix "_ceed")
    endif()
    add_test(NAME amr_same_on_any_ranks_${mode}${suffix}
             COMMAND ${CMAKE_COMMAND}
                     -DRDYCORE_AMR=$<TARGET_FILE:rdycore_amr>
                     -DYAML=${CMAKE_CURRENT_SOURCE_DIR}/channel_wave_amr.yaml
                     -DMESH=${MESH_DIR}/channel_100x4_tri.msh
                     -DMPIEXEC=${MPIEXEC}
                     "-DEXTRA=${${mode}_extra} ${${backend}_ceed}"
                     -DWORK=${CMAKE_CURRENT_BINARY_DIR}/amr_same_on_any_ranks_${mode}${suffix}
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/amr_same_on_any_ranks.cmake)
  endforeach()
endforeach()

# Gate (exactness class: exact degeneration). A lake at rest on a sloping bed
# survives refinement and coarsening: the transfer carries the free surface, so
# a child cell's depth is rebuilt against its own bed rather than inheriting its
# parent's. The box sweeps along the channel so that cells refine ahead of it
# and coarsen behind it, and the driver fails if the surface departs from rest
# by more than 1e-12 of its elevation on either side of any adapt.
#
# -dm_plex_gmsh_spacedim 3 because PETSc's gmsh reader takes the embedding
# dimension to be the topological one unless told otherwise, which would drop
# the bed and leave the case flat.
foreach(np 1 3)
  add_test(amr_still_water_ramp_c_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_c_driver} still_water_ramp.yaml -options_left
           -dm_plex_gmsh_spacedim 3
           -refine_still_water 1.5 -refine_box -5,5,-1,3 -refine_box_speed 10 -refine_allow_coarsening -refine_check_volume)
endforeach()

# Gate: the same case with the box sweeping off the end of the channel, so that
# a rebuild eventually refines nothing and returns the base mesh itself. That is
# the adapt the coarse-DM chain used to close on; see the script.
add_test(NAME amr_relaxes_to_base_mesh
         COMMAND ${CMAKE_COMMAND}
                 -DRDYCORE_AMR=$<TARGET_FILE:rdycore_amr>
                 -DYAML=${CMAKE_CURRENT_SOURCE_DIR}/still_water_ramp.yaml
                 -DMESH_DIR=${MESH_DIR}
                 -DMPIEXEC=${MPIEXEC}
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/amr_relaxes_to_base_mesh
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/amr_relaxes_to_base_mesh.cmake)

# The same gate with the bed read from a file instead of the mesh. It is a
# separate run because it is a separate path: the operator takes the cell-centre
# z the file gives, and the adapt has to carry those values from the base mesh
# rather than re-read a file holding one value per base cell, which is what used
# to make the first adapt fail outright.
foreach(np 1 3)
  add_test(amr_still_water_dem_c_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_c_driver} still_water_dem.yaml -options_left
           -refine_still_water 1.5 -refine_box -5,5,-1,3 -refine_box_speed 10 -refine_allow_coarsening -refine_check_volume)
endforeach()

# Gate (exactness class: exact degeneration -- the refinement level is an
# integer read off an area ratio that is a power of two, so the limit holds or
# it does not; there is no tolerance). On a graded base mesh, refine-only, no
# cell may end up more than one level below -rdy_amr_max_levels: the option caps
# the level a cell may be MARKED at, and refine_sbr's conformity closure splits
# a marked triangle into three children of areas 1/2, 1/4, 1/4 where it needs a
# second edge split, which accounts for the one level and no more.
#
# What this catches, measured at max_levels 2 with the fix reverted:
#   graded,  limit per base cell : depth 2 3 3 3 3 3 3 3,  570 cells at the end
#   graded,  one limit for all   : depth 2 4 6 6 6 6 7 7, 1447 cells at the end
#   uniform, one limit for all   : depth 2 4 4 4 4 4 4 4  -- bounded, so no
#                                  uniform mesh can see the runaway
# The uniform run is one level too deep either way until the test is written >=
# rather than >, which is the second half of the same fix.
foreach(np 1 3)
  add_test(amr_graded_max_level_c_np_${np} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_c_driver} graded_wave_amr.yaml -options_left
           ${amr_jump_args} -refine_check_max_level -rdy_amr_max_levels 2)
endforeach()

# The same lake-at-rest gates under CEED. Running the demonstration cases under
# CEED would not do: they only report the water volume, so a wrong-but-finite
# flux passes them. These assert that the surface stays flat to 1e-12, which a
# wrong flux cannot do -- and the CEED operator scattered its fluxes to the
# wrong cells on any adapted mesh above one rank until the offsets stopped
# going through cells->local_to_owned.
foreach(np 1 3)
  add_test(amr_still_water_ramp_c_np_${np}_ceed ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_c_driver} still_water_ramp.yaml -options_left
           -dm_plex_gmsh_spacedim 3
           -refine_still_water 1.5 -refine_box -5,5,-1,3 -refine_box_speed 10 -refine_allow_coarsening -refine_check_volume ${ceed_args})
endforeach()

# Gate: a lake at rest, the standard well-balancedness benchmark, must not be
# refined by a criterion that measures the jump in the water surface. See the
# script for what the relative VecTagger does with a uniform indicator.
add_test(NAME amr_lake_at_rest_does_not_refine
         COMMAND ${CMAKE_COMMAND}
                 -DRDYCORE_AMR=$<TARGET_FILE:rdycore_amr>
                 -DRAMP_YAML=${CMAKE_CURRENT_SOURCE_DIR}/still_water_ramp.yaml
                 -DWAVE_YAML=${CMAKE_CURRENT_SOURCE_DIR}/channel_wave_amr.yaml
                 -DMESH_DIR=${MESH_DIR}
                 -DMPIEXEC=${MPIEXEC}
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/amr_lake_at_rest_does_not_refine
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/amr_lake_at_rest_does_not_refine.cmake)

# Gate: the adapt machinery with nothing marked leaves the answer untouched
if (NOT H5DIFF)
  message(STATUS "h5diff not found: the AMR bit-identical gate will not be run")
else()
add_test(NAME amr_off_is_bitwise
         COMMAND ${CMAKE_COMMAND}
                 -DRDYCORE=$<TARGET_FILE:rdycore_exe>
                 -DRDYCORE_AMR=$<TARGET_FILE:rdycore_amr>
                 -DYAML=${CMAKE_CURRENT_SOURCE_DIR}/channel_wave_amr.yaml
                 -DMESH=${MESH_DIR}/channel_100x4_tri.msh
                 -DH5DIFF=${H5DIFF}
                 -DWORK=${CMAKE_CURRENT_BINARY_DIR}/amr_off_is_bitwise
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/amr_off_is_bitwise.cmake)
endif()
