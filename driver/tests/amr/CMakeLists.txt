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
      add_test(amr_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_${lang}_driver} amr_dx1.yaml -options_left -dm_adaptor cellrefiner -dm_plex_transform_type refine_sbr -dm_plex_transform_label_match_strata -dm_view -dm_fine_view -refine_data_start_date 2011,01,01,00,01 -refine_data_dir mms_triangles_dx1-${PETSC_ID_TYPE} ${${config}_args})
    endforeach()
  endforeach()
endforeach()

# Demonstration cases refined by the solution itself (water-surface jump), each
# reporting the water volume across every adapt.
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/channel_wave_amr.yaml
     ${CMAKE_CURRENT_SOURCE_DIR}/dam_break_amr.yaml
     ${MESH_DIR}/channel_100x4_tri.msh
     ${MESH_DIR}/planar_dam_20x10_tri.msh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# the transform RDycore's AMR needs is set in the code, so the cases ask only
# for the criterion
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
