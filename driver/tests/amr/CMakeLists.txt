find_program(H5DIFF h5diff REQUIRED
             PATHS ${PETSC_DIR}/${PETSC_ARCH}/bin)
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
      add_test(amr_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_${lang}_driver} amr_dx1.yaml -dm_plex_transform_type refine_sbr -dm_plex_transform_active_adapt -dm_view -dm_fine_view -refine_data_start_date 2011,01,01,00,01 -refine_data_dir mms_triangles_dx1-${PETSC_ID_TYPE} ${${config}_args})
    endforeach()
  endforeach()
endforeach()
