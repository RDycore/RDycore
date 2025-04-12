find_program(H5DIFF h5diff REQUIRED
             PATHS ${PETSC_DIR}/${PETSC_ARCH}/bin)
set(amr_c_driver ${amr_driver})

# arguments for specific configurations ("basic" config has no options)
set(ceed_args -ceed /cpu/self)
set(preload_args -preload)

# AMR driver (C and Fortran)
file(COPY
     ${CMAKE_CURRENT_SOURCE_DIR}/amr_dx1.yaml
     ${MESH_DIR}/mms_triangles_dx1.exo
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
foreach(np 1)
  foreach(config basic)
    foreach(lang c)
      add_test(amr_${lang}_np_${np}_${config} ${MPIEXEC} ${MPIEXEC_FLAGS} -n ${np} ${amr_${lang}_driver} amr_dx1.yaml -dm_plex_transform_type refine_sbr -dm_plex_transform_active adapt -dm_view -dm_fine_view ${${config}_args})
    endforeach()
  endforeach()
endforeach()
