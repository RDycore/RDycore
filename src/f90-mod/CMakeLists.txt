set(RDYCORE_F90_SOURCE_FILES rdycore.F90)
if (ENABLE_SANITIZERS)
  # we rebuild the C source files without Address Sanitizer so the Fortran
  # compiler can link the driver
  foreach(source_file ${RDYCORE_C_SOURCE_FILES})
    list(APPEND RDYCORE_F90_SOURCE_FILES ../${source_file})
  endforeach()
  add_library(rdycore_f90
    ${RDYCORE_F90_SOURCE_FILES}
  )
  target_link_libraries(rdycore_f90 cyaml yaml muparser hdf5 ceed ${SYSTEM_LIBRARIES})
else()
  # stack the Fortran library on top of the C library
  add_library(rdycore_f90
    ${RDYCORE_F90_SOURCE_FILES}
  )
  target_link_libraries(rdycore_f90 rdycore_c cyaml yaml ceed ${SYSTEM_LIBRARIES})
endif()
target_include_directories(rdycore_f90 PUBLIC ${PROJECT_BINARY_DIR}/include/private)

install(
  TARGETS rdycore_f90
  DESTINATION lib
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/rdycore.mod
  DESTINATION include
)

add_subdirectory(tests)
