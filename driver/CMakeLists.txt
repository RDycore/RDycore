# C driver
add_executable(rdycore_exe
  main.c
)
set_target_properties(rdycore_exe PROPERTIES OUTPUT_NAME rdycore)
target_link_libraries(rdycore_exe PRIVATE ${RDYCORE_LIBRARIES})
target_code_coverage(rdycore_exe AUTO ALL)

# Fortran driver
add_executable(rdycore_f90_exe
  main.F90
)
set_target_properties(rdycore_f90_exe PROPERTIES OUTPUT_NAME rdycore_f90)
target_include_directories(rdycore_f90_exe PRIVATE ${PROJECT_BINARY_DIR}/src/f90-mod)
target_link_libraries(rdycore_f90_exe PRIVATE ${RDYCORE_LIBRARIES})
install(
  TARGETS rdycore_exe rdycore_f90_exe
  DESTINATION bin
)

# Tests for the driver.
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()
