# Main library -- we list all source files here (incl. those in subdirectories)
add_library(rdycore
  read_config_file.c
  rdycore.c
  print_config.c
  rdymesh.c
  rdyrun.c
  rdysetup.c
)
add_dependencies(rdycore yaml)
target_code_coverage(rdycore AUTO ALL)

# Fortran library
add_library(rdycore_f90
  rdycore.F90
)
target_code_coverage(rdycore_f90 AUTO ALL)

install(
  TARGETS rdycore rdycore_f90
  DESTINATION lib
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/rdycore.mod
  DESTINATION include
)
set(RDYCORE_LIBRARIES rdycore_f90;rdycore;yaml;${RDYCORE_LIBRARIES})

# Unit tests for the main library
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Pass RDYCORE_LIBRARIES to the parent directory.
set(RDYCORE_LIBRARIES ${RDYCORE_LIBRARIES} PARENT_SCOPE)
