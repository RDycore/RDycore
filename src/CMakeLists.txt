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

install(
  TARGETS rdycore
  DESTINATION lib
)
set(RDYCORE_LIBRARIES rdycore;yaml;${RDYCORE_LIBRARIES})

# Unit tests for the main library
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(f90-mod)

# Pass RDYCORE_LIBRARIES to the parent directory.
set(RDYCORE_LIBRARIES ${RDYCORE_LIBRARIES} PARENT_SCOPE)
