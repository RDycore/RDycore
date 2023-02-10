# Main library -- we list all source files here (incl. those in subdirectories)
add_library(rdycore
  parse_config_file.c
  rdycore.c
  rdyprintf.c
  rdymesh.c
  rdysetup.c
)
add_dependencies(rdycore yaml)
target_code_coverage(rdycore AUTO ALL)
install(
  TARGETS rdycore
  DESTINATION lib
)
set(RDYCORE_LIBRARIES rdycore;yaml;${RDYCORE_LIBRARIES})

# Unit tests for the main library
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Pass RDYCORE_LIBRARIES to the parent directory.
set(RDYCORE_LIBRARIES ${RDYCORE_LIBRARIES} PARENT_SCOPE)
