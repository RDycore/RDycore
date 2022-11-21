# Main library -- we list all source files here (incl. those in subdirectories)
add_library(rdycore
  rdycore.c
  rdymesh.c
)
target_code_coverage(rdycore AUTO ALL)
install(
  TARGETS rdycore
  DESTINATION lib
)
set(RDYCORE_LIBRARIES rdycore;${RDYCORE_LIBRARIES})

# Unit tests for the main library
add_subdirectory(tests)

# Pass RDYCORE_LIBRARIES to the parent directory.
set(RDYCORE_LIBRARIES ${RDYCORE_LIBRARIES} PARENT_SCOPE)
