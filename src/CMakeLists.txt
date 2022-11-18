# Main library
add_library(rdycore
  rdycore.c
  rdymesh.c
)
set(RDYCORE_LIBRARIES rdycore;${RDYCORE_LIBRARIES})

# Unit tests for the main library
add_subdirectory(tests)

# FV and FE libraries are built in subdirectories.
#add_subdirectory(fv)
#add_subdirectory(fe)

# Pass RDYCORE_LIBRARIES to the parent directory.
set(RDYCORE_LIBRARIES ${RDYCORE_LIBRARIES} PARENT_SCOPE)
