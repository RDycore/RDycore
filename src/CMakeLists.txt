# Main library -- we list all source files here (incl. those in subdirectories)
add_library(rdycore
  checkpoint.c
  ensemble.c
  rdycore.c
  rdydata.c
  rdydm.c
  rdymesh.c
  rdyadvance.c
  rdysetup.c
  swe/physics_swe.c
  swe/swe_petsc.c
  swe/swe_ceed.c
  time_series.c
  xdmf_output.c
  yaml_input.c
)
add_dependencies(rdycore cyaml yaml)

install(
  TARGETS rdycore
  DESTINATION lib
)
set(RDYCORE_LIBRARIES rdycore;cyaml;yaml;hdf5;${RDYCORE_LIBRARIES})

# Unit tests for the main library
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(f90-mod)

# Pass RDYCORE_LIBRARIES to the parent directory.
set(RDYCORE_LIBRARIES ${RDYCORE_LIBRARIES} PARENT_SCOPE)
