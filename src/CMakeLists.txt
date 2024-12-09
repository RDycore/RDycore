set(RDYCORE_C_SOURCE_FILES
  ceed.c
  checkpoint.c
  ensemble.c
  rdyadvance.c
  rdycore.c
  rdydata.c
  rdydm.c
  rdymesh.c
  rdymms.c
  rdysetup.c
  operator.c
  operator_diagnostics.c
  swe/swe_petsc.c
  swe/swe_ceed.c
  time_series.c
  xdmf_output.c
  yaml_input.c
)

add_library(rdycore ${RDYCORE_C_SOURCE_FILES})
target_link_libraries(rdycore cyaml yaml muparser hdf5 ceed ${SYSTEM_LIBRARIES})
add_dependencies(rdycore cyaml yaml)

install(
  TARGETS rdycore
  DESTINATION lib
)
if (ENABLE_ASAN)
  set(SANITIZER_OPTIONS address,undefined)
  target_compile_options(rdycore PUBLIC -fsanitize=${SANITIZER_OPTIONS})
  target_link_options(rdycore PUBLIC -fsanitize=${SANITIZER_OPTIONS})
endif()

# Unit tests for the main library
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(f90-mod)
