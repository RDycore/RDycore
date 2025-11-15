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
  rdyamr.c
  rdysetup.c
  operator.c
  operator_fluxes_ceed.c
  operator_fluxes_petsc.c
  operator_sources_ceed.c
  operator_sources_petsc.c
  petsc_operator.c
  swe/swe_ceed.c
  swe/swe_petsc.c
  sediment/sediment_ceed.c
  sediment/sediment_petsc.c
  time_series.c
  xdmf_output.c
  yaml_input.c
)

# generate <physics>_ceed.h headers inside each CEED operator implementation
# directory so the JIT compilers have everything they need
foreach(physics swe;sediment)
  string(TOUPPER ${physics} PHYSICS)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/private/physics_ceed.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/${physics}/${physics}_ceed.h
    @ONLY
  )
endforeach()

add_library(rdycore_c ${RDYCORE_C_SOURCE_FILES})
target_link_libraries(rdycore_c cyaml yaml muparser hdf5 ceed ${SYSTEM_LIBRARIES})
add_dependencies(rdycore_c cyaml yaml)

install(
  TARGETS rdycore_c
  DESTINATION lib
)
if (ENABLE_SANITIZERS)
  target_compile_options(rdycore_c PUBLIC -fsanitize=${SANITIZERS})
  target_link_options(rdycore_c PUBLIC -fsanitize=${SANITIZERS})
endif()

# Unit tests for the main library
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if (ENABLE_FORTRAN)
  add_subdirectory(f90-mod)
endif()
