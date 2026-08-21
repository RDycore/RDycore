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
  operator_fluxes_hydro_recon_ceed.c
  operator_fluxes_petsc.c
  operator_sources_ceed.c
  operator_sources_hydro_recon_ceed.c
  operator_sources_petsc.c
  petsc_operator.c
  swe/swe_ceed.c
  swe/swe_jacobian_petsc.c
  swe/swe_petsc.c
  tracer/tracer_ceed.c
  tracer/tracer_petsc.c
  forcing/rdyforcing.c
  forcing/rdyforcing_dataset.c
  forcing/rdyforcing_map.c
  time_series.c
  xdmf_output.c
  yaml_input.c
)

# generate <physics>_ceed.h headers inside each CEED operator implementation
# directory so the JIT compilers have everything they need
foreach(physics swe;tracer)
  string(TOUPPER ${physics} PHYSICS)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/private/physics_ceed.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/${physics}/${physics}_ceed.h
    @ONLY
  )
endforeach()

# Device assembly TU for the SWE analytic Jacobian: compiled OUTSIDE CMake by
# PETSc's .kokkos.cxx rule (kokkos_compile.mk), which picks nvcc_wrapper on
# CUDA builds; the resulting object is linked into rdycore_c.
if (RDY_HAVE_KOKKOS_JACOBIAN)
  set(swe_jk_obj ${CMAKE_CURRENT_BINARY_DIR}/swe_jacobian_kokkos.o)
  add_custom_command(
    OUTPUT ${swe_jk_obj}
    COMMAND make -f ${CMAKE_CURRENT_SOURCE_DIR}/swe/kokkos_compile.mk
            PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH}
            OBJ=${swe_jk_obj}
            SRC=${CMAKE_CURRENT_SOURCE_DIR}/swe/swe_jacobian_kokkos.kokkos.cxx
            "RDY_INCLUDES=-I${PROJECT_SOURCE_DIR}/include -I${PROJECT_BINARY_DIR}/include"
            ${swe_jk_obj}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/swe/swe_jacobian_kokkos.kokkos.cxx
            ${CMAKE_CURRENT_SOURCE_DIR}/swe/swe_jacobian_kokkos.h
            ${CMAKE_CURRENT_SOURCE_DIR}/swe/swe_roe_flux_jacobian_petsc.h
            ${CMAKE_CURRENT_SOURCE_DIR}/swe/swe_roe_flux_petsc.h
            ${CMAKE_CURRENT_SOURCE_DIR}/swe/swe_types_petsc.h
            ${PROJECT_SOURCE_DIR}/include/private/rdymathimpl.h
            ${CMAKE_CURRENT_SOURCE_DIR}/swe/kokkos_compile.mk
    COMMENT "Compiling swe_jacobian_kokkos.kokkos.cxx with PETSc's Kokkos rule"
    VERBATIM
  )
  set_source_files_properties(${swe_jk_obj} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  list(APPEND RDYCORE_C_SOURCE_FILES ${swe_jk_obj})
endif()

add_library(rdycore_c ${RDYCORE_C_SOURCE_FILES})
target_link_libraries(rdycore_c cyaml yaml muparser hdf5 ceed ${SYSTEM_LIBRARIES})
if (RDY_HAVE_KOKKOS_JACOBIAN)
  target_link_libraries(rdycore_c ${RDY_KOKKOS_LIB_FLAGS})
endif()
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
