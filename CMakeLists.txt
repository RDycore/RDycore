cmake_minimum_required (VERSION 3.12.0)

# Configurable options
option(ENABLE_DRIVER "Enable stand-alone RDycore driver program" ON)
option(ENABLE_TESTS  "Enable unit (and possibly driver) tests" ON)

# Other build parameters
message(STATUS "Setting build parameters:")
function(check_or_set_param name value default)
  if (${value})
    if (${value} LESS_EQUAL 0)
      message(FATAL_ERROR "Invalid ${name}: ${value}")
    endif()
  else()
    set(${name} ${default} PARENT_SCOPE)
  endif()
  message(STATUS "  ${name}: ${value}")
endfunction()
check_or_set_param("MAX_NUM_REGIONS"    ${MAX_NUM_REGIONS}    32)
check_or_set_param("MAX_NUM_SURFACES"   ${MAX_NUM_SURFACES}   32)
check_or_set_param("MAX_NUM_CONDITIONS" ${MAX_NUM_CONDITIONS} 32)
set(MAX_NUM_REGIONS    ${MAX_NUM_REGIONS}    CACHE STRING "Maximum allowable number of regions")
set(MAX_NUM_SURFACES   ${MAX_NUM_SURFACES}   CACHE STRING "Maximum allowable number of surfaces")
set(MAX_NUM_CONDITIONS ${MAX_NUM_CONDITIONS} CACHE STRING "Maximum allowable number of conditions")

# CMake files live in the cmake/ directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Use a specific version of clang-format.
set(CLANG_FORMAT_VERSION 14)

enable_language(C)
enable_language(Fortran)

project(rdycore)

# Find PETSc (must be placed after the project command)
find_package(PETSc COMPONENTS C)

# Use PETSc's compiler config.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER ${PETSC_COMPILER})
set(CMAKE_C_FLAGS    "${PETSC_DEFINITIONS}")

# Default Fortran compiler (if not specified)
if (NOT CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER gfortran)
endif()

message(STATUS "Configuring RDycore (${CMAKE_BUILD_TYPE})")
message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "C compiler is ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")
message(STATUS "Fortran compiler is ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION})")

# Version numbers (written to rdycore.h).
set(RDYCORE_MAJOR_VERSION 0)
set(RDYCORE_MINOR_VERSION 1)
set(RDYCORE_PATCH_VERSION 0)
set(RDYCORE_VERSION "${RDYCORE_MAJOR_VERSION}.${RDYCORE_MINOR_VERSION}.${RDYCORE_PATCH_VERSION}")

message(STATUS "Configuring RDycore v${RDYCORE_VERSION}")

# For now, we build static libraries only.
set(BUILD_SHARED_LIBS OFF)
# ...but for external shared libraries, we set RPATH.
set(CMAKE_BUILD_RPATH ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

# We look for headers in these directories.
include_directories(${PETSC_INCLUDES})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # for generated rdycore.h

# Generate headers from build parameters.
configure_file(
  ${PROJECT_SOURCE_DIR}/include/rdycore.h.in
  ${PROJECT_BINARY_DIR}/include/rdycore.h
  @ONLY
)
configure_file(
  ${PROJECT_SOURCE_DIR}/include/private/rdyconfig.h.in
  ${PROJECT_BINARY_DIR}/include/private/rdyconfig.h
  @ONLY
)

# Inherit libraries from PETSc.
set(RDYCORE_LIBRARIES ${PETSC_LIBRARIES})

# Enable testing.
if (ENABLE_TESTS)
  include(CTest)
  enable_testing()
endif()

# Add code coverage support (enabled using -DENABLE_COVERAGE).
# This adds a `make ccov-all` that produces a single coverage report.
include(code_coverage)
add_code_coverage_all_targets()

# Build any additional external libraries.
add_subdirectory(external)

# Create targets for RDycore library and unit tests.
add_subdirectory(src)

# Build the driver if enabled.
if (ENABLE_DRIVER)
  add_subdirectory(driver)
endif()

#--------------------
# Additional targets
#--------------------

# Add a "make memcheck" target that runs Valgrind on all tests (Linux only).
include(add_memcheck_target)
add_memcheck_target()

# Create the following targets for checking code formatting:
# make format-c       <-- reformats C code to conform to desired style
# make format-c-check <-- checks C code formatting, reporting any errors
include(add_formatting_targets)
add_formatting_targets()

