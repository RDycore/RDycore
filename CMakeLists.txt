cmake_minimum_required (VERSION 3.12.0)

# CMake files live in the cmake/ directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Find PETSc.
find_package(PETSc COMPONENTS C)

# Use PETSc's compiler config.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER ${PETSC_COMPILER})
set(CMAKE_C_FLAGS    "${PETSC_DEFINITIONS}")

# Default Fortran compiler (if not specified)
if (NOT CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER gfortran)
endif()

# Use a specific version of clang-format.
set(CLANG_FORMAT_VERSION 14)

enable_language(C)
enable_language(Fortran)

project(rdycore)

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

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

# We look for headers in these directories.
include_directories(${PETSC_INCLUDES})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # for generated rdycore.h

# Generate rdycore.h from include/rdycore.h.in.
configure_file(
  ${PROJECT_SOURCE_DIR}/include/rdycore.h.in
  ${PROJECT_BINARY_DIR}/include/rdycore.h
  @ONLY
)

# Inherit libraries from PETSc.
set(RDYCORE_LIBRARIES ${PETSC_LIBRARIES})

# Enable testing.
include(CTest)
enable_testing()

# Build any additional external libraries.
add_subdirectory(external)

# Create targets for RDycore library and unit tests.
add_subdirectory(src)

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

