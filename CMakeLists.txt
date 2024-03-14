cmake_minimum_required (VERSION 3.14.0)

# Configurable options
option(ENABLE_DRIVER   "Enable stand-alone RDycore driver program" ON)
option(ENABLE_TESTS    "Enable unit (and possibly driver) tests" ON)
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(ENABLE_ASAN     "Enable Address Sanitizer (if available)" OFF)

# CMake files live in the cmake/ directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Use a specific version of clang-format.
set(CLANG_FORMAT_VERSION 14)

enable_language(C)
enable_language(Fortran)

project(rdycore)

# Find PETSc (must be placed after the project command)
set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH $ENV{PETSC_ARCH})
find_package(PETSc COMPONENTS C)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig")

find_package(PkgConfig REQUIRED)
pkg_check_modules(CEED REQUIRED IMPORTED_TARGET ceed)
include_directories(${CEED_INCLUDE_DIRS})

# Determine PETSc's ID type (int32 or int64) by inspecting petscconf.h
file(READ "${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h" petscconf_h)
string(FIND ${petscconf_h} "#define PETSC_USE_64BIT_INDICES 1" index)
if (index EQUAL -1) # PETSC_USE_64BIT_INDICES not defined
  message(STATUS "PETSc is configured with 32-bit indices")
  set(PETSC_ID_TYPE "int32")
else()
  message(STATUS "PETSc is configured with 64-bit indices")
  set(PETSC_ID_TYPE "int64")
endif()

# Use PETSc's compiler config.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER ${PETSC_COMPILER})
set(CMAKE_C_FLAGS    "${PETSC_DEFINITIONS}")

# Add some compiler flags of our own.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmax-errors=10")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-value")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-format-truncation")
elseif (CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ferror-limit=10")
endif()

# Set linker flags where appropriate
if (APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind")
endif()

if (ENABLE_ASAN)
  message(STATUS "Enabling Address Sanitizer...")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# Sometimes PETSC_MPIEXEC has options embedded in it, which confuses CTest.
# If we find a space in this variable, we break things up into PETSC_MPIEXEC
# and PETSC_MPIEXEC_FLAGS.
string(FIND ${PETSC_MPIEXEC} " " PETSC_MPIEXEC_SPACE)
if (NOT PETSC_MPIEXEC_SPACE EQUAL -1)
  string(SUBSTRING ${PETSC_MPIEXEC} 0 ${PETSC_MPIEXEC_SPACE} PETSC_MPIEXEC)
  string(SUBSTRING ${PETSC_MPIEXEC} ${PETSC_MPIEXEC_SPACE} -1 PETSC_MPIEXEC_FLAGS)
endif()

# Default Fortran compiler (if not specified)
if (NOT CMAKE_Fortran_COMPILER)
  set(CMAKE_Fortran_COMPILER gfortran)
endif()

# flags
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-line-length-none")
endif()

# By default, we build a Debug configuration.
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE not specified. Building Debug configuration.")
  set(CMAKE_BUILD_TYPE "Debug")
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

# Generate a header from our build parameters.
configure_file(
  ${PROJECT_SOURCE_DIR}/include/rdycore.h.in
  ${PROJECT_BINARY_DIR}/include/rdycore.h
  @ONLY
)

# Inherit libraries from PETSc.
set(RDYCORE_LIBRARIES ${PETSC_LIBRARIES} PkgConfig::CEED)

if (ENABLE_TESTS)
  include(CTest)
  enable_testing()
endif()

if (ENABLE_COVERAGE)
  message(STATUS "Enabling code coverage instrumentation")
  find_program(LCOV_EXE lcov DOC "Lcov code coverage tool")
  if (LCOV_EXE MATCHES "NOTFOUND")
    message(FATAL_ERROR "Could not find lcov for code coverage reporting!")
  endif()

  # Add code coverage compiler flags
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} --coverage")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")

  # Add a "make coverage" target.
  add_custom_target(coverage ${LCOV_EXE} --capture --directory . -o unfiltered_coverage.info
    COMMAND ${LCOV_EXE} --remove unfiltered_coverage.info '*/external/*' -o coverage.info
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Generating coverage report (coverage.info)")
endif()

# Build any additional external libraries.
add_subdirectory(external)

# Include miscellaneous data and utilities.
add_subdirectory(share)

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

