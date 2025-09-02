cmake_minimum_required (VERSION 3.14.0)

# Configurable options
option(ENABLE_DRIVER   "Enable stand-alone RDycore driver program(s)" ON)
option(ENABLE_FORTRAN  "Enable Fortran library bindings" ON)
option(ENABLE_TESTS    "Enable unit and driver tests" ON)
option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(ENABLE_SANITIZERS "Enable available sanitizers (ASan, MSan, UBSan)" OFF)

set(MAX_NAME_LEN              128 CACHE STRING "Maximum allowable length of field/variable names")
set(MAX_NUM_SEDIMENT_CLASSES    5 CACHE STRING "Maximum number of sediment size classes")
set(MAX_NUM_FIELDS             10 CACHE STRING "Maximum number of fields in a model's state")

# Material property (integer) identifiers.
# (we generate them here to allow GPU JIT compilers to see them)
set(MATERIAL_PROPERTY_MANNINGS 0)
set(NUM_MATERIAL_PROPERTIES    1)

# options dependent upon above options
math(EXPR MAX_NUM_FIELD_COMPONENTS "3 + ${MAX_NUM_SEDIMENT_CLASSES}")

# CMake files live in the cmake/ directory.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Use a specific version of clang-format.
set(CLANG_FORMAT_VERSION 14)

project(rdycore)
enable_language(C)
if (ENABLE_FORTRAN)
  enable_language(Fortran)
endif()

# Make sure PETSc can be found with pkgconfig
set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH $ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${PETSC_DIR}/${PETSC_ARCH}/lib/pkgconfig")

find_package(PkgConfig REQUIRED)

# PETSc
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)
pkg_get_variable(PETSc_C_COMPILER PETSc ccompiler)
include_directories(${PETSc_INCLUDE_DIRS})
link_directories(${PETSc_LIBRARY_DIRS})
list(APPEND CMAKE_BUILD_RPATH ${PETSC_DIR}/${PETSC_ARCH}/lib)
list(APPEND CMAKE_INSTALL_RPATH ${PETSC_DIR}/${PETSC_ARCH}/lib)

# CEED
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

# Determine MPIEXEC and MPIEXEC_FLAGS using PETSc's configuration.
# (It would be nice if this were included in PETSc.pc)
include(extract_petsc_variable)
extract_petsc_variable("MPIEXEC" MPIEXEC)
# If MPIEXEC includes spaces, put anything after the first space into
# MPIEXEC_FLAGS
string(FIND ${MPIEXEC} " " space)
if (NOT space EQUAL -1)
  math(EXPR start "${space} + 1")
  string(SUBSTRING ${MPIEXEC} ${start} -1 MPIEXEC_FLAGS)
  string(SUBSTRING ${MPIEXEC} 0 ${space} MPIEXEC)
endif()

# Use PETSc's compiler config.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_COMPILER ${PETSc_C_COMPILER})

# Add some compiler flags of our own.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmax-errors=10")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-value")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-format-truncation")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror=vla")
elseif (CMAKE_C_COMPILER_ID MATCHES "Clang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ferror-limit=10")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror=vla")
endif()

# emit compile_commands.json for use with clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set linker flags where appropriate
if (APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_compact_unwind")
endif()

if (ENABLE_SANITIZERS)
  if (APPLE AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64") # no memory sanitizer!
    set(SANITIZERS address,undefined)
  elseif (CMAKE_C_COMPILER_ID MATCHES "Clang")
    # clang should provide MemorySanitizer
    set(SANITIZERS address,memory,undefined)
  elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(SANITIZERS address,undefined)
  else() # probably no sanitizers available for this compiler(?)
    message(FATAL_ERROR "The selected C compiler (${CMAKE_C_COMPILER_ID}) doesn't support sanitizers.")
  endif()
  message(STATUS "Enabling sanitizers (${SANITIZERS})...")
endif()

# Fortran settings
if (ENABLE_FORTRAN)
  pkg_get_variable(PETSc_Fortran_COMPILER PETSc fcompiler)
  include(CheckFortranCompilerFlag)
  set(CMAKE_Fortran_COMPILER ${PETSc_Fortran_COMPILER})
  if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    list(APPEND fflags "-fallow-argument-mismatch")
    list(APPEND fflags "-Wno-allow-argument-mismatch")
    list(APPEND fflags "-ffree-line-length-none")
    foreach(fflag ${fflags})
      check_fortran_compiler_flag(${fflag} result)
      if (result)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${fflag}")
      endif()
    endforeach()

    # make sure libgfortran appears in our rpath
    execute_process(COMMAND ${CMAKE_Fortran_COMPILER} -print-file-name=libgfortran${CMAKE_SHARED_LIBRARY_SUFFIX}
                    OUTPUT_VARIABLE gfortran_library_location)
    get_filename_component(gfortran_library_dir ${gfortran_library_location} DIRECTORY)
    list(APPEND CMAKE_BUILD_RPATH ${gfortran_library_dir})
    list(APPEND CMAKE_INSTALL_RPATH ${gfortran_library_dir})
  endif()
endif()

# By default, we build a Debug configuration.
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE not specified. Building Debug configuration.")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

message(STATUS "Configuring RDycore (${CMAKE_BUILD_TYPE})")
message(STATUS "Generating project files in build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "C compiler is ${CMAKE_C_COMPILER} (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")
if (ENABLE_FORTRAN)
  message(STATUS "Fortran compiler is ${CMAKE_Fortran_COMPILER} (${CMAKE_Fortran_COMPILER_ID} ${CMAKE_Fortran_COMPILER_VERSION})")
endif()

# Version numbers (written to private/config.h).
set(RDYCORE_MAJOR_VERSION 0)
set(RDYCORE_MINOR_VERSION 1)
set(RDYCORE_PATCH_VERSION 0)
set(RDYCORE_VERSION "${RDYCORE_MAJOR_VERSION}.${RDYCORE_MINOR_VERSION}.${RDYCORE_PATCH_VERSION}")

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  OUTPUT_VARIABLE RDYCORE_GIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

if (RDYCORE_GIT_HASH)
  message(STATUS "Configuring RDycore v${RDYCORE_VERSION} (git hash ${RDYCORE_GIT_HASH})")
else()
  set(RDYCORE_GIT_HASH "unknown")
  message(STATUS "Configuring RDycore v${RDYCORE_VERSION}")
endif()

# For now, we build static libraries only.
set(BUILD_SHARED_LIBS OFF)
# ...but for external shared libraries, we set RPATH.
list(APPEND CMAKE_BUILD_RPATH ${PROJECT_BINARY_DIR}/lib)
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

# We look for headers in these directories.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # for generated rdycore.h

# Generate a header from our build parameters.
configure_file(
  ${PROJECT_SOURCE_DIR}/include/private/config.h.in
  ${PROJECT_BINARY_DIR}/include/private/config.h
  @ONLY
)

# Generate a header that handles our maximum number of sediment classes
include(generate_cyaml_sediment_fields)
generate_cyaml_sediment_fields()

# Inherit libraries from PETSc for shared or static libs.
extract_petsc_variable("BUILDSHAREDLIB" petsc_shared)
if (petsc_shared STREQUAL "yes")
  set(SYSTEM_LIBRARIES ${PETSc_LIBRARIES} PkgConfig::CEED m stdc++)
else()
  set(SYSTEM_LIBRARIES ${PETSc_STATIC_LIBRARIES} PkgConfig::CEED m stdc++)
endif()

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
  if (ENABLE_FORTRAN)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} --coverage")
  endif()
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

