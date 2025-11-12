# This file builds any libraries that we need that PETSc doesn't provide.
# It uses CMake's ExternalProject machinery, described here:
# https://cmake.org/cmake/help/latest/module/ExternalProject.html

include(ExternalProject)

# we need this for some finicky library installation paths
include(GNUInstallDirs)

find_package(Patch REQUIRED)

#----------------------------------------
# Build libyaml, a YAML parsing library.
#----------------------------------------
set(LIBYAML_CMAKE_OPTS
  -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)
set(LIBYAML_LIB_NAME libyaml${CMAKE_STATIC_LIBRARY_SUFFIX})
ExternalProject_Add(yaml_proj
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libyaml
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libyaml
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libyaml
  INSTALL_DIR ${CMAKE_BINARY_DIR}
  CMAKE_ARGS ${LIBYAML_CMAKE_OPTS}
  LOG_CONFIGURE TRUE
  LOG_BUILD TRUE
  LOG_INSTALL TRUE
  LOG_OUTPUT_ON_FAILURE TRUE
  BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/lib/${LIBYAML_LIB_NAME}
)
add_library(yaml SHARED IMPORTED GLOBAL)
set_target_properties(yaml PROPERTIES
  IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/${LIBYAML_LIB_NAME})
add_dependencies(yaml yaml_proj)
install(
  FILES ${CMAKE_BINARY_DIR}/lib/${LIBYAML_LIB_NAME}
  DESTINATION lib
)

#---------------------------------------------------------------------
# Build libcyaml, a schema-based YAML parser built on top of libyaml.
#---------------------------------------------------------------------
# I could not get libcyaml to build using ExternalProject_Add, so we're just
# building it as a proper target here. -JNJ 2024/09/12
add_library(cyaml STATIC
  libcyaml/src/mem.c
  libcyaml/src/free.c
  libcyaml/src/load.c
  libcyaml/src/save.c
  libcyaml/src/util.c
  libcyaml/src/utf8.c
)
target_include_directories(cyaml PUBLIC
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/libcyaml/include
)
target_compile_definitions(cyaml PUBLIC
  VERSION_MAJOR=1
  VERSION_MINOR=4
  VERSION_PATCH=2
  VERSION_DEVEL=0
)
add_dependencies(cyaml yaml)
install(
  TARGETS cyaml
  DESTINATION lib
)

#----------------------------------------------------------------------------
# Build mat-ceed, a supplement to libceed that adds support for COO matrices
#----------------------------------------------------------------------------
add_library(mat_ceed STATIC
  mat-ceed/mat-ceed.c
)
target_include_directories(mat_ceed PUBLIC
  ${CMAKE_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/mat-ceed
  ${PETSC_DIR}/${PETSC_ARCH}/externalpackages/git.libceed/include
)
install(
  TARGETS mat_ceed
  DESTINATION lib
)

if (ENABLE_TESTS)
  #----------------------------------------
  # Build cmocka, a C unit testing library.
  #----------------------------------------
  set(CMOCKA_CMAKE_OPTS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DBUILD_SHARED_LIBS=OFF
    -DWITH_EXAMPLES=OFF)
  if (CMAKE_C_COMPILER_ID STREQUAL "Intel" OR CMAKE_C_COMPILER_ID MATCHES "AppleClang")
    set(CMOCKA_CMAKE_OPTS ${CMOCKA_CMAKE_OPTS}
      -DHAVE_WORDS_BIGENDIAN=TRUE
      -DWORDS_BIGENDIAN=FALSE)
  endif()
  set(CMOCKA_LIB_NAME libcmocka${CMAKE_STATIC_LIBRARY_SUFFIX})
  ExternalProject_Add(cmocka_proj
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cmocka
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmocka
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/cmocka
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CMAKE_ARGS ${CMOCKA_CMAKE_OPTS}
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    LOG_OUTPUT_ON_FAILURE TRUE
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMOCKA_LIB_NAME}
  )
  add_library(cmocka STATIC IMPORTED GLOBAL)
  set_target_properties(cmocka PROPERTIES
    IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMOCKA_LIB_NAME})
  add_dependencies(cmocka cmocka_proj)
endif()
