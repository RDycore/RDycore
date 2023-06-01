# This file builds any libraries that we need that PETSc doesn't provide.
# It uses CMake's ExternalProject machinery, described here:
# https://cmake.org/cmake/help/latest/module/ExternalProject.html

include(ExternalProject)

# we need this for some finicky library installation paths
include(GNUInstallDirs)

#----------------------------------------
# Build libyaml, a YAML parsing library.
#----------------------------------------
set(LIBYAML_C_FLAGS ${CMAKE_C_FLAGS})
set(LIBYAML_CMAKE_OPTS
  -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DCMAKE_C_FLAGS=${LIBYAML_C_FLAGS}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
set(LIBYAML_LIB_NAME libyaml${CMAKE_STATIC_LIBRARY_SUFFIX})
ExternalProject_Add(yaml_proj
                    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libyaml
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libyaml
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libyaml
                    INSTALL_DIR ${PROJECT_BINARY_DIR}
                    CMAKE_ARGS ${LIBYAML_CMAKE_OPTS}
                    LOG_CONFIGURE TRUE
                    LOG_BUILD TRUE
                    LOG_INSTALL TRUE
                    BUILD_BYPRODUCTS ${PROJECT_BINARY_DIR}/lib/${LIBYAML_LIB_NAME})
add_library(yaml SHARED IMPORTED GLOBAL)
set_target_properties(yaml PROPERTIES
  IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/${LIBYAML_LIB_NAME})
add_dependencies(yaml yaml_proj)

#---------------------------------------------------------------------
# Build libcyaml, a schema-based YAML parser built on top of libyaml.
#---------------------------------------------------------------------
string(TOLOWER ${CMAKE_BUILD_TYPE} LIBCYAML_VARIANT)
set(LIBCYAML_LIB_NAME libcyaml${CMAKE_STATIC_LIBRARY_SUFFIX})
set(LIBCYAML_YAML_CFLAGS "-I${PROJECT_BINARY_DIR}/include")# ${LIBYAML_C_FLAGS}")
set(LIBCYAML_YAML_LDFLAGS "-L${PROJECT_BINARY_DIR}/lib -lyaml")
ExternalProject_Add(cyaml_proj
                    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/libcyaml
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libcyaml
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libcyaml
                    INSTALL_DIR ${PROJECT_BINARY_DIR}
                    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/libcyaml ${CMAKE_CURRENT_BINARY_DIR}/libcyaml
                    BUILD_COMMAND env PREFIX=${PROJECT_BINARY_DIR} CFLAGS=${LIBCYAML_YAML_CFLAGS} LDFLAGS=${LIBCYAML_YAML_LDFLAGS} make all VARIANT=${LIBCYAML_VARIANT}
                    INSTALL_COMMAND env PREFIX=${PROJECT_BINARY_DIR} make install VARIANT=${LIBCYAML_VARIANT}
                    DEPENDS yaml
                    LOG_BUILD TRUE
                    LOG_INSTALL TRUE
                    BUILD_BYPRODUCTS ${PROJECT_BINARY_DIR}/lib/${LIBCYAML_LIB_NAME})
add_library(cyaml SHARED IMPORTED GLOBAL)
set_target_properties(cyaml PROPERTIES
  IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/${LIBCYAML_LIB_NAME})
add_dependencies(cyaml cyaml_proj)

if (ENABLE_TESTS)
  #----------------------------------------
  # Build cmocka, a C unit testing library.
  #----------------------------------------
  set(CMOCKA_C_FLAGS ${CMAKE_C_FLAGS})
  set(CMOCKA_CMAKE_OPTS
    -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_FLAGS=${CMOCKA_C_FLAGS}
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
                      INSTALL_DIR ${PROJECT_BINARY_DIR}
                      CMAKE_ARGS ${CMOCKA_CMAKE_OPTS}
                      LOG_CONFIGURE TRUE
                      LOG_BUILD TRUE
                      LOG_INSTALL TRUE
                      BUILD_BYPRODUCTS ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMOCKA_LIB_NAME})
  add_library(cmocka STATIC IMPORTED GLOBAL)
  set_target_properties(cmocka PROPERTIES
    IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMOCKA_LIB_NAME})
  add_dependencies(cmocka cmocka_proj)
endif()
