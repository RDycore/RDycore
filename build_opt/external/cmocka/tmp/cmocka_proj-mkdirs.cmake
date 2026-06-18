# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/wsugiarta/Documents/github/RDycore/external/cmocka")
  file(MAKE_DIRECTORY "/Users/wsugiarta/Documents/github/RDycore/external/cmocka")
endif()
file(MAKE_DIRECTORY
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/cmocka"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/cmocka/tmp"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/cmocka/src/cmocka_proj-stamp"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/cmocka/src"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/cmocka/src/cmocka_proj-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/cmocka/src/cmocka_proj-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/cmocka/src/cmocka_proj-stamp${cfgdir}") # cfgdir has leading slash
endif()
