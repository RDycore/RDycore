# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/wsugiarta/Documents/github/RDycore/external/libyaml")
  file(MAKE_DIRECTORY "/Users/wsugiarta/Documents/github/RDycore/external/libyaml")
endif()
file(MAKE_DIRECTORY
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/libyaml"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/libyaml/tmp"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/libyaml/src/yaml_proj-stamp"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/libyaml/src"
  "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/libyaml/src/yaml_proj-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/libyaml/src/yaml_proj-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/wsugiarta/Documents/github/RDycore/build_opt/external/libyaml/src/yaml_proj-stamp${cfgdir}") # cfgdir has leading slash
endif()
