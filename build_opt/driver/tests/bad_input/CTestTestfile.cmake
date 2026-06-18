# CMake generated Testfile for 
# Source directory: /Users/wsugiarta/Documents/github/RDycore/driver/tests/bad_input
# Build directory: /Users/wsugiarta/Documents/github/RDycore/build_opt/driver/tests/bad_input
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(nonexistent_boundary "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "-n" "1" "/Users/wsugiarta/Documents/github/RDycore/build_opt/driver/tests/../rdycore" "nonexistent_boundary.yaml")
set_tests_properties(nonexistent_boundary PROPERTIES  WILL_FAIL "TRUE" _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/driver/tests/bad_input/CMakeLists.txt;11;add_test;/Users/wsugiarta/Documents/github/RDycore/driver/tests/bad_input/CMakeLists.txt;0;")
