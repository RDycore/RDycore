# CMake generated Testfile for 
# Source directory: /Users/wsugiarta/Documents/github/RDycore/src/f90-mod/tests
# Build directory: /Users/wsugiarta/Documents/github/RDycore/build_opt/src/f90-mod/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_coupling_np_1 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "1" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/f90-mod/tests/test_coupling" "coupling.yaml")
set_tests_properties(test_coupling_np_1 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/f90-mod/tests/CMakeLists.txt;12;add_test;/Users/wsugiarta/Documents/github/RDycore/src/f90-mod/tests/CMakeLists.txt;0;")
add_test(test_coupling_np_2 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "2" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/f90-mod/tests/test_coupling" "coupling.yaml")
set_tests_properties(test_coupling_np_2 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/f90-mod/tests/CMakeLists.txt;12;add_test;/Users/wsugiarta/Documents/github/RDycore/src/f90-mod/tests/CMakeLists.txt;0;")
