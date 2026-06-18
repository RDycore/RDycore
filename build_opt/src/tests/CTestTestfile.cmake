# CMake generated Testfile for 
# Source directory: /Users/wsugiarta/Documents/github/RDycore/src/tests
# Build directory: /Users/wsugiarta/Documents/github/RDycore/build_opt/src/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_rdyinit_np_1 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "1" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/tests/test_rdyinit")
set_tests_properties(test_rdyinit_np_1 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;17;add_test;/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;0;")
add_test(test_rdyinit_np_2 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "2" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/tests/test_rdyinit")
set_tests_properties(test_rdyinit_np_2 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;17;add_test;/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;0;")
add_test(test_rdymesh_np_1 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "1" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/tests/test_rdymesh")
set_tests_properties(test_rdymesh_np_1 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;17;add_test;/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;0;")
add_test(test_rdymesh_np_2 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "2" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/tests/test_rdymesh")
set_tests_properties(test_rdymesh_np_2 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;17;add_test;/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;0;")
add_test(test_yaml_input_np_1 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "1" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/tests/test_yaml_input")
set_tests_properties(test_yaml_input_np_1 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;17;add_test;/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;0;")
add_test(test_yaml_input_np_2 "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "2" "/Users/wsugiarta/Documents/github/RDycore/build_opt/src/tests/test_yaml_input")
set_tests_properties(test_yaml_input_np_2 PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;17;add_test;/Users/wsugiarta/Documents/github/RDycore/src/tests/CMakeLists.txt;0;")
