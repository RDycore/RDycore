# CMake generated Testfile for 
# Source directory: /Users/wsugiarta/Documents/github/RDycore/driver/tests
# Build directory: /Users/wsugiarta/Documents/github/RDycore/build_opt/driver/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(c_driver_usage "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "-n" "1" "/Users/wsugiarta/Documents/github/RDycore/build_opt/driver/tests/../rdycore")
set_tests_properties(c_driver_usage PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/driver/tests/CMakeLists.txt;8;add_test;/Users/wsugiarta/Documents/github/RDycore/driver/tests/CMakeLists.txt;0;")
add_test(f90_driver_usage "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "-n" "1" "/Users/wsugiarta/Documents/github/RDycore/build_opt/driver/tests/../rdycore_f90")
set_tests_properties(f90_driver_usage PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/driver/tests/CMakeLists.txt;14;add_test;/Users/wsugiarta/Documents/github/RDycore/driver/tests/CMakeLists.txt;0;")
subdirs("amr")
subdirs("swe_roe")
subdirs("sediment")
subdirs("bad_input")
