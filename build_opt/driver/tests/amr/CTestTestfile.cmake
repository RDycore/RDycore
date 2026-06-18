# CMake generated Testfile for 
# Source directory: /Users/wsugiarta/Documents/github/RDycore/driver/tests/amr
# Build directory: /Users/wsugiarta/Documents/github/RDycore/build_opt/driver/tests/amr
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(amr_c_np_3_basic "/Users/wsugiarta/Documents/github/petsc/arch-opt/bin/mpiexec" "--oversubscribe" "-n" "3" "/Users/wsugiarta/Documents/github/RDycore/build_opt/driver/tests/../rdycore_amr" "amr_dx1.yaml" "-options_left" "-dm_adaptor" "cellrefiner" "-dm_plex_transform_type" "refine_sbr" "-dm_plex_transform_label_match_strata" "-dm_view" "-dm_fine_view" "-refine_data_start_date" "2011,01,01,00,01" "-refine_data_dir" "mms_triangles_dx1-int32")
set_tests_properties(amr_c_np_3_basic PROPERTIES  _BACKTRACE_TRIPLES "/Users/wsugiarta/Documents/github/RDycore/driver/tests/amr/CMakeLists.txt;26;add_test;/Users/wsugiarta/Documents/github/RDycore/driver/tests/amr/CMakeLists.txt;0;")
