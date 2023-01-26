# Main library -- we list all source files here (incl. those in subdirectories)
add_executable(rdycore_exe
  main.c
)
set_target_properties(rdycore_exe PROPERTIES OUTPUT_NAME rdycore)
target_link_libraries(rdycore_exe PRIVATE ${RDYCORE_LIBRARIES})
target_code_coverage(rdycore_exe AUTO ALL)
install(
  TARGETS rdycore_exe
  DESTINATION bin
)

# Tests for the driver.
add_subdirectory(tests)
