# This macro adds a "make memcheck" target that runs Valgrind on all tests.
# It only works on Linux.
macro(add_memcheck_target)
  if (UNIX AND NOT APPLE)
    find_package(Valgrind QUIET)
    if (Valgrind_FOUND)
      set(Valgrind_FOUND 1) # regularize this value
      include_directories(${Valgrind_INCLUDE_DIR})
      set(MEMORYCHECK_COMMAND ${Valgrind_EXECUTABLE})
      # Add "--gen-suppressions=all" to MEMORYCHECK_COMMAND_OPTIONS to generate
      # suppressions for Valgrind's false positives. The suppressions show up
      # right in the MemoryChecker.*.log files.
      set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=definite,possible --track-origins=yes --error-exitcode=1 --trace-children=yes --suppressions=${PROJECT_SOURCE_DIR}/tools/valgrind/scasm.supp" CACHE STRING "Options passed to Valgrind." FORCE)

      # make memcheck target
      add_custom_target(memcheck ctest -T memcheck -j USES_TERMINAL)
    else()
      set(Valgrind_FOUND 0)
    endif()
  else()
    # Valgrind doesn't work on Macs.
    set(Valgrind_FOUND 0)
  endif()

endmacro()
