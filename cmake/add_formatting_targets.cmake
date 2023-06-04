# This macro creates the following targets for checking code formatting:
# make format-c       <-- reformats C code to conform to desired style
# make format-c-check <-- checks C code formatting, reporting any errors
macro(add_formatting_targets)
  find_program(CLANG_FORMAT clang-format)
  if (NOT CLANG_FORMAT STREQUAL "CLANG_FORMAT-NOTFOUND")
    # Is this our blessed version? If not, we create targets that warn the user
    # to obtain the right version.
    execute_process(COMMAND clang-format --version
      OUTPUT_VARIABLE CF_VERSION)
    string(STRIP ${CF_VERSION} CF_VERSION)
    if (NOT ${CF_VERSION} MATCHES ${CLANG_FORMAT_VERSION})
      add_custom_target(format-c
        echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
        "Please make sure this version appears in your path and rerun cmake.")
      add_custom_target(format-c-check
        echo "You have clang-format version ${CF_VERSION}, but ${CLANG_FORMAT_VERSION} is required."
        "Please make sure this version appears in your path and rerun cmake.")
    else()
      add_custom_target(format-c
        find ${PROJECT_SOURCE_DIR}/include -name "*.h" -exec ${CLANG_FORMAT} -i {} \+;
        COMMAND find ${PROJECT_SOURCE_DIR}/src -name "*.c" -exec ${CLANG_FORMAT} -i {} \+;
        VERBATIM
        COMMENT "Auto-formatting C code...")
      add_custom_target(format-c-check
        find ${PROJECT_SOURCE_DIR}/include -name "*.h" -exec ${CLANG_FORMAT} -n --Werror -ferror-limit=1 {} \+;
        COMMAND find ${PROJECT_SOURCE_DIR}/src -name "*.c" -exec ${CLANG_FORMAT} -n --Werror -ferror-limit=1 {} \+;
        VERBATIM
        COMMENT "Checking C formatting...")
    endif()
  endif()
endmacro()
