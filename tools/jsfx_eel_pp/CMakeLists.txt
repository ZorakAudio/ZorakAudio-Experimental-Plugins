cmake_minimum_required(VERSION 3.20)
project(jsfx_eel_pp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ZA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH "Repository root")
get_filename_component(ZA_ROOT "${ZA_ROOT}" ABSOLUTE)

set(WDL_EEL_DIR "${ZA_ROOT}/src/WDL/eel2")
set(WDL_C_SOURCES
  "${WDL_EEL_DIR}/nseel-compiler.c"
  "${WDL_EEL_DIR}/nseel-eval.c"
  "${WDL_EEL_DIR}/nseel-ram.c"
  "${WDL_EEL_DIR}/nseel-cfunc.c"
  "${WDL_EEL_DIR}/nseel-lextab.c"
  "${WDL_EEL_DIR}/nseel-caltab.c"
  "${WDL_EEL_DIR}/nseel-yylex.c"
)
if(WIN32)
  list(APPEND WDL_C_SOURCES "${ZA_ROOT}/src/WDL/win32_utf8.c")
endif()

add_executable(jsfx_eel_pp
  "${CMAKE_CURRENT_LIST_DIR}/jsfx_eel_pp.cpp"
  ${WDL_C_SOURCES}
)

set_source_files_properties(${WDL_C_SOURCES} PROPERTIES LANGUAGE C)

target_include_directories(jsfx_eel_pp PRIVATE "${ZA_ROOT}/src")
target_compile_definitions(jsfx_eel_pp PRIVATE EEL_TARGET_PORTABLE=1 WDL_FFT_REALSIZE=8)

if(MSVC)
  target_compile_options(jsfx_eel_pp PRIVATE /utf-8)
endif()
if(UNIX AND NOT APPLE)
  target_link_libraries(jsfx_eel_pp PRIVATE m)
endif()

# One deterministic path on single- and multi-config generators.
set(_out "${CMAKE_BINARY_DIR}/bin")
set_target_properties(jsfx_eel_pp PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${_out}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${_out}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${_out}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_out}"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${_out}"
)
