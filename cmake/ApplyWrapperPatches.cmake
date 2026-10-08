# Both production AOT and the standalone JIT editor use this dependency setup.
# Run at configure AND build time so a reverted dependency in a reused build
# directory is repaired before its wrapper is compiled.
include_guard(GLOBAL)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(ZA_WRAPPER_PATCH_SCRIPT "${ZA_ROOT}/tools/jit_editor/apply_wrapper_patches.py")
set(ZA_WRAPPER_PATCH_ARGS)
if(NOT ZA_ENABLE_CLAP)
  list(APPEND ZA_WRAPPER_PATCH_ARGS --only juce)
endif()
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${ZA_WRAPPER_PATCH_SCRIPT}" ${ZA_WRAPPER_PATCH_ARGS}
  RESULT_VARIABLE ZA_WRAPPER_PATCH_RESULT
)
if(NOT ZA_WRAPPER_PATCH_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot prepare shared host wrappers; see patch diagnostic above")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${ZA_WRAPPER_PATCH_SCRIPT}"
  "${ZA_ROOT}/tools/jit_editor/wrapper-patches/juce.patch"
  "${ZA_ROOT}/tools/jit_editor/wrapper-patches/clap.patch"
)
add_custom_target(za_wrapper_patches
  COMMAND "${Python3_EXECUTABLE}" "${ZA_WRAPPER_PATCH_SCRIPT}" ${ZA_WRAPPER_PATCH_ARGS}
  COMMENT "Checking shared AOT/JIT host wrapper patches"
  VERBATIM
)
