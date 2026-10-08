cmake_minimum_required(VERSION 3.24)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
project(JITEditor VERSION 0.1.0 LANGUAGES C CXX)
if(NOT WIN32)
  message(FATAL_ERROR "The JIT Editor PoC currently targets Windows x64 only")
endif()
set(CMAKE_CXX_STANDARD 20)
get_filename_component(ZA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/apply_wrapper_patches.py" RESULT_VARIABLE JIT_PATCH_RESULT)
if(NOT JIT_PATCH_RESULT EQUAL 0)
 message(FATAL_ERROR "Could not apply the JIT-only wrapper hooks; see diagnostic above")
endif()

add_subdirectory("${ZA_ROOT}/libs/JUCE" juce)
add_subdirectory("${ZA_ROOT}/libs/clap-juce-extensions" clap_juce_extensions EXCLUDE_FROM_ALL)
juce_add_plugin(JITEditor
  PRODUCT_NAME "ZorakAudio JIT Editor PoC"
  COMPANY_NAME "ZorakAudio" BUNDLE_ID "com.zorakaudio.experimental.jiteditor"
  PLUGIN_MANUFACTURER_CODE Zrak PLUGIN_CODE JtEd
  IS_SYNTH FALSE NEEDS_MIDI_INPUT TRUE NEEDS_MIDI_OUTPUT TRUE
  EDITOR_WANTS_KEYBOARD_FOCUS TRUE COPY_PLUGIN_AFTER_BUILD FALSE FORMATS VST3)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/JitBridge.h"
 COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_bridge.py" "${CMAKE_CURRENT_BINARY_DIR}/JitBridge.h"
 DEPENDS generate_bridge.py "${ZA_ROOT}/dsp_jsfx_aot.py")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/JSFXDSP.h" "#pragma once\n#include \"JitBridge.h\"\n")
target_sources(JITEditor PRIVATE "${ZA_ROOT}/src/DspJsfxRuntime.cpp" "${ZA_ROOT}/src/DspJsfxRuntimeBuiltins.cpp" "${ZA_ROOT}/src/DspJsfxSamplePool.cpp" "${ZA_ROOT}/src/DspJsfxSharedMemory.cpp" "${ZA_ROOT}/src/DspJsfxGmem.cpp" "${ZA_ROOT}/src/DspJsfxMessageBus.cpp" JitEngine.cpp JitProcessor.cpp "${CMAKE_CURRENT_BINARY_DIR}/JitBridge.h")
file(GLOB JIT_EXAMPLE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.jsfx" "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.dsp")
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/JitExamples.h"
 COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_examples.py" "${CMAKE_CURRENT_BINARY_DIR}/JitExamples.h"
 DEPENDS generate_examples.py examples/manifest.json ${JIT_EXAMPLE_FILES})
target_sources(JITEditor PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/JitExamples.h")
target_include_directories(JITEditor PUBLIC "${ZA_ROOT}/src" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(JITEditor PRIVATE ZA_JIT_DYNAMIC_GFX=1 EEL_TARGET_PORTABLE=1 WDL_FFT_REALSIZE=8)
set(WDL_SOURCES)
foreach(name nseel-compiler nseel-eval nseel-ram nseel-cfunc nseel-lextab nseel-caltab nseel-yylex)
  list(APPEND WDL_SOURCES "${ZA_ROOT}/src/WDL/eel2/${name}.c")
endforeach()
target_sources(JITEditor PRIVATE ${WDL_SOURCES} "${ZA_ROOT}/src/WDL/fft.c" "${ZA_ROOT}/src/WDL/win32_utf8.c")
target_compile_definitions(JITEditor PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(JITEditor PRIVATE juce::juce_audio_utils juce::juce_gui_extra PUBLIC juce::juce_recommended_config_flags)
clap_juce_extensions_plugin(TARGET JITEditor CLAP_ID "com.zorakaudio.experimental.jiteditor" CLAP_FEATURES audio-effect)
target_compile_definitions(JITEditor_CLAP PRIVATE ZA_JIT_EDITOR_DYNAMIC_HOST=1)
target_compile_definitions(JITEditor_VST3 PRIVATE ZA_JIT_EDITOR_DYNAMIC_HOST=1)
target_include_directories(JITEditor_VST3 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(jit_editor_check JitCheck.cpp)
target_link_libraries(jit_editor_check PRIVATE JITEditor juce::juce_audio_utils)
target_compile_definitions(jit_editor_check PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
add_executable(jit_editor_bank_check JitCheck.cpp)
target_link_libraries(jit_editor_bank_check PRIVATE JITEditor juce::juce_audio_utils)
target_compile_definitions(jit_editor_bank_check PRIVATE JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
add_executable(jit_editor_clap_check ClapCheck.cpp)
target_include_directories(jit_editor_clap_check PRIVATE "${ZA_ROOT}/libs/clap-juce-extensions/clap-libs/clap/include")

set(VSTSDK "${ZA_ROOT}/libs/JUCE/modules/juce_audio_processors_headless/format_types/VST3_SDK")
add_executable(jit_editor_vst3_check Vst3Check.cpp "${VSTSDK}/pluginterfaces/base/coreiids.cpp" "${VSTSDK}/pluginterfaces/base/funknown.cpp" "${VSTSDK}/public.sdk/source/vst/vstinitiids.cpp")
target_include_directories(jit_editor_vst3_check PRIVATE "${VSTSDK}")
target_link_libraries(jit_editor_vst3_check PRIVATE ole32)
