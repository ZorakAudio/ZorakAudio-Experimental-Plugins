cmake_minimum_required(VERSION 3.21)

set(PLUGIN_NAME "" CACHE STRING "Plugin display name")
set(PLUGIN_SLUG "" CACHE STRING "Target name (no spaces)")
set(PLUGIN_CODE "" CACHE STRING "4-char plugin code")
set(MANUFACTURER_NAME "" CACHE STRING "Manufacturer name")
set(MANUFACTURER_CODE "" CACHE STRING "4-char manufacturer code")
set(BUNDLE_ID "" CACHE STRING "macOS bundle identifier")
set(PLUGIN_VERSION "0.0.0" CACHE STRING "Version string")
set(PLUGIN_DSP "" CACHE FILEPATH "Path to .dsp file")

set(ZA_ROOT "" CACHE PATH "Repo root")
set(ZA_ENABLE_CLAP OFF CACHE BOOL "Build CLAP target too")
set(CLAP_ID "" CACHE STRING "Unique CLAP id")
set(CLAP_FEATURES "audio-effect" CACHE STRING "Space-separated CLAP features")

if(NOT PLUGIN_NAME OR NOT PLUGIN_SLUG OR NOT PLUGIN_CODE OR NOT MANUFACTURER_NAME OR NOT MANUFACTURER_CODE OR NOT PLUGIN_DSP)
  message(FATAL_ERROR "Missing required args. Need PLUGIN_NAME, PLUGIN_SLUG, PLUGIN_CODE, MANUFACTURER_NAME, MANUFACTURER_CODE, PLUGIN_DSP.")
endif()

project(${PLUGIN_SLUG} VERSION ${PLUGIN_VERSION} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ZA_ROOT)
  get_filename_component(ZA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()

# ---- JUCE (required as submodule) ----
if(NOT EXISTS "${ZA_ROOT}/libs/JUCE/CMakeLists.txt")
  message(FATAL_ERROR "JUCE not found at libs/JUCE. Add it as a submodule and run: git submodule update --init --recursive")
endif()
add_subdirectory("${ZA_ROOT}/libs/JUCE" juce)

# ---- CLAP extensions (optional) ----
if(ZA_ENABLE_CLAP)
  if(NOT EXISTS "${ZA_ROOT}/libs/clap-juce-extensions/CMakeLists.txt")
    message(FATAL_ERROR "clap-juce-extensions not found at libs/clap-juce-extensions. Add it as a submodule and run: git submodule update --init --recursive")
  endif()
  add_subdirectory("${ZA_ROOT}/libs/clap-juce-extensions" clap_juce_extensions EXCLUDE_FROM_ALL)

  if(NOT CLAP_ID)
    message(FATAL_ERROR "ZA_ENABLE_CLAP is ON but CLAP_ID is empty")
  endif()
endif()

# ---- Faust codegen ----
set(FAUST_OUT  "${CMAKE_CURRENT_BINARY_DIR}/FaustDSP.h")

add_custom_command(
  OUTPUT "${FAUST_OUT}"
  COMMAND faust -lang cpp -i -cn mydsp -o "${FAUST_OUT}" "${PLUGIN_DSP}"
  DEPENDS "${PLUGIN_DSP}" "${FAUST_ARCH}"
  COMMENT "Faust: generating ${FAUST_OUT}"
  VERBATIM
)

add_custom_target(${PLUGIN_SLUG}_faust DEPENDS "${FAUST_OUT}")

# ---- JUCE plugin ----
juce_add_plugin(${PLUGIN_SLUG}
  PRODUCT_NAME "${PLUGIN_SLUG}"
  COMPANY_NAME "${MANUFACTURER_NAME}"
  BUNDLE_ID "${BUNDLE_ID}"
  PLUGIN_MANUFACTURER_CODE "${MANUFACTURER_CODE}"
  PLUGIN_CODE "${PLUGIN_CODE}"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE
  FORMATS VST3
)

target_sources(${PLUGIN_SLUG}
  PRIVATE
    "${ZA_ROOT}/src/FaustJuceProcessor.cpp"
    "${FAUST_OUT}"
)

target_include_directories(${PLUGIN_SLUG}
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
)

target_compile_definitions(${PLUGIN_SLUG}
  PRIVATE
    ZA_PLUGIN_NAME="${PLUGIN_NAME}"
)

target_link_libraries(${PLUGIN_SLUG}
  PRIVATE
    juce::juce_audio_utils
    juce::juce_audio_processors
    juce::juce_gui_extra
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
)

target_compile_definitions(${PLUGIN_SLUG}
  PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
)

target_compile_definitions(${PLUGIN_SLUG}
  PRIVATE
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_compile_definitions(${PLUGIN_SLUG}
  PRIVATE
    JUCE_VST3_NO_VST2_STEINBERG_ID_CONFLICT=1
)



add_dependencies(${PLUGIN_SLUG} ${PLUGIN_SLUG}_faust)

# ---- CLAP target (creates <target>_CLAP) ----
if(ZA_ENABLE_CLAP)
  separate_arguments(_clap_features NATIVE_COMMAND "${CLAP_FEATURES}")

  clap_juce_extensions_plugin(
    TARGET ${PLUGIN_SLUG}
    CLAP_ID "${CLAP_ID}"
    CLAP_FEATURES ${_clap_features}
  )
endif()
