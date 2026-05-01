cmake_minimum_required(VERSION 3.21)

set(PLUGIN_NAME "" CACHE STRING "Plugin display name")
set(PLUGIN_ARTIFACT_NAME "" CACHE STRING "Filesystem-safe plugin file/bundle basename")
set(PLUGIN_SLUG "" CACHE STRING "Target name (no spaces)")
set(PLUGIN_CODE "" CACHE STRING "4-char plugin code")
set(MANUFACTURER_NAME "" CACHE STRING "Manufacturer name")
set(MANUFACTURER_CODE "" CACHE STRING "4-char manufacturer code")
set(BUNDLE_ID "" CACHE STRING "macOS bundle identifier")
set(PLUGIN_VERSION "0.0.0" CACHE STRING "Version string")
set(PLUGIN_DSP "" CACHE FILEPATH "Path to .dsp file")
set(PLUGIN_TYPE "faust" CACHE STRING "Plugin type: faust or jsfx")
set(PLUGIN_JSFX_OBJ "" CACHE FILEPATH "Path to JSFX AOT object file (.o/.obj)")
set(PLUGIN_KIND "audio_effect" CACHE STRING "Logical plugin kind: audio_effect, midi_effect, instrument, hybrid")
set(PLUGIN_IS_SYNTH OFF CACHE BOOL "Whether JUCE should classify the plugin as a synth/instrument")
set(PLUGIN_NEEDS_MIDI_INPUT OFF CACHE BOOL "Whether the plugin wants MIDI input")
set(PLUGIN_NEEDS_MIDI_OUTPUT OFF CACHE BOOL "Whether the plugin produces MIDI output")
set(PLUGIN_IS_MIDI_EFFECT OFF CACHE BOOL "Whether JUCE should classify the plugin as a MIDI effect")

set(ZA_JSFX_CORRECTNESS_CHECK OFF CACHE BOOL "Enable JSFX shadow-EEL2 correctness monitor/instrumentation")

set(ZA_ROOT "" CACHE PATH "Repo root")
set(ZA_ENABLE_CLAP OFF CACHE BOOL "Build CLAP target too")
set(CLAP_ID "" CACHE STRING "Unique CLAP id")
set(CLAP_FEATURES "audio-effect" CACHE STRING "Space-separated CLAP features")

if(NOT PLUGIN_NAME OR NOT PLUGIN_SLUG OR NOT PLUGIN_CODE OR NOT MANUFACTURER_NAME OR NOT MANUFACTURER_CODE)
  message(FATAL_ERROR "Missing required args. Need PLUGIN_NAME, PLUGIN_SLUG, PLUGIN_CODE, MANUFACTURER_NAME, MANUFACTURER_CODE.")
endif()

if(NOT PLUGIN_ARTIFACT_NAME)
  set(PLUGIN_ARTIFACT_NAME "${PLUGIN_SLUG}")
endif()

if(PLUGIN_TYPE STREQUAL "faust")
  if(NOT PLUGIN_DSP)
    message(FATAL_ERROR "PLUGIN_TYPE=faust but PLUGIN_DSP is empty")
  endif()
elseif(PLUGIN_TYPE STREQUAL "jsfx")
  if(NOT PLUGIN_JSFX_OBJ)
    message(FATAL_ERROR "PLUGIN_TYPE=jsfx but PLUGIN_JSFX_OBJ is empty")
  endif()
else()
  message(FATAL_ERROR "Invalid PLUGIN_TYPE. Expected 'faust' or 'jsfx'.")
endif()

set(ZA_EFFECTIVE_PLUGIN_KIND "${PLUGIN_KIND}")
set(ZA_EFFECTIVE_PLUGIN_IS_SYNTH ${PLUGIN_IS_SYNTH})
set(ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_INPUT ${PLUGIN_NEEDS_MIDI_INPUT})
set(ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_OUTPUT ${PLUGIN_NEEDS_MIDI_OUTPUT})
set(ZA_EFFECTIVE_PLUGIN_IS_MIDI_EFFECT ${PLUGIN_IS_MIDI_EFFECT})

if(PLUGIN_TYPE STREQUAL "jsfx")
  get_filename_component(ZA_JSFX_AOT_DIR "${PLUGIN_JSFX_OBJ}" DIRECTORY)
  set(ZA_JSFX_AOT_HEADER "${ZA_JSFX_AOT_DIR}/JSFXDSP.h")

  if(EXISTS "${ZA_JSFX_AOT_HEADER}")
    file(READ "${ZA_JSFX_AOT_HEADER}" ZA_JSFX_AOT_HEADER_TEXT)

    string(REGEX MATCH "#define[ 	]+DSPJSFX_ACCEPTS_MIDI_INPUT[ 	]+([0-9]+)" ZA_JSFX_ACCEPTS_MIDI_MATCH "${ZA_JSFX_AOT_HEADER_TEXT}")
    if(ZA_JSFX_ACCEPTS_MIDI_MATCH)
      if(CMAKE_MATCH_1 STREQUAL "0")
        set(ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_INPUT OFF)
      else()
        set(ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_INPUT ON)
      endif()
    endif()

    string(REGEX MATCH "#define[ 	]+DSPJSFX_PRODUCES_MIDI_OUTPUT[ 	]+([0-9]+)" ZA_JSFX_PRODUCES_MIDI_MATCH "${ZA_JSFX_AOT_HEADER_TEXT}")
    if(ZA_JSFX_PRODUCES_MIDI_MATCH)
      if(CMAKE_MATCH_1 STREQUAL "0")
        set(ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_OUTPUT OFF)
      else()
        set(ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_OUTPUT ON)
      endif()
    endif()

    string(REGEX MATCH "#define[ \t]+DSPJSFX_PLUGIN_KIND[ \t]+\"([^\"]+)\"" ZA_JSFX_KIND_MATCH "${ZA_JSFX_AOT_HEADER_TEXT}")
    if(ZA_JSFX_KIND_MATCH)
      set(ZA_EFFECTIVE_PLUGIN_KIND "${CMAKE_MATCH_1}")
    endif()

    string(TOLOWER "${ZA_EFFECTIVE_PLUGIN_KIND}" ZA_EFFECTIVE_PLUGIN_KIND_NORMALIZED)
    if(ZA_EFFECTIVE_PLUGIN_KIND_NORMALIZED STREQUAL "instrument")
      set(ZA_EFFECTIVE_PLUGIN_IS_SYNTH ON)
      set(ZA_EFFECTIVE_PLUGIN_IS_MIDI_EFFECT OFF)
    elseif(ZA_EFFECTIVE_PLUGIN_KIND_NORMALIZED STREQUAL "midi_effect")
      set(ZA_EFFECTIVE_PLUGIN_IS_SYNTH OFF)
      set(ZA_EFFECTIVE_PLUGIN_IS_MIDI_EFFECT ON)
    else()
      set(ZA_EFFECTIVE_PLUGIN_IS_SYNTH OFF)
      set(ZA_EFFECTIVE_PLUGIN_IS_MIDI_EFFECT OFF)
    endif()

    message(STATUS "JSFX AOT inferred capabilities: kind=${ZA_EFFECTIVE_PLUGIN_KIND} midi_in=${ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_INPUT} midi_out=${ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_OUTPUT} synth=${ZA_EFFECTIVE_PLUGIN_IS_SYNTH} midi_effect=${ZA_EFFECTIVE_PLUGIN_IS_MIDI_EFFECT}")
  else()
    message(STATUS "JSFX AOT header not found at ${ZA_JSFX_AOT_HEADER}; using PLUGIN_* CMake flags as provided")
  endif()
endif()

project(${PLUGIN_SLUG} VERSION ${PLUGIN_VERSION} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  # Repository-authored UI strings are UTF-8. Make MSVC preserve the
  # source/execution encoding so explicit juce::String::fromUTF8() calls
  # receive UTF-8 bytes instead of the active Windows code page.
  add_compile_options(/utf-8)
endif()

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

if(PLUGIN_TYPE STREQUAL "faust")
  # ---- Faust tool ----
  find_program(FAUST_CMD faust REQUIRED)
  message(STATUS "Using FAUST_CMD=${FAUST_CMD}")

  # ---- Faust codegen ----
  set(FAUST_OUT  "${CMAKE_CURRENT_BINARY_DIR}/FaustDSP.h")

  add_custom_command(
    OUTPUT "${FAUST_OUT}"
    COMMAND "${FAUST_CMD}" -lang cpp -i -cn mydsp -o "${FAUST_OUT}" "${PLUGIN_DSP}"
    DEPENDS "${PLUGIN_DSP}"
    COMMENT "Faust: generating ${FAUST_OUT}"
    VERBATIM
  )

  add_custom_target(${PLUGIN_SLUG}_faust DEPENDS "${FAUST_OUT}")
endif()

# ---- JUCE plugin ----
juce_add_plugin(${PLUGIN_SLUG}
  PRODUCT_NAME "${PLUGIN_ARTIFACT_NAME}"
  PLUGIN_NAME "${PLUGIN_NAME}"
  DESCRIPTION "${PLUGIN_NAME}"
  COMPANY_NAME "${MANUFACTURER_NAME}"
  BUNDLE_ID "${BUNDLE_ID}"
  PLUGIN_MANUFACTURER_CODE "${MANUFACTURER_CODE}"
  PLUGIN_CODE "${PLUGIN_CODE}"
  IS_SYNTH ${ZA_EFFECTIVE_PLUGIN_IS_SYNTH}
  NEEDS_MIDI_INPUT ${ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_INPUT}
  NEEDS_MIDI_OUTPUT ${ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_OUTPUT}
  IS_MIDI_EFFECT ${ZA_EFFECTIVE_PLUGIN_IS_MIDI_EFFECT}
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  COPY_PLUGIN_AFTER_BUILD FALSE
  FORMATS VST3
)

if(PLUGIN_TYPE STREQUAL "faust")
  target_sources(${PLUGIN_SLUG}
    PRIVATE
      "${ZA_ROOT}/src/FaustJuceProcessor.cpp"
      "${FAUST_OUT}"
  )
else()
  # JSFX: JUCE processor + WDL/EEL2 (C) sources.
  target_sources(${PLUGIN_SLUG}
    PRIVATE
      "${ZA_ROOT}/src/JSFXJuceProcessor.cpp"
      "${ZA_ROOT}/src/NSEELHostStubs.cpp"
      "${ZA_ROOT}/src/DspJsfxRuntime.cpp"
      "${ZA_ROOT}/src/DspJsfxRuntimeBuiltins.cpp"
      "${ZA_ROOT}/src/DspJsfxMessageBus.cpp"
      "${ZA_ROOT}/src/DspJsfxGmem.cpp"
      "${ZA_ROOT}/src/DspJsfxSamplePool.cpp"
      "${ZA_ROOT}/src/DspJsfxSharedMemory.cpp"
      "${PLUGIN_JSFX_OBJ}"
  )

  # WDL/EEL2 C sources (must compile as C, not C++).
  set(WDL_C_SOURCES
    "${ZA_ROOT}/src/WDL/fft.c"
    "${ZA_ROOT}/src/WDL/eel2/nseel-compiler.c"
    "${ZA_ROOT}/src/WDL/eel2/nseel-eval.c"
    "${ZA_ROOT}/src/WDL/eel2/nseel-ram.c"
    "${ZA_ROOT}/src/WDL/eel2/nseel-cfunc.c"
    "${ZA_ROOT}/src/WDL/eel2/nseel-lextab.c"
    "${ZA_ROOT}/src/WDL/eel2/nseel-caltab.c"
    "${ZA_ROOT}/src/WDL/eel2/nseel-yylex.c"
  )

  if (WIN32)
    list(APPEND WDL_C_SOURCES
      "${ZA_ROOT}/src/WDL/win32_utf8.c"
    )
  endif()

  target_sources(${PLUGIN_SLUG} PRIVATE ${WDL_C_SOURCES})

  # Force MSVC/VS generators to treat these as C.
  set_source_files_properties(${WDL_C_SOURCES} PROPERTIES LANGUAGE C)

  target_include_directories(${PLUGIN_SLUG}
    PRIVATE
      "${ZA_ROOT}/src"
      "${CMAKE_CURRENT_BINARY_DIR}"
  )

  target_compile_definitions(${PLUGIN_SLUG}
    PRIVATE
      EEL_TARGET_PORTABLE=1
      WDL_FFT_REALSIZE=8
      DSPJSFX_IPC_ENABLE=1

      EELSCRIPT_NO_FILE=1
      EELSCRIPT_NO_NET=1
      EELSCRIPT_NO_MDCT=1
      EELSCRIPT_NO_PREPROC=1
      EELSCRIPT_NO_LICE=1
      # DO NOT define EELSCRIPT_NO_EVAL
      ZA_JSFX_CORRECTNESS_CHECK=$<BOOL:${ZA_JSFX_CORRECTNESS_CHECK}>
  )
endif()

target_include_directories(${PLUGIN_SLUG}
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}"
    # So includes like "WDL/eel2/ns-eel.h" resolve from "${ZA_ROOT}/src/WDL/..."
    "${ZA_ROOT}/src"
)

target_compile_definitions(${PLUGIN_SLUG}
  PRIVATE
    ZA_PLUGIN_NAME="${PLUGIN_NAME}"
    ZA_PLUGIN_KIND="${ZA_EFFECTIVE_PLUGIN_KIND}"
    ZA_PLUGIN_WANTS_MIDI_INPUT=$<BOOL:${ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_INPUT}>
    ZA_PLUGIN_WANTS_MIDI_OUTPUT=$<BOOL:${ZA_EFFECTIVE_PLUGIN_NEEDS_MIDI_OUTPUT}>
    ZA_PLUGIN_IS_MIDI_EFFECT=$<BOOL:${ZA_EFFECTIVE_PLUGIN_IS_MIDI_EFFECT}>
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

if(PLUGIN_TYPE STREQUAL "faust")
  add_dependencies(${PLUGIN_SLUG} ${PLUGIN_SLUG}_faust)
endif()

# ---- CLAP target (creates <target>_CLAP) ----
if(ZA_ENABLE_CLAP)
  separate_arguments(_clap_features NATIVE_COMMAND "${CLAP_FEATURES}")

  clap_juce_extensions_plugin(
    TARGET ${PLUGIN_SLUG}
    CLAP_ID "${CLAP_ID}"
    CLAP_FEATURES ${_clap_features}
  )
endif()
if(PLUGIN_TYPE STREQUAL "jsfx" AND UNIX AND NOT APPLE)
  target_link_libraries(${PLUGIN_SLUG} PRIVATE rt)
endif()
